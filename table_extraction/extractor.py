"""
PDF Text Extraction and Classification Pipeline (MARKDOWN + FOOTNOTES UPGRADE)
Context: Multimodal RAG – Thesis Development
"""

import fitz  # PyMuPDF
import pymupdf4llm
import re
from collections import Counter
from typing import List, Dict
import os
import json
from pdf2image import convert_from_path
import pytesseract

PDF_DIR = "../data/PDFs"
OUTPUT_DIR = "../data/extracted"

# --- LISTAS DE PALAVRAS-CHAVE ---
FRONT_MATTER_BLACKLIST = {
    "agradecimentos", "acknowledgements", "acknowledgments",
    "dedicatória", "dedication", "lista de figuras", "list of figures",
    "lista de tabelas", "list of tables", "lista de abreviaturas", "list of abbreviations",
    "lista de siglas", "list of acronyms", "declaração", "declaração de integridade", 
    "declaration", "folha de rosto", "title page", "epígrafe", "epigraph"
}

APPENDIX_KEYWORDS = {
    "apêndice", "apêndices", "appendix", "appendices",
    "anexo", "anexos", "annexe", "annexes", "attachments"
}

KEYWORDS = {
    "abstract": ["abstract", "resumo", "résumé", "zusammenfassung", "resumen"],
    "references": ["references", "referências", "bibliografia", "bibliografía"],
    "toc": ["table of contents", "contents", "índice", "sumário"]
}

# ---------------------------
# 1. EXTRAÇÃO DE TEXTO E CLASSIFICAÇÃO
# ---------------------------

def extract_text_from_pdf_markdown(pdf_path: str) -> List[Dict]:
    """Extrai texto com pymupdf4llm, forçando os números de página corretos."""
    md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages = []
    
    for i, md_page in enumerate(md_pages):
        # GARANTIA ABSOLUTA de que a página 1 é a 1, a 2 é a 2...
        pages.append({
            "page_num": i + 1, 
            "text": md_page.get("text", "")
        })
        
    return pages

def extract_text_from_scanned_pdf(pdf_path, languages="eng+por", dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    full_text = []
    for page_num, image in enumerate(pages):
        text = pytesseract.image_to_string(image, lang=languages, config="--psm 1")
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        full_text.append(f"\n# PAGE {page_num + 1}\n{text}")
    return "\n".join(full_text)

def is_scanned_pdf(pages_text: List[Dict], min_chars: int = 1000) -> bool:
    total_chars = sum(len(p["text"]) for p in pages_text)
    return total_chars < min_chars

def has_keyword(text: str, keyword_type: str) -> bool:
    return any(kw in text for kw in KEYWORDS.get(keyword_type, []))

def classify_pdf(pages_text: List[Dict]) -> str:
    """Classifica com base em TODO o texto extraído para não falhar os artigos."""
    if is_scanned_pdf(pages_text): return "scanned"

    total_pages = len(pages_text)
    total_words = sum(len(p["text"].split()) for p in pages_text)
    avg_words_per_page = total_words / max(total_pages, 1)

    full_text = " ".join(p["text"].lower() for p in pages_text)

    if avg_words_per_page < 60: return "slides"
    if has_keyword(full_text, "toc") and total_pages > 30: return "thesis"
    if has_keyword(full_text, "abstract") and has_keyword(full_text, "references"): return "article"
    
    return "report"

# ---------------------------
# 2. EXTRAÇÃO DE FOOTNOTES (MÉTODO CLÁSSICO)
# ---------------------------

def extract_footnotes_only(pdf_path: str) -> str:
    """Corre o teu algoritmo original APENAS para capturar as notas de rodapé no fundo da página."""
    doc = fitz.open(pdf_path)
    clean_footnotes = []
    FOOTNOTE_START_RE = re.compile(r'^\s*(\(?\d+\)?|[\*\†\‡])[\.\)\s]\s*[A-ZÀ-Úa-z]')

    for page in doc:
        page_dict = page.get_text("dict")
        page_height = page.rect.height

        sizes = []
        for b in page_dict.get("blocks", []):
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        if s["text"].strip(): sizes.append(round(s["size"], 1))
        body_font_size = Counter(sizes).most_common(1)[0][0] if sizes else 11.0

        drawings = page.get_drawings()
        min_y = page_height * 0.66
        candidates = [d["rect"].y0 for d in drawings if d["rect"].y0 > min_y and d["rect"].height < 5 and d["rect"].width > 50]
        footnote_zone_y = min(candidates) if candidates else (page_height * 0.85)

        footnote_candidates = []
        for b in page_dict.get("blocks", []):
            if "lines" not in b: continue
            for l in b["lines"]:
                l_y0 = l["bbox"][1]
                full_text = " ".join([s["text"] for s in l["spans"]]).strip()
                if not full_text or re.match(r'^\s*\d+\s*$', full_text): continue
                valid_sizes = [s["size"] for s in l["spans"] if s["text"].strip()]
                avg_size = sum(valid_sizes)/len(valid_sizes) if valid_sizes else body_font_size

                if l_y0 >= footnote_zone_y and avg_size <= (body_font_size - 0.5):
                    footnote_candidates.append({"text": full_text, "y0": l_y0})

        footnote_candidates.sort(key=lambda x: x["y0"])
        current_note = []
        active = False
        for item in footnote_candidates:
            if FOOTNOTE_START_RE.match(item["text"]):
                if current_note: clean_footnotes.append(" ".join(current_note))
                current_note = [item["text"]]; active = True
            elif active: current_note.append(item["text"])
        if current_note: clean_footnotes.append(" ".join(current_note))

    return "\n\n".join(clean_footnotes)

# ---------------------------
# 3. LIMPEZA DE DADOS
# ---------------------------

def remove_front_matter_pages(pages: List[Dict], doc_type: str) -> List[Dict]:
    if doc_type == "article": return pages
    cleaned_pages = []
    
    for i, page in enumerate(pages):
        if i >= 15:
            cleaned_pages.append(page)
            continue

        lines = page["text"].strip().splitlines()
        if not lines: continue
        
        header_text = " ".join(lines[:3]).lower().replace("#", "").strip()
        if any(kw in header_text for kw in FRONT_MATTER_BLACKLIST):
            print(f"  -> [Front-Matter] Removida Pag {page['page_num']}")
            continue
            
        cleaned_pages.append(page)
    return cleaned_pages

def remove_toc_pages(pages: List[Dict]) -> List[Dict]:
    TOC_LINK = re.compile(r'\[.*?\]\(#page=\d+\)')
    # NOVO REGEX: Apanha números ou numeração romana, e ignora os símbolos da Tabela Markdown no fim da linha
    TOC_DOTS = re.compile(r'(\.{3,}|\s{4,})\s*(\d+|[ivxlcdmIVXLCDM]+)[\*\|\s]*$') 
    filtered_pages = []
    
    for page in pages:
        lines = page["text"].splitlines()
        if not lines: continue
        
        toc_lines = sum(1 for line in lines if TOC_LINK.search(line) or TOC_DOTS.search(line.strip()))
        ratio = toc_lines / len(lines) if len(lines) > 5 else 0
        
        if ratio > 0.15: 
            print(f"  -> [TOC] Removido Índice: Página {page['page_num']}")
            continue 
            
        filtered_pages.append(page)
    return filtered_pages

def truncate_appendices(pages: List[Dict]) -> List[Dict]:
    cutoff_index = -1
    total_pages = len(pages)
    search_start = int(total_pages * 0.5) # Só procura na SEGUNDA METADE
    
    for i in range(search_start, total_pages):
        lines = pages[i]["text"].strip().splitlines()
        for line in lines[:5]:
            if line.startswith("#"):
                clean_header = line.replace("#", "").strip().lower()
                if any(kw in clean_header for kw in APPENDIX_KEYWORDS) and len(clean_header) < 60:
                    print(f"  -> [Apêndices] Corte detetado na Pag {pages[i]['page_num']}.")
                    cutoff_index = i
                    break
        if cutoff_index != -1: break
            
    if cutoff_index != -1: return pages[:cutoff_index]
    return pages

def remove_references_section(pages: List[Dict]) -> List[Dict]:
    REF_HEADERS = ["referências", "referencias", "bibliografia", "bibliography", "references", "literaturverzeichnis"]
    cutoff_page_index = -1
    cutoff_line_index = -1
    total_pages = len(pages)
    search_limit = max(0, total_pages - int(total_pages * 0.4)) 
    
    for i in range(total_pages - 1, search_limit - 1, -1):
        lines = pages[i]["text"].splitlines()
        for j, line in enumerate(lines):
            if line.strip().startswith("#"):
                # SUPER LIMPEZA: Tira o # e todos os asteriscos (**) antes de comparar
                clean_line = line.replace("#", "").replace("*", "").strip().lower()
                clean_line = re.sub(r'^\d+(\.\d+)*\s+', '', clean_line)
                
                if clean_line in REF_HEADERS:
                    cutoff_page_index = i
                    cutoff_line_index = j
                    print(f"  -> [Ref] Referências detetadas na Pag {pages[i]['page_num']}. Cortando...")
                    break
        if cutoff_page_index != -1: break
            
    if cutoff_page_index == -1: return pages

    final_pages = pages[:cutoff_page_index]
    if cutoff_line_index > 0:
        content_lines = pages[cutoff_page_index]["text"].splitlines()[:cutoff_line_index]
        texto_limpo = "\n".join(content_lines).strip()
        if texto_limpo: # Só adiciona a página se houver texto ANTES das Referências
            final_pages.append({
                "page_num": pages[cutoff_page_index]["page_num"],
                "text": texto_limpo
            })
    return final_pages

def remove_repeated_headers_footers(pages: List[Dict], threshold=0.6) -> List[Dict]:
    line_counter = Counter()
    for page in pages:
        for line in page["text"].splitlines():
            clean = line.strip()
            if 3 < len(clean) < 150 and not clean.startswith("|") and not clean.startswith("#"):
                line_counter[clean] += 1

    total_pages = len(pages)
    repeated = {line for line, count in line_counter.items() if count / total_pages >= threshold}

    cleaned_pages = []
    for page in pages:
        body = [l for l in page["text"].splitlines() if l.strip() not in repeated]
        cleaned_pages.append({
            "page_num": page["page_num"],
            "text": "\n".join(body)
        })
    return cleaned_pages

# ---------------------------
# 4. TEXT NORMALISATION E CHUNKING
# ---------------------------

def normalize_text(text: str) -> str:
    # 1. Remover citações [1, 2] e (Autor, Ano)
    text = re.sub(r'\([^\)]*?\b(?:18|19|20)\d{2}\b[^\)]*?\)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\s*\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*\s*\]', '', text)
    
    # 2. NOVO: Destruir TODO o bloco "**----- Start of picture text -----**" e o seu conteúdo
    text = re.sub(r'\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*(?:<br>|\n)*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Remover a mensagem de imagens omitidas clássica
    text = re.sub(r'\*\*==>\s*picture.*?intentionally omitted\s*<==\*\*', '', text, flags=re.IGNORECASE)
    
    # 4. Remover hiperligações Markdown puras para imagens
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    return text

def split_into_sections_markdown(pages):
    sections = []
    current_title = "Início do Documento"
    current_text = []
    current_page = 1 

    for page in pages:
        page_num = page.get("page_num", 1)
        for line in page["text"].splitlines():
            if line.startswith("#"):
                if current_text:
                    sections.append({"title": current_title, "text": "\n".join(current_text), "page_num": current_page})
                current_title = line.replace("#", "").strip()
                current_text = [line] 
                current_page = page_num 
            else:
                if not current_text and current_title == "Início do Documento": current_page = page_num
                current_text.append(line)
    
    if current_text:
        sections.append({"title": current_title, "text": "\n".join(current_text), "page_num": current_page})
    return sections

def chunk_text_smart(text: str, chunk_size=300, overlap=50):
    """Corta texto protegendo o interior das Tabelas Markdown."""
    lines = text.splitlines()
    chunks = []
    current_chunk_words, current_word_count = [], 0
    in_table = False
    chunk_id = 0
    
    for line in lines:
        line_str = line.strip()
        words_in_line = line.split()
        
        if line_str.startswith("|") and line_str.endswith("|"): in_table = True
        elif in_table and not line_str.startswith("|"): in_table = False
            
        current_chunk_words.extend(words_in_line)
        current_chunk_words.append("\n")
        current_word_count += len(words_in_line)
        
        if current_word_count >= chunk_size and not in_table:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk_words).replace(" \n ", "\n"),
                "start_word": chunk_id * chunk_size, 
                "end_word": (chunk_id * chunk_size) + current_word_count
            })
            chunk_id += 1
            
            overlap_words, overlap_count = [], 0
            for l in reversed(lines[:lines.index(line)+1]):
                if overlap_count >= overlap: break
                w = l.split()
                overlap_words = w + ["\n"] + overlap_words
                overlap_count += len(w)
                
            current_chunk_words = overlap_words
            current_word_count = overlap_count

    if current_word_count > 0:
        chunks.append({
            "chunk_id": chunk_id, "text": " ".join(current_chunk_words).replace(" \n ", "\n"),
            "start_word": chunk_id * chunk_size, "end_word": (chunk_id * chunk_size) + current_word_count
        })
    return chunks

def chunk_sections(sections: List[Dict], chunk_size=300, overlap=50) -> List[Dict]:
    all_chunks = []
    global_chunk_id = 0
    for sec in sections:
        sec_chunks = chunk_text_smart(sec["text"], chunk_size, overlap)
        for ch in sec_chunks:
            all_chunks.append({
                "chunk_id": global_chunk_id, "section": sec["title"], "text": ch["text"],
                "start_word": ch["start_word"], "end_word": ch["end_word"],
                "type": "body", "page_num": sec.get("page_num", 1)
            })
            global_chunk_id += 1
    return all_chunks

def chunk_individual_footnotes(footnotes_text: str, chunk_size=200, overlap=0) -> List[Dict]:
    raw_notes = [n.strip() for n in footnotes_text.split('\n\n') if n.strip()]
    chunks = []
    global_chunk_id = 0
    for note in raw_notes:
        words = note.split()
        if len(words) <= chunk_size:
            chunks.append({
                "chunk_id": f"fn_{global_chunk_id}", "text": note,
                "type": "footnote_whole", "word_count": len(words)
            })
            global_chunk_id += 1
        else:
            sub_chunks = chunk_text_smart(note, chunk_size, overlap)
            for sc in sub_chunks:
                chunks.append({
                    "chunk_id": f"fn_{global_chunk_id}_{sc['chunk_id']}", "text": sc["text"],
                    "type": "footnote_part", "word_count": len(sc["text"].split())
                })
            global_chunk_id += 1
    return chunks

# ---------------------------
# 5. FULL PIPELINE
# ---------------------------

def extract_doi(text: str) -> str:
    match = re.search(r'\b(10\.\d{4,}/[^\s]+)', text)
    if match: return re.sub(r'[.,;:\]\)>]+$', '', match.group(1))
    return ""

def extract_pdf_metadata(pdf_path: str, pages_text: List[Dict] = None) -> Dict:
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    doi = ""
    if pages_text:
        doi = extract_doi(" ".join(p["text"] for p in pages_text[:5]))
    return {
        "filename": os.path.basename(pdf_path), "filepath": pdf_path,
        "title": meta.get("title", ""), "author": meta.get("author", ""),
        "total_pages": len(doc), "doi": doi
    }

def process_pdf(pdf_path: str) -> Dict:
    # 1️⃣ Extração Baseada em Markdown
    pages = extract_text_from_pdf_markdown(pdf_path)
    
    # 2️⃣ Classificação Completa (usando o texto todo)
    doc_type = classify_pdf(pages)
    metadata = extract_pdf_metadata(pdf_path, pages)

    if doc_type == "scanned":
        ocr_text = extract_text_from_scanned_pdf(pdf_path)
        pages = [{"page_num": i + 1, "text": text} for i, text in enumerate(ocr_text.split("\n# PAGE"))]
    else:
        # 3️⃣ Limpezas Inteligentes (com páginas bem numeradas)
        pages = remove_front_matter_pages(pages, doc_type)
        pages = remove_toc_pages(pages)
        pages = truncate_appendices(pages)
        pages = remove_references_section(pages)
        pages = remove_repeated_headers_footers(pages)

    # 4️⃣ Normalização (Apaga blocos de imagem e formatações indesejadas)
    for p in pages:
        p["text"] = normalize_text(p["text"])

    # 5️⃣ Split por Secções (Markdown #)
    sections = split_into_sections_markdown(pages)

    # 6️⃣ Chunking Protetor de Tabelas
    chunks = chunk_sections(sections, chunk_size=300, overlap=50)

    # 7️⃣ O Retorno Triunfal das Footnotes!
    footnotes_text = ""
    if doc_type != "scanned":
        footnotes_text = extract_footnotes_only(pdf_path)
    footnote_chunks = chunk_individual_footnotes(footnotes_text, chunk_size=200)

    return {
        "metadata": metadata,
        "document_type": doc_type,
        "chunks": chunks,
        "footnote_chunks": footnote_chunks
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in", PDF_DIR)
        return

    for pdf_name in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        print(f"\nProcessing: {pdf_name}")
        try:
            output = process_pdf(pdf_path)
            output_file = os.path.splitext(pdf_name)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_file)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print("  Document type:", output["document_type"])
            print("  Chunks criados:", len(output["chunks"]))
        except Exception as e:
            print(f"  ❌ Error processing {pdf_name}: {e}")

if __name__ == "__main__":
    main()