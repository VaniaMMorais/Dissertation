# """
# PDF Text Extraction Pipeline (MARKDOWN + FOOTNOTES + MULTIMODAL VISION 👁️)
# Context: Multimodal RAG – Thesis Development
# """

# import fitz  # PyMuPDF
# import pymupdf4llm
# import re
# from collections import Counter
# from typing import List, Dict
# import os
# import json
# from pdf2image import convert_from_path
# import pytesseract
# import time
# import google.generativeai as genai
# from PIL import Image
# from dotenv import load_dotenv

# # --- CONFIGURAÇÃO ---
# PDF_DIR = "../data/PDFs"
# OUTPUT_DIR = "../data/extracted"
# FIGURES_DIR = "../data/figures" # Pasta onde as imagens vão ser guardadas


# load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)
# vision_model = genai.GenerativeModel('gemini-2.5-flash')

# # --- LISTAS DE PALAVRAS-CHAVE ---
# FRONT_MATTER_BLACKLIST = {
#     "agradecimentos", "acknowledgements", "acknowledgments",
#     "dedicatória", "dedication", "lista de figuras", "list of figures",
#     "lista de tabelas", "list of tables", "lista de abreviaturas", "list of abbreviations",
#     "lista de siglas", "list of acronyms", "declaração", "declaração de integridade", 
#     "declaration", "folha de rosto", "title page", "epígrafe", "epigraph"
# }

# APPENDIX_KEYWORDS = {"apêndice", "apêndices", "appendix", "appendices", "anexo", "anexos", "annexe", "annexes", "attachments"}

# KEYWORDS = {
#     "abstract": ["abstract", "resumo", "résumé", "zusammenfassung", "resumen"],
#     "references": ["references", "referências", "bibliografia", "bibliografía"],
#     "toc": ["table of contents", "contents", "índice", "sumário"]
# }

# # ---------------------------
# # 1. EXTRAÇÃO DE TEXTO, IMAGENS E CLASSIFICAÇÃO
# # ---------------------------

# def extract_text_and_figures(pdf_path: str) -> List[Dict]:
#     """Extrai texto com pymupdf4llm e guarda as imagens na pasta figures."""
#     os.makedirs(FIGURES_DIR, exist_ok=True)
    
#     # write_images=True é o que extrai as imagens reais!
#     md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=True, image_path=FIGURES_DIR)
#     pages = []
    
#     for i, md_page in enumerate(md_pages):
#         pages.append({
#             "page_num": i + 1, 
#             "text": md_page.get("text", "")
#         })
        
#     return pages

# def process_images_with_vlm(pages: List[Dict]) -> tuple[List[Dict], List[Dict]]:
#     """Lê imagens, descreve-as com VLM (Gemini) e cria chunks de imagem."""
#     image_chunks = []
#     global_img_id = 0
#     img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    
#     for p in pages:
#         text = p["text"]
#         matches = img_pattern.findall(text)
        
#         for img_path in matches:
#             if not os.path.exists(img_path): continue
            
#             try:
#                 img = Image.open(img_path)
#                 # Ignorar imagens minúsculas (ex: logotipos, ícones de rodapé)
#                 if img.width < 150 or img.height < 150:
#                     text = img_pattern.sub('', text, count=1)
#                     img.close() # Fechamos o ficheiro na memória do Python
#                     os.remove(img_path) 
#                     continue
                
#                 print(f"    👁️ Gemini a analisar imagem (Pag {p['page_num']}): {os.path.basename(img_path)}...")
                
#                 # O Prompt com Contexto
#                 prompt = f"""
#                 You are a scientific and medical expert. Please describe this figure, graph or diagram in detail. 
#                 Identify all the numerical data, trends, categories and conclusions that can be seen. 
#                 Here is the text from the page where the image was embedded, to provide some context: 
#                 {text[:1500]}
#                 """
                
#                 response = vision_model.generate_content([prompt, img])
#                 descricao = response.text.strip()
                
#                 image_chunks.append({
#                     "chunk_id": f"img_{global_img_id}",
#                     "section": "Figura Extraída",
#                     "text": descricao,
#                     "image_path": img_path,
#                     "page_num": p["page_num"],
#                     "type": "image"
#                 })
#                 global_img_id += 1
#                 time.sleep(6) # Pausa para respeitar limite da API gratuita (15/minuto)
                
#             except Exception as e:
#                 print(f"    ⚠️ Erro no VLM para a imagem {os.path.basename(img_path)}: {e}")
#                 if 'img' in locals(): img.close()
#                 if os.path.exists(img_path): os.remove(img_path) # Apaga também se der erro!
            
#             # Limpar a hiperligação original para não sujar o texto
#             text = text.replace(f"![{img_path}]", "")
#             text = re.sub(r'!\[.*?\]\(' + re.escape(img_path) + r'\)', '', text)
            
#         p["text"] = text
        
#     return image_chunks, pages

# def extract_text_from_scanned_pdf(pdf_path, languages="eng+por", dpi=300):
#     pages = convert_from_path(pdf_path, dpi=dpi)
#     full_text = []
#     for page_num, image in enumerate(pages):
#         text = pytesseract.image_to_string(image, lang=languages, config="--psm 1")
#         text = re.sub(r"-\n", "", text)
#         text = re.sub(r"\n+", "\n", text)
#         text = re.sub(r"[ \t]+", " ", text).strip()
#         full_text.append(f"\n# PAGE {page_num + 1}\n{text}")
#     return "\n".join(full_text)

# def is_scanned_pdf(pages_text: List[Dict], min_chars: int = 1000) -> bool:
#     total_chars = sum(len(p["text"]) for p in pages_text)
#     return total_chars < min_chars

# def has_keyword(text: str, keyword_type: str) -> bool:
#     return any(kw in text for kw in KEYWORDS.get(keyword_type, []))

# def classify_pdf(pages_text: List[Dict]) -> str:
#     if is_scanned_pdf(pages_text): return "scanned"
#     total_pages = len(pages_text)
#     total_words = sum(len(p["text"].split()) for p in pages_text)
#     avg_words_per_page = total_words / max(total_pages, 1)
#     full_text = " ".join(p["text"].lower() for p in pages_text)
#     if avg_words_per_page < 60: return "slides"
#     if has_keyword(full_text, "toc") and total_pages > 30: return "thesis"
#     if has_keyword(full_text, "abstract") and has_keyword(full_text, "references"): return "article"
#     return "report"

# # ---------------------------
# # 2. EXTRAÇÃO DE FOOTNOTES
# # ---------------------------

# def extract_footnotes_only(pdf_path: str) -> str:
#     doc = fitz.open(pdf_path)
#     clean_footnotes = []
#     FOOTNOTE_START_RE = re.compile(r'^\s*(\(?\d+\)?|[\*\†\‡])[\.\)\s]\s*[A-ZÀ-Úa-z]')

#     for page in doc:
#         page_dict = page.get_text("dict")
#         page_height = page.rect.height
#         sizes = []
#         for b in page_dict.get("blocks", []):
#             if "lines" in b:
#                 for l in b["lines"]:
#                     for s in l["spans"]:
#                         if s["text"].strip(): sizes.append(round(s["size"], 1))
#         body_font_size = Counter(sizes).most_common(1)[0][0] if sizes else 11.0

#         drawings = page.get_drawings()
#         min_y = page_height * 0.66
#         candidates = [d["rect"].y0 for d in drawings if d["rect"].y0 > min_y and d["rect"].height < 5 and d["rect"].width > 50]
#         footnote_zone_y = min(candidates) if candidates else (page_height * 0.85)

#         footnote_candidates = []
#         for b in page_dict.get("blocks", []):
#             if "lines" not in b: continue
#             for l in b["lines"]:
#                 l_y0 = l["bbox"][1]
#                 full_text = " ".join([s["text"] for s in l["spans"]]).strip()
#                 if not full_text or re.match(r'^\s*\d+\s*$', full_text): continue
#                 valid_sizes = [s["size"] for s in l["spans"] if s["text"].strip()]
#                 avg_size = sum(valid_sizes)/len(valid_sizes) if valid_sizes else body_font_size
#                 if l_y0 >= footnote_zone_y and avg_size <= (body_font_size - 0.5):
#                     footnote_candidates.append({"text": full_text, "y0": l_y0})

#         footnote_candidates.sort(key=lambda x: x["y0"])
#         current_note = []
#         active = False
#         for item in footnote_candidates:
#             if FOOTNOTE_START_RE.match(item["text"]):
#                 if current_note: clean_footnotes.append(" ".join(current_note))
#                 current_note = [item["text"]]; active = True
#             elif active: current_note.append(item["text"])
#         if current_note: clean_footnotes.append(" ".join(current_note))

#     return "\n\n".join(clean_footnotes)

# # ---------------------------
# # 3. LIMPEZA DE DADOS
# # ---------------------------

# def remove_front_matter_pages(pages: List[Dict], doc_type: str) -> List[Dict]:
#     if doc_type == "article": return pages
#     cleaned_pages = []
#     for i, page in enumerate(pages):
#         if i >= 15:
#             cleaned_pages.append(page)
#             continue
#         lines = page["text"].strip().splitlines()
#         if not lines: continue
#         header_text = " ".join(lines[:3]).lower().replace("#", "").strip()
#         if any(kw in header_text for kw in FRONT_MATTER_BLACKLIST):
#             print(f"  -> [Front-Matter] Removida Pag {page['page_num']}")
#             continue
#         cleaned_pages.append(page)
#     return cleaned_pages

# def remove_toc_pages(pages: List[Dict]) -> List[Dict]:
#     TOC_LINK = re.compile(r'\[.*?\]\(#page=\d+\)')
#     TOC_DOTS = re.compile(r'(\.{3,}|\s{4,})\s*(\d+|[ivxlcdmIVXLCDM]+)[\*\|\s]*$') 
#     filtered_pages = []
#     for page in pages:
#         lines = page["text"].splitlines()
#         if not lines: continue
#         toc_lines = sum(1 for line in lines if TOC_LINK.search(line) or TOC_DOTS.search(line.strip()))
#         ratio = toc_lines / len(lines) if len(lines) > 5 else 0
#         if ratio > 0.15: 
#             print(f"  -> [TOC] Removido Índice: Página {page['page_num']}")
#             continue 
#         filtered_pages.append(page)
#     return filtered_pages

# def truncate_appendices(pages: List[Dict]) -> List[Dict]:
#     cutoff_index = -1
#     total_pages = len(pages)
#     search_start = int(total_pages * 0.5)
#     for i in range(search_start, total_pages):
#         lines = pages[i]["text"].strip().splitlines()
#         for line in lines[:5]:
#             if line.startswith("#"):
#                 clean_header = line.replace("#", "").strip().lower()
#                 if any(kw in clean_header for kw in APPENDIX_KEYWORDS) and len(clean_header) < 60:
#                     print(f"  -> [Apêndices] Corte detetado na Pag {pages[i]['page_num']}.")
#                     cutoff_index = i
#                     break
#         if cutoff_index != -1: break
#     if cutoff_index != -1: return pages[:cutoff_index]
#     return pages

# def remove_references_section(pages: List[Dict]) -> List[Dict]:
#     REF_HEADERS = ["referências", "referencias", "bibliografia", "bibliography", "references", "literaturverzeichnis"]
#     cutoff_page_index = -1
#     cutoff_line_index = -1
#     total_pages = len(pages)
#     search_limit = max(0, total_pages - int(total_pages * 0.4)) 
#     for i in range(total_pages - 1, search_limit - 1, -1):
#         lines = pages[i]["text"].splitlines()
#         for j, line in enumerate(lines):
#             if line.strip().startswith("#"):
#                 clean_line = line.replace("#", "").replace("*", "").strip().lower()
#                 clean_line = re.sub(r'^\d+(\.\d+)*\s+', '', clean_line)
#                 if clean_line in REF_HEADERS:
#                     cutoff_page_index = i
#                     cutoff_line_index = j
#                     print(f"  -> [Ref] Referências detetadas na Pag {pages[i]['page_num']}. Cortando...")
#                     break
#         if cutoff_page_index != -1: break
#     if cutoff_page_index == -1: return pages
#     final_pages = pages[:cutoff_page_index]
#     if cutoff_line_index > 0:
#         content_lines = pages[cutoff_page_index]["text"].splitlines()[:cutoff_line_index]
#         texto_limpo = "\n".join(content_lines).strip()
#         if texto_limpo:
#             final_pages.append({"page_num": pages[cutoff_page_index]["page_num"], "text": texto_limpo})
#     return final_pages

# def remove_repeated_headers_footers(pages: List[Dict], threshold=0.6) -> List[Dict]:
#     line_counter = Counter()
#     for page in pages:
#         for line in page["text"].splitlines():
#             clean = line.strip()
#             if 3 < len(clean) < 150 and not clean.startswith("|") and not clean.startswith("#"):
#                 line_counter[clean] += 1
#     total_pages = len(pages)
#     repeated = {line for line, count in line_counter.items() if count / total_pages >= threshold}
#     cleaned_pages = []
#     for page in pages:
#         body = [l for l in page["text"].splitlines() if l.strip() not in repeated]
#         cleaned_pages.append({"page_num": page["page_num"], "text": "\n".join(body)})
#     return cleaned_pages

# # ---------------------------
# # 4. TEXT NORMALISATION E CHUNKING
# # ---------------------------

# def normalize_text(text: str) -> str:
#     text = re.sub(r'\([^\)]*?\b(?:18|19|20)\d{2}\b[^\)]*?\)', '', text, flags=re.DOTALL)
#     text = re.sub(r'\[\s*\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*\s*\]', '', text)
#     text = re.sub(r'\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*(?:<br>|\n)*', '', text, flags=re.DOTALL | re.IGNORECASE)
#     text = re.sub(r'\*\*==>\s*picture.*?intentionally omitted\s*<==\*\*', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
#     return text

# def split_into_sections_markdown(pages):
#     sections = []
#     current_title = "Início do Documento"
#     current_text = []
#     current_page = 1 
#     for page in pages:
#         page_num = page.get("page_num", 1)
#         for line in page["text"].splitlines():
#             if line.startswith("#"):
#                 if current_text:
#                     sections.append({"title": current_title, "text": "\n".join(current_text), "page_num": current_page})
#                 current_title = line.replace("#", "").strip()
#                 current_text = [line] 
#                 current_page = page_num 
#             else:
#                 if not current_text and current_title == "Início do Documento": current_page = page_num
#                 current_text.append(line)
#     if current_text:
#         sections.append({"title": current_title, "text": "\n".join(current_text), "page_num": current_page})
#     return sections

# def chunk_text_smart(text: str, chunk_size=300, overlap=50):
#     lines = text.splitlines()
#     chunks = []
#     current_chunk_words, current_word_count = [], 0
#     in_table = False
#     chunk_id = 0
#     for line in lines:
#         line_str = line.strip()
#         words_in_line = line.split()
#         if line_str.startswith("|") and line_str.endswith("|"): in_table = True
#         elif in_table and not line_str.startswith("|"): in_table = False
#         current_chunk_words.extend(words_in_line)
#         current_chunk_words.append("\n")
#         current_word_count += len(words_in_line)
#         if current_word_count >= chunk_size and not in_table:
#             chunks.append({
#                 "chunk_id": chunk_id, "text": " ".join(current_chunk_words).replace(" \n ", "\n"),
#                 "start_word": chunk_id * chunk_size, "end_word": (chunk_id * chunk_size) + current_word_count
#             })
#             chunk_id += 1
#             overlap_words, overlap_count = [], 0
#             for l in reversed(lines[:lines.index(line)+1]):
#                 if overlap_count >= overlap: break
#                 w = l.split()
#                 overlap_words = w + ["\n"] + overlap_words
#                 overlap_count += len(w)
#             current_chunk_words = overlap_words
#             current_word_count = overlap_count
#     if current_word_count > 0:
#         chunks.append({
#             "chunk_id": chunk_id, "text": " ".join(current_chunk_words).replace(" \n ", "\n"),
#             "start_word": chunk_id * chunk_size, "end_word": (chunk_id * chunk_size) + current_word_count
#         })
#     return chunks

# def chunk_sections(sections: List[Dict], chunk_size=300, overlap=50) -> List[Dict]:
#     all_chunks = []
#     global_chunk_id = 0
#     for sec in sections:
#         sec_chunks = chunk_text_smart(sec["text"], chunk_size, overlap)
#         for ch in sec_chunks:
#             all_chunks.append({
#                 "chunk_id": global_chunk_id, "section": sec["title"], "text": ch["text"],
#                 "start_word": ch["start_word"], "end_word": ch["end_word"],
#                 "type": "body", "page_num": sec.get("page_num", 1)
#             })
#             global_chunk_id += 1
#     return all_chunks

# def chunk_individual_footnotes(footnotes_text: str, chunk_size=200, overlap=0) -> List[Dict]:
#     raw_notes = [n.strip() for n in footnotes_text.split('\n\n') if n.strip()]
#     chunks = []
#     global_chunk_id = 0
#     for note in raw_notes:
#         words = note.split()
#         if len(words) <= chunk_size:
#             chunks.append({
#                 "chunk_id": f"fn_{global_chunk_id}", "text": note,
#                 "type": "footnote_whole", "word_count": len(words)
#             })
#             global_chunk_id += 1
#         else:
#             sub_chunks = chunk_text_smart(note, chunk_size, overlap)
#             for sc in sub_chunks:
#                 chunks.append({
#                     "chunk_id": f"fn_{global_chunk_id}_{sc['chunk_id']}", "text": sc["text"],
#                     "type": "footnote_part", "word_count": len(sc["text"].split())
#                 })
#             global_chunk_id += 1
#     return chunks

# # ---------------------------
# # 5. FULL PIPELINE
# # ---------------------------

# def extract_doi(text: str) -> str:
#     match = re.search(r'\b(10\.\d{4,}/[^\s]+)', text)
#     if match: return re.sub(r'[.,;:\]\)>]+$', '', match.group(1))
#     return ""

# def extract_pdf_metadata(pdf_path: str, pages_text: List[Dict] = None) -> Dict:
#     doc = fitz.open(pdf_path)
#     meta = doc.metadata
#     doi = ""
#     if pages_text:
#         doi = extract_doi(" ".join(p["text"] for p in pages_text[:5]))
#     return {
#         "filename": os.path.basename(pdf_path), "filepath": pdf_path,
#         "title": meta.get("title", ""), "author": meta.get("author", ""),
#         "total_pages": len(doc), "doi": doi
#     }

# def process_pdf(pdf_path: str) -> Dict:
#     # 1️⃣ Extração Baseada em Markdown E Imagens
#     pages = extract_text_and_figures(pdf_path)
    
#     doc_type = classify_pdf(pages)
#     metadata = extract_pdf_metadata(pdf_path, pages)
#     image_chunks = []

#     if doc_type == "scanned":
#         ocr_text = extract_text_from_scanned_pdf(pdf_path)
#         pages = [{"page_num": i + 1, "text": text} for i, text in enumerate(ocr_text.split("\n# PAGE"))]
#     else:
#         # 2️⃣ O "Olho Clínico" (Gemini) entra em ação
#         image_chunks, pages = process_images_with_vlm(pages)
        
#         # 3️⃣ Limpezas Inteligentes
#         pages = remove_front_matter_pages(pages, doc_type)
#         pages = remove_toc_pages(pages)
#         pages = truncate_appendices(pages)
#         pages = remove_references_section(pages)
#         pages = remove_repeated_headers_footers(pages)

#     for p in pages: p["text"] = normalize_text(p["text"])

#     sections = split_into_sections_markdown(pages)
#     chunks = chunk_sections(sections, chunk_size=300, overlap=50)

#     footnotes_text = ""
#     if doc_type != "scanned": footnotes_text = extract_footnotes_only(pdf_path)
#     footnote_chunks = chunk_individual_footnotes(footnotes_text, chunk_size=200)

#     return {
#         "metadata": metadata,
#         "document_type": doc_type,
#         "chunks": chunks,
#         "footnote_chunks": footnote_chunks,
#         "image_chunks": image_chunks # A chave Nova!
#     }

# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

#     if not pdf_files:
#         print("No PDF files found in", PDF_DIR)
#         return

#     for pdf_name in pdf_files:
#         pdf_path = os.path.join(PDF_DIR, pdf_name)
#         print(f"\nProcessing: {pdf_name}")
#         try:
#             output = process_pdf(pdf_path)
#             output_file = os.path.splitext(pdf_name)[0] + ".json"
#             output_path = os.path.join(OUTPUT_DIR, output_file)

#             with open(output_path, "w", encoding="utf-8") as f:
#                 json.dump(output, f, ensure_ascii=False, indent=2)

#             print("  Document type:", output["document_type"])
#             print("  Chunks criados:", len(output["chunks"]))
#             print(f"  Imagens extraídas/descritas: {len(output.get('image_chunks', []))}")
#         except Exception as e:
#             print(f"  ❌ Error processing {pdf_name}: {e}")

# if __name__ == "__main__":
#     main()



"""
PDF Text Extraction Pipeline (MARKDOWN + FOOTNOTES + MULTIMODAL VISION 👁️ + CAPTIONS)
Context: Multimodal RAG – Thesis Development
UPGRADE: Now extracts figure captions and passes them to VLM for contextualized descriptions
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
import time
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
PDF_DIR = "../data/PDFs"
OUTPUT_DIR = "../data/extracted"
FIGURES_DIR = "../data/figures"

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Changed to 1.5 for better quota

# --- LISTAS DE PALAVRAS-CHAVE ---
FRONT_MATTER_BLACKLIST = {
    "agradecimentos", "acknowledgements", "acknowledgments",
    "dedicatória", "dedication", "lista de figuras", "list of figures",
    "lista de tabelas", "list of tables", "lista de abreviaturas", "list of abbreviations",
    "lista de siglas", "list of acronyms", "declaração", "declaração de integridade", 
    "declaration", "folha de rosto", "title page", "epígrafe", "epigraph"
}

APPENDIX_KEYWORDS = {"apêndice", "apêndices", "appendix", "appendices", "anexo", "anexos", "annexe", "annexes", "attachments"}

KEYWORDS = {
    "abstract": ["abstract", "resumo", "résumé", "zusammenfassung", "resumen"],
    "references": ["references", "referências", "bibliografia", "bibliografía"],
    "toc": ["table of contents", "contents", "índice", "sumário"]
}

# ---------------------------
# 1. EXTRAÇÃO DE TEXTO, IMAGENS E CLASSIFICAÇÃO
# ---------------------------

def extract_caption_near_image(doc: fitz.Document, page_num: int, img_path: str, max_distance: int = 150) -> str:
    """
    SOLUÇÃO 1: Extrai legenda abaixo da imagem.
    Procura padrões: "Figure X", "Fig. X", "Figura X", "Table X"
    """
    try:
        page = doc[page_num]
        
        # Tentar encontrar a imagem pelo nome do ficheiro
        img_filename = os.path.basename(img_path)
        
        # Procurar referências de imagem no markdown extraído
        # Como fallback, procuramos texto em toda a parte inferior da página
        page_height = page.rect.height
        
        # Procurar na metade inferior da página
        search_rect = fitz.Rect(0, page_height * 0.4, page.rect.width, page_height)
        
        # Extrair texto estruturado
        text_dict = page.get_text("dict", clip=search_rect)
        
        caption_lines = []
        found_caption = False
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Texto
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    # Procurar padrões de legenda
                    # Padrões: "Figure 1", "Fig. 2:", "Figura 3.", "Table 1", etc.
                    if re.search(r'\b(Figure|Fig\.|Figura|FIGURE|Table|Tabela|TABLE)\s*\d+', line_text, re.IGNORECASE):
                        caption_lines.append(line_text.strip())
                        found_caption = True
                    # Se já encontrou uma legenda, continua a adicionar linhas até quebra
                    elif found_caption and line_text.strip() and len(caption_lines) < 5:
                        # Verificar se parece continuação (não começa com maiúscula isolada)
                        if not re.match(r'^[A-Z][a-z]+\s+[A-Z]', line_text.strip()):
                            caption_lines.append(line_text.strip())
                        else:
                            break
                    # Para se encontrar linha vazia após legenda
                    elif found_caption and not line_text.strip():
                        break
            
            # Se já tem legenda suficiente, para
            if len(caption_lines) >= 5:
                break
        
        # Juntar linhas da legenda
        caption = " ".join(caption_lines) if caption_lines else ""
        
        # Limpar excesso de espaços e quebras
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Limitar tamanho (legendas não devem ser muito longas)
        if len(caption) > 500:
            caption = caption[:500] + "..."
        
        return caption
    
    except Exception as e:
        print(f"    ⚠️ Warning: Could not extract caption: {e}")
        return ""


def extract_text_and_figures(pdf_path: str) -> List[Dict]:
    """Extrai texto com pymupdf4llm e guarda as imagens na pasta figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # write_images=True é o que extrai as imagens reais!
    md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=True, image_path=FIGURES_DIR)
    pages = []
    
    for i, md_page in enumerate(md_pages):
        pages.append({
            "page_num": i + 1, 
            "text": md_page.get("text", "")
        })
        
    return pages


def process_images_with_vlm(pdf_path: str, pages: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    SOLUÇÃO 2 + 3: Lê imagens, extrai legendas, descreve com VLM contextualizado,
    e cria chunks completos com caption + description.
    """
    # Abrir PDF para extração de legendas
    doc = fitz.open(pdf_path)
    
    image_chunks = []
    global_img_id = 0
    img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    
    for p in pages:
        text = p["text"]
        matches = img_pattern.findall(text)
        
        for img_path in matches:
            if not os.path.exists(img_path): 
                continue
            
            try:
                img = Image.open(img_path)
                
                # Ignorar imagens minúsculas (ex: logotipos, ícones de rodapé)
                if img.width < 150 or img.height < 150:
                    text = img_pattern.sub('', text, count=1)
                    img.close()
                    os.remove(img_path) 
                    continue
                
                # ✅ SOLUÇÃO 1: EXTRAIR LEGENDA
                caption = extract_caption_near_image(doc, p["page_num"] - 1, img_path)
                
                caption_preview = caption[:60] + "..." if len(caption) > 60 else caption
                if caption:
                    print(f"    📋 Caption extracted (Page {p['page_num']}): {caption_preview}")
                
                print(f"    👁️ Gemini analyzing image (Page {p['page_num']}): {os.path.basename(img_path)}...")
                
                # ✅ SOLUÇÃO 2: VLM COM LEGENDA CONTEXTUALIZADA
                # Prompt em INGLÊS como pedido
                if caption:
                    prompt = f"""You are a scientific and medical expert analyzing figures from research papers.

ORIGINAL CAPTION FROM THE DOCUMENT:
"{caption}"

Based on the caption AND the visual content, provide a comprehensive description in ENGLISH that includes:
1. Type of visualization (bar chart, line graph, flowchart, microscopy, diagram, etc.)
2. Main visual elements (axes, legends, data series, annotations, colors, symbols)
3. Data patterns or trends shown (comparisons, distributions, relationships)
4. Any visible text, labels, or numerical values in the figure

Your description should complement the original caption with specific visual details.
Be precise, technical, and factual. Respond in ENGLISH."""
                else:
                    # Fallback se não houver legenda
                    prompt = f"""You are a scientific and medical expert. Describe this figure in detail in ENGLISH.

Include:
1. Type of visualization (graph, diagram, flowchart, microscopy, etc.)
2. Main visual elements visible (axes, legends, data series, annotations)
3. Key patterns, trends, or conclusions that can be observed
4. Any visible text, labels, or numerical values

Context from the page where this image appears:
{text[:1000]}

Be precise, technical, and factual. Respond in ENGLISH."""
                
                response = vision_model.generate_content(
                    [prompt, img],
                    generation_config={
                        'temperature': 0.4,  # More consistent descriptions
                        'top_p': 0.9
                    }
                )
                vlm_description = response.text.strip()
                
                # ✅ SOLUÇÃO 3: CHUNK COMPLETO COM CAPTION + DESCRIPTION
                image_chunks.append({
                    "chunk_id": f"img_{global_img_id}",
                    "section": "Extracted Figure",
                    "text": vlm_description,           # VLM description
                    "caption": caption,                # Original caption
                    "image_path": img_path,
                    "page_num": p["page_num"],
                    "image_size": f"{img.width}x{img.height}",
                    "type": "figure"
                })
                
                global_img_id += 1
                time.sleep(4)  # Rate limiting for API
                
            except Exception as e:
                print(f"    ⚠️ Error processing image {os.path.basename(img_path)}: {e}")
                if 'img' in locals(): 
                    img.close()
                if os.path.exists(img_path): 
                    os.remove(img_path)
            
            # Limpar a hiperligação original para não sujar o texto
            text = text.replace(f"![{img_path}]", "")
            text = re.sub(r'!\[.*?\]\(' + re.escape(img_path) + r'\)', '', text)
            
        p["text"] = text
    
    doc.close()
    return image_chunks, pages


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
# 2. EXTRAÇÃO DE FOOTNOTES
# ---------------------------

def extract_footnotes_only(pdf_path: str) -> str:
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
            print(f"  -> [Front-Matter] Removed Page {page['page_num']}")
            continue
        cleaned_pages.append(page)
    return cleaned_pages

def remove_toc_pages(pages: List[Dict]) -> List[Dict]:
    TOC_LINK = re.compile(r'\[.*?\]\(#page=\d+\)')
    TOC_DOTS = re.compile(r'(\.{3,}|\s{4,})\s*(\d+|[ivxlcdmIVXLCDM]+)[\*\|\s]*$') 
    filtered_pages = []
    for page in pages:
        lines = page["text"].splitlines()
        if not lines: continue
        toc_lines = sum(1 for line in lines if TOC_LINK.search(line) or TOC_DOTS.search(line.strip()))
        ratio = toc_lines / len(lines) if len(lines) > 5 else 0
        if ratio > 0.15: 
            print(f"  -> [TOC] Removed Table of Contents: Page {page['page_num']}")
            continue 
        filtered_pages.append(page)
    return filtered_pages

def truncate_appendices(pages: List[Dict]) -> List[Dict]:
    cutoff_index = -1
    total_pages = len(pages)
    search_start = int(total_pages * 0.5)
    for i in range(search_start, total_pages):
        lines = pages[i]["text"].strip().splitlines()
        for line in lines[:5]:
            if line.startswith("#"):
                clean_header = line.replace("#", "").strip().lower()
                if any(kw in clean_header for kw in APPENDIX_KEYWORDS) and len(clean_header) < 60:
                    print(f"  -> [Appendices] Cut detected at Page {pages[i]['page_num']}.")
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
                clean_line = line.replace("#", "").replace("*", "").strip().lower()
                clean_line = re.sub(r'^\d+(\.\d+)*\s+', '', clean_line)
                if clean_line in REF_HEADERS:
                    cutoff_page_index = i
                    cutoff_line_index = j
                    print(f"  -> [Ref] References detected at Page {pages[i]['page_num']}. Cutting...")
                    break
        if cutoff_page_index != -1: break
    if cutoff_page_index == -1: return pages
    final_pages = pages[:cutoff_page_index]
    if cutoff_line_index > 0:
        content_lines = pages[cutoff_page_index]["text"].splitlines()[:cutoff_line_index]
        texto_limpo = "\n".join(content_lines).strip()
        if texto_limpo:
            final_pages.append({"page_num": pages[cutoff_page_index]["page_num"], "text": texto_limpo})
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
        cleaned_pages.append({"page_num": page["page_num"], "text": "\n".join(body)})
    return cleaned_pages

# ---------------------------
# 4. TEXT NORMALISATION E CHUNKING
# ---------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r'\([^\)]*?\b(?:18|19|20)\d{2}\b[^\)]*?\)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\s*\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*\s*\]', '', text)
    text = re.sub(r'\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*(?:<br>|\n)*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\*\*==>\s*picture.*?intentionally omitted\s*<==\*\*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    return text

def split_into_sections_markdown(pages):
    sections = []
    current_title = "Document Start"
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
                if not current_text and current_title == "Document Start": current_page = page_num
                current_text.append(line)
    if current_text:
        sections.append({"title": current_title, "text": "\n".join(current_text), "page_num": current_page})
    return sections

def chunk_text_smart(text: str, chunk_size=300, overlap=50):
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
                "chunk_id": chunk_id, "text": " ".join(current_chunk_words).replace(" \n ", "\n"),
                "start_word": chunk_id * chunk_size, "end_word": (chunk_id * chunk_size) + current_word_count
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
    # 1️⃣ Extração Baseada em Markdown E Imagens
    pages = extract_text_and_figures(pdf_path)
    
    doc_type = classify_pdf(pages)
    metadata = extract_pdf_metadata(pdf_path, pages)
    image_chunks = []

    if doc_type == "scanned":
        ocr_text = extract_text_from_scanned_pdf(pdf_path)
        pages = [{"page_num": i + 1, "text": text} for i, text in enumerate(ocr_text.split("\n# PAGE"))]
    else:
        # 2️⃣ O "Olho Clínico" (Gemini) com Legendas
        image_chunks, pages = process_images_with_vlm(pdf_path, pages)
        
        # 3️⃣ Limpezas Inteligentes
        pages = remove_front_matter_pages(pages, doc_type)
        pages = remove_toc_pages(pages)
        pages = truncate_appendices(pages)
        pages = remove_references_section(pages)
        pages = remove_repeated_headers_footers(pages)

    for p in pages: p["text"] = normalize_text(p["text"])

    sections = split_into_sections_markdown(pages)
    chunks = chunk_sections(sections, chunk_size=300, overlap=50)

    footnotes_text = ""
    if doc_type != "scanned": footnotes_text = extract_footnotes_only(pdf_path)
    footnote_chunks = chunk_individual_footnotes(footnotes_text, chunk_size=200)

    return {
        "metadata": metadata,
        "document_type": doc_type,
        "chunks": chunks,
        "footnote_chunks": footnote_chunks,
        "image_chunks": image_chunks  # Agora com caption + description!
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
            print("  Text chunks created:", len(output["chunks"]))
            print("  Footnote chunks:", len(output["footnote_chunks"]))
            print(f"  Images extracted with captions: {len(output.get('image_chunks', []))}")
        except Exception as e:
            print(f"  ❌ Error processing {pdf_name}: {e}")

if __name__ == "__main__":
    main()