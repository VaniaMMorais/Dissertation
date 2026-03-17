"""
PDF Text Extraction and Classification Pipeline
Author: [o teu nome]
Context: Multimodal RAG – Thesis Development

This module performs:
1. Text extraction from PDFs
2. Detection of scanned PDFs
3. Document type classification
4. Basic header/footer cleaning
5. Text chunking for retrieval
"""

import fitz  # PyMuPDF
import re
from collections import Counter
from typing import List, Dict
import os
import json
from pdf2image import convert_from_path
import pytesseract
import re
PDF_DIR = "data/PDFs"
OUTPUT_DIR = "data/extracted"
margin_ratio = 0.12

# Páginas inteiras a remover APENAS em Teses/Relatórios (Front Matter)
FRONT_MATTER_BLACKLIST = {
    # Agradecimentos
    "agradecimentos", "acknowledgements", "acknowledgments",
    "remerciements", "agradecimientos", "danksagung", "ringraziamenti",

    # Dedicação
    "dedicatória", "dedication",
    "dédicace", "dedicatoria", "widmung", "dedica",

    # Listas
    "lista de figuras", "list of figures",
    "liste des figures", "lista de figuras", "abbildungsverzeichnis", "elenco delle figure",

    "lista de tabelas", "list of tables",
    "liste des tableaux", "lista de tablas", "tabellenverzeichnis", "elenco delle tabelle",

    "lista de abreviaturas", "list of abbreviations",
    "liste des abréviations", "lista de abreviaturas", "abkürzungsverzeichnis", "elenco delle abbreviazioni",

    "lista de siglas", "list of acronyms",
    "liste des sigles", "lista de siglas", "abkürzungen", "elenco degli acronimi",

    # Declarações
    "declaração", "declaração de integridade", "declaration",
    "déclaration", "declaración", "erklärung", "dichiarazione",

    # Página de título
    "folha de rosto", "title page",
    "page de titre", "página de título", "titelseite", "frontespizio",

    # Epígrafe
    "epígrafe", "epigraph",
    "épigraphe", "epígrafe", "epigraph", "epigrafe"
}

APPENDIX_KEYWORDS = {
    "apêndice", "apêndices",
    "appendix", "appendices",
    "annexe", "annexes",
    "anexo", "anexos",
    "anhang", "anhänge",
    "appendice", "appendici",
    "allegato", "allegati",
    "attachments"
}

SECTION_NUMBER_RE = re.compile(r'^\d+(\.\d+)*\s+')
ALL_CAPS_RE = re.compile(r'^[A-ZÀ-Ü\s\-]{4,}$')
CAPTION_START_RE = re.compile(r'^(figura|figure|fig\.|tabela|table|tab\.|quadro|chart|graph|imagem|image|fonte|source)\b', re.IGNORECASE)
BAD_TITLE_PATTERNS = re.compile(r'^\s*[\-\.\,\)\:\u2013\u2014\u2022\u25E6]\s+|;|^\d+\s*$|\.{3,}')
SOURCE_RE = re.compile(r'^(fonte|source|adaptado de|adapted from)\b', re.IGNORECASE)
BAD_TITLE_STARTS_RE = re.compile(
    r'^(de facto|no entanto|entretanto|todavia|assim|portanto|deste modo|'
    r'furthermore|moreover|however|nevertheless|according to|consequently|'
    r'apesar|embora|consoante|tendo em conta|relativamente|quanto|outro exemplo)',
    re.IGNORECASE
)
BAD_TITLE_ENDS_RE = re.compile(
    r'\b(a|o|as|os|um|uma|de|da|do|das|dos|em|na|no|nas|nos|por|para|com|se|que|e|ou|à|às|'
    r'of|the|and|or|in|on|at|to|for|with|by|from|is|are)\s*$',
    re.IGNORECASE
)


# ---------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------

FOOTNOTE_START_RE = re.compile(r'^\s*(\(?\d+\)?|[\*\†\‡])[\.\)\s]\s*[A-ZÀ-Úa-z]')
PAGE_NUMBER_RE = re.compile(r'^\s*\d+\s*$')

def get_body_font_size(page_dict: Dict) -> float:
    """Calcula a moda do tamanho da fonte (assumindo ser o corpo do texto)."""
    sizes = []
    for block in page_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    
    if not sizes: return 0.0
    return Counter(sizes).most_common(1)[0][0]

def find_footnote_separator_y(page, page_height) -> float:
    """Procura uma linha horizontal no terço inferior da página."""
    drawings = page.get_drawings()
    candidates = []
    
    # Procurar no terço inferior (y > 0.66 * height)
    min_y = page_height * 0.66
    
    for draw in drawings:
        rect = draw["rect"]
        # Verifica se está em baixo, é fina (altura < 5) e larga (largura > 50)
        if rect.y0 > min_y and rect.height < 5 and rect.width > 50:
            candidates.append(rect.y0)
            
    return min(candidates) if candidates else None

TOC_PATTERN_RE = re.compile(r'.*[\.\s]{3,}\s*\d+$')
TABLE_ROW_RE = re.compile(r'\d+\s{2,}\d+')

def is_table_row(text: str) -> bool:
    """
    Deteta se a linha parece pertencer a uma tabela (dados soltos).
    """
    # Se for uma frase normal que termina em ponto, salvamos (mesmo que tenha números)
    if text.strip().endswith(('.', ':', ';')):
        return False
        
    # Critério 1: Padrão visual de colunas numéricas
    if TABLE_ROW_RE.search(text):
        return True
    
    # Critério 2: Densidade. Se tiver < 50% de letras, provavelmente é dados/tabela
    clean = text.replace(" ", "")
    if len(clean) > 0:
        alpha_count = sum(c.isalpha() for c in clean)
        if (alpha_count / len(clean)) < 0.5:
            return True

    return False



def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []

    first_page = doc[0]
    is_slide_layout = first_page.rect.width > first_page.rect.height

    for i, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_width = page.rect.width
        page_height = page.rect.height
        
        # 1. Tabelas (PyMuPDF Nativo)
        tables = page.find_tables()
        table_rects = [tab.bbox for tab in tables]

        def is_inside_table(x, y):
            for rect in table_rects:
                if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                    return True
            return False

        # --- ESTATÍSTICAS DE FONTE E COR (NOVO) ---
        sizes = []
        colors = []
        for block in page_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        txt = span["text"].strip()
                        if txt:
                            sizes.append(round(span["size"], 1))
                            colors.append(span["color"]) # Guardar a cor
        
        # Moda da Fonte (Tamanho do corpo)
        body_font_size = Counter(sizes).most_common(1)[0][0] if sizes else 11.0
        
        # Moda da Cor (Cor do corpo - geralmente Preto/0)
        body_color = Counter(colors).most_common(1)[0][0] if colors else 0
        
        # ------------------------------------------

        TITLE_THRESHOLD = body_font_size + (4.0 if is_slide_layout else 0.5)

        separator_y = find_footnote_separator_y(page, page_height)
        footnote_zone_y = separator_y if separator_y else (page_height * 0.85)
        
        SAFE_TITLE_MIN_Y = page_height * 0.10 if not is_slide_layout else 0
        SAFE_TITLE_MAX_Y = page_height * 0.90

        body_blocks = []
        footnote_candidates = []

        for block in page_dict["blocks"]:
            if "lines" not in block: continue
            
            b_x0, b_y0, b_x1, b_y1 = block["bbox"]
            if b_x1 < (page_width * margin_ratio) or b_x0 > (page_width * (1 - margin_ratio)):
                continue
            
            center_x = (b_x0 + b_x1) / 2
            center_y = (b_y0 + b_y1) / 2
            if is_inside_table(center_x, center_y):
                continue

            for line in block["lines"]:
                l_y0 = line["bbox"][1]
                spans = line["spans"] # Acesso aos spans para ver a cor
                
                # Reconstruir texto e verificar propriedades
                full_text = " ".join([s["text"] for s in spans]).strip()
                
                if not full_text: continue
                if PAGE_NUMBER_RE.match(full_text): continue
                if is_table_row(full_text): continue 

                # Calcular tamanho médio e verificar cor
                valid_sizes = [s["size"] for s in spans if s["text"].strip()]
                avg_size = sum(valid_sizes)/len(valid_sizes) if valid_sizes else body_font_size
                
                # Verificar se a linha tem cor diferente do corpo (NOVO)
                # Assumimos que se o primeiro pedaço da linha for colorido, a linha é colorida
                line_color = spans[0]["color"]
                is_different_color = (line_color != body_color)

                # --- CLASSIFICAÇÃO ---
                is_footnote = (l_y0 >= footnote_zone_y and avg_size <= (body_font_size - 0.5))
                is_title = False
                
                # LÓGICA DE TÍTULO (COM COR)
                # Se for colorido, somos mais flexíveis com o tamanho
                effective_threshold = TITLE_THRESHOLD
                if is_different_color:
                    # Se for colorido, basta ser do mesmo tamanho do corpo ou maior (bold colorido)
                    effective_threshold = body_font_size 
                
                # Regra A: Tamanho (ou Cor + Tamanho)
                if avg_size >= effective_threshold:
                    is_title = True
                
                # Regra B: Zonas Proibidas
                if l_y0 < SAFE_TITLE_MIN_Y or l_y0 > SAFE_TITLE_MAX_Y:
                    is_title = False

                if is_title:
                    # FILTROS DE EXCLUSÃO
                    if BAD_TITLE_PATTERNS.search(full_text): is_title = False
                    elif CAPTION_START_RE.match(full_text): is_title = False
                    elif BAD_TITLE_STARTS_RE.match(full_text): is_title = False
                    elif BAD_TITLE_ENDS_RE.search(full_text): is_title = False
                    elif len(full_text) > 120: is_title = False
                    elif "," in full_text[:20]: is_title = False
                    elif full_text.count("  ") > 3: is_title = False
                    elif len(full_text) < 4 and not re.match(r'^\d+\.', full_text):is_title = False
                    elif "," in full_text: is_title = False

                    primeiro_alfa = next((c for c in full_text if c.isalpha()), None)
                    if primeiro_alfa and primeiro_alfa.islower():
                        # Verifica se NÃO tem número antes
                        pos = full_text.index(primeiro_alfa)
                        if pos == 0 or not any(c.isdigit() for c in full_text[:pos]):
                            is_title = False
                    
                    # Filtro extra para cor: Links (azul) no meio do texto não são títulos
                    # Se for colorido mas começar com minúscula, é link ou destaque, não título
                    if is_different_color and full_text[0].islower():
                        is_title = False
                    
                    elif full_text[0].islower() and not re.match(r'^[\d\-\u2022]', full_text): 
                        is_title = False
                    
                    elif len(full_text) < 6 and full_text.isupper() and not re.match(r'^(I|V|X|L|M)+\.?$', full_text):
                        if full_text not in ["RESUMO", "ABSTRACT", "INTRO"]:
                            is_title = False

                if is_footnote:
                    footnote_candidates.append({"text": full_text, "y0": l_y0})
                else:
                    if CAPTION_START_RE.match(full_text) or SOURCE_RE.match(full_text):
                        continue

                    if is_title:
                        full_text = f"###SEC### {full_text}"
                    body_blocks.append(full_text)

        # Ordenar e limpar footnotes
        footnote_candidates.sort(key=lambda x: x["y0"])
        clean_footnotes = []
        current_note = []
        active = False
        for item in footnote_candidates:
            if FOOTNOTE_START_RE.match(item["text"]):
                if current_note: clean_footnotes.append(" ".join(current_note))
                current_note = [item["text"]]; active = True
            elif active: current_note.append(item["text"])
        if current_note: clean_footnotes.append(" ".join(current_note))

        page_content = "\n".join(body_blocks)
        if is_slide_layout and page_content and "###SEC###" not in page_content[:100]:
             page_content = f"###SEC### Slide {i+1}\n" + page_content

        pages.append({
            "page_num": i + 1,
            "text": page_content,
            "footnotes": "\n\n".join(clean_footnotes)
        })

    return pages


def extract_text_from_scanned_pdf(
    pdf_path,
    languages="eng+por",
    dpi=300
):
    """
    Extracts text from a scanned PDF using OCR.
    Returns raw concatenated text.
    """

    pages = convert_from_path(pdf_path, dpi=dpi)
    full_text = []

    for page_num, image in enumerate(pages):
        text = pytesseract.image_to_string(
            image,
            lang=languages,
            config="--psm 1"
        )

        # limpeza básica
        text = clean_ocr_text(text)

        full_text.append(f"\n--- PAGE {page_num + 1} ---\n{text}")

    return "\n".join(full_text)

# ---------------------------
# 2. SCANNED PDF DETECTION
# ---------------------------

def is_scanned_pdf(pages_text: List[Dict], min_chars: int = 200) -> bool:
    """
    Heuristic to detect scanned PDFs.
    If total extracted characters < min_chars → scanned
    """
    total_chars = sum(len(p["text"]) for p in pages_text)
    return total_chars < min_chars


# ---------------------------
# 3. DOCUMENT TYPE CLASSIFICATION
# ---------------------------

KEYWORDS = {
    "abstract": [
        "abstract", "resumo", "résumé", "zusammenfassung", "resumen", "sommario"
    ],
    "references": [
        "references", "referências", "bibliografia", "bibliografía",
        "références", "literaturverzeichnis", "bibliography"
    ],
    "toc": [
        "table of contents", "contents", "índice", "sumário",
        "table des matières", "inhaltsverzeichnis", "indice"
    ]
}


def has_keyword(text: str, keyword_type: str) -> bool:
    """Check if text contains any keyword variant for the given type."""
    return any(kw in text for kw in KEYWORDS.get(keyword_type, []))


def classify_pdf(pages_text: List[Dict]) -> str:
    """
    Classifies PDF into:
    - scanned
    - article
    - thesis
    - slides
    - report
    """

    if is_scanned_pdf(pages_text):
        return "scanned"

    total_pages = len(pages_text)
    total_words = sum(len(p["text"].split()) for p in pages_text)
    avg_words_per_page = total_words / max(total_pages, 1)

    full_text = " ".join(p["text"].lower() for p in pages_text)

    # Slides: pouco texto por página
    if avg_words_per_page < 100:
        return "slides"

    # Thesis / Dissertation
    if has_keyword(full_text, "toc") and total_pages > 30:
        return "thesis"

    # Scientific article
    if has_keyword(full_text, "abstract") and has_keyword(full_text, "references"):
        return "article"

    # Fallback
    return "report"


# ---------------------------
# 4. HEADER / FOOTER REMOVAL
# ---------------------------

def remove_repeated_headers_footers(pages_text, threshold=0.6):
    line_counter = Counter()

    for page in pages_text:
        for line in page["text"].splitlines():
            clean = line.strip()
            if 3 < len(clean) < 300:
                line_counter[clean] += 1

    total_pages = len(pages_text)
    repeated = {
        line for line, count in line_counter.items()
        if count / total_pages >= threshold
    }

    cleaned_pages = []
    for page in pages_text:
        body = [
            l for l in page["text"].splitlines()
            if l.strip() and l.strip() not in repeated
        ]

        cleaned_pages.append({
            "page_num": page["page_num"],
            "text": "\n".join(body),
            "footnotes": page.get("footnotes", "")
        })

    return cleaned_pages


def remove_front_matter_pages(pages: List[Dict], doc_type: str) -> List[Dict]:
    """
    Remove páginas pré-textuais (Capa, Agradecimentos, Listas).
    CRÍTICO: Só corre se for 'thesis' ou 'report'. Se for 'article', não faz nada.
    """
    # SE FOR ARTIGO, SALTA ESTA LIMPEZA! 
    # Artigos começam logo com conteúdo relevante na pág 1.
    if doc_type == "article":
        return pages

    cleaned_pages = []
    # Verificar apenas as primeiras 15 páginas para teses
    check_limit = 15 
    
    for i, page in enumerate(pages):
        # Se já passámos o limite, aceitamos tudo o resto
        if i >= check_limit:
            cleaned_pages.append(page)
            continue

        text = page["text"].strip()
        lines = text.splitlines()
        
        if not lines: continue

        # Verifica o "título" da página (primeiras 3 linhas)
        header_text = " ".join(lines[:3]).lower()
        
        # Se contiver palavras proibidas, ignoramos a página
        if any(kw in header_text for kw in FRONT_MATTER_BLACKLIST):
            print(f"  -> [Front-Matter] Removida Pag {page['page_num']}: {lines[0][:30]}...")
            continue
            
        cleaned_pages.append(page)

    return cleaned_pages


def truncate_appendices(pages: List[Dict]) -> List[Dict]:
    """
    Corta o documento assim que encontra 'Apêndices' ou 'Anexos'.
    Válido para Teses e Artigos.
    """
    cutoff_index = -1

    for i, page in enumerate(pages):
        text = page["text"].strip()
        lines = text.splitlines()
        if not lines: continue
        
        # Procura título nas primeiras linhas
        header_text = " ".join(lines[:3]).lower()
        
        # Heurística: Tem palavra "Apêndice" E o título é curto (não é uma frase no meio do texto)
        is_appendix = any(kw in header_text for kw in APPENDIX_KEYWORDS)
        
        if is_appendix and len(lines[0]) < 60:
            print(f"  -> [Apêndices] Corte detetado na Pag {page['page_num']}.")
            cutoff_index = i
            break
    
    # Se encontrou corte, devolve apenas até lá
    if cutoff_index != -1:
        return pages[:cutoff_index]
        
    return pages

TOC_RELAXED_RE = re.compile(r'.+\s{2,}\d+$')

def remove_toc_pages(pages: List[Dict], threshold_ratio=0.4) -> List[Dict]:
    """
    Remove páginas de Índice.
    Melhoria: Se o título da página for 'Índice', o threshold desce para 10%.
    """
    TOC_HEADERS = ["table of contents", "contents", "índice", "sumário", "table des matières", "inhaltsverzeichnis", "indice"]
    
    filtered_pages = []
    
    for page in pages:
        lines = page["text"].splitlines()
        if not lines: continue
        
        # Verificar se o título da página diz explicitamente que é um Índice
        header = " ".join(lines[:3]).lower()
        is_explicit_toc = any(h in header for h in TOC_HEADERS)
        
        # Ajustar a sensibilidade baseada no título
        # Se diz "Índice", basta 15% das linhas parecerem índice para cortar.
        # Se não diz, exigimos 40% (para não cortar texto normal com números).
        current_threshold = 0.15 if is_explicit_toc else threshold_ratio
        
        toc_lines_count = 0
        total_lines = len(lines)
        
        for line in lines:
            # Verifica padrão clássico (.... 12) OU padrão relaxado (   12)
            if TOC_PATTERN_RE.match(line) or TOC_RELAXED_RE.match(line):
                toc_lines_count += 1
        
        ratio = toc_lines_count / total_lines if total_lines > 0 else 0
        
        if ratio > current_threshold:
            print(f"  -> [TOC] Removido Índice: Página {page['page_num']} (Confiança: {ratio:.0%})")
            continue 
            
        filtered_pages.append(page)
        
    return filtered_pages

def remove_references_section(pages: List[Dict]) -> List[Dict]:
    """
    Corta o documento quando encontra o cabeçalho 'Referências'.
    Ignora a tag ###SEC### para fazer a comparação.
    """
    REF_HEADERS = [
        "referências", "referencias", "bibliografia", "bibliography", 
        "references", "obras citadas", "literaturverzeichnis", "quellenverzeichnis"
    ]
    
    cutoff_page_index = -1
    cutoff_line_index = -1
    
    # Procura de trás para a frente
    for i in range(len(pages) - 1, -1, -1):
        lines = pages[i]["text"].splitlines()
        for j, line in enumerate(lines):
            # 1. Limpar a tag de secção e converter para minúsculas
            clean_line = line.replace("###SEC###", "").strip().lower()
            
            # 2. Remover numeração (ex: "6. Referências") e pontuação final
            clean_line = re.sub(r'^\d+(\.\d+)*\s+', '', clean_line)
            clean_line = re.sub(r'[.:;]+$', '', clean_line)
            
            # 3. Comparar
            if clean_line in REF_HEADERS and len(clean_line) < 30:
                cutoff_page_index = i
                cutoff_line_index = j
                print(f"  -> [Ref] Referências detetadas na Pag {pages[i]['page_num']}. Cortando...")
                break
        if cutoff_page_index != -1: break
            
    if cutoff_page_index == -1: return pages

    final_pages = []
    for i in range(cutoff_page_index + 1):
        if i < cutoff_page_index:
            final_pages.append(pages[i])
        elif i == cutoff_page_index:
            content_lines = pages[i]["text"].splitlines()[:cutoff_line_index]
            final_pages.append({
                "page_num": pages[i]["page_num"],
                "text": "\n".join(content_lines),
                "footnotes": pages[i]["footnotes"]
            })
            
    return final_pages




# ---------------------------
# 5. TEXT NORMALISATION
# ---------------------------

def remove_citations(text: str) -> str:
    """
    Remove citações académicas.
    """
    # 1. APA/Harvard (anos)
    text = re.sub(r'\([^\)]*?\b(?:18|19|20)\d{2}\b[^\)]*?\)', '', text, flags=re.DOTALL)
    
    # 2. Brackets [1, 2]
    text = re.sub(r'\[\s*\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*\s*\]', '', text)
    
    # 3. Superscritos colados (texto1)
    text = re.sub(r'(?<=[a-z])\d{1,3}(?=[.,;:]|\s)', '', text)
    text = re.sub(r'(?<=[a-z])\s+\d{1,3}(?=[.,;:])', '', text)

    # 4. Números flutuantes (Vancouver)
    # CORREÇÃO: Aceita letra maiúscula DEPOIS OU fim da string ($)
    # Ex: "text. 94 Next" OU "text. 94"
    text = re.sub(r'(?<=\.)\s+\d{1,3}\s*(?=[A-Z]|$)', ' ', text)
    
    return text


# def normalize_text(text: str) -> str:
#     """
#     Normaliza o texto mas MANTÉM as quebras de linha para detetar títulos.
#     """
#     text = remove_citations(text)
    
#     # Divide em linhas, limpa cada linha individualmente
#     lines = text.splitlines()
#     cleaned_lines = []
#     for line in lines:
#         # Remove espaços extra dentro da linha
#         clean = re.sub(r"\s+", " ", line).strip()
#         if clean:
#             cleaned_lines.append(clean)
            
#     # Junta de volta com \n
#     return "\n".join(cleaned_lines)

def normalize_text(text: str) -> str:
    text = remove_citations(text)
    
    # Dividir em linhas e limpar cada uma
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Manter marcadores de secção
        if line.startswith("###SEC###"):
            cleaned_lines.append(line)
        else:
            # Remover espaços extra dentro da linha
            clean = re.sub(r"\s+", " ", line).strip()
            if clean:
                cleaned_lines.append(clean)
    
    # Juntar com \n (para preservar estrutura)
    return "\n".join(cleaned_lines)

def clean_ocr_text(text):
    # remover hifenização de quebra de linha
    text = re.sub(r"-\n", "", text)

    # juntar linhas quebradas
    text = re.sub(r"\n+", "\n", text)

    # remover espaços excessivos
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

# ---------------------------
# 6. CHUNKING FOR RAG
# ---------------------------

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": min(end, len(words))
        })

        chunk_id += 1
        start = end - overlap
    return chunks

def is_section_title(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    # demasiado longo → nunca é título
    if len(line) > 120:
        return False

    words = line.split()
    if len(words) > 12:
        return False

    # títulos não acabam com ponto
    if line.endswith("."):
        return False

    # frases com vírgulas são quase sempre texto
    if "," in line:
        return False

    # padrões válidos
    if SECTION_NUMBER_RE.match(line):
        return True

    if ALL_CAPS_RE.match(line):
        return True

    if line.istitle():
        return True

    return False





# def split_into_sections(pages):
#     sections = []
#     current_title = None
#     current_text = []

#     for page in pages:
#         for line in page["text"].splitlines():
#             clean = line.strip()
#             if not clean:
#                 continue
            
#             # Usar APENAS o marcador explícito
#             if line.startswith("###SEC###"):
#                 # Guardar secção anterior
#                 if current_text:
#                     full_sec_text = " ".join(current_text)
#                     sections.append({
#                         "title": current_title or "UNLABELED_SECTION",
#                         "text": full_sec_text
#                     })
                
#                 # Nova secção
#                 current_title = clean.replace("###SEC###", "").strip()
#                 current_text = []
#             else:
#                 current_text.append(clean)
    
#     # Última secção
#     if current_text:
#         full_sec_text = " ".join(current_text)
#         sections.append({
#             "title": current_title or "UNLABELED_SECTION",
#             "text": full_sec_text
#         })
    
#     return sections
def split_into_sections(pages):
    sections = []
    current_title = None
    current_text = []
    current_page = 1  # Guarda a página atual

    for page in pages:
        page_num = page.get("page_num", 1) # Vai buscar a página atual ao dicionário
        
        for line in page["text"].splitlines():
            clean = line.strip()
            if not clean:
                continue
            
            # Usar APENAS o marcador explícito
            if line.startswith("###SEC###"):
                # Guardar secção anterior
                if current_text:
                    full_sec_text = " ".join(current_text)
                    sections.append({
                        "title": current_title or "UNLABELED_SECTION",
                        "text": full_sec_text,
                        "page_num": current_page  # <-- Guardar a página!
                    })
                
                # Nova secção
                current_title = clean.replace("###SEC###", "").strip()
                current_text = []
                current_page = page_num  # <-- A nova secção começa nesta página!
            else:
                if not current_text and current_title is None:
                    current_page = page_num
                current_text.append(clean)
    
    # Última secção
    if current_text:
        full_sec_text = " ".join(current_text)
        sections.append({
            "title": current_title or "UNLABELED_SECTION",
            "text": full_sec_text,
            "page_num": current_page  # <-- Guardar a página!
        })
    
    return sections

def segment_by_headers(lines: List[Dict], body_font_size: float) -> List[Dict]:
    """
    Agrupa linhas em secções baseando-se em:
    1. Tamanho da fonte (> corpo + tolerância)
    2. Negrito + Caixa Alta
    3. Regex (Capítulo X, 1.1, etc)
    """
    sections = []
    
    current_section = {
        "title": "Introduction/Preamble",
        "content": []
    }

    # Tolerância: Se for 1pt maior que o corpo, é título
    SIZE_THRESHOLD = body_font_size + 0.5
    
    for line in lines:
        text = line["text"]
        size = line["size"]
        is_bold = line["is_bold"]
        
        is_header = False

        # --- HEURÍSTICAS DE TÍTULO ---
        
        # 1. Tamanho da Fonte (Muito forte)
        if size > SIZE_THRESHOLD and len(text) < 150:
             # Ignorar se terminar em pontuação de frase (exceto se for numero tipo 1.)
            if not text.strip().endswith((".", ":", ";")) or SECTION_NUMBER_RE.match(text):
                is_header = True
        
        # 2. Regex (Forte para teses)
        elif SECTION_NUMBER_RE.match(text) and len(text) < 100:
            is_header = True
            
        # 3. Visual (Bold + Caps)
        elif is_bold and ALL_CAPS_RE.match(text) and len(text) < 100:
            is_header = True

        # --- CRIAÇÃO DE SECÇÃO ---
        
        if is_header:
            # Fechar secção anterior
            if current_section["content"]:
                # Normalizar o texto apenas AGORA, depois de agrupado
                full_text = " ".join(current_section["content"])
                current_section["text"] = normalize_text(full_text)
                del current_section["content"] # libertar memória
                sections.append(current_section)
            
            # Abrir nova secção
            current_section = {
                "title": text, # O texto da linha é o título
                "content": []
            }
        else:
            # É texto normal
            current_section["content"].append(text)

    # Adicionar a última
    if current_section["content"]:
        full_text = " ".join(current_section["content"])
        current_section["text"] = normalize_text(full_text)
        del current_section["content"]
        sections.append(current_section)

    return sections

# def chunk_sections(sections: List[Dict], chunk_size=300, overlap=50) -> List[Dict]:
#     """
#     Cria chunks mantendo o contexto da secção.
#     """
#     all_chunks = []
#     global_chunk_id = 0

#     for sec in sections:
#         sec_chunks = chunk_text(sec["text"], chunk_size, overlap)

#         for ch in sec_chunks:
#             all_chunks.append({
#                 "chunk_id": global_chunk_id,
#                 "section": sec["title"],
#                 "text": ch["text"],
#                 "start_word": ch["start_word"],
#                 "end_word": ch["end_word"],
#                 "type": "body"
#             })
#             global_chunk_id += 1

#     return all_chunks

def chunk_sections(sections: List[Dict], chunk_size=300, overlap=50) -> List[Dict]:
    """
    Cria chunks mantendo o contexto da secção.
    """
    all_chunks = []
    global_chunk_id = 0

    for sec in sections:
        sec_chunks = chunk_text(sec["text"], chunk_size, overlap)

        for ch in sec_chunks:
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "section": sec["title"],
                "text": ch["text"],
                "start_word": ch["start_word"],
                "end_word": ch["end_word"],
                "type": "body",
                "page_num": sec.get("page_num", 1)  # <--- O SEGREDO ESTÁ AQUI!
            })
            global_chunk_id += 1

    return all_chunks



# ---------------------------
# 7. FULL PIPELINE
# ---------------------------

def extract_doi(text: str) -> str:
    """
    Extract DOI from text using regex.
    DOI format: 10.XXXX/... (e.g., 10.1000/xyz123)
    """
    doi_pattern = r'\b(10\.\d{4,}/[^\s]+)'
    match = re.search(doi_pattern, text)
    if match:
        doi = match.group(1)
        # Clean trailing punctuation
        doi = re.sub(r'[.,;:\]\)>]+$', '', doi)
        return doi
    return ""


def extract_pdf_metadata(pdf_path: str, pages_text: List[Dict] = None) -> Dict:
    """
    Extract metadata from PDF for citation/reference purposes.
    """
    doc = fitz.open(pdf_path)
    meta = doc.metadata

    # Try to find DOI in first few pages
    doi = ""
    if pages_text:
        first_pages_text = " ".join(p["text"] for p in pages_text[:5])
        doi = extract_doi(first_pages_text)

    return {
        "filename": os.path.basename(pdf_path),
        "filepath": pdf_path,
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "creation_date": meta.get("creationDate", ""),
        "modification_date": meta.get("modDate", ""),
        "total_pages": len(doc),
        "doi": doi
    }


def chunk_individual_footnotes(footnotes_text: str, chunk_size=200, overlap=0) -> List[Dict]:
    """
    Trata cada footnote como uma unidade semântica separada.
    Evita misturar a Footnote 1 com a Footnote 2 no mesmo chunk.
    """
    # 1. Recuperar a lista original (assumindo que usámos \n\n como separador na extração)
    # O filtro 'if n.strip()' remove entradas vazias causadas por quebras duplas
    raw_notes = [n.strip() for n in footnotes_text.split('\n\n') if n.strip()]
    
    chunks = []
    global_chunk_id = 0

    for note in raw_notes:
        words = note.split()
        
        # CASO 1: Footnote curta (cabe num chunk)
        # Criamos um chunk único e perfeito, sem overlap, para não haver repetição.
        if len(words) <= chunk_size:
            chunks.append({
                "chunk_id": f"fn_{global_chunk_id}",
                "text": note,
                "type": "footnote_whole",
                "word_count": len(words)
            })
            global_chunk_id += 1
            
        # CASO 2: Footnote longa (ex: jurídica ou explicativa longa)
        # Só aqui aplicamos o chunking deslizante
        else:
            sub_chunks = chunk_text(note, chunk_size, overlap)
            for sc in sub_chunks:
                # Ajustar o ID para identificar que é parte de uma nota
                chunks.append({
                    "chunk_id": f"fn_{global_chunk_id}_{sc['chunk_id']}",
                    "text": sc["text"],
                    "type": "footnote_part",
                    "word_count": len(sc["text"].split())
                })
            global_chunk_id += 1

    return chunks

def process_pdf(pdf_path: str) -> Dict:
    """
    Full pipeline:
    - extract text
    - extract metadata
    - classify document
    - clean headers/footers
    - chunk text
    - chunk footnotes separadamente
    """

    # 1️⃣ Extração
    pages = extract_text_from_pdf(pdf_path)
    doc_type = classify_pdf(pages)
    metadata = extract_pdf_metadata(pdf_path, pages)

    if doc_type == "scanned":
        ocr_text = extract_text_from_scanned_pdf(pdf_path)
        pages = [{
            "page_num": i + 1,
            "text": text,
            "footnotes": ""  # OCR não detecta footnotes automaticamente
        } for i, text in enumerate(ocr_text.split("\n--- PAGE"))]
    else:
        pages = remove_front_matter_pages(pages, doc_type)
        pages = remove_toc_pages(pages)
        pages = truncate_appendices(pages)
        pages = remove_references_section(pages)

    # 2️⃣ Limpeza de headers/footers
    cleaned_pages = remove_repeated_headers_footers(pages)

    # # 3️⃣ Normalizar texto do corpo
    # full_text = " ".join(p["text"] for p in cleaned_pages)
    # full_text = normalize_text(full_text)

    # # 4️⃣ Criar chunks do corpo
    # chunks = chunk_text(full_text)

    # 3️⃣ Normalizar texto do corpo (por página)
    for p in cleaned_pages:
        p["text"] = normalize_text(p["text"])

    # 4️⃣ Split por secções
    sections = split_into_sections(cleaned_pages)

    # 5️⃣ Chunking por secção
    chunks = chunk_sections(sections, chunk_size=300, overlap=50)

    # 5️⃣ Normalizar e chunkar footnotes
    all_footnotes_list = []
    for p in cleaned_pages:
        fn_text = p.get("footnotes", "").strip()
        if fn_text:
            all_footnotes_list.append(fn_text)
    
    # Juntamos com \n\n para manter a estrutura, e passamos ao chunker inteligente
    full_footnotes_str = "\n\n".join(all_footnotes_list)
    footnote_chunks = chunk_individual_footnotes(full_footnotes_str, chunk_size=200)

    # 6️⃣ Retornar resultado
    return {
        "metadata": metadata,
        "document_type": doc_type,
        "chunks": chunks,
        "footnote_chunks": footnote_chunks  # para vetorização
    }


# ---------------------------
# 8. EXAMPLE USAGE
# ---------------------------

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
            print("  Chunks:", len(output["chunks"]))

            if output["chunks"]:
                preview = output["chunks"][0]["text"][:300]
                print("  Sample chunk:", preview, "...")

        except Exception as e:
            print(f"  ❌ Error processing {pdf_name}: {e}")


if __name__ == "__main__":
    main()
