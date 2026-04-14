import streamlit as st
import psycopg2
import torch
from google import genai
from FlagEmbedding import BGEM3FlagModel
import os
import re
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ API Key não encontrada! Verifica o .env")
    st.stop()

DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
MODEL_NAME = "BAAI/bge-m3"

client = genai.Client(api_key=GEMINI_API_KEY)

# --- CACHE ---
@st.cache_resource
def load_embedding_model():
    use_fp16 = torch.cuda.is_available()
    return BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)

embed_model = load_embedding_model()


# --- HELPER FUNCTIONS ---
def extract_cited_sources(gemini_response: str) -> set:
    """Extrai documentos citados."""
    cited = set()
    pattern1 = r'\[([^,\]]+),\s*Page\s+\d+\]'
    matches1 = re.findall(pattern1, gemini_response, re.IGNORECASE)
    cited.update(m.strip() for m in matches1)
    
    pattern2 = r'\[([^,\]]+),\s*Page\s+\d+\]\([^\)]+\)'
    matches2 = re.findall(pattern2, gemini_response, re.IGNORECASE)
    cited.update(m.strip() for m in matches2)
    
    return cited


def match_source_to_citation(source_file: str, metadata: dict, cited_names: set) -> bool:
    """Verifica se source corresponde a citação."""
    clean_source = source_file.replace(".json", "").replace(".pdf", "").replace("_", " ")
    doc_title = metadata.get("title", "") if metadata else ""
    
    for cited in cited_names:
        cited_lower = cited.lower()
        if (cited_lower in clean_source.lower() or 
            cited_lower in doc_title.lower() or
            clean_source.lower() in cited_lower or
            doc_title.lower() in cited_lower):
            return True
    return False


# --- FUNÇÕES CORE RAG ---
def hybrid_search(query_text, top_k=6):
    """Busca híbrida - SEM coluna fts."""
    output = embed_model.encode([query_text], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()
    lexical_query = " | ".join(query_text.replace("'", "").split())

    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    
    # Query MODIFICADA - cria tsvector on-the-fly
    sql = """
    WITH semantic_search AS (
        SELECT id, source_file, page_num, text, metadata,
               RANK() OVER (ORDER BY embedding_dense <=> %s::vector) AS rank
        FROM document_chunks
        ORDER BY embedding_dense <=> %s::vector
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, source_file, page_num, text, metadata,
               RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('simple', text), to_tsquery('simple', %s)) DESC) AS rank
        FROM document_chunks
        WHERE to_tsvector('simple', text) @@ to_tsquery('simple', %s)
        ORDER BY ts_rank_cd(to_tsvector('simple', text), to_tsquery('simple', %s)) DESC
        LIMIT 20
    )
    SELECT 
        COALESCE(s.source_file, k.source_file) as source_file,
        COALESCE(s.page_num, k.page_num) as page_num,
        COALESCE(s.text, k.text) as text,
        COALESCE(s.metadata, k.metadata) as metadata,
        COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """
    
    cur.execute(sql, (query_vector, query_vector, lexical_query, lexical_query, lexical_query, top_k))
    results = cur.fetchall()
    conn.close()
    
    return results


# def ask_gemini(query, context_results):
#     """Gera resposta - FILTRA imagens do contexto."""
#     context_text = ""
    
#     # Filtrar só chunks de TEXTO
#     text_chunks = []
#     for source, page, text, metadata, rrf_score in context_results:
#         chunk_type = metadata.get("chunk_type", "text") if metadata else "text"
#         if chunk_type != "image":  # ← LER DO METADATA
#             text_chunks.append((source, page, text, metadata, rrf_score))
    
#     for i, (source, page, text, metadata, rrf_score) in enumerate(text_chunks):
#         clean_source = source.replace(".json", "").replace(".pdf", "")
#         document_name = clean_source 
#         doi_link = ""
        
#         if metadata and isinstance(metadata, dict):
#             if metadata.get("title"):
#                 document_name = metadata["title"]
#             if metadata.get("doi"):
#                 doi_cru = metadata["doi"]
#                 if doi_cru.startswith("http"):
#                     doi_link = doi_cru
#                 else:
#                     doi_link = f"https://doi.org/{doi_cru}"
        
#         doi_info = f"\nLink/DOI: {doi_link}" if doi_link else ""
#         context_text += f"\n--- INÍCIO DA FONTE {i+1} ---\nDocumento: {document_name}\nPágina: {page}{doi_info}\nTexto: {text}\n--- FIM DA FONTE {i+1} ---\n"

#     prompt = f"""
# You are an elite academic research assistant, specialized in analyzing documents and extracting precise answers.
# Below, I provide you with context extracted from scientific databases and a user query.

# MANDATORY RULES:
# 1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that SAME LANGUAGE.
# 2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context. Do not use outside knowledge or hallucinate.
# 3. COMPLETE ANSWERS: Provide COMPLETE, DETAILED answers based on the text context.
#    - DO NOT simply refer to figures or tables (e.g., "see Figure 1")
#    - EXTRACT and EXPLAIN the key information from the text
#    - If relevant visual content exists, it will be shown separately to the user
# 4. CITATIONS WITH LINKS: Whenever you make a claim, you MUST cite the source at the end of the sentence.
#    - Always use the provided 'Document' field to name the source.
#    - If the source has a 'Link/DOI', format the citation as a clickable Markdown link: [Document Name, Page X](URL_DO_DOI)
#    - If there is no DOI, use plain text: [Document Name, Page X]
# 5. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information in the documents.

# PROVIDED CONTEXT:
# {context_text}

# USER QUERY:
# {query}

# FORMATTED RESPONSE:
# """

#     try:
#         response = client.models.generate_content(
#             model='gemini-3-flash-preview',
#             contents=prompt,
#             config={
#                 'temperature': 0.4,  
#                 'top_p': 0.9
#             }
#         )
#         return response.text

#     except Exception as e:
#         error_msg = str(e).lower()

#         print(f"\n[ERRO REAL DO GEMINI]: {e}\n")
#         # Deteta se é erro de quota (429) ou excesso de pedidos
#         if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
#             return "⚠️ **Reached Daily Limit:** O sistema atingiu o limite de perguntas gratuitas. Por favor, tenta amanhã."
#         elif "503" in error_msg or "service unavailable" in error_msg:
#             return "⚠️ **High Demand:** A rede está sobrecarregada. Tenta novamente em alguns minutos."
#         # Deteta se o modelo não existe (o erro 404 que tiveste ontem)
#         elif "404" in error_msg or "not found" in error_msg:
#             return "⚠️ **Model Not Found:** O modelo de IA configurado não está disponível neste momento."
#         # Erro genérico
#         else:
#             return "⚠️ **System Error:** Ocorreu um erro de comunicação com a IA. Tenta novamente."

def ask_gemini(query, context_results):
    """Gera resposta com TODO o contexto (Textos + Descrições das Imagens)."""
    context_text = ""
    
    # REMOVIDO o filtro! Agora o Gemini vai voltar a ler as descrições dos gráficos!
    for i, (source, page, text, metadata, rrf_score) in enumerate(context_results):
        clean_source = source.replace(".json", "").replace(".pdf", "")
        document_name = clean_source 
        doi_link = ""
        
        if metadata and isinstance(metadata, dict):
            if metadata.get("title"):
                document_name = metadata["title"]
            if metadata.get("doi"):
                doi_cru = metadata["doi"]
                if doi_cru.startswith("http"):
                    doi_link = doi_cru
                else:
                    doi_link = f"https://doi.org/{doi_cru}"
        
        doi_info = f"\nLink/DOI: {doi_link}" if doi_link else ""
        context_text += f"\n--- INÍCIO DA FONTE {i+1} ---\nDocumento: {document_name}\nPágina: {page}{doi_info}\nTexto: {text}\n--- FIM DA FONTE {i+1} ---\n"

    prompt = f"""
You are an elite academic research assistant, specialized in analyzing documents and extracting precise answers.
Below, I provide you with context extracted from scientific databases and a user query.

MANDATORY RULES:
1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that SAME LANGUAGE.
2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context. Do not use outside knowledge or hallucinate.
3. COMPLETE ANSWERS: Provide COMPLETE, DETAILED answers based on the text context.
   - DO NOT simply refer to figures or tables (e.g., "see Figure 1")
   - EXTRACT and EXPLAIN the key information from the text
   - If relevant visual content exists, it will be shown separately to the user
4. CITATIONS WITH LINKS: Whenever you make a claim, you MUST cite the source at the end of the sentence.
   - Always use the provided 'Document' field to name the source.
   - If the source has a 'Link/DOI', format the citation as a clickable Markdown link: [Document Name, Page X](URL_DO_DOI)
   - If there is no DOI, use plain text: [Document Name, Page X]
5. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information in the documents.

PROVIDED CONTEXT:
{context_text}

USER QUERY:
{query}

FORMATTED RESPONSE:
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite', # <--- CORRIGIDO PARA O MODELO QUE FUNCIONA
            contents=prompt,
            config={
                'temperature': 0.4,  
                'top_p': 0.9
            }
        )
        return response.text

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n[ERRO REAL DO GEMINI]: {e}\n")
        # Deteta se é erro de quota (429) ou excesso de pedidos
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            return "⚠️ **Reached Daily Limit:** O sistema atingiu o limite de perguntas gratuitas. Por favor, tenta amanhã."
        elif "503" in error_msg or "service unavailable" in error_msg:
            return "⚠️ **High Demand:** A rede está sobrecarregada. Tenta novamente em alguns minutos."
        # Deteta se o modelo não existe
        elif "404" in error_msg or "not found" in error_msg:
            return "⚠️ **Model Not Found:** O modelo de IA configurado não está disponível neste momento."
        # Erro genérico
        else:
            return "⚠️ **System Error:** Ocorreu um erro de comunicação com a IA. Tenta novamente."


def optimize_search_query(user_query):
    """Expande query PT/EN."""
    prompt = f"""
You are an expert search engine optimizer. 
Analyze the user's question and extract the most important keywords.
Generate a list of keywords in BOTH Portuguese and English to maximize database retrieval.
Return ONLY the keywords separated by spaces. No punctuation, no quotes, no conversational text.

User question: {user_query}
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        # Se falhar (por exemplo, por falta de quota), NÃO mostramos erro!
        # Simplesmente devolvemos a pergunta original do utilizador para que
        # a pesquisa híbrida tente encontrar os documentos à mesma.
        return user_query   


# --- INTERFACE ---
st.set_page_config(page_title="Dissertation", page_icon="🎓", layout="centered")
st.title("🎓 AI-based Platform for Extracting and Processing Scientific Articles")
st.caption("Question-Answering system with multimodal retrieval (text + figures)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "images" in message:
            for img_info in message["images"]:
                st.image(
                    img_info["path"],
                    caption=f"📄 {img_info['source']}, Página {img_info['page']}",
                    width='stretch'
                )

# Input
if prompt := st.chat_input("Faz uma pergunta sobre os documentos..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 A pesquisar e gerar resposta..."):
            
            # 1. Otimização
            optimized_query = optimize_search_query(prompt)
            st.caption(f"*Query expandida: {optimized_query}*")
            
            # 2. Retrieval
            context_results = hybrid_search(optimized_query, top_k=10)
            
            if not context_results:
                resposta_final = "❌ Sem informação relevante na base de dados para responder a esta pergunta."
                retrieved_images = []
            else:
                # 3. Geração
                resposta_final = ask_gemini(prompt, context_results)
                
                # 4. EXTRAÇÃO DE IMAGENS
                cited_names = extract_cited_sources(resposta_final)
                retrieved_images = []
                
                for source, page, text, metadata, rrf_score in context_results:
                    if not metadata:
                        continue
                    
                    chunk_type = metadata.get("chunk_type", "text")
                    
                    if chunk_type == "image":
                        match_result = match_source_to_citation(source, metadata, cited_names)
                        
                        if match_result:
                            image_path = metadata.get("image_path")
                            caption = metadata.get("caption", "") 
                            
                            if image_path and os.path.exists(image_path):
                                retrieved_images.append({
                                    "path": image_path,
                                    "source": source.replace(".json", ""),
                                    "page": page,
                                    "caption": caption,
                                    "description": text
                                })
                
                # Fallback: Se não encontrou imagens pelas citações, tenta apanhar a primeira imagem do contexto
                if not retrieved_images:
                    for source, page, text, metadata, rrf_score in context_results:
                        if metadata and metadata.get("chunk_type") == "image":
                            image_path = metadata.get("image_path")
                            caption = metadata.get("caption", "")

                            if image_path and os.path.exists(image_path):
                                retrieved_images.append({
                                    "path": image_path,
                                    "source": source.replace(".json", ""),
                                    "page": page,
                                    "caption": caption,
                                    "description": text
                                })
                                break
            
            # 5. Mostrar resposta em texto
            st.markdown(resposta_final)
            
            # 6. Mostrar figuras (se existirem)
            if retrieved_images:
                st.divider()
                st.markdown("### 📊 Figuras Relacionadas")
                
                for img_info in retrieved_images:
                    caption_text = f"📄 {img_info['source']}, Página {img_info['page']}"
                    if img_info.get('caption'):
                        caption_text += f"\n{img_info['caption']}"
                    
                    st.image(
                        img_info["path"],
                        caption=caption_text,
                        use_container_width=True
                    )
            
            # 7. Guardar Histórico
            st.session_state.messages.append({
                "role": "assistant", 
                "content": resposta_final,
                "images": retrieved_images
            })