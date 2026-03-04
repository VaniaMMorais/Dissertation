import streamlit as st
import psycopg2
import torch
from google import genai
from FlagEmbedding import BGEM3FlagModel
import os
from dotenv import load_dotenv

# --- CONFIGURAÇÃO INICIAL E SEGURANÇA ---
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ API Key não encontrada! Verifica o teu ficheiro .env")
    st.stop()

DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
MODEL_NAME = "BAAI/bge-m3"

client = genai.Client(api_key=GEMINI_API_KEY)

# --- CACHE DO MODELO BGE-M3 ---
# O Streamlit guarda isto em memória para a app não ficar lenta
@st.cache_resource
def load_embedding_model():
    use_fp16 = torch.cuda.is_available()
    return BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)

embed_model = load_embedding_model()

# --- FUNÇÕES CORE DO RAG ---
def dense_search(query_text, top_k=5):
    """Busca os parágrafos mais relevantes no PostgreSQL."""
    output = embed_model.encode([query_text], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()

    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    
    sql = """
        SELECT source_file, page_num, text, metadata 
        FROM document_chunks
        ORDER BY embedding_dense <=> %s::vector
        LIMIT %s;
    """
    cur.execute(sql, (query_vector, top_k))
    results = cur.fetchall()
    conn.close()
    
    return results

def ask_gemini(query, context_results):
    """Constrói o prompt com o contexto e envia para o Gemini."""
    context_text = ""
    for i, (source, page, text, metadata) in enumerate(context_results):
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
    1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that same language.
    2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context. Do not use outside knowledge or hallucinate.
    3. CITATIONS WITH LINKS: Whenever you make a claim, you MUST cite the source at the end of the sentence.
       - Always use the provided 'Document' field to name the source.
       - If the source has a 'Link/DOI', format the citation as a clickable Markdown link: [Document Name, Page X](URL_DO_DOI)
       - If there is no DOI, use plain text: [Document Name, Page X]
    4. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information in the documents.
    
    PROVIDED CONTEXT:
    {context_text}
    
    USER QUERY:
    {query}
    
    FORMATTED RESPONSE:
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text

# --- INTERFACE GRÁFICA (STREAMLIT UI) ---
st.set_page_config(page_title="Dissertation", page_icon="🎓", layout="centered")
st.title("AI-based Platform for Extracting and Processing Information from Scientific Articles")
st.caption("Question-Answering system based on indexed documents.")

# Inicializar o histórico de mensagens na memória do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Desenhar as mensagens antigas no ecrã
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de texto para o utilizador escrever
if prompt := st.chat_input("Ask me anything"):
    
    # 1. Adicionar e mostrar a pergunta do utilizador
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Mostrar o balão a dizer que está a pensar...
    with st.chat_message("assistant"):
        with st.spinner("Searching through documents and generating responses..."):
            
            # Executar a nossa Pipeline RAG
            context_results = dense_search(prompt, top_k=4)
            
            if not context_results:
                resposta_final = "Sorry, I couldn't find any relevant information in the database."
            else:
                resposta_final = ask_gemini(prompt, context_results)
            
            # Mostrar a resposta no ecrã
            st.markdown(resposta_final)
            
            # Adicionar a resposta da IA ao histórico
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})