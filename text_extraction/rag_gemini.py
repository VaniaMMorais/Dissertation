import psycopg2
import torch
from google import genai
from FlagEmbedding import BGEM3FlagModel
import os
from dotenv import load_dotenv

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API Key não encontrada! Verifica o teu ficheiro .env")

# --- CONFIGURAÇÃO ---
DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
MODEL_NAME = "BAAI/bge-m3"

# Configurar o NOVO cliente do Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

def dense_search(query_text, top_k=5):
    """Busca os parágrafos mais relevantes e os seus metadados na base de dados."""
    use_fp16 = torch.cuda.is_available()
    embed_model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)
    
    output = embed_model.encode([query_text], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()

    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    
    # ADICIONADO: Agora pedimos também a coluna 'metadata' ao PostgreSQL
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
    """Constrói o prompt com o contexto, priorizando o Título e incluindo o link do DOI."""
    
    context_text = ""
    for i, (source, page, text, metadata) in enumerate(context_results):
        
        # 1. Fallback base: Nome do ficheiro limpo
        clean_source = source.replace(".json", "").replace(".pdf", "")
        document_name = clean_source # Começamos por assumir o nome do ficheiro
        
        doi_link = ""
        
        # 2. Inspecionar os Metadados (Onde a magia acontece)
        if metadata and isinstance(metadata, dict):
            # Substituir pelo TÍTULO, se existir e não estiver vazio
            if metadata.get("title"):
                document_name = metadata["title"]
                
            # Extrair e formatar o DOI, se existir
            if metadata.get("doi"):
                doi_cru = metadata["doi"]
                if doi_cru.startswith("http"):
                    doi_link = doi_cru
                else:
                    doi_link = f"https://doi.org/{doi_cru}"
        
        doi_info = f"\nLink/DOI: {doi_link}" if doi_link else ""
        
        # 3. Construir o bloco de contexto para o Gemini
        context_text += f"\n--- INÍCIO DA FONTE {i+1} ---\nDocumento: {document_name}\nPágina: {page}{doi_info}\nTexto: {text}\n--- FIM DA FONTE {i+1} ---\n"

    # 4. O Prompt Atualizado
    prompt = f"""
    You are an elite academic research assistant, specialized in analyzing documents and extracting precise answers.
    Below, I provide you with context extracted from scientific databases and a user query.
    
    MANDATORY RULES:
    1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that same language. If the query is in English, reply in English. If it is in Portuguese, reply in Portuguese.
    2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context. Do not use outside knowledge or hallucinate.
    3. CITATIONS WITH LINKS: Whenever you make a claim, you MUST cite the source at the end of the sentence.
       - Always use the provided 'Document' field to name the source.
       - If the source has a 'Link/DOI', format the citation as a clickable Markdown link: [Document Name, Page X](URL_DO_DOI)
       - If there is no DOI, use plain text: [Document Name, Page X]
       Example: "... topical treatments are effective [Dermatology Article, Page 12](https://doi.org/10.1111/jdv.18946)."
    4. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information in the documents.
    
    PROVIDED CONTEXT:
    {context_results}
    
    USER QUERY:
    {query}
    
    FORMATTED RESPONSE:
    """

    print("⏳ A pedir ao Gemini para formular a resposta (com citações ricas)...")
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text

def main():
    print("="*50)
    print("🤖 BEM-VINDO AO TEU SISTEMA RAG (TESE)")
    print("="*50)
    
    query = input("\nFaz a tua pergunta: ")
    
    if not query.strip():
        return
        
    print("\n🔎 A procurar na base de dados (PostgreSQL + pgvector)...")
    results = dense_search(query, top_k=4) 
    
    if not results:
        print("❌ Não foi encontrada informação relevante na base de dados.")
        return
        
    resposta = ask_gemini(query, results)
    
    print("\n✨ RESPOSTA DO GEMINI ✨")
    print("-" * 50)
    print(resposta)
    print("-" * 50)

if __name__ == "__main__":
    main()