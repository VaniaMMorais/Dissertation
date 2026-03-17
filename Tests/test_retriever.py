import psycopg2
import torch
from FlagEmbedding import BGEM3FlagModel

# --- CONFIGURAÇÃO ---
DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
MODEL_NAME = "BAAI/bge-m3"

def connect_db():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

def dense_search(query_text, top_k=5):
    """
    Pesquisa Semântica usando os embeddings Densos e pgvector.
    """
    print(f"🧠 A gerar vetor para a pergunta: '{query_text}'...")
    use_fp16 = torch.cuda.is_available()
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)
    
    # 1. Converter a pergunta num vetor
    output = model.encode([query_text], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()

    # 2. Pesquisar na BD
    conn = connect_db()
    cur = conn.cursor()
    
    # O operador <=> calcula a distância de cosseno. 
    # Fazemos 1 - distância para obter a Similaridade (0 a 1)
    sql = """
        SELECT chunk_id, source_file, section, page_num, text, 
               1 - (embedding_dense <=> %s::vector) as similarity
        FROM document_chunks
        ORDER BY embedding_dense <=> %s::vector
        LIMIT %s;
    """
    
    cur.execute(sql, (query_vector, query_vector, top_k))
    results = cur.fetchall()
    
    conn.close()
    return results

def format_context_for_llm(results):
    """
    Formata os resultados exatamente como vamos enviar para o Gemini no próximo passo.
    """
    context_blocks = []
    for rank, (chunk_id, source, section, page, text, sim) in enumerate(results):
        block = (
            f"--- DOCUMENTO {rank + 1} ---\n"
            f"Fonte: {source} (Página {page})\n"
            f"Secção: {section}\n"
            f"Relevância Semântica: {sim:.2f}\n"
            f"Texto: {text}\n"
        )
        context_blocks.append(block)
    
    return "\n".join(context_blocks)

def main():
    # 💡 MUDA ESTA PERGUNTA PARA ALGO ESPECÍFICO DOS TEUS PDFS
    # Ex: Algo sobre o Nudging (Tese) ou Eczema (Artigo)
    query = "O que é uma doença?"
    
    print("\n🔎 A iniciar pesquisa...\n")
    results = dense_search(query, top_k=3)
    
    if not results:
        print("❌ Nenhum resultado encontrado. A BD está vazia?")
        return
        
    print("\n✅ CONTEXTO RECUPERADO (Pronto para o Gemini):\n")
    context_string = format_context_for_llm(results)
    print(context_string)

if __name__ == "__main__":
    main()