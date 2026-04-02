import json
import os
import psycopg2
from psycopg2.extras import Json
import numpy as np

# --- CONFIGURAÇÃO DA BASE DE DADOS ---
DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
INPUT_DIR = "../data/embeddings" # Pasta onde estão os JSONs acabados de gerar

VECTOR_DIM = 1024 

def connect_db():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

def setup_database():
    conn = connect_db()
    conn.autocommit = True
    cur = conn.cursor()

    print("🛠️  A preparar a base de dados para o RAG Multimodal...")
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 1. APAGAR A TABELA ANTIGA PARA COMEÇAR DO ZERO
    print("   -> A apagar dados antigos (Drop Table)...")
    cur.execute("DROP TABLE IF EXISTS document_chunks;")
    
    # 2. CRIAR A NOVA TABELA
    cur.execute(f"""
        CREATE TABLE document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(255),
            source_file TEXT,
            section TEXT,
            text TEXT,
            page_num INTEGER,
            embedding_dense vector({VECTOR_DIM}),
            embedding_sparse JSONB,
            metadata JSONB
        );
    """)
    
    print("   -> A criar índice HNSW para pesquisa ultrarrápida...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS dense_vector_index 
        ON document_chunks 
        USING hnsw (embedding_dense vector_cosine_ops);
    """)
    
    cur.close()
    conn.close()
    print("✅ Tabela fresquinha e pronta a receber imagens e texto!")

def insert_chunks(filename):
    filepath = os.path.join(INPUT_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JUNTAR TUDO: Texto + Footnotes + Imagens!
    chunks = data.get("chunks", [])
    footnotes = data.get("footnote_chunks", [])
    images = data.get("image_chunks", [])
    
    all_chunks = chunks + footnotes + images

    if not all_chunks: return

    conn = connect_db()
    cur = conn.cursor()
    
    print(f"📥 A inserir {len(all_chunks)} blocos ({len(chunks)} texto, {len(footnotes)} notas, {len(images)} imagens) de: {filename}...")
    
    for chunk in all_chunks:
        dense_vec = chunk.get("embedding_dense")
        sparse_vec = chunk.get("embedding_sparse", {})
        
        if not dense_vec or len(dense_vec) != VECTOR_DIM:
            continue

        # Vamos criar um metadata específico para esta linha
        row_metadata = data.get("metadata", {}).copy()
        
        # SE FOR IMAGEM, GUARDAMOS O CAMINHO PARA O STREAMLIT A PODER MOSTRAR!
        if chunk.get("type") == "image":
            row_metadata["image_path"] = chunk.get("image_path")
            row_metadata["chunk_type"] = "image"
        else:
            row_metadata["chunk_type"] = "text"

        sql = """
            INSERT INTO document_chunks 
            (chunk_id, source_file, section, text, page_num, embedding_dense, embedding_sparse, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try: page = int(chunk.get("page_num", 0))
        except: page = 0

        values = (
            str(chunk.get("chunk_id")),
            filename,
            chunk.get("section", "Geral"),
            chunk.get("text", ""),
            page,
            dense_vec,
            Json(sparse_vec),
            Json(row_metadata) # Metadata agora tem o image_path!
        )
        
        cur.execute(sql, values)

    conn.commit()
    cur.close()
    conn.close()

def main():
    setup_database()
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not files:
        print(f"❌ Nenhum ficheiro encontrado na pasta {INPUT_DIR}.")
        return

    for f in files:
        try:
            insert_chunks(f)
        except Exception as e:
            print(f"❌ Erro ao processar {f}: {e}")

    print("\n🎉 INGESTÃO CONCLUÍDA! O teu PostgreSQL é agora 100% Multimodal.")

if __name__ == "__main__":
    main()