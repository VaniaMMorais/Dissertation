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
INPUT_DIR = "../data/embeddings" # Ajustado para o caminho correto se correres dentro da pasta table_extraction

# Dimensão do BGE-M3 (não alterar)
VECTOR_DIM = 1024 

def connect_db():
    """Cria a ligação ao PostgreSQL"""
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

def setup_database():
    """Configura a extensão pgvector e cria a tabela"""
    conn = connect_db()
    conn.autocommit = True
    cur = conn.cursor()

    print("🛠️  A preparar a base de dados...")
    
    # 1. Ativar a extensão pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Criar a tabela para guardar os chunks
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
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
    
    # 3. Criar Índice HNSW para pesquisa super rápida
    print("   -> A criar/verificar índice HNSW...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS dense_vector_index 
        ON document_chunks 
        USING hnsw (embedding_dense vector_cosine_ops);
    """)
    
    cur.close()
    conn.close()
    print("✅ Tabela e extensão prontas!")

def insert_chunks(filename):
    """Lê um JSON e insere os seus chunks na base de dados"""
    filepath = os.path.join(INPUT_DIR, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JUNTAR OS CHUNKS NORMAIS E AS NOTAS DE RODAPÉ!
    chunks = data.get("chunks", [])
    footnotes = data.get("footnote_chunks", [])
    all_chunks = chunks + footnotes

    if not all_chunks: 
        return

    conn = connect_db()
    cur = conn.cursor()
    
    # PROTEÇÃO CONTRA DUPLICADOS: Apaga os dados deste ficheiro se já existirem na BD
    cur.execute("DELETE FROM document_chunks WHERE source_file = %s", (filename,))
    
    print(f"📥 A inserir {len(all_chunks)} blocos ({len(chunks)} texto, {len(footnotes)} notas) do documento: {filename}...")
    
    for chunk in all_chunks:
        # Extrair os vetores
        dense_vec = chunk.get("embedding_dense")
        sparse_vec = chunk.get("embedding_sparse", {})
        
        if not dense_vec or len(dense_vec) != VECTOR_DIM:
            print(f"⚠️  Chunk {chunk.get('chunk_id')} ignorado: sem vetor ou tamanho errado.")
            continue

        # Identificar se é uma nota de rodapé pelo ID
        is_footnote = str(chunk.get("chunk_id", "")).startswith("fn_")
        section_name = chunk.get("section", "Nota de Rodapé" if is_footnote else "Geral")

        # Comando SQL
        sql = """
            INSERT INTO document_chunks 
            (chunk_id, source_file, section, text, page_num, embedding_dense, embedding_sparse, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try: 
            page = int(chunk.get("page_num", 0))
        except: 
            page = 0

        values = (
            str(chunk.get("chunk_id")),
            filename,
            section_name,
            chunk.get("text", ""),
            page,
            dense_vec,
            Json(sparse_vec),
            Json(data.get("metadata", {}))
        )
        
        cur.execute(sql, values)

    conn.commit()
    cur.close()
    conn.close()

def main():
    setup_database()
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ A pasta {INPUT_DIR} não existe! Verifica o caminho.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    if not files:
        print(f"❌ Nenhum ficheiro encontrado na pasta {INPUT_DIR}. Correste o script de embeddings?")
        return

    for f in files:
        try:
            insert_chunks(f)
        except Exception as e:
            print(f"❌ Erro ao processar {f}: {e}")

    print("\n🎉 Ingestão concluída! A tua base de dados vetorial está carregada com sucesso.")

if __name__ == "__main__":
    main()