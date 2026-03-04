import json
import os
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
import numpy as np

# --- CONFIGURAÇÃO DA BASE DE DADOS ---
DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
INPUT_DIR = "data/embeddings" # A pasta onde o script anterior guardou os vetores

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
    
    # 1. Ativar a extensão pgvector (só funciona porque fizeste o 'make install' antes)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Criar a tabela para guardar os chunks
    # Repara que o 'embedding_dense' tem o tipo especial 'vector(1024)'
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
    print("   -> A criar índice HNSW (ajuda o RAG a pesquisar em milissegundos)...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS dense_vector_index 
        ON document_chunks 
        USING hnsw (embedding_dense vector_cosine_ops);
    """)
    
    cur.close()
    conn.close()
    print("✅ Tabela e extensão configuradas com sucesso!")

def insert_chunks(filename):
    """Lê um JSON e insere os seus chunks na base de dados"""
    filepath = os.path.join(INPUT_DIR, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    if not chunks: return

    conn = connect_db()
    cur = conn.cursor()
    
    print(f"📥 A inserir {len(chunks)} chunks do documento: {filename}...")
    
    for chunk in chunks:
        # Extrair os vetores
        dense_vec = chunk.get("embedding_dense")
        sparse_vec = chunk.get("embedding_sparse", {})
        
        # Verificar se o vetor denso existe e tem o tamanho correto (1024)
        if not dense_vec or len(dense_vec) != VECTOR_DIM:
            print(f"⚠️  Chunk {chunk.get('chunk_id')} ignorado: sem vetor ou tamanho errado.")
            continue

        # Comando SQL para inserir os dados
        sql = """
            INSERT INTO document_chunks 
            (chunk_id, source_file, section, text, page_num, embedding_dense, embedding_sparse, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Garantir que a página é um número inteiro (fallback para 0)
        try: 
            page = int(chunk.get("page_num", 0))
        except: 
            page = 0

        # Mapear os dados para as colunas do SQL
        values = (
            str(chunk.get("chunk_id")),
            filename,
            chunk.get("section", "General"),
            chunk.get("text", ""),
            page,
            dense_vec,           # O psycopg2 + pgvector convertem a lista de Python para vetor SQL automaticamente
            Json(sparse_vec),    # Guardamos as palavras-chave (sparse) no formato JSONB
            Json(data.get("metadata", {})) # Metadados do PDF (título, autor, etc.)
        )
        
        cur.execute(sql, values)

    conn.commit() # Gravar as alterações
    cur.close()
    conn.close()

def main():
    # 1. Configura a BD (só faz efeito na primeira vez, depois ignora se já existir)
    setup_database()
    
    # 2. Procura os JSONs na pasta data/embeddings
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    if not files:
        print(f"❌ Nenhum ficheiro encontrado na pasta {INPUT_DIR}. Correste o script de embeddings?")
        return

    # 3. Inserir cada ficheiro
    for f in files:
        try:
            insert_chunks(f)
        except Exception as e:
            print(f"❌ Erro ao processar {f}: {e}")

    print("\n🎉 Ingestão concluída! A tua base de dados vetorial está pronta.")

if __name__ == "__main__":
    main()