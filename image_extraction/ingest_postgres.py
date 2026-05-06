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
INPUT_DIR = "../data/embeddings"

VECTOR_DIM = 1024 

def connect_db():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

def setup_database():
    """Cria tabela SE NÃO EXISTIR (não apaga dados!)"""
    conn = connect_db()
    conn.autocommit = True
    cur = conn.cursor()

    print("🛠️  A verificar/criar estrutura da base de dados...")
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # ✅ CRIAR TABELA SÓ SE NÃO EXISTIR (não apaga nada!)
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
    
    # Verificar se índice já existe
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM pg_indexes 
            WHERE indexname = 'dense_vector_index'
        );
    """)
    index_exists = cur.fetchone()[0]
    
    if not index_exists:
        print("   -> A criar índice HNSW para pesquisa ultrarrápida...")
        cur.execute("""
            CREATE INDEX dense_vector_index 
            ON document_chunks 
            USING hnsw (embedding_dense vector_cosine_ops);
        """)
    else:
        print("   -> Índice HNSW já existe.")
    
    cur.close()
    conn.close()
    print("✅ Base de dados pronta!")

def document_exists(cursor, source_file):
    """Verifica se documento já foi ingerido."""
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM document_chunks WHERE source_file = %s)",
        (source_file,)
    )
    return cursor.fetchone()[0]

def insert_chunks(filename):
    filepath = os.path.join(INPUT_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JUNTAR TUDO: Texto + Footnotes + Imagens!
    chunks = data.get("chunks", [])
    footnotes = data.get("footnote_chunks", [])
    images = data.get("image_chunks", [])
    
    all_chunks = chunks + footnotes + images

    if not all_chunks: 
        return 0

    conn = connect_db()
    cur = conn.cursor()
    
    print(f"📥 A inserir {len(all_chunks)} blocos ({len(chunks)} texto, {len(footnotes)} notas, {len(images)} imagens) de: {filename}...")
    
    inserted = 0
    for chunk in all_chunks:
        dense_vec = chunk.get("embedding_dense")
        sparse_vec = chunk.get("embedding_sparse", {})
        
        if not dense_vec or len(dense_vec) != VECTOR_DIM:
            continue

        # Metadata específico para esta linha
        row_metadata = data.get("metadata", {}).copy()
        
        # SE FOR IMAGEM OU FIGURA, GUARDAMOS O CAMINHO E LEGENDA!
        tipo = chunk.get("type", "")
        if tipo == "image" or tipo == "figure":
            row_metadata["image_path"] = chunk.get("image_path")
            row_metadata["caption"] = chunk.get("caption", "") 
            row_metadata["chunk_type"] = "image"
        else:
            row_metadata["chunk_type"] = "text"

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
            chunk.get("section", "Geral"),
            chunk.get("text", ""),
            page,
            dense_vec,
            Json(sparse_vec),
            Json(row_metadata)
        )
        
        cur.execute(sql, values)
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    
    return inserted

def main():
    setup_database()
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not files:
        print(f"❌ Nenhum ficheiro encontrado na pasta {INPUT_DIR}.")
        return

    # ✅ VERIFICAR QUAIS JÁ ESTÃO NA BD
    conn = connect_db()
    cur = conn.cursor()
    
    already_ingested = []
    new_files = []
    
    for filename in files:
        if document_exists(cur, filename):
            already_ingested.append(filename)
        else:
            new_files.append(filename)
    
    cur.close()
    conn.close()
    
    print(f"\n📊 Encontrados {len(files)} ficheiros com embeddings")
    print(f"✅ Já na base de dados: {len(already_ingested)}")
    print(f"🆕 Novos para ingerir: {len(new_files)}\n")
    
    if not new_files:
        print("🎉 Todos os documentos já estão na base de dados!")
        return
    
    # ✅ PROCESSAR SÓ OS NOVOS
    total_chunks = 0
    for filename in new_files:
        try:
            chunks_inserted = insert_chunks(filename)
            total_chunks += chunks_inserted
            print(f"   ✅ {chunks_inserted} chunks inseridos")
        except Exception as e:
            print(f"   ❌ Erro ao processar {filename}: {e}")

    print(f"\n🎉 INGESTÃO CONCLUÍDA! {total_chunks} chunks adicionados ao PostgreSQL.")

if __name__ == "__main__":
    main()