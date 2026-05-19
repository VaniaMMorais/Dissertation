"""
Estatísticas da Base de Dados (PostgreSQL + pgvector)
Ajustado à tabela 'document_chunks' do ingest_postgres.py

Corre no Linux com acesso à BD:
  python3 corpus_stats_db.py
"""

import psycopg2

# --- CONFIGURAÇÃO (igual ao ingest_postgres.py) ---
DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"

def connect_db():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

def run_query(cursor, query, label):
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        value = result[0] if result else "N/A"
        print(f"  {label:<50} {value}")
        return value
    except Exception as e:
        print(f"  {label:<50} ERRO: {e}")
        return None

def run_query_rows(cursor, query, label):
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"\n  {label}")
        print(f"  {'─'*60}")
        for row in rows:
            print(f"    {row[0]:<45} {row[1]}")
        return rows
    except Exception as e:
        print(f"  {label:<50} ERRO: {e}")
        return None

def main():
    print(f"\n{'='*65}")
    print("ESTATÍSTICAS DA BASE DE DADOS")
    print(f"{'='*65}\n")

    try:
        conn = connect_db()
        cur = conn.cursor()
    except Exception as e:
        print(f"Erro de conexão: {e}")
        return

    # ─── TOTAIS ───
    print("TOTAIS:")
    run_query(cur, "SELECT COUNT(*) FROM document_chunks", "Total de chunks")
    run_query(cur, "SELECT COUNT(DISTINCT source_file) FROM document_chunks", "Documentos distintos")

    # ─── POR TIPO (text vs image) ───
    print(f"\n{'─'*65}")
    print("POR TIPO DE CHUNK:")
    run_query(cur,
        "SELECT COUNT(*) FROM document_chunks WHERE metadata->>'chunk_type' = 'text'",
        "Chunks de texto")
    run_query(cur,
        "SELECT COUNT(*) FROM document_chunks WHERE metadata->>'chunk_type' = 'image'",
        "Chunks de imagem")

    # ─── ESTATÍSTICAS DE TEXTO ───
    print(f"\n{'─'*65}")
    print("ESTATÍSTICAS DE TEXTO (palavras por chunk):")
    run_query(cur,
        "SELECT ROUND(AVG(array_length(string_to_array(text, ' '), 1))) FROM document_chunks WHERE metadata->>'chunk_type' = 'text'",
        "Média de palavras por chunk (texto)")
    run_query(cur,
        "SELECT MIN(array_length(string_to_array(text, ' '), 1)) FROM document_chunks WHERE metadata->>'chunk_type' = 'text'",
        "Chunk mais curto (palavras)")
    run_query(cur,
        "SELECT MAX(array_length(string_to_array(text, ' '), 1)) FROM document_chunks WHERE metadata->>'chunk_type' = 'text'",
        "Chunk mais longo (palavras)")

    # ─── POR DOCUMENTO ───
    print(f"\n{'─'*65}")
    run_query_rows(cur,
        """SELECT source_file, COUNT(*) as total 
           FROM document_chunks 
           GROUP BY source_file 
           ORDER BY total DESC""",
        "CHUNKS POR DOCUMENTO:")

    # ─── IMAGENS POR DOCUMENTO ───
    print(f"\n{'─'*65}")
    run_query_rows(cur,
        """SELECT source_file, COUNT(*) as total 
           FROM document_chunks 
           WHERE metadata->>'chunk_type' = 'image'
           GROUP BY source_file 
           ORDER BY total DESC""",
        "IMAGENS POR DOCUMENTO:")

    # ─── SECÇÕES MAIS COMUNS ───
    print(f"\n{'─'*65}")
    run_query_rows(cur,
        """SELECT section, COUNT(*) as total 
           FROM document_chunks 
           GROUP BY section 
           ORDER BY total DESC 
           LIMIT 15""",
        "TOP 15 SECÇÕES MAIS FREQUENTES:")

    cur.close()
    conn.close()
    print(f"\n✅ Concluído!")

if __name__ == "__main__":
    main()