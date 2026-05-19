"""
Performance Benchmark — Multimodal RAG Pipeline
Mede a latência de cada fase do pipeline para as 20 queries de teste.

Fases cronometradas:
  1. Query Optimization (Gemini)
  2. Query Embedding (BGE-M3)
  3. Hybrid Search (PostgreSQL + pgvector)
  4. Answer Generation (Gemini)

Corre com:
  python3 benchmark_performance.py
"""

import json
import time
import os
import csv
import psycopg2
import torch
from google import genai
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
from datetime import datetime

# --- CONFIGURAÇÃO ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
MODEL_NAME = "BAAI/bge-m3"

# Ficheiro com as 20 queries (ajusta o caminho se necessário)
QUERIES_FILE = "ragas_dataset.json"
OUTPUT_FILE = "benchmark_results.csv"

client = genai.Client(api_key=GEMINI_API_KEY)

# --- CARREGAR MODELO DE EMBEDDINGS ---
print("🚀 A carregar modelo BGE-M3...")
use_fp16 = torch.cuda.is_available()
embed_model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)
print(f"✅ Modelo carregado ({'GPU' if use_fp16 else 'CPU'})")

# --- FUNÇÕES DO PIPELINE (copiadas do app.py) ---

MAX_RETRIES = 3
RETRY_WAIT = 30  # segundos entre retries

def call_gemini_with_retry(model, contents, config=None):
    """Wrapper com retry automático para chamadas ao Gemini."""
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {"model": model, "contents": contents}
            if config:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                print(f"    ⚠️ Rate limit (tentativa {attempt+1}/{MAX_RETRIES}). A esperar {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
            elif "503" in error_msg or "service unavailable" in error_msg or "overloaded" in error_msg:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"    ⚠️ Servidor ocupado (tentativa {attempt+1}/{MAX_RETRIES}). A esperar {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ❌ Erro não-recuperável: {e}")
                raise
    raise Exception(f"Gemini falhou após {MAX_RETRIES} tentativas")


def optimize_search_query(user_query):
    prompt = f"""
You are an expert search engine optimizer. 
Analyze the user's question and extract the most important keywords.
Generate a list of keywords in BOTH Portuguese and English to maximize database retrieval.
Return ONLY the keywords separated by spaces. No punctuation, no quotes, no conversational text.

User question: {user_query}
"""
    try:
        return call_gemini_with_retry('gemini-3.1-flash-lite-preview', prompt).strip()
    except Exception:
        return user_query


def embed_query(query_text):
    output = embed_model.encode([query_text], return_dense=True)
    return output['dense_vecs'][0].tolist()


def hybrid_search(query_vector, query_text, top_k=10):
    lexical_query = " | ".join(query_text.replace("'", "").split())
    
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    
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
    cur.close()
    conn.close()
    
    return results


def generate_answer(query, context_results):
    context_text = ""
    for i, (source, page, text, metadata, rrf_score) in enumerate(context_results):
        clean_source = source.replace(".json", "").replace(".pdf", "")
        document_name = clean_source
        doi_link = ""
        
        if metadata and isinstance(metadata, dict):
            if metadata.get("title"):
                document_name = metadata["title"]
            if metadata.get("doi"):
                doi_cru = metadata["doi"]
                doi_link = doi_cru if doi_cru.startswith("http") else f"https://doi.org/{doi_cru}"
        
        doi_info = f"\nLink/DOI: {doi_link}" if doi_link else ""
        context_text += f"\n--- SOURCE {i+1} ---\nDocument: {document_name}\nPage: {page}{doi_info}\nText: {text}\n--- END SOURCE {i+1} ---\n"

    prompt = f"""
You are an elite academic research assistant, specialized in analyzing documents and extracting precise answers.
Below, I provide you with context extracted from scientific databases and a user query.

MANDATORY RULES:
1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that SAME LANGUAGE.
2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context. Do not use outside knowledge or hallucinate.
3. COMPLETE ANSWERS: Provide COMPLETE, DETAILED answers based on the text context.
4. CITATIONS WITH LINKS: Whenever you make a claim, cite the source: [Document Name, Page X]
5. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information in the documents.

PROVIDED CONTEXT:
{context_text}

USER QUERY:
{query}

FORMATTED RESPONSE:
"""

    try:
        return call_gemini_with_retry(
            'gemini-3.1-flash-lite-preview', 
            prompt, 
            config={'temperature': 0.4, 'top_p': 0.9}
        )
    except Exception as e:
        return f"ERROR: {e}"


# --- BENCHMARK ---
def run_benchmark():
    # Carregar queries
    if not os.path.exists(QUERIES_FILE):
        print(f"❌ Ficheiro {QUERIES_FILE} não encontrado!")
        print(f"   Coloca-o na mesma pasta que este script.")
        return
    
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
        # Suporta ambos os formatos: lista direta ou {"queries": [...]}
        queries_data = raw["queries"] if isinstance(raw, dict) and "queries" in raw else raw
    
    # --- RESUME: Carregar progresso anterior ---
    results = []
    completed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Converter tipos
                for key in ["t_optimize_s", "t_embed_s", "t_search_s", "t_generate_s", "t_total_s"]:
                    row[key] = float(row[key])
                for key in ["query_id", "chunks_retrieved", "images_retrieved", "answer_length"]:
                    row[key] = int(row[key])
                results.append(row)
                completed_ids.add(int(row["query_id"]))
        print(f"📂 Progresso anterior encontrado: {len(completed_ids)} queries já completas")
    
    remaining = [(i, q) for i, q in enumerate(queries_data) if (i + 1) not in completed_ids]
    
    if not remaining:
        print("✅ Todas as queries já foram processadas!")
    else:
        print(f"\n{'='*70}")
        print(f"BENCHMARK DE PERFORMANCE — {len(remaining)} queries restantes (de {len(queries_data)})")
        print(f"{'='*70}")
        print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Hardware: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}\n")
        
        for i, q in remaining:
            query = q["query"]
            category = q.get("category", "unknown")
            language = q.get("language", "unknown")
            
            print(f"[{i+1}/{len(queries_data)}] {query[:60]}...")
            
            try:
                # --- FASE 1: Query Optimization ---
                t0 = time.time()
                optimized_query = optimize_search_query(query)
                t_optimize = time.time() - t0
                
                # --- FASE 2: Query Embedding ---
                t0 = time.time()
                query_vector = embed_query(optimized_query)
                t_embed = time.time() - t0
                
                # --- FASE 3: Hybrid Search ---
                t0 = time.time()
                context_results = hybrid_search(query_vector, optimized_query, top_k=10)
                t_search = time.time() - t0
                
                # --- FASE 4: Answer Generation ---
                t0 = time.time()
                answer = generate_answer(query, context_results)
                t_generate = time.time() - t0
                
                # --- Total ---
                t_total = t_optimize + t_embed + t_search + t_generate
                
                # Contar chunks e imagens recuperadas
                n_chunks = len(context_results)
                n_images = sum(1 for _, _, _, m, _ in context_results if m and m.get("chunk_type") == "image")
                has_error = "yes" if answer.startswith("ERROR") else "no"
                
            except Exception as e:
                print(f"  ❌ Erro na query {i+1}: {e}")
                print(f"  ⏭️  A saltar para a próxima...")
                t_optimize = t_embed = t_search = t_generate = t_total = 0
                n_chunks = n_images = 0
                has_error = "yes"
                answer = ""
            
            result = {
                "query_id": i + 1,
                "category": category,
                "language": language,
                "query": query,
                "t_optimize_s": round(t_optimize, 3),
                "t_embed_s": round(t_embed, 3),
                "t_search_s": round(t_search, 3),
                "t_generate_s": round(t_generate, 3),
                "t_total_s": round(t_total, 3),
                "chunks_retrieved": n_chunks,
                "images_retrieved": n_images,
                "answer_length": len(answer.split()) if answer else 0,
                "error": has_error
            }
            
            results.append(result)
            
            print(f"  Optimize: {t_optimize:.2f}s | Embed: {t_embed:.2f}s | "
                  f"Search: {t_search:.2f}s | Generate: {t_generate:.2f}s | "
                  f"TOTAL: {t_total:.2f}s")
            
            # --- SALVAMENTO INCREMENTAL ---
            results_sorted = sorted(results, key=lambda r: int(r["query_id"]))
            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results_sorted[0].keys())
                writer.writeheader()
                writer.writerows(results_sorted)
            
            # Pausa entre queries para não exceder rate limits
            time.sleep(2)
    
    # --- RESUMO FINAL ---
    # Garantir tipos numéricos (necessário quando carregados do CSV no resume)
    for r in results:
        for key in ["t_optimize_s", "t_embed_s", "t_search_s", "t_generate_s", "t_total_s"]:
            r[key] = float(r[key])
        for key in ["chunks_retrieved", "images_retrieved", "answer_length"]:
            r[key] = int(r[key])
    
    print(f"\n{'='*70}")
    print("RESUMO")
    print(f"{'='*70}")
    
    valid = [r for r in results if r["error"] == "no"]
    
    if not valid:
        print("❌ Todas as queries falharam!")
        return
    
    def avg(key): return sum(r[key] for r in valid) / len(valid)
    def minn(key): return min(r[key] for r in valid)
    def maxx(key): return max(r[key] for r in valid)
    
    print(f"\n  {'Fase':<25} {'Média':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*55}")
    
    for phase, key in [
        ("Query Optimization", "t_optimize_s"),
        ("Query Embedding", "t_embed_s"),
        ("Hybrid Search", "t_search_s"),
        ("Answer Generation", "t_generate_s"),
        ("TOTAL", "t_total_s")
    ]:
        print(f"  {phase:<25} {avg(key):>7.2f}s {minn(key):>7.2f}s {maxx(key):>7.2f}s")
    
    print(f"\n  Queries com sucesso: {len(valid)}/{len(results)}")
    print(f"  Chunks recuperados (média): {avg('chunks_retrieved'):.1f}")
    print(f"  Imagens recuperadas (média): {avg('images_retrieved'):.1f}")
    print(f"  Palavras na resposta (média): {avg('answer_length'):.0f}")
    
    # Breakdown por categoria
    categories = set(r["category"] for r in valid)
    if len(categories) > 1:
        print(f"\n  {'Categoria':<20} {'Média Total':>12} {'Nº Queries':>12}")
        print(f"  {'─'*48}")
        for cat in sorted(categories):
            cat_results = [r for r in valid if r["category"] == cat]
            cat_avg = sum(r["t_total_s"] for r in cat_results) / len(cat_results)
            print(f"  {cat:<20} {cat_avg:>11.2f}s {len(cat_results):>12}")
    
    print(f"\n✅ Resultados guardados em: {OUTPUT_FILE}")
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_benchmark()