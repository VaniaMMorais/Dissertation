"""
FASE 1 — Gerar e guardar todas as respostas do RAG
===================================================
Corre este script primeiro. Ele processa cada pergunta do dataset,
gera a resposta via RAG, e guarda TUDO num CSV intermédio.
Só depois corre o phase2_evaluate.py.
"""

import pandas as pd
import psycopg2
import torch
import os
import time
import warnings
import json
from dotenv import load_dotenv
from google import genai

warnings.filterwarnings("ignore", category=DeprecationWarning)

from FlagEmbedding import BGEM3FlagModel

# --- CONFIGURAÇÕES ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB_HOST = "127.0.0.1"
DB_NAME = "tese_rag"
DB_USER = "admin"
DB_PASS = "password123"
MODEL_NAME = "BAAI/bge-m3"

# Ficheiro de saída desta fase
OUTPUT_CSV = "rag_answers.csv"

# Pausa entre perguntas (segundos) — 2 chamadas Gemini por pergunta,
# com 10 RPM o mínimo seguro é ~15s. Usa 20s para ter margem.
PAUSE_BETWEEN_QUESTIONS = 20

client = genai.Client(api_key=GEMINI_API_KEY)

print("⏳ A carregar o modelo BGE-M3...")
use_fp16 = torch.cuda.is_available()
embed_model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)
print("✅ Modelo carregado!")


# --- FUNÇÕES RAG ---

def call_gemini(prompt, model="gemini-2.5-flash-lite", max_retries=5):
    """Wrapper com retry automático para qualquer chamada Gemini."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()
        except Exception as e:
            wait = 65
            print(f"  ⚠️  Rate limit (tentativa {attempt+1}/{max_retries}). A aguardar {wait}s... [{e}]")
            time.sleep(wait)
    return None  # Falhou todas as tentativas


def optimize_search_query(user_query):
    prompt = (
        f"Extract keywords in Portuguese and English from this query. "
        f"Return ONLY space-separated keywords: {user_query}"
    )
    result = call_gemini(prompt)
    return result if result else user_query


def hybrid_search(query_text, top_k=4):
    output = embed_model.encode([query_text], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()
    lexical_query = " | ".join(query_text.replace("'", "").split())

    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()

    sql = """
    WITH semantic_search AS (
        SELECT id, source_file, page_num, text, metadata,
               RANK() OVER (ORDER BY embedding_dense <=> %s::vector) AS rank
        FROM document_chunks
        ORDER BY embedding_dense <=> %s::vector LIMIT 20
    ),
    keyword_search AS (
        SELECT id, source_file, page_num, text, metadata,
               RANK() OVER (ORDER BY ts_rank_cd(fts, to_tsquery('simple', %s)) DESC) AS rank
        FROM document_chunks
        WHERE fts @@ to_tsquery('simple', %s)
        ORDER BY ts_rank_cd(fts, to_tsquery('simple', %s)) DESC LIMIT 20
    )
    SELECT
        COALESCE(s.source_file, k.source_file),
        COALESCE(s.page_num, k.page_num),
        COALESCE(s.text, k.text),
        COALESCE(s.metadata, k.metadata),
        COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC LIMIT %s;
    """
    cur.execute(sql, (query_vector, query_vector, lexical_query, lexical_query, lexical_query, top_k))
    results = cur.fetchall()
    conn.close()
    return results


def ask_gemini(query, context_results):
    context_text = ""
    for i, (source, page, text, metadata, rrf) in enumerate(context_results):
        context_text += f"\n--- INÍCIO DA FONTE {i+1} ---\nTexto: {text}\n--- FIM DA FONTE {i+1} ---\n"

    prompt = f"""You are an elite academic research assistant.
MANDATORY RULES:
1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that same language.
2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context.
3. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information.

PROVIDED CONTEXT:
{context_text}

USER QUERY:
{query}"""

    result = call_gemini(prompt)
    return result if result else "Erro: Não foi possível contactar o modelo."


# --- PIPELINE FASE 1 ---

def main():
    print("🚀 FASE 1 — A gerar respostas RAG...\n")

    df = pd.read_csv("dataset_evaluation.csv")
    total = len(df)

    # Retomar de onde ficou, se o CSV já existir parcialmente
    if os.path.exists(OUTPUT_CSV):
        df_done = pd.read_csv(OUTPUT_CSV)
        start_index = len(df_done)
        rows = df_done.to_dict("records")
        print(f"📂 Retomando do índice {start_index} (já existem {start_index} respostas guardadas).\n")
    else:
        start_index = 0
        rows = []

    for index, row in df.iloc[start_index:].iterrows():
        q = row["Question"]
        gt = row["Ground Truth"]
        print(f"[{index+1}/{total}] {q}")

        # 1ª chamada Gemini — otimizar query
        otimizada = optimize_search_query(q)
        print(f"  🔍 Query otimizada: {otimizada}")

        # Busca híbrida (sem LLM, só embeddings + SQL)
        resultados = hybrid_search(otimizada, top_k=4)
        textos = [r[2] for r in resultados]

        # 2ª chamada Gemini — gerar resposta
        if textos:
            resposta = ask_gemini(q, resultados)
        else:
            resposta = "Não encontrei informação suficiente no contexto."

        print(f"  ✅ Resposta gerada ({len(resposta)} chars)")

        # Guardar contextos como JSON string para poder recarregar depois
        rows.append({
            "question": q,
            "ground_truth": gt,
            "answer": resposta,
            "contexts": json.dumps(textos, ensure_ascii=False),
        })

        # Guardar incrementalmente (crash-safe)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

        if index + 1 < total:
            print(f"  ⏸️  Pausa de {PAUSE_BETWEEN_QUESTIONS}s...\n")
            time.sleep(PAUSE_BETWEEN_QUESTIONS)

    print(f"\n✅ FASE 1 CONCLUÍDA! {total} respostas guardadas em '{OUTPUT_CSV}'.")
    print("➡️  Agora corre: python phase2_evaluate.py")


if __name__ == "__main__":
    main()