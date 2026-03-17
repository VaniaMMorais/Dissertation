"""
FASE 2 — Avaliar com RAGAS (com retoma incremental)
=====================================================
Avalia pergunta a pergunta e guarda progressivamente.
Se crashar ou bater no rate limit, retoma de onde ficou.
Corre este script quantas vezes precisares — salta as perguntas
que já têm todas as métricas preenchidas.
"""

import pandas as pd
import json
import os
import time
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURAÇÕES ---
# Carregar .env da raiz do projeto (um nível acima da pasta Tests/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "BAAI/bge-m3"
_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(_DIR, "rag_answers.csv")
OUTPUT_CSV = os.path.join(_DIR, "final_results.csv")

# Pausa entre perguntas (segundos)
# RAGAS faz ~4 chamadas por pergunta. Com 15 RPM (Flash-Lite) -> mínimo ~20s
PAUSE_BETWEEN_QUESTIONS = 45

# --- MODELOS ---
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,
)
evaluator_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

ragas_run_config = RunConfig(
    max_workers=1,
    timeout=120,
    max_retries=5,
)

METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def evaluate_single(question, answer, contexts, ground_truth):
    """Avalia uma única pergunta com RAGAS. Devolve dict com as métricas."""
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=ragas_run_config,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    return {metric: df[metric].iloc[0] if metric in df.columns else None for metric in METRIC_NAMES}


def main():
    print("🚀 FASE 2 — Avaliação RAGAS incremental\n")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ '{INPUT_CSV}' não encontrado. Corre primeiro o phase1_generate_answers.py!")
        return

    df_input = pd.read_csv(INPUT_CSV)
    total = len(df_input)

    # Carregar progresso anterior, se existir
    if os.path.exists(OUTPUT_CSV):
        df_done = pd.read_csv(OUTPUT_CSV)
        # Perguntas já avaliadas com TODAS as métricas preenchidas
        done_mask = df_done[METRIC_NAMES].notna().all(axis=1)
        done_questions = set(df_done.loc[done_mask, "question"].tolist())
        rows = df_done.to_dict("records")
        print(f"📂 Progresso anterior encontrado: {len(done_questions)}/{total} perguntas já avaliadas.\n")
    else:
        done_questions = set()
        rows = []
        # Inicializar com os dados base (sem métricas)
        for _, row in df_input.iterrows():
            rows.append({
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
                **{m: None for m in METRIC_NAMES}
            })

    for i, row in df_input.iterrows():
        question = row["question"]

        if question in done_questions:
            print(f"[{i+1}/{total}] ⏭️  Já avaliada, a saltar: {question[:60]}...")
            continue

        print(f"[{i+1}/{total}] A avaliar: {question[:70]}")

        contexts = json.loads(row["contexts"]) if isinstance(row["contexts"], str) else []

        metrics_result = evaluate_single(
            question=question,
            answer=row["answer"],
            contexts=contexts,
            ground_truth=row["ground_truth"],
        )

        print(f"  📊 {metrics_result}")

        # Atualizar ou adicionar linha no array de resultados
        found = False
        for r in rows:
            if r["question"] == question:
                r.update(metrics_result)
                found = True
                break
        if not found:
            rows.append({
                "question": question,
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
                **metrics_result,
            })

        # Guardar incrementalmente após cada pergunta
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

        if i + 1 < total:
            print(f"  ⏸️  Pausa de {PAUSE_BETWEEN_QUESTIONS}s...\n")
            time.sleep(PAUSE_BETWEEN_QUESTIONS)

    # Sumário final
    df_final = pd.read_csv(OUTPUT_CSV)
    filled = df_final[METRIC_NAMES].notna().all(axis=1).sum()
    print(f"\n{'='*50}")
    print(f"✅ CONCLUÍDO: {filled}/{total} perguntas com todas as métricas.")

    if filled > 0:
        print("\n📊 MÉDIAS FINAIS:")
        for metric in METRIC_NAMES:
            val = df_final[metric].mean()
            print(f"   {metric}: {val:.4f}" if pd.notna(val) else f"   {metric}: N/A")

    if filled < total:
        print(f"\n⚠️  {total - filled} perguntas ainda sem avaliação completa.")
        print("   Volta a correr este script para retomar!")


if __name__ == "__main__":
    main()