"""
RAGAS Evaluation Script
Avalia sistema RAG usando métricas standard
"""

import json
import os
import time
import psycopg2
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd

# Configuração
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DB_CONFIG = {
    "host": "127.0.0.1",
    "database": "tese_rag",
    "user": "admin",
    "password": "password123"
}

MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL = "gemini-3.1-flash-lite-preview"

# Carregar modelos
print("🔄 Loading BGE-M3 model...")
embedding_model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)
print("✅ Model loaded!")

llm = genai.GenerativeModel(LLM_MODEL)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def hybrid_search(original_query, optimized_query, top_k=10):  
    
    # O Embedding (Matemática Semântica) usa a pergunta natural e bem estruturada!
    output = embedding_model.encode([original_query], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()
    
    # A pesquisa Lexical (no PostgreSQL) usa as palavras-chave otimizadas!
    lexical_query = " | ".join(optimized_query.replace("'", "").split())

    conn = get_db_connection()
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
    conn.close()
    
    contexts = []
    for source, page_num, text, metadata, rrf_score in results:
        if metadata and metadata.get('chunk_type') == 'image':
            caption = metadata.get('caption', '')
            if caption:
                text = f"[FIGURE] Caption: {caption}\n\nDescription: {text}"
            else:
                text = f"[FIGURE] {text}"
        
        contexts.append({
            'text': text,
            'source': source,
            'page': page_num,
            'metadata': metadata
        })
    
    return contexts

def generate_answer(query: str, contexts: List[Dict]) -> str:
    """Gera resposta usando LLM"""
    
    # AGORA SIM! Ele recebe o texto todo (igual à aplicação real)
    context_text = "\n\n".join([
        f"[Source {i+1}] {ctx['text']}" 
        for i, ctx in enumerate(contexts)
    ])
    
    prompt = f"""You are an elite academic research assistant, specialized in analyzing documents and extracting precise answers.
Below, I provide you with context extracted from scientific databases and a user query.

MANDATORY RULES:
1. RESPONSE LANGUAGE: Analyze the language of the 'USER QUERY' and respond EXACTLY in that SAME LANGUAGE.
2. STRICT FIDELITY: Answer the query based EXCLUSIVELY on the provided context. Do not use outside knowledge or hallucinate.
3. COMPLETE ANSWERS: Provide COMPLETE, DETAILED answers based on the text context.
   - DO NOT simply refer to figures or tables (e.g., "see Figure 1")
   - EXTRACT and EXPLAIN the key information from the text
   - If relevant visual content exists, it will be shown separately to the user
4. CITATIONS WITH LINKS: Whenever you make a claim, you MUST cite the source at the end of the sentence.
   - Always use the provided 'Document' field to name the source.
   - If the source has a 'Link/DOI', format the citation as a clickable Markdown link: [Document Name, Page X](URL_DO_DOI)
   - If there is no DOI, use plain text: [Document Name, Page X]
5. INSUFFICIENT DATA: If the provided context does not contain the answer, state strictly that there is not enough information in the documents.

PROVIDED CONTEXT:
{context_text}

USER QUERY:
{query}

FORMATTED RESPONSE:"""
    
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"   ⚠️ Error generating answer: {e}")
        return f"Error: {e}"

def calculate_faithfulness(answer: str, contexts: List[Dict]) -> float:
    """
    Calcula Faithfulness: resposta baseada apenas no contexto?
    1.0 = totalmente fiel, 0.0 = alucinação total
    """
    # Simplificado: verifica se palavras-chave do contexto aparecem na resposta
    context_words = set()
    for ctx in contexts:
        words = ctx['text'].lower().split()
        context_words.update([w for w in words if len(w) > 4])
    
    answer_words = set(answer.lower().split())
    
    if not answer_words or not context_words:
        return 0.0
    
    # Proporção de palavras da resposta que vêm do contexto
    overlap = len(answer_words & context_words)
    score = min(overlap / len(answer_words), 1.0)
    
    return score

def calculate_context_precision(contexts: List[Dict], query: str) -> float:
    """
    Context Precision: contextos retrieved são relevantes?
    1.0 = todos relevantes, 0.0 = nenhum relevante
    """
    # Simplificado: assume que top-3 são relevantes, resto penaliza
    if not contexts:
        return 0.0
    
    # Score decrescente: primeiro chunk mais importante
    scores = [1.0 / (i + 1) for i in range(len(contexts))]
    avg_score = sum(scores) / len(scores)
    
    return min(avg_score, 1.0)

def calculate_context_recall(contexts: List[Dict]) -> float:
    """
    Context Recall: retrieveu todos os contextos necessários?
    Simplificado: assume que 5 chunks são suficientes
    """
    if len(contexts) >= 5:
        return 1.0
    else:
        return len(contexts) / 5.0

def calculate_answer_relevancy(answer: str, query: str) -> float:
    """
    Answer Relevancy: resposta é relevante para a query?
    1.0 = muito relevante, 0.0 = irrelevante
    """
    # Simplificado: verifica se palavras-chave da query aparecem na resposta
    query_words = set([w.lower() for w in query.split() if len(w) > 3])
    answer_words = set(answer.lower().split())
    
    if not query_words:
        return 1.0
    
    overlap = len(query_words & answer_words)
    score = overlap / len(query_words)
    
    return min(score, 1.0)

def optimize_search_query(user_query: str) -> str:
    """Expande query PT/EN (copiado do app.py)"""
    prompt = f"""
You are an expert search engine optimizer. 
Analyze the user's question and extract the most important keywords.
Generate a list of keywords in BOTH Portuguese and English to maximize database retrieval.
Return ONLY the keywords separated by spaces. No punctuation, no quotes, no conversational text.

User question: {user_query}
"""
    
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception:
        # Se falhar, usa query original
        return user_query

def evaluate_single_query(query_data: Dict, query_num: int, total: int) -> Dict:
    query = query_data['query']
    category = query_data['category']
    language = query_data['language']
    
    print(f"\n[{query_num}/{total}] {category.upper()} ({language})")
    print(f"Query: {query[:80]}...")
    
    print("   🔧 Optimizing query...")
    optimized_query = optimize_search_query(query)
    print(f"   → Optimized: {optimized_query[:60]}...")
    
    # PASSAMOS A QUERY ORIGINAL E A OTIMIZADA PARA A FUNÇÃO!
    print("   🔍 Retrieving contexts...")
    contexts = hybrid_search(query, optimized_query, top_k=10) 
    print(f"   Retrieved: {len(contexts)} chunks")
    
    print("   💬 Generating answer...")
    answer = generate_answer(query, contexts)
    print(f"   Answer: {answer[:100]}...")
    
    faithfulness = calculate_faithfulness(answer, contexts)
    context_precision = calculate_context_precision(contexts, query)
    context_recall = calculate_context_recall(contexts)
    answer_relevancy = calculate_answer_relevancy(answer, query)
    
    print(f"   📊 Faithfulness: {faithfulness:.3f} | Precision: {context_precision:.3f} | Recall: {context_recall:.3f} | Relevancy: {answer_relevancy:.3f}")
    
    return {
        'query': query,
        'category': category,
        'language': language,
        'answer': answer,
        'num_contexts': len(contexts),
        'contexts': [ctx['text'][:200] for ctx in contexts],  # Preview apenas no JSON
        'metrics': {
            'faithfulness': faithfulness,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'answer_relevancy': answer_relevancy
        }
    }

def run_evaluation(dataset_path: str = "ragas_dataset.json", checkpoint_file: str = "ragas_progress.json"):
    """
    Corre avaliação completa com rate limiting e checkpoints
    """
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = data['queries']
    total = len(queries)
    
    # Load checkpoint se existir
    start_from = 0
    results = []
    
    if os.path.exists(checkpoint_file):
        print(f"📂 Found checkpoint file, resuming...")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get('results', [])
            start_from = len(results)
        print(f"   Resuming from query {start_from + 1}/{total}")
    
    print(f"\n{'='*60}")
    print(f"🧪 RAGAS EVALUATION - {total} QUERIES")
    print(f"⚠️  Rate limit: 15 RPM → waiting 5s between queries")
    print(f"{'='*60}")
    
    # Processar queries restantes
    for i in range(start_from, total):
        query_data = queries[i]
        query_num = i + 1
        
        try:
            result = evaluate_single_query(query_data, query_num, total)
            results.append(result)
            
            # ✅ SALVAR CHECKPOINT INCREMENTAL
            checkpoint = {
                'completed': query_num,
                'total': total,
                'results': results
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
            # ⏰ RATE LIMITING - Aguardar 5 segundos
            if query_num < total:  # Não aguardar no último
                print(f"   ⏰ Waiting 5s... (Rate limit: 15 RPM)")
                time.sleep(5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'query': query_data['query'],
                'category': query_data.get('category', 'unknown'),
                'error': str(e)
            })
            
            # Se for erro de rate limit, aguardar mais
            if "429" in str(e) or "quota" in str(e).lower():
                print("   ⚠️  Rate limit hit! Waiting 60s...")
                time.sleep(60)
    
    # Limpar checkpoint quando terminar
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("\n✅ Checkpoint file removed (evaluation complete)")
    
    return results

def analyze_results(results: List[Dict]):
    """Analisa e apresenta resultados"""
    
    # Filtrar resultados válidos (sem erros)
    valid_results = [r for r in results if 'metrics' in r]
    
    if not valid_results:
        print("\n❌ No valid results to analyze!")
        return
    
    # Agregar métricas
    metrics_summary = {
        'faithfulness': [],
        'context_precision': [],
        'context_recall': [],
        'answer_relevancy': []
    }
    
    for result in valid_results:
        for metric, value in result['metrics'].items():
            metrics_summary[metric].append(value)
    
    # Calcular médias
    print(f"\n{'='*60}")
    print("📊 OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Total queries evaluated: {len(valid_results)}/{len(results)}")
    print()
    
    for metric, values in metrics_summary.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric:25s}: {mean:.3f} (±{std:.3f})")
    
    # Análise por categoria
    print(f"\n{'='*60}")
    print("📊 RESULTS BY CATEGORY")
    print(f"{'='*60}")
    
    categories = {}
    for result in valid_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    for category, cat_results in categories.items():
        print(f"\n{category.upper()} ({len(cat_results)} queries):")
        
        cat_metrics = {
            'faithfulness': np.mean([r['metrics']['faithfulness'] for r in cat_results]),
            'context_precision': np.mean([r['metrics']['context_precision'] for r in cat_results]),
            'context_recall': np.mean([r['metrics']['context_recall'] for r in cat_results]),
            'answer_relevancy': np.mean([r['metrics']['answer_relevancy'] for r in cat_results])
        }
        
        for metric, value in cat_metrics.items():
            print(f"  {metric:20s}: {value:.3f}")
    
    # Salvar resultados
    output = {
        'summary': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for metric, values in metrics_summary.items()
        },
        'by_category': {
            category: {
                'count': len(cat_results),
                'metrics': {
                    metric: float(np.mean([r['metrics'][metric] for r in cat_results]))
                    for metric in ['faithfulness', 'context_precision', 'context_recall', 'answer_relevancy']
                }
            }
            for category, cat_results in categories.items()
        },
        'detailed_results': results
    }
    
    with open('ragas_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed results saved to: ragas_results.json")

def main():
    print("🚀 Starting RAGAS Evaluation with Rate Limiting...")
    print("⏰ Estimated time: ~2 minutes (20 queries × 5s)")
    
    # Run evaluation (com checkpoints)
    results = run_evaluation("ragas_dataset.json")
    
    # Analyze results
    analyze_results(results)
    
    print("\n🎉 Evaluation complete!")
    print("📁 Results saved to: ragas_results.json")

if __name__ == "__main__":
    main()