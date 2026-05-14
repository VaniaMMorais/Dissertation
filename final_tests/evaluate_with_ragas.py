"""
STEP 2: Avalia respostas com RAGAS (CRASH-SAFE)
Adiciona checkpoint incremental - pode parar e retomar!
"""

import json
import os
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd

# RAGAS metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Configuração
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

INPUT_FILE = "answers_complete.json"
OUTPUT_FILE = "ragas_official_results.json"
CHECKPOINT_FILE = "ragas_checkpoint.json"  # ← NOVO!

# Configurar modelos
print("🔄 Configuring models...")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0.4,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Models configured!")

def load_answers():
    """Carrega respostas já geradas"""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ {INPUT_FILE} not found! Run generate_answers.py first!")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_single_sample(sample_dict, llm, embeddings):
    """
    Avalia UMA amostra de cada vez
    Retorna métricas ou None se falhar
    """
    try:
        # Criar dataset com 1 amostra
        dataset = Dataset.from_dict({
            'question': [sample_dict['question']],
            'answer': [sample_dict['answer']],
            'contexts': [sample_dict['contexts']],
            'ground_truth': [sample_dict['ground_truth']]
        })
        
        # Avaliar
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )
        
        # Extrair métricas
        df = results.to_pandas()
        metrics = df.iloc[0].to_dict()
        
        return metrics
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None

def run_evaluation_incremental(answers):
    """
    Avalia amostra por amostra com checkpoint
    """
    print(f"\n{'='*70}")
    print("🧪 RUNNING RAGAS EVALUATION (INCREMENTAL)")
    print(f"{'='*70}")
    
    # ✅ Carregar checkpoint se existir
    evaluated_results = []
    start_index = 0
    
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            evaluated_results = json.load(f)
            start_index = len(evaluated_results)
            print(f"\n✅ Found checkpoint: {start_index} samples already evaluated")
            print(f"   Resuming from sample {start_index + 1}...\n")
    
    total = len(answers)
    
    if start_index >= total:
        print(f"✅ All {total} samples already evaluated!")
        return evaluated_results
    
    print(f"\n📊 Evaluating {total - start_index} samples...")
    print(f"⚠️  This will make ~10 LLM calls per sample")
    print(f"   Estimated time: ~{(total - start_index) * 2} minutes\n")
    
    # ✅ Avaliar sample por sample
    for i in range(start_index, total):
        answer = answers[i]
        
        print(f"\n[{i+1}/{total}] {answer['category'].upper()}")
        print(f"Query: {answer['query'][:60]}...")
        
        # Preparar sample
        sample = {
            'question': answer['query'],
            'answer': answer['answer'],
            'contexts': answer['contexts'],
            'ground_truth': answer['ground_truth'] # <--- CORRIGIDO AQUI!
        }
        
        # Avaliar
        print(f"   🧪 Evaluating metrics...")
        metrics = evaluate_single_sample(sample, llm, embeddings)
        
        if metrics:
            # Adicionar info original
            result = {
                'query': answer['query'],
                'category': answer['category'],
                'language': answer['language'],
                'answer': answer['answer'],
                'contexts': answer['contexts'],
                'metrics': {
                    'faithfulness': float(metrics.get('faithfulness', 0)),
                    'answer_relevancy': float(metrics.get('answer_relevancy', 0)),
                    'context_recall': float(metrics.get('context_recall', 0)),
                    'context_precision': float(metrics.get('context_precision', 0))
                }
            }
            
            evaluated_results.append(result)
            
            # Mostrar métricas
            print(f"   📊 Faithfulness: {result['metrics']['faithfulness']:.3f}")
            print(f"   📊 Relevancy: {result['metrics']['answer_relevancy']:.3f}")
            print(f"   📊 Recall: {result['metrics']['context_recall']:.3f}")
            print(f"   📊 Precision: {result['metrics']['context_precision']:.3f}")
        else:
            # Se falhar, adiciona placeholder
            result = {
                'query': answer['query'],
                'category': answer['category'],
                'language': answer['language'],
                'answer': answer['answer'],
                'contexts': answer['contexts'],
                'metrics': {
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_recall': 0.0,
                    'context_precision': 0.0
                },
                'error': True
            }
            evaluated_results.append(result)
        
        # 💾 Salvar checkpoint DEPOIS de cada amostra
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(evaluated_results, f, indent=2, ensure_ascii=False)
        print(f"   💾 Checkpoint saved!")
    
    return evaluated_results

def analyze_results(results):
    """Analisa e salva resultados finais"""
    print(f"\n{'='*70}")
    print("📊 RAGAS EVALUATION RESULTS")
    print(f"{'='*70}")
    
    # Filtrar erros
    valid_results = [r for r in results if not r.get('error', False)]
    
    if not valid_results:
        print("❌ No valid results!")
        return
    
    # Calcular estatísticas
    metrics_list = [r['metrics'] for r in valid_results]
    
    print(f"\n📈 OVERALL METRICS (n={len(valid_results)}):")
    for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']:
        values = [m[metric] for m in metrics_list]
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        print(f"  {metric:20s}: {mean_val:.3f} [{min_val:.3f}, {max_val:.3f}]")
    
    # Salvar resultados finais
    output = {
        'summary': {
            metric: {
                'mean': sum([m[metric] for m in metrics_list]) / len(metrics_list),
                'min': min([m[metric] for m in metrics_list]),
                'max': max([m[metric] for m in metrics_list]),
            }
            for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
        },
        'detailed_results': results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Final results saved to: {OUTPUT_FILE}")
    
    # Limpar checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"✅ Checkpoint file removed (evaluation complete)")

def main():
    print("🚀 STEP 2: RAGAS EVALUATION (CRASH-SAFE)")
    print("="*70)
    
    # Carregar respostas
    print("\n📂 Loading answers...")
    answers = load_answers()
    print(f"✅ Loaded {len(answers)} answers from {INPUT_FILE}")
    
    # Avaliar (incremental)
    results = run_evaluation_incremental(answers)
    
    # Analisar
    print("\n📊 Analyzing results...")
    analyze_results(results)
    
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main()