"""
RAGAS Evaluation usando biblioteca oficial
CORRIGIDO: Usa Gemini para LLM e HuggingFace para embeddings (local)
Adicionado: Sistema de Cache para o Dataset
"""

import json
import os
import time
import psycopg2
import pandas as pd
from datasets import Dataset
from FlagEmbedding import BGEM3FlagModel

# Langchain para Gemini e Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# RAGAS metrics
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

# Configuração
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API Key não encontrada! Verifica o teu ficheiro .env")

DB_CONFIG = {
    "host": "127.0.0.1",
    "database": "tese_rag",
    "user": "admin",
    "password": "password123"
}

MODEL_NAME = "BAAI/bge-m3"
CACHE_FILE = "dataset_cache.json"

# ✅ Configurar LLM para RAGAS e Geração
print("🔄 Configuring Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0.4,
    max_retries=15,  # <-- NOVO: Diz à biblioteca para tentar 15 vezes internamente
)

# ✅ Configurar Embeddings locais para a avaliação matemática do RAGAS
print("🔄 Configuring local RAGAS embeddings...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Carregar modelo BGE-M3 para o Retrieval (Pesquisa na Base de Dados)
print("🔄 Loading BGE-M3 model for retrieval...")
embedding_model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)
print("✅ Models loaded!")

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def extract_text(response) -> str:
    """Garante que a resposta do Langchain é sempre uma string limpa."""
    content = response.content
    if isinstance(content, list):
        return " ".join([str(block.get("text", "")) for block in content if isinstance(block, dict)])
    return str(content)

def optimize_search_query(user_query: str) -> str:
    """Expande a query para Português e Inglês."""
    prompt = f"""
You are an expert search engine optimizer. 
Analyze the user's question and extract the most important keywords.
Generate a list of keywords in BOTH Portuguese and English to maximize database retrieval.
Return ONLY the keywords separated by spaces. No punctuation, no quotes, no conversational text.

User question: {user_query}
"""
    # LOOP TEIMOSO: Tenta até 6 vezes antes de desistir
    for attempt in range(6):
        try:
            response = llm.invoke(prompt)
            return extract_text(response).strip() 
        except Exception as e:
            error_msg = str(e).lower()
            if "503" in error_msg or "429" in error_msg or "quota" in error_msg:
                print(f"   ⚠️ Servidor Google ocupado. A aguardar 15s antes de repetir a otimização... (Tentativa {attempt+1}/6)")
                time.sleep(15)
            else:
                return user_query # Se for outro erro bizarro, avança
    return user_query

def hybrid_search(original_query, optimized_query, top_k=10):
    """Busca híbrida com dupla query"""
    output = embedding_model.encode([original_query], return_dense=True)
    query_vector = output['dense_vecs'][0].tolist()
    
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
        COALESCE(s.text, k.text) as text,
        COALESCE(s.metadata, k.metadata) as metadata
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) DESC
    LIMIT %s;
    """
    
    cur.execute(sql, (query_vector, query_vector, lexical_query, lexical_query, lexical_query, top_k))
    results = cur.fetchall()
    conn.close()
    
    contexts = []
    for text, metadata in results:
        if metadata and metadata.get('chunk_type') == 'image':
            caption = metadata.get('caption', '')
            if caption:
                contexts.append(f"[FIGURE] Caption: {caption}\n\nDescription: {text}")
            else:
                contexts.append(f"[FIGURE] {text}")
        else:
            contexts.append(text)
            
    return contexts

def generate_answer(query: str, contexts: list) -> str:
    """Gera resposta baseada no contexto com o Langchain/Gemini"""
    context_text = "\n\n".join([
        f"[Source {i+1}] {ctx}" 
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
    
    # LOOP TEIMOSO: Tenta até 6 vezes antes de desistir
    for attempt in range(6):
        try:
            response = llm.invoke(prompt)
            return extract_text(response).strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "503" in error_msg or "429" in error_msg or "quota" in error_msg:
                print(f"   ⚠️ Servidor Google ocupado. A aguardar 15s antes de tentar gerar a resposta novamente... (Tentativa {attempt+1}/6)")
                time.sleep(15)
            else:
                return f"Error: {e}"
    
    return "Error: 503 - Falha ao comunicar com os servidores da Google após várias tentativas."

def prepare_ragas_dataset(dataset_path: str = "ragas_dataset.json"):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = data['queries']
    
    dataset_dict = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': [] 
    }
    
    start_index = 0
    
    # 💾 1. VERIFICAR CACHE: Se o ficheiro já existe, carregamos o que lá está
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if 'question' in cached_data and len(cached_data['question']) > 0:
                    dataset_dict = cached_data
                    start_index = len(dataset_dict['question'])
                    print(f"\n✅ Cache encontrada! A retomar a partir da pergunta {start_index + 1} de {len(queries)}...")
        except Exception as e:
            print(f"⚠️ Erro ao ler a cache: {e}. A começar do zero...")
            
    # Se já fizemos todas as perguntas, avançamos direto para a avaliação
    if start_index >= len(queries):
        print(f"\n✅ Todas as {len(queries)} respostas já estão geradas na cache!")
        return Dataset.from_dict(dataset_dict)
    
    print(f"\n{'='*70}")
    print(f"🔄 PREPARING RAGAS DATASET - FALTAM {len(queries) - start_index} QUERIES")
    print(f"{'='*70}")
    
    # 2. CONTINUAR DE ONDE PARAMOS
    for i in range(start_index, len(queries)):
        query_data = queries[i]
        query = query_data['query']
        category = query_data['category']
        
        print(f"\n[{i+1}/{len(queries)}] {category.upper()}")
        print(f"Query: {query[:60]}...")
        
        print("   🔧 Optimizing query...")
        optimized_query = optimize_search_query(query)
        
        print("   🔍 Retrieving contexts...")
        contexts = hybrid_search(query, optimized_query, top_k=10)
        
        print("   💬 Generating answer...")
        answer = generate_answer(query, contexts)
        print(f"   Answer: {answer[:80]}...")
        
        # Adicionar as novas informações ao dicionário
        dataset_dict['question'].append(query)
        dataset_dict['answer'].append(answer)
        dataset_dict['contexts'].append(contexts)
        dataset_dict['ground_truth'].append(answer) 
        
        # 💾 3. SALVAMENTO INCREMENTAL: Guardar no disco a CADA pergunta concluída
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        print("   💾 Progresso guardado na cache!")
        
        # Esperar 5s apenas se não for a última pergunta
        if i < len(queries) - 1:
            print(f"   ⏰ Waiting 5s...")
            time.sleep(5)
    
    return Dataset.from_dict(dataset_dict)

def run_ragas_evaluation(dataset):
    print(f"\n{'='*70}")
    print("🧪 RUNNING RAGAS EVALUATION")
    print(f"{'='*70}")
    
    # Classes Inicializadas!
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRecall(),
        ContextPrecision(),
    ]
    
    print(f"\nMetrics to evaluate: Faithfulness, Answer Relevancy, Context Recall, Context Precision")
    print(f"⚠️  This will make multiple LLM calls per query. Grab a coffee! ☕")
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,                   # LLM Gemini
            embeddings=ragas_embeddings, # Embeddings locais HuggingFace
            raise_exceptions=False 
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_results(results, dataset):
    if results is None:
        print("❌ No results to analyze!")
        return
    
    print(f"\n{'='*70}")
    print("📊 RAGAS EVALUATION RESULTS")
    print(f"{'='*70}")
    
    df = results.to_pandas()
    
    print(f"\n📈 OVERALL METRICS:")
    for col in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {col:20s}: {mean_val:.3f} (±{std_val:.3f})")
    
    output = {
        'summary': {
            metric: {
                'mean': float(df[metric].mean()),
                'std': float(df[metric].std()),
                'min': float(df[metric].min()),
                'max': float(df[metric].max())
            }
            for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
            if metric in df.columns
        },
        'detailed_results': df.to_dict('records')
    }
    
    with open('ragas_official_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: ragas_official_results.json")

def main():
    print("🚀 Starting RAGAS Official Evaluation")
    print("=" * 70)
    
    if not os.path.exists("ragas_dataset.json"):
        print("❌ ragas_dataset.json not found!")
        return
    
    print("\n📦 STEP 1: Preparing dataset...")
    dataset = prepare_ragas_dataset("ragas_dataset.json")
    print(f"✅ Dataset prepared: {len(dataset)} samples")
    
    print("\n🧪 STEP 2: Running RAGAS evaluation...")
    results = run_ragas_evaluation(dataset)
    
    print("\n📊 STEP 3: Analyzing results...")
    analyze_results(results, dataset)
    
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main()