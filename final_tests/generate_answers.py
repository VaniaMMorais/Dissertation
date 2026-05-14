"""
STEP 1: Gera respostas para todas as queries
Salva tudo num ficheiro JSON para posterior avaliação
"""

import json
import os
import time
import psycopg2
from FlagEmbedding import BGEM3FlagModel
from google import genai
from dotenv import load_dotenv

# Configuração
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API Key não encontrada! Verifica o .env")

DB_CONFIG = {
    "host": "127.0.0.1",
    "database": "tese_rag",
    "user": "admin",
    "password": "password123"
}

MODEL_NAME = "BAAI/bge-m3"
OUTPUT_FILE = "answers_complete.json"

# Cliente Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Carregar BGE-M3
print("🔄 Loading BGE-M3 model...")
embedding_model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)
print("✅ Model loaded!")

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def optimize_search_query(user_query: str) -> str:
    """Expande query PT/EN"""
    prompt = f"""You are an expert search engine optimizer. 
Analyze the user's question and extract the most important keywords.
Generate a list of keywords in BOTH Portuguese and English to maximize database retrieval.
Return ONLY the keywords separated by spaces. No punctuation, no quotes, no conversational text.

User question: {user_query}
"""
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                print(f"   ⚠️ Retry {attempt+1}/3 after 10s...")
                time.sleep(10)
            else:
                return user_query
    return user_query

def hybrid_search(original_query, optimized_query, top_k=10):
    """Busca híbrida"""
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
            contexts.append(f"[FIGURE] {caption}\n{text}" if caption else f"[FIGURE] {text}")
        else:
            contexts.append(text)
    
    return contexts

def generate_answer(query: str, contexts: list) -> str:
    """Gera resposta usando Gemini"""
    context_text = "\n\n".join([f"[Source {i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    
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
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                print(f"   ⚠️ Retry {attempt+1}/3 after 10s...")
                time.sleep(10)
            else:
                return f"Error: {e}"
    
    return "Error: Failed after 3 retries"

def main():
    print("🚀 STEP 1: GENERATING ANSWERS")
    print("="*70)
    
    # Carregar dataset
    if not os.path.exists("ragas_dataset.json"):
        print("❌ ragas_dataset.json not found!")
        return
    
    with open("ragas_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = data['queries']
    
    # Verificar se já existe output parcial
    results = []
    start_index = 0
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
            start_index = len(results)
            print(f"✅ Found {start_index} existing answers. Resuming...\n")
    
    # Gerar respostas
    print(f"📝 Generating {len(queries) - start_index} answers...")
    print("="*70)
    
    for i in range(start_index, len(queries)):
        query_data = queries[i]
        query = query_data['query']
        category = query_data['category']
        language = query_data['language']
        
        print(f"\n[{i+1}/{len(queries)}] {category.upper()} ({language})")
        print(f"Query: {query[:60]}...")
        
        # Optimize
        print("   🔧 Optimizing query...")
        optimized_query = optimize_search_query(query)
        print(f"   → {optimized_query[:60]}...")
        
        # Retrieve
        print("   🔍 Retrieving contexts...")
        contexts = hybrid_search(query, optimized_query, top_k=10)
        print(f"   Retrieved: {len(contexts)} chunks")
        
        # Generate
        print("   💬 Generating answer...")
        answer = generate_answer(query, contexts)
        print(f"   Answer: {answer[:80]}...")
        
        # Salvar resultado
        result = {
            'query': query,
            'category': category,
            'language': language,
            'optimized_query': optimized_query,
            'contexts': contexts,
            'answer': answer
        }
        
        results.append(result)
        
        # 💾 Salvar incrementalmente
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("   💾 Saved to cache!")
        
        # Rate limiting
        if i < len(queries) - 1:
            print("   ⏰ Waiting 5s...")
            time.sleep(5)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE! Generated {len(results)} answers")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()