import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse



def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["chunks"]
    

def build_vectorizer(chunks):
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        min_df=2
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def intra_chunk_coherence(chunks):
    scores = []

    for ch in chunks:
        sentences = [s.strip() for s in ch["text"].split(".") if len(s.strip()) > 30]

        if len(sentences) < 2:
            continue

        vec = TfidfVectorizer(stop_words="english").fit_transform(sentences)
        sim = cosine_similarity(vec)

        n = sim.shape[0]
        coherence = (sim.sum() - n) / (n * (n - 1))
        scores.append(coherence)

    return float(np.mean(scores)) if scores else 0.0

def boundary_integrity(chunks):
    sims = []

    for i in range(len(chunks)-1):
        end_text = " ".join(chunks[i]["text"].split()[-40:])
        start_text = " ".join(chunks[i+1]["text"].split()[:40])

        vec = TfidfVectorizer(stop_words="english").fit_transform([end_text, start_text])
        sim = cosine_similarity(vec)[0,1]
        sims.append(sim)

    return float(np.mean(sims)) if sims else 0.0

def retrieval_score(chunks, vectorizer, matrix):
    correct = 0

    for i, ch in enumerate(chunks):
        words = ch["text"].split()
        query = " ".join(words[:15])  # synthetic query

        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(matrix, q_vec).flatten()

        retrieved = int(np.argmax(sims))
        if retrieved == i:
            correct += 1

    return correct / len(chunks)

def chunk_stats(chunks):
    sizes = [len(c["text"].split()) for c in chunks]

    return {
        "num_chunks": len(chunks),
        "avg_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "min_size": int(np.min(sizes)),
        "max_size": int(np.max(sizes)),
    }

def evaluate(json_path):
    print(f"\nEvaluating: {json_path}")

    chunks = load_chunks(json_path)

    vectorizer, matrix = build_vectorizer(chunks)

    icc = intra_chunk_coherence(chunks)
    bis = boundary_integrity(chunks)
    rss = retrieval_score(chunks, vectorizer, matrix)
    stats = chunk_stats(chunks)

    print("\n============= RESULTS =============")
    print(f"Chunks:              {stats['num_chunks']}")
    print(f"Avg size:            {stats['avg_size']:.1f} words")
    print(f"Std size:            {stats['std_size']:.1f}")
    print("-----------------------------------")
    print(f"Intra-Chunk Coherence: {icc:.3f}")
    print(f"Boundary Integrity:    {bis:.3f}")
    print(f"Retrieval Score:       {rss:.3f}")
    print("===================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to processed JSON")
    
    args = parser.parse_args()
    evaluate(args.json_file)