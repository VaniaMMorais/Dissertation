# validate_embeddings.py
import json
import os
import numpy as np

EMBEDDINGS_DIR = "data/embeddings"

def validate_file(filepath):
    issues = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    if not chunks:
        return ["❌ Sem chunks"]
    
    dense_dims = []
    
    for i, chunk in enumerate(chunks):
        # 1. Tem embeddings?
        if "embedding_dense" not in chunk:
            issues.append(f"  Chunk {i}: sem embedding_dense")
            continue
        if "embedding_sparse" not in chunk:
            issues.append(f"  Chunk {i}: sem embedding_sparse")
        
        dense = chunk["embedding_dense"]
        
        # 2. É uma lista de floats?
        if not isinstance(dense, list):
            issues.append(f"  Chunk {i}: embedding_dense não é lista")
            continue
        
        # 3. Dimensão consistente?
        dense_dims.append(len(dense))
        
        # 4. Tem valores NaN ou zeros?
        arr = np.array(dense)
        if np.any(np.isnan(arr)):
            issues.append(f"  Chunk {i}: contém NaN")
        if np.allclose(arr, 0):
            issues.append(f"  Chunk {i}: vetor todo zeros (embedding falhou)")
        
        # 5. Está normalizado? (BGE-M3 retorna vetores normalizados, norma ≈ 1.0)
        norm = np.linalg.norm(arr)
        if not (0.95 < norm < 1.05):
            issues.append(f"  Chunk {i}: norma inesperada ({norm:.3f}, esperado ≈ 1.0)")
    
    # 6. Todas as dimensões iguais?
    if len(set(dense_dims)) > 1:
        issues.append(f"  Dimensões inconsistentes: {set(dense_dims)}")
    
    return issues, dense_dims[0] if dense_dims else 0, len(chunks)


def main():
    files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".json")]
    
    if not files:
        print("❌ Nenhum ficheiro encontrado em data/embeddings")
        return
    
    print(f"🔍 A validar {len(files)} ficheiros...\n")
    all_ok = True
    
    for filename in files:
        path = os.path.join(EMBEDDINGS_DIR, filename)
        result = validate_file(path)
        
        issues, dim, n_chunks = result if len(result) == 3 else (result[0], 0, 0)
        
        if not issues:
            print(f"✅ {filename}: {n_chunks} chunks, dim={dim}")
        else:
            all_ok = False
            print(f"❌ {filename}: {n_chunks} chunks, dim={dim}")
            for issue in issues:
                print(issue)
    
    print()
    if all_ok:
        print("🎉 Tudo válido! Podes avançar para o vector database.")
    else:
        print("⚠️  Alguns ficheiros têm problemas. Corrige antes de indexar.")

if __name__ == "__main__":
    main()