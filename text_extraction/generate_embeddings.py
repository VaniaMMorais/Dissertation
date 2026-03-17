import json
import os
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# --- CONFIGURAÇÃO ---
INPUT_DIR = "data/extracted"
OUTPUT_DIR = "data/embeddings"
MODEL_NAME = "BAAI/bge-m3"
# Reduzi o Batch Size para 1 para ser mais "gentil" com o teu CPU
BATCH_SIZE = 1 

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath):
    # Agora guardamos sem funções complexas, pois já convertemos os dados antes
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    use_fp16 = torch.cuda.is_available()
    print(f"🚀 A carregar modelo {MODEL_NAME}...")
    print(f"🖥️  Hardware detetado: {'GPU (CUDA)' if use_fp16 else 'CPU (Prepare-se para ouvir as ventoinhas!)'}")

    model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not files:
        print("❌ Nenhum ficheiro encontrado em data/extracted")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 A processar {len(files)} ficheiros...")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            data = load_json(input_path)
            if "chunks" not in data or not data["chunks"]:
                print(f"⚠️  Saltar {filename}: sem chunks.")
                continue

            chunks = data["chunks"]
            texts = [c["text"] for c in chunks]
            
            print(f"Processing {filename} ({len(chunks)} chunks)...")

            # Gerar Embeddings
            output = model.encode(
                texts, 
                batch_size=BATCH_SIZE, 
                max_length=8192, 
                return_dense=True, 
                return_sparse=True, 
                return_colbert_vecs=False
            )

            dense_vecs = output['dense_vecs']
            sparse_vecs = output['lexical_weights']

            # --- CONVERSÃO SEGURA (A CORREÇÃO) ---
            for i, chunk in enumerate(chunks):
                # 1. Converter Vetor Denso (Numpy -> Lista Python)
                d_vec = dense_vecs[i]
                if isinstance(d_vec, (np.ndarray, torch.Tensor)):
                    chunk["embedding_dense"] = d_vec.tolist()
                else:
                    chunk["embedding_dense"] = d_vec
                
                # 2. Converter Vetor Esparso (Dict Numpy -> Dict Python puro)
                # O BGE-M3 devolve pesos como numpy.float, o que baralha o JSON.
                s_vec = sparse_vecs[i]
                clean_sparse = {}
                for k, v in s_vec.items():
                    # Converter chave para string e valor para float nativo
                    clean_sparse[str(k)] = float(v)
                
                chunk["embedding_sparse"] = clean_sparse

            # Guardar
            data["chunks"] = chunks 
            save_json(data, output_path)
            
            print(f"✅ Guardado com sucesso: {output_path}")

        except Exception as e:
            print(f"❌ Erro crítico em {filename}: {e}")

    print("\n🎉 Processo concluído! Podes respirar fundo (e o teu PC também).")

if __name__ == "__main__":
    main()