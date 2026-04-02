import json
import os
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# --- CONFIGURAÇÃO ---
INPUT_DIR = "../data/extracted" # Ajusta os caminhos se necessário
OUTPUT_DIR = "../data/embeddings"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 1 

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_chunk_list(chunk_list, model):
    """Função auxiliar que injeta contexto e gera os embeddings para uma lista de chunks."""
    if not chunk_list:
        return chunk_list
        
    texts = []
    for c in chunk_list:
        # A GRANDE MAGIA DO CONTEXTO (Funciona para texto, tabelas e imagens!):
        if "section" in c:
            texto_enriquecido = f"Secção: {c.get('section', 'Sem Título')}\nConteúdo: {c['text']}"
        else:
            texto_enriquecido = f"Nota de Rodapé: {c.get('text', '')}"
            
        texts.append(texto_enriquecido)

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

    # --- CONVERSÃO SEGURA ---
    for i, chunk in enumerate(chunk_list):
        # 1. Converter Vetor Denso
        d_vec = dense_vecs[i]
        if isinstance(d_vec, (np.ndarray, torch.Tensor)):
            chunk["embedding_dense"] = d_vec.tolist()
        else:
            chunk["embedding_dense"] = d_vec
        
        # 2. Converter Vetor Esparso
        s_vec = sparse_vecs[i]
        clean_sparse = {}
        for k, v in s_vec.items():
            clean_sparse[str(k)] = float(v)
        
        chunk["embedding_sparse"] = clean_sparse

    return chunk_list

def main():
    use_fp16 = torch.cuda.is_available()
    print(f"🚀 A carregar modelo {MODEL_NAME}...")
    print(f"🖥️ Hardware detetado: {'GPU (CUDA)' if use_fp16 else 'CPU (Prepara-te para ouvir as ventoinhas!)'}")

    model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not files:
        print(f"❌ Nenhum ficheiro encontrado na pasta {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📂 A processar {len(files)} ficheiros...")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            data = load_json(input_path)
            print(f"\nA processar ficheiro: {filename}")
            
            # 1. Processar os Chunks normais (corpo e tabelas)
            if "chunks" in data and data["chunks"]:
                data["chunks"] = process_chunk_list(data["chunks"], model)
                print(f"  -> Embeddings gerados para {len(data['chunks'])} secções de texto.")
            
            # 2. Processar os Chunks das Footnotes
            if "footnote_chunks" in data and data["footnote_chunks"]:
                data["footnote_chunks"] = process_chunk_list(data["footnote_chunks"], model)
                print(f"  -> Embeddings gerados para {len(data['footnote_chunks'])} notas de rodapé.")

            # 3. Processar os Chunks de Imagens (A NOVA MAGIA! 👁️✨)
            if "image_chunks" in data and data["image_chunks"]:
                data["image_chunks"] = process_chunk_list(data["image_chunks"], model)
                print(f"  -> Embeddings gerados para {len(data['image_chunks'])} imagens descritas.")

            # Guardar
            save_json(data, output_path)
            print(f"✅ Guardado com sucesso: {output_path}")

        except Exception as e:
            print(f"❌ Erro crítico em {filename}: {e}")

    print("\n🎉 Processo concluído! O motor semântico leu e interpretou todas as tuas tabelas, textos e imagens.")

if __name__ == "__main__":
    main()