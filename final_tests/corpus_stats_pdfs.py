"""
Estatísticas do Corpus de PDFs
Corre na pasta onde tens os 22 PDFs:
  python corpus_stats_pdfs.py /caminho/para/pasta/dos/pdfs
"""

import os
import sys
import fitz  # PyMuPDF

def analyze_corpus(pdf_folder):
    results = []
    
    for filename in sorted(os.listdir(pdf_folder)):
        if not filename.lower().endswith('.pdf'):
            continue
        
        filepath = os.path.join(pdf_folder, filename)
        try:
            doc = fitz.open(filepath)
            num_pages = len(doc)
            num_images = 0
            num_tables_approx = 0
            
            for page in doc:
                # Contar imagens por página
                img_list = page.get_images(full=True)
                num_images += len(img_list)
                
                # Aproximação de tabelas (blocos com muitas linhas curtas)
                text = page.get_text()
                if '|' in text or '\t' in text:
                    num_tables_approx += 1
            
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            results.append({
                'filename': filename,
                'pages': num_pages,
                'images_raw': num_images,
                'pages_with_tables': num_tables_approx,
                'size_mb': file_size_mb,
            })
            
            doc.close()
            
        except Exception as e:
            print(f"  ERRO em {filename}: {e}")
    
    # Resumo
    print(f"\n{'='*70}")
    print(f"ESTATÍSTICAS DO CORPUS DE PDFs")
    print(f"{'='*70}")
    print(f"Total de ficheiros: {len(results)}")
    print(f"Total de páginas:   {sum(r['pages'] for r in results)}")
    print(f"Total de imagens (raw): {sum(r['images_raw'] for r in results)}")
    print(f"Páginas com tabelas (aprox): {sum(r['pages_with_tables'] for r in results)}")
    print(f"Tamanho total: {sum(r['size_mb'] for r in results):.1f} MB")
    
    print(f"\n{'─'*70}")
    print(f"{'Ficheiro':<45} {'Pág':>5} {'Img':>5} {'Tab':>5} {'MB':>7}")
    print(f"{'─'*70}")
    for r in results:
        print(f"{r['filename'][:44]:<45} {r['pages']:>5} {r['images_raw']:>5} {r['pages_with_tables']:>5} {r['size_mb']:>7.1f}")
    
    # Estatísticas por tipo (para preencheres manualmente)
    print(f"\n{'─'*70}")
    print("PREENCHE MANUALMENTE:")
    print("  Artigos científicos: ___")
    print("  Dissertações/teses:  ___")
    print("  Diretrizes clínicas: ___")
    print("  Slides/apresentações: ___")
    print("  Cartilhas/manuais:   ___")
    print("  Outros:              ___")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_corpus(folder)
