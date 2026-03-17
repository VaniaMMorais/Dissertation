import pymupdf4llm

# Coloca aqui o nome de um PDF teu que tenha uma tabela
pdf_path = "data/PDFs/Documento_Emanuel_Gonçalves.pdf"

print(f"🪄 A converter o ficheiro '{pdf_path}' para formato Inteligência Artificial...")

# Esta é a linha mágica que faz tudo: lê texto, deteta tabelas e formata em Markdown
md_text = pymupdf4llm.to_markdown(pdf_path)

# Vamos guardar o resultado num ficheiro para tu leres
output_file = "resultado_tabelas.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(md_text)

print(f"✅ Feito! Abre o ficheiro '{output_file}' no teu editor e procura a zona da tabela.")