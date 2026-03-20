import pypdf

pdf_path = r"c:\Users\Arnold's\Documents\Repositorios Machine Learning\Proyecto 2 Machine Learning\data\raw\taller2-ml1-premier-league.pdf"
output_path = "pdf_text_extracted.txt"

try:
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Success: Extracted text to {output_path}")
except Exception as e:
    print(f"Error: {e}")
