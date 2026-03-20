import pypdf
import sys

pdf_path = r"c:\Users\Arnold's\Documents\Repositorios Machine Learning\Proyecto 2 Machine Learning\data\raw\taller2-ml1-premier-league.pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    # Use sys.stdout.buffer.write for binary safe output or just encode
    print(full_text.encode('utf-8', errors='replace').decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
