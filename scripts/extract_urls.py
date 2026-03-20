import pypdf
import re

pdf_path = r"c:\Users\Arnold's\Documents\Repositorios Machine Learning\Proyecto 2 Machine Learning\data\raw\taller2-ml1-premier-league.pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    urls = []
    
    # Extract from text
    for page in reader.pages:
        text = page.extract_text()
        found = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        urls.extend(found)
    
    # Extract from annotations (links)
    for page in reader.pages:
        if "/Annots" in page:
            for annot in page["/Annots"]:
                obj = annot.get_object()
                if "/A" in obj and "/URI" in obj["/A"]:
                    urls.append(obj["/A"]["/URI"])

    # Unique and filter for CSV
    unique_urls = sorted(list(set(urls)))
    for url in unique_urls:
        print(url)

except Exception as e:
    print(f"Error: {e}")
