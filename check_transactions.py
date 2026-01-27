"""
Script pour examiner le format exact des transactions dans les PDFs
"""
import fitz
import re
import os

PDF_FILES = [
    r"c:\Users\matdi\Downloads\2025-04 Relevé de compte.pdf",
    r"c:\Users\matdi\Downloads\2025-03 Relevé bancaire.pdf",
    r"c:\Users\matdi\Downloads\2025-02 - Relevé.pdf",
    r"c:\Users\matdi\Downloads\2025-12 Relevé de compte1189313D034_12-01-2026.pdf"
]

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un PDF"""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(full_text)
    except Exception as e:
        return f"Erreur: {str(e)}"

def find_transaction_section(text: str):
    """Trouve et affiche la section des transactions"""
    # Chercher la section "Vos opérations" ou similaire
    lines = text.split('\n')
    
    in_transactions = False
    transaction_lines = []
    
    for i, line in enumerate(lines):
        if 'opérations' in line.lower() or 'operation' in line.lower() or 'Date' in line and 'Opération' in line:
            in_transactions = True
            # Prendre les 100 lignes suivantes
            transaction_lines = lines[i:min(i+100, len(lines))]
            break
    
    return '\n'.join(transaction_lines)

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    for pdf_path in PDF_FILES:
        if not os.path.exists(pdf_path):
            continue
            
        print(f"\n{'='*80}")
        print(f"Fichier: {os.path.basename(pdf_path)}")
        print(f"{'='*80}")
        
        text = extract_text_from_pdf(pdf_path)
        
        # Trouver toutes les dates au format JJ/MM
        date_pattern = r'\b\d{1,2}/\d{1,2}\b'
        dates = re.findall(date_pattern, text)
        
        # Filtrer pour ne garder que celles qui ressemblent à des dates (JJ/MM où MM <= 12)
        valid_dates = []
        for date in dates:
            parts = date.split('/')
            if len(parts) == 2:
                try:
                    day, month = int(parts[0]), int(parts[1])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        valid_dates.append(date)
                except:
                    pass
        
        print(f"\nDates trouvees (format JJ/MM): {len(valid_dates)}")
        unique_dates = sorted(set(valid_dates), key=lambda x: (int(x.split('/')[1]), int(x.split('/')[0])))
        print(f"Dates uniques: {', '.join(unique_dates[:30])}")
        if len(unique_dates) > 30:
            print(f"... et {len(unique_dates) - 30} autres")
        
        # Afficher la section des transactions
        transaction_section = find_transaction_section(text)
        if transaction_section:
            print(f"\nSection des transactions (premiers 2000 caracteres):")
            print(transaction_section[:2000])
            print("...")
