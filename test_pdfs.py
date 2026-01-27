"""
Script de test pour analyser les PDFs et identifier les opérations manquantes
"""
import fitz
import re
from datetime import datetime
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

# Chemins des PDFs à tester
PDF_FILES = [
    r"c:\Users\matdi\Downloads\2025-04 Relevé de compte.pdf",
    r"c:\Users\matdi\Downloads\2025-03 Relevé bancaire.pdf",
    r"c:\Users\matdi\Downloads\2025-02 - Relevé.pdf",
    r"c:\Users\matdi\Downloads\2025-12 Relevé de compte1189313D034_12-01-2026.pdf"
]

# Dates problématiques à vérifier
PROBLEMATIC_DATES = {
    "février": ["25/02/2025", "26/02/2025", "27/02/2025", "28/02/2025"],
    "mars": ["13/03/2025", "14/03/2025", "15/03/2025", "16/03/2025", "17/03/2025", 
             "18/03/2025", "19/03/2025", "20/03/2025", "21/03/2025", "22/03/2025",
             "23/03/2025", "24/03/2025", "25/03/2025", "26/03/2025", "27/03/2025",
             "28/03/2025", "29/03/2025", "30/03/2025", "31/03/2025"],
    "avril": ["14/04/2025", "15/04/2025", "16/04/2025", "17/04/2025", "18/04/2025",
              "19/04/2025", "20/04/2025", "21/04/2025", "22/04/2025", "23/04/2025",
              "24/04/2025", "25/04/2025", "26/04/2025", "27/04/2025", "28/04/2025",
              "29/04/2025", "30/04/2025"],
    "décembre": ["09/12/2025", "10/12/2025", "11/12/2025", "12/12/2025", "13/12/2025",
                 "14/12/2025", "15/12/2025", "16/12/2025", "17/12/2025", "18/12/2025",
                 "19/12/2025", "20/12/2025", "21/12/2025", "22/12/2025", "23/12/2025",
                 "24/12/2025", "25/12/2025", "26/12/2025", "27/12/2025", "28/12/2025",
                 "29/12/2025", "30/12/2025", "31/12/2025"]
}

def extract_text_from_pdf(pdf_path: str) -> tuple:
    """Extrait le texte d'un PDF et retourne (texte, nombre_de_pages)"""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        full_text = []
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
        result_text = "\n\n".join(full_text)
        doc.close()
        return result_text, page_count
    except Exception as e:
        return f"Erreur: {str(e)}", 0

def find_dates_in_text(text: str, date_list: list) -> dict:
    """Cherche les dates dans le texte et retourne un dictionnaire avec les occurrences"""
    found = {}
    for date in date_list:
        # Chercher la date au format JJ/MM/AAAA
        pattern = date.replace("/", r"[/\-\.]")
        matches = re.findall(rf'\b{date}\b', text)
        if matches:
            found[date] = len(matches)
        else:
            found[date] = 0
    return found

def analyze_pdf(pdf_path: str):
    """Analyse un PDF et vérifie la présence des dates problématiques"""
    print(f"\n{'='*80}")
    print(f"Analyse de: {os.path.basename(pdf_path)}")
    print(f"{'='*80}")
    
    # Extraire le texte
    text, page_count = extract_text_from_pdf(pdf_path)
    
    if text.startswith("Erreur"):
        print(f"[ERROR] Erreur lors de l'extraction: {text}")
        return
    
    print(f"[INFO] Nombre de pages: {page_count}")
    print(f"[INFO] Longueur du texte extrait: {len(text)} caracteres")
    
    # Déterminer quelles dates chercher selon le nom du fichier
    filename = os.path.basename(pdf_path).lower()
    dates_to_check = []
    month_name = None
    
    if "2025-02" in filename or "février" in filename or "fevrier" in filename:
        dates_to_check = PROBLEMATIC_DATES["février"]
        month_name = "février"
    elif "2025-03" in filename or "mars" in filename:
        dates_to_check = PROBLEMATIC_DATES["mars"]
        month_name = "mars"
    elif "2025-04" in filename or "avril" in filename:
        dates_to_check = PROBLEMATIC_DATES["avril"]
        month_name = "avril"
    elif "2025-12" in filename or "décembre" in filename or "decembre" in filename:
        dates_to_check = PROBLEMATIC_DATES["décembre"]
        month_name = "décembre"
    else:
        # Si on ne peut pas déterminer, chercher toutes les dates
        dates_to_check = []
        for month_dates in PROBLEMATIC_DATES.values():
            dates_to_check.extend(month_dates)
    
    if dates_to_check:
        print(f"\n[SEARCH] Recherche des dates problematiques pour {month_name}:")
        found_dates = find_dates_in_text(text, dates_to_check)
        
        missing = []
        found = []
        for date, count in found_dates.items():
            if count == 0:
                missing.append(date)
            else:
                found.append((date, count))
        
        if found:
            print(f"\n[FOUND] Dates trouvees dans le PDF:")
            for date, count in found:
                print(f"   - {date}: {count} occurrence(s)")
        
        if missing:
            print(f"\n[MISSING] Dates MANQUANTES dans le texte extrait:")
            for date in missing:
                print(f"   - {date}")
        else:
            print(f"\n[OK] Toutes les dates problematiques sont presentes dans le texte extrait!")
    
    # Chercher tous les formats de dates possibles dans le texte
    print(f"\n[DATE_FORMATS] Recherche de tous les formats de dates dans le texte:")
    # Formats possibles: JJ/MM/AAAA, JJ-MM-AAAA, JJ.MM.AAAA, JJ MM AAAA, etc.
    date_patterns = [
        (r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}', 'JJ/MM/AAAA'),
        (r'\d{1,2}\s+\d{1,2}\s+\d{4}', 'JJ MM AAAA'),
        (r'\d{1,2}\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+\d{4}', 'JJ Mois AAAA'),
    ]
    
    all_dates_found = []
    for pattern, desc in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            unique_matches = list(set(matches))[:20]  # Limiter à 20 pour l'affichage
            all_dates_found.extend(unique_matches)
            print(f"   Format {desc}: {len(matches)} occurrences")
            if len(unique_matches) <= 10:
                for match in unique_matches:
                    print(f"      - {match}")
    
    # Afficher un echantillon du texte pour verification
    print(f"\n[SAMPLE] Echantillon du texte (premiers 1000 caracteres):")
    print(text[:1000])
    print("...")
    
    # Afficher aussi la fin du texte pour voir s'il y a des transactions à la fin
    if len(text) > 1000:
        print(f"\n[SAMPLE_END] Fin du texte (derniers 500 caracteres):")
        print(text[-500:])

if __name__ == "__main__":
    import sys
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("Analyse des PDFs pour identifier les operations manquantes")
    print("="*80)
    
    for pdf_path in PDF_FILES:
        if os.path.exists(pdf_path):
            analyze_pdf(pdf_path)
        else:
            print(f"\n[WARNING] Fichier non trouve: {pdf_path}")
    
    print(f"\n{'='*80}")
    print("[OK] Analyse terminee")
