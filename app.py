"""
Extracteur Comptable IA
Application Streamlit pour extraire des lignes comptables depuis des relevés bancaires PDF
via Gemini 1.5 Flash et les exporter en Excel.
"""

import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import pandas as pd
import json
import io
import re
import time
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Extracteur Comptable IA",
    page_icon="📊",
    layout="wide"
)

# Constantes
MAX_FILES = 15
GEMINI_MODEL = "gemini-2.5-flash"  # Mise à jour : gemini-1.5-flash n'est plus disponible

# Prompt système pour Gemini
SYSTEM_PROMPT = """Tu es un assistant comptable expert spécialisé dans l'analyse de relevés bancaires français.

MISSION : Analyse le relevé bancaire fourni et extrais TOUTES les lignes de transactions, SANS EN OMETTRE AUCUNE.

Pour chaque transaction, retourne un objet JSON avec :
- "date": la date de la transaction au format JJ/MM/AAAA (IMPORTANT: si la date dans le PDF est au format JJ/MM seulement, reconstitue la date complète en utilisant l'année mentionnée dans l'en-tête du relevé, par exemple "Arrêté mensuel du 1 au 30 avril 2025" indique que l'année est 2025)
- "libelle": le libellé de l'opération. Si le libellé dépasse 50 caractères, résume-le de manière concise en gardant les mots-clés essentiels (nom du bénéficiaire, type d'opération, référence importante).
- "debit": le montant en débit sous forme de nombre flottant (ex: 1234.56). Mettre null si c'est un crédit.
- "credit": le montant en crédit sous forme de nombre flottant (ex: 1234.56). Mettre null si c'est un débit.

RÈGLES IMPORTANTES :
1. NORMALISATION DES MONTANTS : Convertis tous les formats de montants en nombres flottants standard.
   - "1 000,50" → 1000.50
   - "1.000,50" → 1000.50
   - "1,000.50" → 1000.50
   - "1000,50" → 1000.50
2. DISTINCTION DÉBIT/CRÉDIT (CRITIQUE) :
   - DÉBIT (sortie d'argent) : paiements CB, prélèvements, virements VERS quelqu'un
   - CRÉDIT (entrée d'argent) : virements REÇUS, remboursements
   - ATTENTION aux virements : "VIREMENT A [nom]" ou "VIREMENT INSTANTANE A [nom]" = DÉBIT (argent qui SORT)
   - "VIREMENT DE [nom]" ou "VIREMENT INSTANTANE DE [nom]" = CRÉDIT (argent qui ENTRE)
   - Dans le PDF, le montant est généralement dans la colonne Débit ou Crédit - respecte cette position.
3. Ignore les lignes qui ne sont pas des transactions (soldes, totaux, en-têtes, etc.).
4. Si une transaction s'étend sur plusieurs lignes dans le PDF, reconstitue-la correctement.
5. CRITIQUE : Extrais ABSOLUMENT TOUTES les transactions, y compris celles en fin de relevé. Ne tronque pas ta réponse même si elle est longue.
6. Les dates peuvent être au format JJ/MM dans le PDF - reconstitue-les en JJ/MM/AAAA en utilisant l'année du relevé.
7. Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après.

FORMAT DE RÉPONSE ATTENDU (JSON uniquement) :
[
  {"date": "15/01/2024", "libelle": "VIREMENT SALAIRE ENTREPRISE XYZ", "debit": null, "credit": 2500.00},
  {"date": "16/01/2024", "libelle": "CB CARREFOUR", "debit": 85.32, "credit": null}
]

Analyse maintenant le relevé suivant :
"""


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extrait le texte de toutes les pages d'un fichier PDF.
    
    Args:
        pdf_file: Fichier PDF uploadé via Streamlit
        
    Returns:
        str: Texte brut concaténé de toutes les pages
    """
    try:
        # Lire le contenu du fichier uploadé
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset pour permettre une relecture si nécessaire
        
        # Ouvrir le PDF avec PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        return "\n\n".join(full_text)
    
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du PDF: {str(e)}")


def is_json_truncated(json_string: str) -> bool:
    """
    Vérifie si une chaîne JSON semble tronquée.
    
    Args:
        json_string: Chaîne JSON à vérifier
        
    Returns:
        bool: True si le JSON semble tronqué
    """
    cleaned = json_string.strip()
    
    # Enlever les marqueurs markdown pour l'analyse
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Vérifier si les brackets sont équilibrés
    open_braces = cleaned.count('{') - cleaned.count('}')
    open_brackets = cleaned.count('[') - cleaned.count(']')
    
    # Si déséquilibré, c'est tronqué
    if open_braces != 0 or open_brackets != 0:
        return True
    
    # Vérifier si ça se termine correctement pour un tableau JSON
    if not cleaned.endswith(']'):
        return True
    
    return False


def analyze_with_gemini(text: str, api_key: str, max_retries: int = 3) -> str:
    """
    Envoie le texte au modèle Gemini pour analyse avec retry automatique.
    Inclut une détection de troncature avec retry.
    
    Args:
        text: Texte extrait du PDF
        api_key: Clé API Gemini
        max_retries: Nombre maximum de tentatives en cas d'erreur
        
    Returns:
        str: Réponse du modèle (JSON attendu)
    """
    best_response = None
    best_length = 0
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            # Construire le prompt complet
            full_prompt = SYSTEM_PROMPT + text
            
            # Générer la réponse avec mode JSON structuré
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,  # Température 0 pour résultats déterministes
                    max_output_tokens=32768,  # Limite max de Gemini
                    response_mime_type="application/json",  # Force Gemini à produire un JSON valide
                )
            )
            
            response_text = response.text
            
            # Garder la meilleure réponse (la plus longue)
            if len(response_text) > best_length:
                best_length = len(response_text)
                best_response = response_text
            
            # Vérifier si la réponse semble tronquée
            if is_json_truncated(response_text):
                print(f"[WARNING] Réponse potentiellement tronquée (tentative {attempt + 1}/{max_retries}), retry...")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Petit délai avant retry
                    continue
            
            # Réponse complète, on peut la retourner
            return response_text
        
        except Exception as e:
            error_msg = str(e).lower()
            # Vérifier si c'est une erreur de rate limit
            is_rate_limit = any(keyword in error_msg for keyword in ['rate limit', 'quota', '429', 'too many requests'])
            
            if is_rate_limit and attempt < max_retries - 1:
                # Attendre progressivement plus longtemps à chaque retry
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            elif attempt < max_retries - 1:
                # Pour les autres erreurs, attendre un peu avant de réessayer
                time.sleep(1)
                continue
            else:
                # Si on a une réponse partielle, la retourner plutôt que de lever une erreur
                if best_response:
                    print(f"[WARNING] Utilisation de la meilleure réponse partielle après {max_retries} tentatives")
                    return best_response
                raise Exception(f"Erreur API Gemini après {max_retries} tentatives: {str(e)}")
    
    # Retourner la meilleure réponse obtenue
    if best_response:
        return best_response
    raise Exception("Aucune réponse valide obtenue de Gemini")


def repair_json(json_string: str) -> str:
    """
    Répare un JSON potentiellement malformé ou tronqué.
    
    Args:
        json_string: Chaîne JSON potentiellement malformée
        
    Returns:
        str: Chaîne JSON nettoyée et réparée
    """
    cleaned = json_string.strip()
    
    # Enlever les marqueurs markdown
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Supprimer les virgules traînantes: ,} ou ,]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Vérifier si le JSON est tronqué (brackets non fermés)
    open_braces = cleaned.count('{') - cleaned.count('}')
    open_brackets = cleaned.count('[') - cleaned.count(']')
    
    if open_braces > 0 or open_brackets > 0:
        # Tronquer au dernier objet complet et fermer le tableau
        last_complete = cleaned.rfind('},')
        if last_complete > 0:
            cleaned = cleaned[:last_complete + 1] + ']'
        else:
            # Essayer de trouver le dernier objet complet sans virgule
            last_obj = cleaned.rfind('}')
            if last_obj > 0:
                cleaned = cleaned[:last_obj + 1] + ']'
    
    return cleaned


def validate_and_fix_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide et corrige les erreurs de classification débit/crédit.
    
    Priorité des règles (du plus spécifique au plus générique) :
    1. VIREMENTS (priorité haute - déterminent clairement la direction)
       - "VIREMENT A" ou "VIREMENT INSTANTANE A" = DÉBIT (sortie d'argent)
       - "VIREMENT DE" ou "VIREMENT INSTANTANE DE" = CRÉDIT (entrée d'argent)
    2. Autres patterns (priorité basse - seulement si pas de virement)
       - "ACHAT CB", "PRELEVEMENT", "RETRAIT" = DÉBIT
       - "CREDIT CARTE", "REMBOURSEMENT" = CRÉDIT
    
    Args:
        df: DataFrame avec les transactions
        
    Returns:
        pd.DataFrame: DataFrame corrigé
    """
    if df.empty or "Libellé" not in df.columns:
        return df
    
    # PRIORITÉ 1 : Patterns de virement (les plus spécifiques)
    virement_debit_patterns = [
        r'VIREMENT\s+(INSTANTANE\s+)?A\s+',  # VIREMENT A ou VIREMENT INSTANTANE A
        r'VIREMENT\s+POUR\s+',  # VIREMENT POUR
    ]
    
    virement_credit_patterns = [
        r'VIREMENT\s+(INSTANTANE\s+)?DE\s+',  # VIREMENT DE ou VIREMENT INSTANTANE DE
    ]
    
    # PRIORITÉ 2 : Autres patterns (moins spécifiques)
    other_debit_patterns = [
        r'ACHAT\s+CB',
        r'PRELEVEMENT',
        r'RETRAIT\s+DAB',
        # Retirés : COMMISSION, COTISATION, FRAIS (ambigus - la direction dépend du contexte)
    ]
    
    other_credit_patterns = [
        r'CREDIT\s+CARTE',
        r'REMBOURSEMENT',
    ]
    
    corrections_count = 0
    
    for idx, row in df.iterrows():
        libelle = str(row.get("Libellé", "")).upper()
        debit = row.get("Débit")
        credit = row.get("Crédit")
        
        pattern_matched = False
        
        # ÉTAPE 1 : Vérifier les patterns de virement EN PREMIER (priorité haute)
        # Virement sortant (A) = DÉBIT
        for pattern in virement_debit_patterns:
            if re.search(pattern, libelle, re.IGNORECASE):
                if pd.notna(credit) and pd.isna(debit):
                    df.at[idx, "Débit"] = credit
                    df.at[idx, "Crédit"] = None
                    corrections_count += 1
                pattern_matched = True
                break
        
        # Virement entrant (DE) = CRÉDIT
        if not pattern_matched:
            for pattern in virement_credit_patterns:
                if re.search(pattern, libelle, re.IGNORECASE):
                    if pd.notna(debit) and pd.isna(credit):
                        df.at[idx, "Crédit"] = debit
                        df.at[idx, "Débit"] = None
                        corrections_count += 1
                    pattern_matched = True
                    break
        
        # ÉTAPE 2 : Seulement si pas de virement, vérifier les autres patterns
        if not pattern_matched:
            # Autres débits
            for pattern in other_debit_patterns:
                if re.search(pattern, libelle, re.IGNORECASE):
                    if pd.notna(credit) and pd.isna(debit):
                        df.at[idx, "Débit"] = credit
                        df.at[idx, "Crédit"] = None
                        corrections_count += 1
                    pattern_matched = True
                    break
            
            # Autres crédits
            if not pattern_matched:
                for pattern in other_credit_patterns:
                    if re.search(pattern, libelle, re.IGNORECASE):
                        if pd.notna(debit) and pd.isna(credit):
                            df.at[idx, "Crédit"] = debit
                            df.at[idx, "Débit"] = None
                            corrections_count += 1
                        break
    
    if corrections_count > 0:
        print(f"[INFO] {corrections_count} transaction(s) corrigée(s) (débit/crédit)")
    
    return df


def parse_llm_response(response: str, filename: str) -> pd.DataFrame:
    """
    Parse la réponse JSON du LLM et la convertit en DataFrame.
    
    Args:
        response: Réponse texte du LLM
        filename: Nom du fichier source pour la colonne Source
        
    Returns:
        pd.DataFrame: DataFrame avec les transactions
    """
    try:
        # Utiliser repair_json pour nettoyer et réparer la réponse
        cleaned = repair_json(response)
        
        # Parser le JSON
        transactions = json.loads(cleaned)
        
        if not isinstance(transactions, list):
            raise ValueError("La réponse n'est pas une liste de transactions")
        
        if len(transactions) == 0:
            return pd.DataFrame(columns=["Date", "Libellé", "Débit", "Crédit", "Source"])
        
        # Créer le DataFrame
        df = pd.DataFrame(transactions)
        
        # Renommer les colonnes pour le français
        column_mapping = {
            "date": "Date",
            "libelle": "Libellé",
            "debit": "Débit",
            "credit": "Crédit"
        }
        df = df.rename(columns=column_mapping)
        
        # Ajouter la colonne source
        df["Source"] = filename
        
        # S'assurer que les colonnes numériques sont bien des nombres
        for col in ["Débit", "Crédit"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Valider et corriger les erreurs débit/crédit
        df = validate_and_fix_debit_credit(df)
        
        # Réordonner les colonnes
        expected_cols = ["Date", "Libellé", "Débit", "Crédit", "Source"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        df = df[expected_cols]
        
        return df
    
    except json.JSONDecodeError as e:
        # Tentative de récupération: extraire les transactions valides avant l'erreur
        try:
            # Chercher le dernier objet JSON complet
            last_valid = cleaned.rfind('},')
            if last_valid > 0:
                truncated = cleaned[:last_valid + 1] + ']'
                transactions = json.loads(truncated)
                if isinstance(transactions, list) and len(transactions) > 0:
                    df = pd.DataFrame(transactions)
                    column_mapping = {"date": "Date", "libelle": "Libellé", "debit": "Débit", "credit": "Crédit"}
                    df = df.rename(columns=column_mapping)
                    df["Source"] = filename
                    for col in ["Débit", "Crédit"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    expected_cols = ["Date", "Libellé", "Débit", "Crédit", "Source"]
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = None
                    # Retourner les données partielles si possible
                    return df[expected_cols]
        except Exception:
            pass  # Si la récupération échoue, on lève l'erreur originale
        
        raise ValueError(f"Erreur de parsing JSON: {str(e)}\nRéponse reçue (500 premiers caractères): {response[:500]}...")
    except Exception as e:
        raise ValueError(f"Erreur lors du traitement de la réponse: {str(e)}")


def aggregate_results(dataframes: list) -> pd.DataFrame:
    """
    Agrège tous les DataFrames en un seul et les trie par date chronologique.
    
    Args:
        dataframes: Liste de DataFrames à combiner
        
    Returns:
        pd.DataFrame: DataFrame unifié trié par date
    """
    if not dataframes:
        return pd.DataFrame(columns=["Date", "Libellé", "Débit", "Crédit", "Source"])
    
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Trier par date chronologique
    if "Date" in combined.columns and len(combined) > 0:
        # Créer une colonne temporaire avec les dates converties en datetime
        def parse_date(date_str):
            """Convertit une date au format JJ/MM/AAAA en datetime"""
            if pd.isna(date_str) or date_str is None:
                return pd.NaT
            try:
                return pd.to_datetime(date_str, format="%d/%m/%Y", errors='coerce')
            except:
                return pd.NaT
        
        combined['_date_sort'] = combined['Date'].apply(parse_date)
        # Trier par date (les NaT seront en dernier)
        combined = combined.sort_values('_date_sort', na_position='last')
        # Supprimer la colonne temporaire
        combined = combined.drop(columns=['_date_sort'])
        # Réinitialiser l'index
        combined = combined.reset_index(drop=True)
    
    return combined


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en fichier Excel.
    
    Args:
        df: DataFrame à convertir
        
    Returns:
        bytes: Contenu du fichier Excel
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Transactions')
        
        # Ajuster la largeur des colonnes
        worksheet = writer.sheets['Transactions']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max() if len(df) > 0 else 0,
                len(col)
            ) + 2
            # Limiter la largeur maximale
            max_length = min(max_length, 50)
            worksheet.column_dimensions[chr(65 + idx)].width = max_length
    
    return output.getvalue()


def process_single_pdf(pdf_file, api_key: str, progress_lock: Lock, progress_dict: dict):
    """
    Traite un seul fichier PDF et retourne le résultat.
    
    Args:
        pdf_file: Fichier PDF uploadé via Streamlit
        api_key: Clé API Gemini
        progress_lock: Lock pour synchroniser les mises à jour de progression
        progress_dict: Dictionnaire partagé pour suivre la progression
        
    Returns:
        tuple: (filename, df, error) où df est un DataFrame ou None, et error est un message d'erreur ou None
    """
    filename = pdf_file.name
    try:
        # Étape 1: Extraction du texte
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            return (filename, None, "Le PDF ne contient pas de texte extractible")
        
        # Étape 2: Analyse avec Gemini
        response = analyze_with_gemini(text, api_key)
        
        # Étape 3: Parsing de la réponse
        df = parse_llm_response(response, filename)
        
        # Mettre à jour la progression
        with progress_lock:
            progress_dict['completed'] = progress_dict.get('completed', 0) + 1
        
        return (filename, df, None)
    
    except Exception as e:
        # Mettre à jour la progression même en cas d'erreur
        with progress_lock:
            progress_dict['completed'] = progress_dict.get('completed', 0) + 1
            progress_dict['errors'] = progress_dict.get('errors', [])
            progress_dict['errors'].append((filename, str(e)))
        
        return (filename, None, str(e))


def main():
    """Fonction principale de l'application."""
    
    # CSS pour cacher la sidebar par défaut
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal
    st.title("📊 Extracteur Comptable IA")
    st.markdown("*Extrayez automatiquement les lignes comptables de vos relevés bancaires PDF*")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gestion de la clé API
        api_key = os.getenv("GEMINI_API_KEY", "")
        
        if not api_key or api_key == "your_api_key_here":
            api_key = st.text_input("Clé API Gemini", type="password")
        
        st.divider()
        st.markdown("### ⚡ Performance")
        num_workers = st.slider(
            "Nombre de fichiers traités en parallèle",
            min_value=1,
            max_value=8,
            value=8,
            help="Augmentez ce nombre pour traiter plus de fichiers simultanément. Attention aux limites de l'API Gemini.",
            key="num_workers"
        )
        
        st.divider()
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Uploadez vos relevés PDF (max 15)
        2. Cliquez sur "Lancer l'analyse"
        3. Téléchargez le fichier Excel
        """)
        
        st.divider()
        st.markdown("### ℹ️ À propos")
        st.markdown("""
        Cette application utilise **Gemini 2.5 Flash** 
        pour analyser vos relevés bancaires et extraire 
        automatiquement les transactions.
        
        """)
    
    # Zone principale
    st.header("📁 Upload des fichiers")
    
    uploaded_files = st.file_uploader(
        "Glissez vos fichiers PDF ici",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILES} fichiers"
    )
    
    # Vérification du nombre de fichiers
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.error(f"❌ Trop de fichiers ! Maximum autorisé : {MAX_FILES}")
        uploaded_files = uploaded_files[:MAX_FILES]
        st.warning(f"Seuls les {MAX_FILES} premiers fichiers seront traités.")
    
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} fichier(s) sélectionné(s)")
        
        # Afficher la liste des fichiers
        with st.expander("Voir les fichiers"):
            for f in uploaded_files:
                st.text(f"• {f.name}")
    
    # Bouton d'analyse
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Lancer l'analyse",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files or not api_key or api_key == "your_api_key_here"
        )
    
    if not api_key or api_key == "your_api_key_here":
        st.warning("⚠️ Veuillez configurer votre clé API Gemini dans la sidebar.")
    
    # Traitement
    if analyze_button and uploaded_files and api_key:
        all_dataframes = []
        errors = []
        
        # Récupérer le nombre de workers depuis la session state ou utiliser la valeur par défaut
        num_workers = st.session_state.get('num_workers', 8)
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_container = st.container()
        
        # Dictionnaire partagé pour suivre la progression
        progress_dict = {'completed': 0, 'total': len(uploaded_files), 'errors': []}
        progress_lock = Lock()
        
        # Afficher le nombre de workers utilisés
        status_text.text(f"🚀 Démarrage du traitement parallèle ({num_workers} fichiers simultanés)...")
        
        # Traitement parallèle avec ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Soumettre toutes les tâches
            future_to_file = {
                executor.submit(process_single_pdf, pdf_file, api_key, progress_lock, progress_dict): pdf_file
                for pdf_file in uploaded_files
            }
            
            # Créer un conteneur pour les messages de statut par fichier
            status_placeholders = {}
            for pdf_file in uploaded_files:
                status_placeholders[pdf_file.name] = status_container.empty()
            
            # Traiter les résultats au fur et à mesure qu'ils arrivent
            for future in as_completed(future_to_file):
                filename, df, error = future.result()
                
                # Mettre à jour la barre de progression
                completed = progress_dict['completed']
                total = progress_dict['total']
                progress = completed / total
                progress_bar.progress(progress)
                
                # Afficher le statut
                status_text.text(f"📊 Progression : {completed}/{total} fichiers traités ({int(progress * 100)}%)")
                
                if error:
                    error_msg = f"❌ {filename} : {error}"
                    errors.append(error_msg)
                    status_placeholders[filename].error(error_msg)
                elif df is not None and len(df) > 0:
                    all_dataframes.append(df)
                    status_placeholders[filename].success(f"✅ {filename} : {len(df)} transactions extraites")
                else:
                    status_placeholders[filename].warning(f"⚠️ {filename} : Aucune transaction trouvée")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Traitement terminé !")
        
        # Agrégation et affichage des résultats
        if all_dataframes:
            st.divider()
            st.header("📊 Résultats")
            
            final_df = aggregate_results(all_dataframes)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total transactions", len(final_df))
            with col2:
                total_debit = final_df["Débit"].sum()
                st.metric("Total débits", f"{total_debit:,.2f} €" if pd.notna(total_debit) else "0,00 €")
            with col3:
                total_credit = final_df["Crédit"].sum()
                st.metric("Total crédits", f"{total_credit:,.2f} €" if pd.notna(total_credit) else "0,00 €")
            with col4:
                st.metric("Fichiers traités", len(all_dataframes))
            
            # Affichage du tableau
            st.dataframe(
                final_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Libellé": st.column_config.TextColumn("Libellé", width="large"),
                    "Débit": st.column_config.NumberColumn("Débit", format="%.2f €"),
                    "Crédit": st.column_config.NumberColumn("Crédit", format="%.2f €"),
                    "Source": st.column_config.TextColumn("Source", width="medium"),
                }
            )
            
            # Bouton de téléchargement
            st.divider()
            excel_data = convert_df_to_excel(final_df)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Télécharger le fichier Excel",
                    data=excel_data,
                    file_name="extraction_comptable.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
        
        elif errors:
            st.error("❌ Aucune transaction n'a pu être extraite. Vérifiez vos fichiers.")


if __name__ == "__main__":
    main()
