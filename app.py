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

MISSION : Analyse le relevé bancaire fourni et extrais TOUTES les lignes de transactions.

Pour chaque transaction, retourne un objet JSON avec :
- "date": la date de la transaction au format JJ/MM/AAAA
- "libelle": le libellé de l'opération. Si le libellé dépasse 50 caractères, résume-le de manière concise en gardant les mots-clés essentiels (nom du bénéficiaire, type d'opération, référence importante).
- "debit": le montant en débit sous forme de nombre flottant (ex: 1234.56). Mettre null si c'est un crédit.
- "credit": le montant en crédit sous forme de nombre flottant (ex: 1234.56). Mettre null si c'est un débit.

RÈGLES IMPORTANTES :
1. NORMALISATION DES MONTANTS : Convertis tous les formats de montants en nombres flottants standard.
   - "1 000,50" → 1000.50
   - "1.000,50" → 1000.50
   - "1,000.50" → 1000.50
   - "1000,50" → 1000.50
2. Ne confonds pas débit et crédit. Un débit est une sortie d'argent (paiement), un crédit est une entrée (virement reçu).
3. Ignore les lignes qui ne sont pas des transactions (soldes, totaux, en-têtes, etc.).
4. Si une transaction s'étend sur plusieurs lignes dans le PDF, reconstitue-la correctement.
5. Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après.

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


def analyze_with_gemini(text: str, api_key: str) -> str:
    """
    Envoie le texte au modèle Gemini pour analyse.
    
    Args:
        text: Texte extrait du PDF
        api_key: Clé API Gemini
        
    Returns:
        str: Réponse du modèle (JSON attendu)
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Construire le prompt complet
        full_prompt = SYSTEM_PROMPT + text
        
        # Générer la réponse avec mode JSON structuré
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Basse température pour plus de précision
                max_output_tokens=16384,  # Augmenté pour les relevés avec beaucoup de transactions
                response_mime_type="application/json",  # Force Gemini à produire un JSON valide
            )
        )
        
        return response.text
    
    except Exception as e:
        raise Exception(f"Erreur API Gemini: {str(e)}")


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
    Agrège tous les DataFrames en un seul.
    
    Args:
        dataframes: Liste de DataFrames à combiner
        
    Returns:
        pd.DataFrame: DataFrame unifié
    """
    if not dataframes:
        return pd.DataFrame(columns=["Date", "Libellé", "Débit", "Crédit", "Source"])
    
    combined = pd.concat(dataframes, ignore_index=True)
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


def verify_api_key(api_key: str) -> bool:
    """
    Vérifie si la clé API Gemini est valide.
    
    Args:
        api_key: Clé API à vérifier
        
    Returns:
        bool: True si la clé est valide
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        # Test simple
        response = model.generate_content("Dis 'OK'")
        return True
    except Exception:
        return False


def main():
    """Fonction principale de l'application."""
    
    # Titre principal
    st.title("📊 Extracteur Comptable IA")
    st.markdown("*Extrayez automatiquement les lignes comptables de vos relevés bancaires PDF*")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gestion de la clé API
        api_key = os.getenv("GEMINI_API_KEY", "")
        
        if api_key and api_key != "your_api_key_here":
            st.success("✅ Clé API chargée depuis .env")
            use_env_key = st.checkbox("Utiliser la clé du fichier .env", value=True)
            if not use_env_key:
                api_key = st.text_input("Clé API Gemini", type="password")
        else:
            st.warning("⚠️ Aucune clé API trouvée dans .env")
            st.markdown("""
            **Pour configurer votre clé API :**
            1. Créez un fichier `.env` à la racine du projet
            2. Ajoutez : `GEMINI_API_KEY=votre_clé`
            
            Ou entrez-la directement ci-dessous :
            """)
            api_key = st.text_input("Clé API Gemini", type="password")
        
        # Vérification de la clé
        if api_key and api_key != "your_api_key_here":
            if st.button("🔍 Vérifier la connexion"):
                with st.spinner("Vérification..."):
                    if verify_api_key(api_key):
                        st.success("✅ Connexion réussie !")
                    else:
                        st.error("❌ Clé API invalide")
        
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
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            status_text.text(f"📄 Traitement de {pdf_file.name}... ({i + 1}/{len(uploaded_files)})")
            
            try:
                # Étape 1: Extraction du texte
                status_text.text(f"📄 {pdf_file.name} - Extraction du texte...")
                text = extract_text_from_pdf(pdf_file)
                
                if not text.strip():
                    raise ValueError("Le PDF ne contient pas de texte extractible")
                
                # Étape 2: Analyse avec Gemini
                status_text.text(f"📄 {pdf_file.name} - Analyse IA en cours...")
                response = analyze_with_gemini(text, api_key)
                
                # Étape 3: Parsing de la réponse
                status_text.text(f"📄 {pdf_file.name} - Traitement des données...")
                df = parse_llm_response(response, pdf_file.name)
                
                if len(df) > 0:
                    all_dataframes.append(df)
                    st.success(f"✅ {pdf_file.name} : {len(df)} transactions extraites")
                else:
                    st.warning(f"⚠️ {pdf_file.name} : Aucune transaction trouvée")
                
            except Exception as e:
                error_msg = f"❌ {pdf_file.name} : {str(e)}"
                errors.append(error_msg)
                st.error(error_msg)
            
            progress_bar.progress(progress)
            
            # Petit délai pour éviter le rate limiting
            if i < len(uploaded_files) - 1:
                time.sleep(0.5)
        
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
