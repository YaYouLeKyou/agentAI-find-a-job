import streamlit as st
from groq import Groq
import PyPDF2
import json
import os
from dotenv import load_dotenv
import pandas as pd
import io
import time
from datetime import datetime
import zipfile
import google.generativeai as genai
import base64

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ Clé API GROQ non trouvée. Veuillez vérifier votre fichier .env")

client = Groq(api_key=api_key) if api_key else None

if gemini_key:
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
else:
    st.warning("⚠️ Clé API GEMINI non trouvée. L'analyse d'images et de PDFs complexes sera désactivée.")
    gemini_model = None

def normalize_columns(df):
    """Normalise les noms de colonnes pour qu'ils soient reconnus par l'application."""
    col_map = {
        'Date': ['date', 'date opération', 'date op', 'dateop', 'valeur', 'jour'],
        'Libellé': ['libellé', 'libelle', 'description', 'désignation', 'libelleop', 'détails', 'motif', 'objet'],
        'Montant': ['montant', 'valeur', 'somme', 'montantop', 'total', 'euro', 'prix'],
        'Débit': ['débit', 'debit', 'dépenses', 'sorties', 'paiement'],
        'Crédit': ['crédit', 'credit', 'recettes', 'entrées', 'versement']
    }
    
    # Nettoyage des noms actuels (minuscules et sans espaces)
    current_cols = {str(c).strip().lower(): c for c in df.columns}
    
    for target, variations in col_map.items():
        if target in df.columns: continue
        for var in variations:
            if var in current_cols:
                df[target] = df[current_cols[var]]
                break
    return df

def clean_amount(val):
    """Nettoie et convertit une valeur en float robuste."""
    if pd.isna(val) or val == "": return 0.0
    if isinstance(val, (int, float)): return float(val)
    
    # Remove spaces, currency symbols and common separators
    s = str(val).upper().replace(' ', '').replace('€', '').replace('\xa0', '').replace('\u202f', '').replace('\u00a0', '')
    is_neg = "-" in s or "(" in s or "DEBIT" in s
    
    # Gestion des séparateurs de milliers (1.234,56 ou 1,234.56)
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '') # Le point est le millier
        else:
            s = s.replace(',', '') # La virgule est le millier
            
    # Garde chiffres, points et virgules pour la conversion finale
    cleaned = "".join(c for c in s if c.isdigit() or c in ".,")
    if not cleaned: return 0.0
    
    try:
        amount = float(cleaned.replace(',', '.'))
        return -amount if is_neg else amount
    except ValueError: return 0.0

def extract_text_from_pdf(file):
    """Extrait le texte d'un fichier PDF de manière sécurisée."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        if len(pdf_reader.pages) == 0:
            return None
            
        for page_num, page in enumerate(pdf_reader.pages):
            content = page.extract_text() or ""
            if content:
                text += content
        return text
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        return None

def handle_zip_file(uploaded_file, target_exts):
    """Extrait les fichiers d'un ZIP correspondant à l'extension voulue."""
    extracted_files = []
    try:
        with zipfile.ZipFile(uploaded_file) as z:
            for file_name in z.namelist():
                if file_name.lower().endswith(target_exts):
                    with z.open(file_name) as f:
                        content = io.BytesIO(f.read())
                        content.name = file_name # Simulation de l'attribut name
                        extracted_files.append(content)
        return extracted_files
    except Exception as e:
        st.error(f"Erreur ZIP : {e}")
        return []

def analyze_invoice_ai(file_data, mime_type, text_content=None):
    """Utilise l'IA (Groq ou Gemini) pour extraire les informations clés d'une facture."""
    prompt = """
    Tu es un expert comptable. Analyse ce document (facture ou reçu) et extrait les informations en JSON :
    {
        "fournisseur": "nom",
        "date": "YYYY-MM-DD",
        "total_ttc": 0.0,
        "tva": 0.0,
        "categorie": "Loyer/Transport/Logiciel/etc",
        "description": "bref résumé"
    }
    Réponds uniquement avec l'objet JSON, sans texte additionnel.
    Si une info est manquante, mets null.
    """
    
    # 1. Essai avec Groq si texte disponible (PDF numérique)
    if text_content and client:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": f"{prompt}\n\nTexte du document :\n{text_content}"}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception:
            pass

    if not gemini_model:
        return None

    try:
        response = gemini_model.generate_content([
            prompt,
            {'mime_type': mime_type, 'data': file_data}
        ])
        if not response.parts:
            st.error("L'IA n'a pas pu générer de réponse (possiblement bloquée par les filtres de sécurité).")
            return None
        # Nettoyage pour s'assurer d'avoir un JSON valide
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        st.error(f"Erreur d'analyse IA : {e}")
        return None

def analyze_bank_statement_ai(file_data, mime_type, text_content=None):
    """Utilise l'IA (Groq ou Gemini) pour extraire la liste des transactions d'un relevé bancaire."""
    prompt = """
    Tu es un expert comptable. Analyse ce relevé bancaire et extrait la liste de TOUTES les transactions sous forme de tableau JSON.
    Chaque objet du tableau doit avoir exactement ces clés :
    {
        "Date": "YYYY-MM-DD",
        "Libellé": "description complète de l'opération",
        "Montant": "-12.50" (utilise un signe moins pour les dépenses, pas de signe pour les recettes)
    }
    Réponds uniquement avec le tableau JSON [ ... ], sans texte avant ou après.
    """
    
    # 1. Essai avec Groq si texte disponible
    if text_content and client:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": f"{prompt}\n\nTexte du relevé :\n{text_content}"}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            st.warning(f"Groq a échoué pour le relevé... {e}")

    if not gemini_model:
        return None

    try:
        response = gemini_model.generate_content([
            prompt,
            {'mime_type': mime_type, 'data': file_data}
        ])
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        st.error(f"Erreur d'analyse du relevé : {e}")
        return None

def get_mime_type(filename):
    ext = filename.lower()
    if ext.endswith(".pdf"): return "application/pdf"
    if ext.endswith(".png"): return "image/png"
    if ext.endswith(".jpg") or ext.endswith(".jpeg"): return "image/jpeg"
    if ext.endswith(".csv"): return "text/csv"
    return "application/octet-stream"

def categorize_bank_labels(unmatched_rows):
    """Demande à l'IA de catégoriser les lignes sans factures par lot."""
    if not client or not unmatched_rows: return {}
    labels = [r['Libellé'] for r in unmatched_rows]
    prompt = f"""
    En tant qu'expert comptable, catégorise ces libellés bancaires. 
    Utilise ces catégories : Ventes, Loyer, Logiciel, Transport, Alimentation, Salaire, Impôts, Divers.
    Retourne UNIQUEMENT un JSON : {{"libellé": "catégorie"}}
    
    Libellés : {labels}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except: return {}

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="My Accounter AI", page_icon="💰", layout="wide")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    /* Amélioration globale */
    .main { background-color: #f9fafb; }
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Style des cartes */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #edf2f7 !important;
        background-color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    /* Optimisation Mobile */
    @media (max-width: 640px) {
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        h1 { font-size: 1.8rem !important; }
        .stMarkdown div p { font-size: 0.95rem; }
    }
    </style>
""", unsafe_allow_html=True)

st.title("💰 My Accounter AI")
st.markdown("#### Votre comptabilité simplifiée par l'Intelligence Artificielle")

# --- INITIALISATION DE L'ÉTAT ---
if 'bank_data' not in st.session_state: st.session_state['bank_data'] = None
if 'invoices' not in st.session_state: st.session_state['invoices'] = []

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.info("Uploadez vos documents pour générer votre bilan.")
    if st.button("Réinitialiser les données"):
        st.session_state['bank_data'] = None
        st.session_state['invoices'] = []
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏦 Relevé Bancaire (CSV, PDF, Images)")
    bank_files = st.file_uploader("Upload CSV(s), PDF(s), Image(s) ou ZIP", type=["csv", "zip", "pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if bank_files and st.button("Charger les relevés"):
        all_dfs = []
        valid_bank_exts = ('.csv', '.pdf', '.png', '.jpg', '.jpeg')
        
        for f in bank_files:
            files_to_process = [f] if f.name.lower().endswith(valid_bank_exts) else handle_zip_file(f, valid_bank_exts)
            for b_f in files_to_process:
                if b_f.name.lower().endswith('.csv'):
                    try:
                        df_temp = pd.read_csv(b_f, sep=None, engine='python')
                        df_temp = normalize_columns(df_temp)
                        all_dfs.append(df_temp)
                    except:
                        st.error(f"Erreur sur le fichier CSV {b_f.name}")
                else:
                    with st.spinner(f"Analyse IA du relevé : {b_f.name}..."):
                        m_type = get_mime_type(b_f.name)
                        # Si b_f est un BytesIO (depuis ZIP), on utilise getvalue(), sinon read()
                        file_content = b_f.getvalue() if hasattr(b_f, 'getvalue') else b_f.read()
                        
                        # Extraction de texte pour Groq
                        txt = extract_text_from_pdf(b_f) if m_type == "application/pdf" else None
                        data = analyze_bank_statement_ai(file_content, m_type, text_content=txt)
                        if data:
                            df_temp = pd.DataFrame(data)
                            df_temp = normalize_columns(df_temp)
                            all_dfs.append(df_temp)
        
        if all_dfs:
            st.session_state['bank_data'] = pd.concat(all_dfs, ignore_index=True)
            st.success(f"✅ {len(all_dfs)} relevé(s) consolidé(s) !")

with col2:
    st.subheader("🧾 Factures & Reçus (PDF, PNG, JPG)")
    invoice_files_input = st.file_uploader("Upload PDF(s), Image(s) ou ZIP", type=["pdf", "zip", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if invoice_files_input and st.button("Lancer l'analyse IA"):
        # Collecte de tous les documents (directs ou dans ZIP)
        all_docs = []
        valid_exts = ('.pdf', '.png', '.jpg', '.jpeg')
        for f in invoice_files_input:
            if f.name.lower().endswith(valid_exts):
                all_docs.append(f)
            elif f.name.lower().endswith('.zip'):
                all_docs.extend(handle_zip_file(f, valid_exts))

        with st.spinner("L'IA analyse vos factures..."):
            new_invoices = 0
            for file in all_docs:
                # Éviter de traiter deux fois le même fichier
                if any(inv.get('filename') == file.name for inv in st.session_state['invoices']):
                    continue
                    
                # Préparation des données pour Gemini (plus besoin d'extraire le texte manuellement)
                file_bytes = file.getvalue()
                m_type = get_mime_type(file.name)

                # Extraction de texte pour Groq
                txt = extract_text_from_pdf(file) if m_type == "application/pdf" else None
                data = analyze_invoice_ai(file_bytes, m_type, text_content=txt)
                if data:
                    data['filename'] = file.name
                    st.session_state['invoices'].append(data)
                    new_invoices += 1
            st.success(f"✨ {new_invoices} nouvelles factures ajoutées !")

# --- TRAITEMENT & RECONCILIATION ---
if st.session_state['bank_data'] is not None:
    st.divider()
    st.header("📊 Analyse du Bilan Professionnel")
    
    df = st.session_state['bank_data']
    invoices = st.session_state['invoices']
    
    # Vérification de sécurité
    critical_cols = ['Date', 'Libellé']
    if not any(c in df.columns for c in critical_cols):
        st.warning("⚠️ Les colonnes du relevé n'ont pas été reconnues. Vérifiez les titres de votre fichier CSV.")

    results = []
    for idx, row in df.iterrows():
        if 'Débit' in df.columns or 'Crédit' in df.columns:
            d = clean_amount(row.get('Débit', row.get('Debit', 0)))
            c = clean_amount(row.get('Crédit', row.get('Credit', 0)))
            montant_net = c if abs(c) > 0 else -abs(d)
        else: montant_net = clean_amount(row.get('Montant', 0))

        match = None
        abs_montant = abs(montant_net)
        for inv in invoices:
            if abs(inv.get('total_ttc', 0) - abs_montant) < 0.05:
                match = inv
                break
        
        results.append({
            "Date": row.get('Date', 'N/A'),
            "Libellé": row.get('Libellé', 'N/A'),
            "Montant": montant_net,
            "Statut": "✅ Lié" if match else "❌ Manquant",
            "Facture": match.get('filename') if match else "-",
            "Catégorie": match.get('categorie') if match else "À définir"
        })

    # Catégorisation intelligente pour les manquants
    unmatched = [r for r in results if r['Statut'] == "❌ Manquant"]
    if unmatched and st.button("🪄 Catégoriser intelligemment les manquants"):
        with st.spinner("L'IA catégorise vos dépenses sans factures..."):
            ai_cats = categorize_bank_labels(unmatched)
            for r in results:
                if r['Statut'] == "❌ Manquant" and r['Libellé'] in ai_cats:
                    r['Catégorie'] = ai_cats[r['Libellé']]
        st.rerun()

    # --- AFFICHAGE COMPTE DE RÉSULTAT ---
    res_df = pd.DataFrame(results)
    
    st.subheader("📊 Compte de Résultat (Simplifié)")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("**Produits (Recettes)**")
        recettes_df = res_df[res_df['Montant'] > 0].groupby('Catégorie')['Montant'].sum().reset_index()
        if recettes_df.empty: st.write("Aucune recette détectée.")
        else: st.table(recettes_df.style.format({'Montant': '{:.2f} €'}))
        
    with col_res2:
        st.markdown("**Charges (Dépenses)**")
        depenses_df = res_df[res_df['Montant'] < 0].groupby('Catégorie')['Montant'].sum().abs().reset_index()
        if depenses_df.empty: st.write("Aucune dépense détectée.")
        else: st.table(depenses_df.style.format({'Montant': '{:.2f} €'}))

    st.subheader("🔍 Détail des opérations")
    st.dataframe(res_df.style.applymap(lambda x: 'color: red' if x == "❌ Manquant" else 'color: green', subset=['Statut']), use_container_width=True)

    # --- RÉSUMÉ FINANCIER ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    total_depenses = abs(res_df[res_df['Montant'] < 0]['Montant'].sum())
    total_recettes = res_df[res_df['Montant'] > 0]['Montant'].sum()
    
    c1.metric("Total Recettes", f"{total_recettes:.2f} €")
    c2.metric("Total Dépenses", f"{total_depenses:.2f} €")
    c3.metric("Bilan (Net)", f"{(total_recettes - total_depenses):.2f} €", delta=f"{(total_recettes - total_depenses):.2f} €")

st.caption("My Accounter AI - Propulsé par Streamlit, Gemini 1.5 Flash & Llama 3.3")
