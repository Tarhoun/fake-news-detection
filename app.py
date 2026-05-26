import subprocess
import sys

# --- حيلة ذكية لصب المكتبات غصباً عن السيرفر في الأول ---
try:
    import streamlit as st
    import sklearn
    import nltk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "nltk"])
    import streamlit as st

import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# تحميل ملفات الـ NLTK الضرورية للـ Cloud
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Chargement des objets sauvegardés
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
svd = pickle.load(open('svd.pkl', 'rb'))
model_lr = pickle.load(open('model_lr.pkl', 'rb'))
model_nb = pickle.load(open('model_nb.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# --- Design de l'interface ---
st.set_page_config(page_title="Détecteur de Fake News", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Système de Détection de Fake News")
st.write("**Projet PFA - Apprentissage Supervisé (iTeam 2026)**")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Analyse d'un article")
    user_input = st.text_area("Saisir ou coller le texte de l'article de presse à tester :", height=200)

    if st.button("Lancer l'Analyse"):
        if user_input.strip() == "":
            st.warning("Veuillez introduire un texte valide.")
        else:
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])
            red = svd.transform(vec)

            pred_lr = "✅ VRAI (True News)" if model_lr.predict(red)[0] == 1 else "🚨 FAUX (Fake News)"
            pred_nb = "✅ VRAI (True News)" if model_nb.predict(red)[0] == 1 else "🚨 FAUX (Fake News)"
            pred_rf = "✅ VRAI (True News)" if model_rf.predict(red)[0] == 1 else "🚨 FAUX (Fake News)"

            st.info("### 🤖 Verdicts des modèles :")
            st.write(f"**1. Régression Logistique :** {pred_lr}")
            st.write(f"**2. Naïve Bayes :** {pred_nb}")
            st.write(f"**3. Random Forest :** {pred_rf}")

with col2:
    st.subheader("📊 Performance globale des modèles")
    st.write("Voici les métriques d'évaluation obtenues lors de la phase de test :")
    st.table({
        "Modèle": ["Régression Logistique", "Naïve Bayes", "Random Forest"],
        "Accuracy": ["98.99%", "81.24%", "96.43%"],
        "F1-Score": ["98.96%", "80.24%", "96.36%"]
    })
