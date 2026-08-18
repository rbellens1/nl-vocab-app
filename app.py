import streamlit as st
import pandas as pd
import os
import random

# Fichier de sauvegarde
DATA_FILE = "vocabulaire_b2.csv"

# Initialisation des données
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=["Néerlandais", "Français", "Contexte", "Catégorie", "Score"])
        df.to_csv(DATA_FILE, index=False)
        return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

df = load_data()

# Interface utilisateur
st.set_page_config(page_title="Mon Vocabulaire Néerlandais", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Aller à", ["📚 Bibliothèque", "➕ Ajouter un mot", "🏋️ Exercices"])

# --- PAGE 1 : LA BIBLIOTHÈQUE ---
if menu == "📚 Bibliothèque":
    st.title("Ma Bibliothèque de Vocabulaire")
    
    if df.empty:
        st.info("Ta bibliothèque est vide. Ajoute des mots via le menu !")
    else:
        # Filtre par catégorie
        categories = ["Toutes"] + df["Catégorie"].dropna().unique().tolist()
        filtre = st.selectbox("Filtrer par catégorie", categories)
        
        if filtre != "Toutes":
            st.dataframe(df[df["Catégorie"] == filtre], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

# --- PAGE 2 : AJOUTER UN MOT ---
elif menu == "➕ Ajouter un mot":
    st.title("Ajouter du nouveau vocabulaire")
    
    with st.form("add_word_form"):
        col1, col2 = st.columns(2)
        with col1:
            nl_word = st.text_input("Mot en néerlandais (ex: de nettowinst)")
            cat = st.selectbox("Catégorie", ["Finance & Économie", "Juridique", "Expressions Idiomatiques", "Général"])
        with col2:
            fr_word = st.text_input("Traduction en français (ex: le bénéfice net)")
            context = st.text_area("Contexte / Phrase d'exemple")
            
        submit = st.form_submit_button("Ajouter à la base")
        
        if submit and nl_word and fr_word:
            new_row = pd.DataFrame({
                "Néerlandais": [nl_word],
                "Français": [fr_word],
                "Contexte": [context],
                "Catégorie": [cat],
                "Score": [0] # Pour suivre tes bonnes réponses plus tard
            })
            df = pd.concat([df, new_row], ignore_index=True)
            save_data(df)
            st.success(f"Le mot '{nl_word}' a été ajouté avec succès !")

# --- PAGE 3 : EXERCICES (QUIZ) ---
elif menu == "🏋️ Exercices":
    st.title("Entraînement Actif")
    
    if df.empty:
        st.warning("Ajoute d'abord des mots dans ta bibliothèque pour t'entraîner !")
    else:
        st.write("Traduis le mot suivant :")
        
        # Tirer un mot au hasard
        if "current_word" not in st.session_state:
            st.session_state.current_word = df.sample(1).iloc[0]
            st.session_state.answered = False
            
        word_to_guess = st.session_state.current_word
        
        st.subheader(f"🇫🇷 {word_to_guess['Français']}")
        
        # Formulaire de réponse
        with st.form("quiz_form"):
            answer = st.text_input("Ta réponse en néerlandais :")
            check = st.form_submit_button("Vérifier")
            
            if check:
                st.session_state.answered = True
                if answer.strip().lower() == word_to_guess['Néerlandais'].strip().lower():
                    st.success("Correct ! 🎉")
                    # Bonus: Afficher le contexte
                    st.info(f"Contexte : {word_to_guess['Contexte']}")
                else:
                    st.error(f"Faux. La bonne réponse était : **{word_to_guess['Néerlandais']}**")
                    st.info(f"Contexte : {word_to_guess['Contexte']}")
        
        # Bouton pour passer au mot suivant
        if st.session_state.get("answered", False):
            if st.button("Mot suivant ➔"):
                st.session_state.current_word = df.sample(1).iloc[0]
                st.session_state.answered = False
                st.rerun()