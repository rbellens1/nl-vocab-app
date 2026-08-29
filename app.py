import streamlit as st
import pandas as pd
import os
import random
from io import BytesIO
from datetime import date, timedelta
from gtts import gTTS

# ============================================================
# INITIALISATION DE L'ÉTAT (SESSION STATE)
# ============================================================
if "words_done_today" not in st.session_state:
    st.session_state.words_done_today = 0
if "session_goal" not in st.session_state:
    st.session_state.session_goal = 20

# ============================================================
# AUDIO (Text-to-Speech via gTTS)
# ============================================================
@st.cache_data(show_spinner=False)
def generate_audio(text, lang):
    tts = gTTS(text=text, lang=lang)
    buf = BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()

def speak_button(text, lang="nl", label="🔊 Écouter", key=None):
    text = "" if text is None else str(text)
    if not text.strip():
        return
    if st.button(label, key=key):
        try:
            with st.spinner("Génération..."):
                audio_bytes = generate_audio(text, lang)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception:
            st.warning("Audio indisponible (connexion internet requise).")

# ============================================================
# CONFIGURATION & DONNÉES
# ============================================================
DATA_FILE = "vocabulaire_b2.csv"

COLUMNS = [
    "ID", "Néerlandais", "Français", "Contexte", "Catégorie",
    "EaseFactor", "Interval", "Repetitions",
    "NextReview", "LastReview", "TotalReviews", "CorrectReviews",
]

DEFAULTS = {
    "EaseFactor": 2.5, "Interval": 0, "Repetitions": 0,
    "LastReview": "", "TotalReviews": 0, "CorrectReviews": 0,
}

TEXT_COLUMNS = ["Néerlandais", "Français", "Contexte", "Catégorie", "NextReview", "LastReview"]
INT_COLUMNS = ["ID", "Interval", "Repetitions", "TotalReviews", "CorrectReviews"]

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=COLUMNS)
        save_data(df)
        return df

    changed = False
    today_str = str(date.today())

    if "Score" in df.columns and "Repetitions" not in df.columns:
        df = df.drop(columns=["Score"])
        changed = True

    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))
        changed = True

    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            changed = True

    if "NextReview" not in df.columns:
        df["NextReview"] = today_str
        changed = True

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, "")

    df = df[COLUMNS]

    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["EaseFactor"] = pd.to_numeric(df["EaseFactor"], errors="coerce").fillna(2.5).astype(float)
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)
    
    df.loc[df["NextReview"].isin(["", "nan"]), "NextReview"] = today_str

    if changed:
        save_data(df)
    return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def next_id(df):
    if df.empty:
        return 1
    return int(df["ID"].max()) + 1

# ============================================================
# RÉPÉTITION ESPACÉE (Algorithme SM-2)
# ============================================================
def sm2_update(df, idx, quality):
    ef = float(df.at[idx, "EaseFactor"])
    reps = int(df.at[idx, "Repetitions"])
    interval = int(df.at[idx, "Interval"])

    if quality < 3:
        reps = 0
        interval = 1
    else:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        reps += 1

    ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ef = max(1.3, ef)

    df.at[idx, "EaseFactor"] = round(ef, 2)
    df.at[idx, "Repetitions"] = reps
    df.at[idx, "Interval"] = interval
    df.at[idx, "NextReview"] = str(date.today() + timedelta(days=interval))
    df.at[idx, "LastReview"] = str(date.today())
    df.at[idx, "TotalReviews"] = int(df.at[idx, "TotalReviews"]) + 1
    if quality >= 3:
        df.at[idx, "CorrectReviews"] = int(df.at[idx, "CorrectReviews"]) + 1
    return df

def update_word(df, word_id, quality):
    idx = df.index[df["ID"] == word_id][0]
    df = sm2_update(df, idx, quality)
    save_data(df)
    return df

def mark_as_known(df, word_id):
    """Marque un mot comme totalement maîtrisé (repoussé de 10 ans)."""
    idx = df.index[df["ID"] == word_id][0]
    df.at[idx, "Repetitions"] = 10
    df.at[idx, "Interval"] = 3650
    df.at[idx, "EaseFactor"] = 2.5
    df.at[idx, "NextReview"] = str(date.today() + timedelta(days=3650))
    df.at[idx, "LastReview"] = str(date.today())
    save_data(df)
    return df

def pick_word(df):
    today_str = str(date.today())
    due = df[df["NextReview"] <= today_str]
    if not due.empty:
        due_sorted = due.sort_values("NextReview")
        top = due_sorted.head(max(3, len(due_sorted) // 2))
        return top.sample(1).iloc[0], True
    return df.sample(1).iloc[0], False

def generate_choices(df, correct_row, direction):
    target_col = "Néerlandais" if direction == "fr_to_nl" else "Français"
    correct_answer = str(correct_row[target_col])

    same_cat = df[(df["Catégorie"] == correct_row["Catégorie"]) & (df[target_col].astype(str) != correct_answer)]
    pool = same_cat[target_col].dropna().astype(str).unique().tolist()
    random.shuffle(pool)
    distractors = pool[:3]

    if len(distractors) < 3:
        others = df[df[target_col].astype(str) != correct_answer]
        extra = [v for v in others[target_col].dropna().astype(str).unique().tolist() if v not in distractors]
        random.shuffle(extra)
        distractors += extra[: 3 - len(distractors)]

    choices = distractors + [correct_answer]
    random.shuffle(choices)
    return choices, correct_answer

def new_question(df, direction_pref):
    row, was_due = pick_word(df)
    st.session_state.current_word = row
    st.session_state.was_due = was_due
    if direction_pref == "Aléatoire":
        st.session_state.current_direction = random.choice(["fr_to_nl", "nl_to_fr"])
    elif direction_pref == "FR → NL":
        st.session_state.current_direction = "fr_to_nl"
    else:
        st.session_state.current_direction = "nl_to_fr"
    st.session_state.answered = False
    st.session_state.is_correct = None
    st.session_state.qcm_choices = None
    st.session_state.qcm_correct = None

# ============================================================
# INTERFACE
# ============================================================
df = load_data()

st.set_page_config(page_title="Vocabulaire Néerlandais", page_icon="🇳🇱", layout="wide")

st.sidebar.title("Navigation")
if not df.empty:
    due_count = len(df[df["NextReview"] <= str(date.today())])
    st.sidebar.caption(f"📅 {due_count} mot(s) en attente de révision")
    
menu = st.sidebar.radio("Aller à", ["📚 Bibliothèque", "➕ Ajouter / Importer", "🏋️ Exercices", "📊 Statistiques"])

# --- PAGE 1 : LA BIBLIOTHÈQUE ET GESTION ---
if menu == "📚 Bibliothèque":
    st.title("Ma Bibliothèque de Vocabulaire")

    if df.empty:
        st.info("Ta bibliothèque est vide. Ajoute des mots via le menu !")
    else:
        tab1, tab2 = st.tabs(["📖 Consulter", "✏️ Éditer / Supprimer"])
        
        with tab1:
            categories = ["Toutes"] + sorted(df["Catégorie"].dropna().unique().tolist())
            filtre = st.selectbox("Filtrer par catégorie", categories)

            shown = df if filtre == "Toutes" else df[df["Catégorie"] == filtre]
            shown = shown.sort_values("NextReview")
            display = shown.rename(columns={"Repetitions": "Niveau", "NextReview": "Prochaine révision"})[
                ["Néerlandais", "Français", "Contexte", "Catégorie", "Niveau", "Prochaine révision"]
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)
            
        with tab2:
            st.subheader("Modifier ou supprimer un mot")
            word_list = df["Néerlandais"] + " - " + df["Français"]
            word_to_edit = st.selectbox("Rechercher un mot :", word_list.tolist())
            
            if word_to_edit:
                selected_nl = word_to_edit.split(" - ")[0]
                row_idx = df.index[df["Néerlandais"] == selected_nl][0]
                current_data = df.iloc[row_idx]
                
                with st.form("edit_form"):
                    col1, col2 = st.columns(2)
                    edit_nl = col1.text_input("Néerlandais", current_data["Néerlandais"])
                    edit_fr = col2.text_input("Français", current_data["Français"])
                    edit_ctx = st.text_input("Contexte", current_data["Contexte"])
                    edit_cat = st.text_input("Catégorie", current_data["Catégorie"])
                    
                    c_save, c_del = st.columns([1, 1])
                    if c_save.form_submit_button("💾 Enregistrer les modifications"):
                        df.at[row_idx, "Néerlandais"] = edit_nl
                        df.at[row_idx, "Français"] = edit_fr
                        df.at[row_idx, "Contexte"] = edit_ctx
                        df.at[row_idx, "Catégorie"] = edit_cat
                        save_data(df)
                        st.success("Modifications sauvegardées !")
                        st.rerun()
                        
                    if c_del.form_submit_button("🗑️ Supprimer ce mot (Définitif)"):
                        df = df.drop(row_idx)
                        save_data(df)
                        st.warning("Mot supprimé !")
                        st.rerun()

# --- PAGE 2 : AJOUTER OU IMPORTER ---
elif menu == "➕ Ajouter / Importer":
    st.title("Alimenter la base de données")

    tab1, tab2 = st.tabs(["✍️ Ajout Manuel", "📥 Importation CSV"])
    
    with tab1:
        with st.form("add_word_form"):
            col1, col2 = st.columns(2)
            nl_word = col1.text_input("Mot en néerlandais (ex: de nettowinst)")
            fr_word = col2.text_input("Traduction en français (ex: le bénéfice net)")
            cat = col1.selectbox("Catégorie", ["Finance & Économie", "Juridique", "Expressions Idiomatiques", "Général"])
            context = col2.text_area("Contexte / Phrase d'exemple")

            if st.form_submit_button("Ajouter à la base") and nl_word and fr_word:
                new_row = pd.DataFrame({
                    "ID": [next_id(df)], "Néerlandais": [nl_word], "Français": [fr_word],
                    "Contexte": [context], "Catégorie": [cat], "EaseFactor": [2.5],
                    "Interval": [0], "Repetitions": [0], "NextReview": [str(date.today())],
                    "LastReview": [""], "TotalReviews": [0], "CorrectReviews": [0],
                })
                df = pd.concat([df, new_row], ignore_index=True)
                save_data(df)
                st.success(f"'{nl_word}' ajouté avec succès !")

    with tab2:
        st.info("Importe une liste de mots via un fichier CSV. Assure-toi d'avoir au moins les colonnes 'Néerlandais' et 'Français'. L'importation ignorera les mots déjà existants pour ne pas écraser ta progression.")
        uploaded_file = st.file_uploader("Choisis un fichier CSV", type="csv")
        
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            if "Néerlandais" in new_df.columns and "Français" in new_df.columns:
                existing_nl = df["Néerlandais"].tolist()
                # Filtrer les mots qui ne sont pas déjà dans la base
                to_add = new_df[~new_df["Néerlandais"].isin(existing_nl)].copy()
                
                if st.button(f"Importer {len(to_add)} nouveaux mots"):
                    for col in COLUMNS:
                        if col not in to_add.columns:
                            to_add[col] = DEFAULTS.get(col, "")
                            if col == "ID":
                                # Attribuer des IDs uniques
                                start_id = next_id(df)
                                to_add["ID"] = range(start_id, start_id + len(to_add))
                            elif col == "NextReview":
                                to_add["NextReview"] = str(date.today())
                    
                    df = pd.concat([df, to_add[COLUMNS]], ignore_index=True)
                    save_data(df)
                    st.success(f"{len(to_add)} mots importés avec succès ! Rafraîchis la page.")
            else:
                st.error("Le CSV doit contenir au moins les colonnes 'Néerlandais' et 'Français'.")

# --- PAGE 3 : EXERCICES ---
elif menu == "🏋️ Exercices":
    st.title("Entraînement actif")

    if df.empty:
        st.warning("Ajoute d'abord des mots dans ta bibliothèque pour t'entraîner !")
    else:
        # Configuration de la session (Objectif)
        col_goal, col_prog = st.columns([1, 2])
        with col_goal:
            st.session_state.session_goal = st.number_input("Objectif de la session (mots) :", min_value=5, max_value=200, value=st.session_state.session_goal, step=5)
        with col_prog:
            progress = min(st.session_state.words_done_today / st.session_state.session_goal, 1.0)
            st.progress(progress, text=f"Progression : {st.session_state.words_done_today} / {st.session_state.session_goal} mots")

        if st.session_state.words_done_today >= st.session_state.session_goal:
            st.success("🎉 Objectif atteint ! Excellente session.")
            st.balloons()
            if st.button("Continuer l'entraînement (+10 mots)"):
                st.session_state.session_goal += 10
                st.rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                mode = st.radio("Mode", ["✍️ Texte libre", "🔘 QCM"], horizontal=True, key="ex_mode")
            with col_b:
                direction = st.radio("Sens", ["FR → NL", "NL → FR", "Aléatoire"], horizontal=True, key="ex_direction")

            mode_key = "qcm" if "QCM" in mode else "texte"
            if mode_key == "qcm" and len(df) < 4:
                st.info("Il faut au moins 4 mots pour un QCM. Mode texte forcé.")
                mode_key = "texte"

            if "current_word" not in st.session_state:
                new_question(df, direction)

            word = st.session_state.current_word
            d = st.session_state.current_direction

            if d == "fr_to_nl":
                prompt_col, correct_col, answer_lang, flag = "Français", "Néerlandais", "néerlandais", "🇫🇷"
            else:
                prompt_col, correct_col, answer_lang, flag = "Néerlandais", "Français", "français", "🇳🇱"

            if not st.session_state.get("was_due", True):
                st.caption("Aucun mot à réviser aujourd'hui — entraînement libre 🎯")

            # TOP BAR AVEC LES BOUTONS
            top_row = st.columns([5, 2, 2])
            with top_row[0]:
                st.subheader(f"{flag} {word[prompt_col]}")
                prompt_lang = "nl" if d == "nl_to_fr" else "fr"
                speak_button(word[prompt_col], lang=prompt_lang, label="🔊 Écouter", key=f"speak_prompt_{word['ID']}")
            with top_row[1]:
                if st.button("⏭️ Passer"):
                    new_question(df, direction)
                    st.rerun()
            with top_row[2]:
                if st.button("✅ Déjà connu", help="Repousse la révision de ce mot à dans 10 ans."):
                    updated_df = mark_as_known(df, word["ID"])
                    st.session_state.words_done_today += 1
                    new_question(updated_df, direction)
                    st.rerun()

            st.write("---")

            # ZONE DE QUESTION
            if not st.session_state.answered:
                if mode_key == "texte":
                    with st.form("quiz_form_texte"):
                        answer = st.text_input(f"Ta réponse en {answer_lang} :")
                        check = st.form_submit_button("Vérifier")
                        if check:
                            st.session_state.answered = True
                            st.session_state.is_correct = answer.strip().lower() == str(word[correct_col]).strip().lower()
                            st.rerun()
                else:
                    if st.session_state.qcm_choices is None:
                        choices, correct_answer = generate_choices(df, word, d)
                        st.session_state.qcm_choices = choices
                        st.session_state.qcm_correct = correct_answer
                    with st.form("quiz_form_qcm"):
                        choice = st.radio(f"Choisis la traduction en {answer_lang} :", st.session_state.qcm_choices)
                        check = st.form_submit_button("Vérifier")
                        if check:
                            st.session_state.answered = True
                            st.session_state.is_correct = (choice == st.session_state.qcm_correct)
                            st.rerun()
            else:
                # ZONE DE RÉSULTAT
                if st.session_state.is_correct:
                    st.success("Correct ! 🎉")
                else:
                    st.error(f"Faux. La bonne réponse était : **{word[correct_col]}**")

                if str(word.get("Contexte", "")).strip():
                    st.info(f"Contexte : {word['Contexte']}")

                if correct_col == "Néerlandais":
                    speak_button(word["Néerlandais"], lang="nl", label="🔊 Écouter la prononciation", key=f"speak_answer_{word['ID']}")

                if st.session_state.is_correct:
                    if mode_key == "texte":
                        st.write("Cette réponse t'a semblé...")
                        c1, c2, c3 = st.columns(3)
                        if c1.button("😓 Difficile"):
                            updated_df = update_word(df, word["ID"], 3)
                            st.session_state.words_done_today += 1
                            new_question(updated_df, direction)
                            st.rerun()
                        if c2.button("🙂 Bien"):
                            updated_df = update_word(df, word["ID"], 4)
                            st.session_state.words_done_today += 1
                            new_question(updated_df, direction)
                            st.rerun()
                        if c3.button("😎 Facile"):
                            updated_df = update_word(df, word["ID"], 5)
                            st.session_state.words_done_today += 1
                            new_question(updated_df, direction)
                            st.rerun()
                    else:
                        if st.button("Mot suivant ➔"):
                            updated_df = update_word(df, word["ID"], 4)
                            st.session_state.words_done_today += 1
                            new_question(updated_df, direction)
                            st.rerun()
                else:
                    if st.button("Continuer ➔"):
                        updated_df = update_word(df, word["ID"], 1)
                        st.session_state.words_done_today += 1
                        new_question(updated_df, direction)
                        st.rerun()

# --- PAGE 4 : STATISTIQUES ---
elif menu == "📊 Statistiques":
    st.title("Statistiques d'apprentissage")

    if df.empty:
        st.info("Pas encore de statistiques, ajoute des mots pour commencer !")
    else:
        today_str = str(date.today())
        total = len(df)
        due_today = len(df[df["NextReview"] <= today_str])
        mastered = len(df[df["Repetitions"] >= 5])
        total_reviews = int(df["TotalReviews"].sum())
        correct_reviews = int(df["CorrectReviews"].sum())
        success_rate = (correct_reviews / total_reviews * 100) if total_reviews > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mots au total", total)
        c2.metric("À réviser aujourd'hui", due_today)
        c3.metric("Mots maîtrisés", mastered)
        c4.metric("Taux de réussite", f"{success_rate:.0f} %")

        st.subheader("Répartition par catégorie")
        st.bar_chart(df["Catégorie"].value_counts())

        st.subheader("Révisions prévues (7 prochains jours)")
        upcoming = []
        for i in range(7):
            d_ = date.today() + timedelta(days=i)
            count = len(df[df["NextReview"] == str(d_)])
            upcoming.append({"Date": d_.strftime("%d/%m"), "Mots": count})
        upcoming_df = pd.DataFrame(upcoming).set_index("Date")
        st.bar_chart(upcoming_df)

        difficult = df[(df["TotalReviews"] >= 2) & (df["EaseFactor"] < 2.0)]
        if not difficult.empty:
            st.subheader("⚠️ Mots à travailler en priorité")
            st.dataframe(
                difficult[["Néerlandais", "Français", "Catégorie", "EaseFactor", "TotalReviews"]],
                use_container_width=True,
                hide_index=True,
            )