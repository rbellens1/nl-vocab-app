import streamlit as st
import pandas as pd
import os
import random
from io import BytesIO
from datetime import date, timedelta
from gtts import gTTS
import plotly.express as px
import plotly.graph_objects as go
import re

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

def get_frequency_rank(categorie):
    """
    Extrait le rang de fréquence approximatif à partir du nom de catégorie
    (ex : 'Top 2001-3000 - Base' -> 2001). Renvoie None si aucun chiffre n'est
    trouvé (catégories ajoutées manuellement comme 'Finance & Économie').
    """
    match = re.search(r'\d+', str(categorie))
    return int(match.group()) if match else None


def get_frequency_band_label(rank):
    """Regroupe un rang de fréquence en tranche lisible, pour les statistiques."""
    if rank is None:
        return "Sans fréquence (ajout manuel)"
    if rank <= 1000:
        return "Top 1-1000 (très fréquent)"
    elif rank <= 2000:
        return "Top 1001-2000"
    elif rank <= 3000:
        return "Top 2001-3000"
    elif rank <= 4000:
        return "Top 3001-4000"
    else:
        return "Top 4001-5000+ (rare)"


def compute_priority_weight(row, today, rarity_bias=1.0, difficulty_bias=1.0):
    """
    Calcule un poids de sélection pour un mot : plus il est élevé, plus le mot
    a de chances d'être tiré au sort pendant l'exercice. Combine trois signaux :

    1. L'ancienneté du retard (SM-2) : un mot en retard depuis longtemps remonte plus vite.
    2. La rareté du mot (rarity_bias) : favorise les mots à fréquence plus faible
       (frequentieklasse élevée dans "Catégorie"), pour t'entraîner sur du vocabulaire
       moins courant plutôt que de revoir sans cesse les mots les plus fréquents.
    3. La difficulté perçue (difficulty_bias) : favorise les mots que tu maîtrises mal
       (faible taux de réussite, EaseFactor bas), pour cibler tes lacunes.
    """
    weight = 1.0

    # 1. Ancienneté de la révision
    try:
        next_review = date.fromisoformat(str(row["NextReview"]))
        overdue_days = max(0, (today - next_review).days)
    except (ValueError, TypeError):
        overdue_days = 0
    weight *= 1 + min(overdue_days, 30) * 0.05  # jusqu'à +150 % si très en retard

    # 2. Rareté du mot (frequentieklasse) : plus le rang est élevé, plus le mot est rare
    if rarity_bias > 0:
        rank = get_frequency_rank(row.get("Catégorie", ""))
        if rank is not None:
            normalized_rank = min(rank, 5000) / 5000  # 0 (très fréquent) → 1 (rare)
            weight *= 1 + rarity_bias * normalized_rank * 3

    # 3. Difficulté perçue : taux de réussite bas et/ou EaseFactor bas = priorité plus haute
    if difficulty_bias > 0:
        total = int(row.get("TotalReviews", 0))
        correct = int(row.get("CorrectReviews", 0))
        ease = float(row.get("EaseFactor", 2.5))
        if total > 0:
            success_rate = correct / total
            weight *= 1 + difficulty_bias * (1 - success_rate) * 2
            if ease < 2.0:
                weight *= 1 + difficulty_bias * (2.0 - ease)

    return weight


def pick_word(df, rarity_bias=1.0, difficulty_bias=1.0, exclude_id=None):
    """
    Sélection intelligente et pondérée du prochain mot à réviser.

    Parmi les mots dus (ou toute la base s'il n'y en a pas), chaque mot reçoit un
    poids (voir compute_priority_weight) puis un tirage aléatoire pondéré est
    effectué : les mots en retard, rares et/ou mal maîtrisés ont statistiquement
    plus de chances d'apparaître, sans jamais totalement exclure les autres.
    """
    today = date.today()
    today_str = str(today)
    due = df[df["NextReview"] <= today_str]
    pool = due if not due.empty else df
    was_due = not due.empty

    # Évite de reposer immédiatement le même mot deux fois de suite
    if exclude_id is not None and len(pool) > 1:
        filtered = pool[pool["ID"] != exclude_id]
        if not filtered.empty:
            pool = filtered

    weights = pool.apply(lambda r: compute_priority_weight(r, today, rarity_bias, difficulty_bias), axis=1)
    if weights.sum() <= 0:
        chosen = pool.sample(1).iloc[0]
    else:
        chosen = pool.sample(1, weights=weights).iloc[0]
    return chosen, was_due

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
    rarity_bias = st.session_state.get("rarity_bias", 1.5)
    difficulty_bias = st.session_state.get("difficulty_bias", 1.0)
    exclude_id = st.session_state.current_word["ID"] if "current_word" in st.session_state else None

    row, was_due = pick_word(df, rarity_bias=rarity_bias, difficulty_bias=difficulty_bias, exclude_id=exclude_id)
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

def get_category_color(categorie):
    # Réutilise le même parsing de rang que le reste de l'algorithme intelligent
    num = get_frequency_rank(categorie)

    if num is None:
        # Si aucun chiffre n'est trouvé (ex: catégories manuelles "Finance", "Juridique")
        return "inherit"
    if num <= 1000:
        return "#10B981"  # Vert (1 à 1000)
    elif num <= 2000:
        return "#3B82F6"  # Bleu (1001 à 2000)
    elif num <= 3000:
        return "#F97316"  # Orange (2001 à 3000)
    elif num <= 4000:
        return "#EF4444"  # Rouge (3001 à 4000)
    else:
        return "#8B5CF6"  # Violet (4001 et +)

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

            # --- SÉLECTION INTELLIGENTE DES MOTS ---
            with st.expander("⚙️ Sélection intelligente des mots", expanded=False):
                col_r, col_d = st.columns(2)
                with col_r:
                    st.session_state.rarity_bias = st.slider(
                        "🎯 Prioriser les mots rares", 0.0, 3.0,
                        st.session_state.get("rarity_bias", 1.5), 0.1,
                        help="0 = tous les mots ont autant de chances d'apparaître. Plus la "
                             "valeur est élevée, plus les mots peu fréquents (frequentieklasse "
                             "haute, ex. Top 4001-5000) reviennent souvent — pour élargir ton "
                             "vocabulaire au-delà des mots les plus courants."
                    )
                with col_d:
                    st.session_state.difficulty_bias = st.slider(
                        "🔥 Prioriser les mots difficiles", 0.0, 3.0,
                        st.session_state.get("difficulty_bias", 1.0), 0.1,
                        help="Fait revenir plus souvent les mots sur lesquels tu échoues le "
                             "plus (faible taux de réussite ou EaseFactor bas)."
                    )
                freq_options = ["Toutes les catégories"] + sorted(df["Catégorie"].dropna().unique().tolist())
                freq_filter = st.selectbox(
                    "📂 Se concentrer sur une seule catégorie / tranche (optionnel)", freq_options
                )

            def get_pool(base_df):
                """Sous-ensemble de mots utilisé pour le tirage, selon le filtre choisi.
                Ne sert jamais à sauvegarder : les mises à jour SM-2 s'appliquent
                toujours sur la base complète (voir update_word / mark_as_known)."""
                if freq_filter == "Toutes les catégories":
                    return base_df
                filtered = base_df[base_df["Catégorie"] == freq_filter]
                return filtered if not filtered.empty else base_df

            pool_df = get_pool(df)

            mode_key = "qcm" if "QCM" in mode else "texte"
            if mode_key == "qcm" and len(pool_df) < 4:
                st.info("Il faut au moins 4 mots (dans cette catégorie) pour un QCM. Mode texte forcé.")
                mode_key = "texte"

            if "current_word" not in st.session_state:
                new_question(pool_df, direction)

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
                # Nouvel affichage avec code couleur :
                word_color = get_category_color(word["Catégorie"])
                st.markdown(
                            f"<h3 style='color: {word_color}; margin-bottom: 0px;'>{flag} {word[prompt_col]}</h3>", 
                 unsafe_allow_html=True
                            )
                st.caption(f"Catégorie : {word['Catégorie']}") # Optionnel : affiche la tranche sous le mot
                prompt_lang = "nl" if d == "nl_to_fr" else "fr"
                speak_button(word[prompt_col], lang=prompt_lang, label="🔊 Écouter", key=f"speak_prompt_{word['ID']}")
            with top_row[1]:
                if st.button("⏭️ Passer"):
                    new_question(get_pool(df), direction)
                    st.rerun()
            with top_row[2]:
                if st.button("✅ Déjà connu", help="Repousse la révision de ce mot à dans 10 ans."):
                    updated_df = mark_as_known(df, word["ID"])
                    st.session_state.words_done_today += 1
                    new_question(get_pool(updated_df), direction)
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
                            new_question(get_pool(updated_df), direction)
                            st.rerun()
                        if c2.button("🙂 Bien"):
                            updated_df = update_word(df, word["ID"], 4)
                            st.session_state.words_done_today += 1
                            new_question(get_pool(updated_df), direction)
                            st.rerun()
                        if c3.button("😎 Facile"):
                            updated_df = update_word(df, word["ID"], 5)
                            st.session_state.words_done_today += 1
                            new_question(get_pool(updated_df), direction)
                            st.rerun()
                    else:
                        if st.button("Mot suivant ➔"):
                            updated_df = update_word(df, word["ID"], 4)
                            st.session_state.words_done_today += 1
                            new_question(get_pool(updated_df), direction)
                            st.rerun()
                else:
                    if st.button("Continuer ➔"):
                        updated_df = update_word(df, word["ID"], 1)
                        st.session_state.words_done_today += 1
                        new_question(get_pool(updated_df), direction)
                        st.rerun()

# --- PAGE 4 : STATISTIQUES ---
elif menu == "📊 Statistiques":
    st.title("📊 Tableau de Bord d'Apprentissage")

    if df.empty:
        st.info("Pas encore de statistiques, ajoute des mots pour commencer !")
    else:
        # --- CALCUL DES DONNÉES ---
        today_str = str(date.today())
        total = len(df)
        due_today = len(df[df["NextReview"] <= today_str])
        
        nouveaux = len(df[df["TotalReviews"] == 0])
        maitrises = len(df[df["Repetitions"] >= 5])
        en_cours = total - nouveaux - maitrises
        
        total_reviews = int(df["TotalReviews"].sum())
        correct_reviews = int(df["CorrectReviews"].sum())
        success_rate = (correct_reviews / total_reviews * 100) if total_reviews > 0 else 0

        # --- LIGNE 1 : METRICS GLOBAUX ---
        st.write("### Vue d'ensemble")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📚 Mots dans la base", total)
        c2.metric("🔥 À réviser aujourd'hui", due_today)
        c3.metric("🧠 Mots maîtrisés", maitrises)
        c4.metric("🎯 Taux de précision", f"{success_rate:.0f} %")

        st.divider()

        # --- LIGNE 2 : GRAPHIQUES CIRCULAIRES ---
        col_donut, col_gauge = st.columns(2)

        with col_donut:
            st.write("#### Progression du vocabulaire")
            # Graphique Donut pour la répartition des mots
            labels = ['Nouveaux', "En cours d'apprentissage", 'Maîtrisés']
            values = [nouveaux, en_cours, maitrises]
            colors = ['#E2E8F0', '#F97316', '#10B981'] # Gris, Orange (Néerlandais!), Vert

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.6, 
                marker_colors=colors,
                textinfo='percent',
                hoverinfo='label+value'
            )])
            fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
            
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_gauge:
            st.write("#### Taux de réussite global")
            # Jauge pour le taux de succès
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=success_rate,
                number={'suffix': "%", 'font': {'size': 50}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#3B82F6"}, # Bleu
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': "#FEE2E2"},   # Rouge clair
                        {'range': [50, 80], 'color': "#FEF3C7"},  # Jaune clair
                        {'range': [80, 100], 'color': "#D1FAE5"}  # Vert clair
                    ],
                }
            ))
            fig_gauge.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # --- LIGNE 3 : CHARGE DE TRAVAIL FUTURE (BAR CHART MODERNE) ---
        st.write("#### 📅 Prévisions des 7 prochains jours")
        upcoming = []
        for i in range(7):
            d_ = date.today() + timedelta(days=i)
            count = len(df[df["NextReview"] == str(d_)])
            # Formater la date en jj/mm
            upcoming.append({"Date": d_.strftime("%d/%m"), "Mots": count})
        
        df_upcoming = pd.DataFrame(upcoming)
        
        # Joli graphique en barre Plotly
        fig_bar = px.bar(
            df_upcoming, 
            x='Date', 
            y='Mots', 
            text_auto=True,
            color='Mots',
            color_continuous_scale='Blues' # Dégradé de bleu selon l'intensité
        )
        fig_bar.update_layout(
            xaxis_title="", 
            yaxis_title="", 
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False # Cache la légende de couleur inutile
        )
        fig_bar.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # --- LIGNE 3.5 : PROGRESSION PAR TRANCHE DE FRÉQUENCE ---
        st.write("#### 📈 Progression par tranche de fréquence")
        st.caption(
            "Vérifie si tu progresses aussi sur le vocabulaire rare, et pas seulement "
            "sur les mots les plus courants."
        )

        df_bands = df.copy()
        df_bands["Rang"] = df_bands["Catégorie"].apply(get_frequency_rank)
        df_bands["Tranche"] = df_bands["Rang"].apply(get_frequency_band_label)

        band_order = [
            "Top 1-1000 (très fréquent)", "Top 1001-2000", "Top 2001-3000",
            "Top 3001-4000", "Top 4001-5000+ (rare)", "Sans fréquence (ajout manuel)",
        ]
        band_colors = {
            "Top 1-1000 (très fréquent)": "#10B981",
            "Top 1001-2000": "#3B82F6",
            "Top 2001-3000": "#F97316",
            "Top 3001-4000": "#EF4444",
            "Top 4001-5000+ (rare)": "#8B5CF6",
            "Sans fréquence (ajout manuel)": "#94A3B8",
        }

        band_stats = []
        for band in band_order:
            sub = df_bands[df_bands["Tranche"] == band]
            if sub.empty:
                continue
            band_total = len(sub)
            band_mastered = len(sub[sub["Repetitions"] >= 5])
            band_stats.append({
                "Tranche": band,
                "Mots": band_total,
                "Maîtrisés": band_mastered,
                "% Maîtrisé": round(band_mastered / band_total * 100, 1) if band_total else 0,
            })

        if band_stats:
            df_band_stats = pd.DataFrame(band_stats)
            fig_bands = px.bar(
                df_band_stats, x="Tranche", y="% Maîtrisé", text="% Maîtrisé",
                color="Tranche", color_discrete_map=band_colors,
            )
            fig_bands.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_bands.update_layout(
                showlegend=False, yaxis_range=[0, 100],
                xaxis_title="", margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_bands, use_container_width=True)
            st.dataframe(df_band_stats, use_container_width=True, hide_index=True)

        st.divider()

        # --- LIGNE 4 : LES MOTS DIFFICILES (DANS UN MENU DÉROULANT POUR NE PAS POLLUER) ---
        difficult = df[(df["TotalReviews"] >= 2) & (df["EaseFactor"] < 2.0)]
        if not difficult.empty:
            with st.expander(f"⚠️ Afficher les mots à retravailler en priorité ({len(difficult)} mots)"):
                st.dataframe(
                    difficult[["Néerlandais", "Français", "Catégorie", "EaseFactor", "TotalReviews"]].sort_values("EaseFactor"),
                    use_container_width=True,
                    hide_index=True,
                )