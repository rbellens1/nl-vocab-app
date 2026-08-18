import streamlit as st
import pandas as pd
import os
import random
from io import BytesIO
from datetime import date, timedelta
from gtts import gTTS

# ============================================================
# AUDIO (Text-to-Speech serveur via gTTS — plus fiable que la
# synthèse vocale du navigateur, qui est bridée dans l'iframe de
# Streamlit Cloud. Nécessite une connexion internet.)
# ============================================================

@st.cache_data(show_spinner=False)
def generate_audio(text, lang):
    tts = gTTS(text=text, lang=lang)
    buf = BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def speak_button(text, lang="nl", label="🔊 Écouter", key=None):
    """Bouton qui génère et joue la prononciation du mot (voix Google)."""
    text = "" if text is None else str(text)
    if not text.strip():
        return
    if st.button(label, key=key):
        try:
            with st.spinner("Génération de l'audio..."):
                audio_bytes = generate_audio(text, lang)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception:
            st.warning("Audio indisponible pour le moment (connexion internet requise).")

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
    "EaseFactor": 2.5,
    "Interval": 0,
    "Repetitions": 0,
    "LastReview": "",
    "TotalReviews": 0,
    "CorrectReviews": 0,
}


TEXT_COLUMNS = ["Néerlandais", "Français", "Contexte", "Catégorie", "NextReview", "LastReview"]
INT_COLUMNS = ["ID", "Interval", "Repetitions", "TotalReviews", "CorrectReviews"]


def load_data():
    """Charge le CSV et migre automatiquement les anciennes sauvegardes
    (celles qui n'avaient qu'une colonne 'Score') vers le nouveau schéma
    compatible avec la répétition espacée.

    IMPORTANT : keep_default_na=False évite que pandas transforme les
    cases vides (ex: Contexte non renseigné) en NaN, ce qui faisait
    passer certaines colonnes en dtype float64 et provoquait un
    TypeError dès qu'on y réécrivait une date ou un texte."""
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

    # Types forcés explicitement : évite tout souci de dtype (float64 vs
    # str/int) lors des mises à jour ultérieures, quelle que soit la
    # version de pandas.
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
# RÉPÉTITION ESPACÉE (algorithme SM-2, façon Anki)
# ============================================================

def sm2_update(df, idx, quality):
    """quality va de 0 (échec total) à 5 (parfait, sans effort).
    0-2 -> réinitialise le mot. 3-5 -> espace la prochaine révision.
    Met à jour la ligne idx cellule par cellule (df.at) plutôt que par
    réaffectation de ligne entière, pour rester compatible avec les
    dtypes stricts de pandas récent."""
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


def pick_word(df):
    """Priorise les mots dus (en retard en premier). S'il n'y en a aucun,
    propose un entraînement libre sur un mot au hasard."""
    today_str = str(date.today())
    due = df[df["NextReview"] <= today_str]
    if not due.empty:
        due_sorted = due.sort_values("NextReview")
        top = due_sorted.head(max(3, len(due_sorted) // 2))
        return top.sample(1).iloc[0], True
    return df.sample(1).iloc[0], False


def generate_choices(df, correct_row, direction):
    """Génère les propositions du QCM, en piochant si possible des
    distracteurs de la même catégorie (plus pertinent pédagogiquement)."""
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

st.set_page_config(page_title="Mon Vocabulaire Néerlandais", page_icon="🇳🇱", layout="wide")

st.sidebar.title("Navigation")
if not df.empty:
    due_count = len(df[df["NextReview"] <= str(date.today())])
    st.sidebar.caption(f"📅 {due_count} mot(s) à réviser aujourd'hui")
menu = st.sidebar.radio("Aller à", ["📚 Bibliothèque", "➕ Ajouter un mot", "🏋️ Exercices", "📊 Statistiques"])

# --- PAGE 1 : LA BIBLIOTHÈQUE ---
if menu == "📚 Bibliothèque":
    st.title("Ma Bibliothèque de Vocabulaire")

    if df.empty:
        st.info("Ta bibliothèque est vide. Ajoute des mots via le menu !")
    else:
        categories = ["Toutes"] + sorted(df["Catégorie"].dropna().unique().tolist())
        filtre = st.selectbox("Filtrer par catégorie", categories)

        shown = df if filtre == "Toutes" else df[df["Catégorie"] == filtre]
        shown = shown.sort_values("NextReview")

        display = shown.rename(columns={"Repetitions": "Niveau", "NextReview": "Prochaine révision"})[
            ["Néerlandais", "Français", "Contexte", "Catégorie", "Niveau", "Prochaine révision"]
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)

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
                "ID": [next_id(df)],
                "Néerlandais": [nl_word],
                "Français": [fr_word],
                "Contexte": [context],
                "Catégorie": [cat],
                "EaseFactor": [2.5],
                "Interval": [0],
                "Repetitions": [0],
                "NextReview": [str(date.today())],
                "LastReview": [""],
                "TotalReviews": [0],
                "CorrectReviews": [0],
            })
            df = pd.concat([df, new_row], ignore_index=True)
            save_data(df)
            st.success(f"Le mot '{nl_word}' a été ajouté ! Il apparaîtra dès ta prochaine session d'exercices.")

# --- PAGE 3 : EXERCICES ---
elif menu == "🏋️ Exercices":
    st.title("Entraînement actif")

    if df.empty:
        st.warning("Ajoute d'abord des mots dans ta bibliothèque pour t'entraîner !")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            mode = st.radio("Mode", ["✍️ Texte libre", "🔘 QCM"], horizontal=True, key="ex_mode")
        with col_b:
            direction = st.radio("Sens", ["FR → NL", "NL → FR", "Aléatoire"], horizontal=True, key="ex_direction")

        mode_key = "qcm" if "QCM" in mode else "texte"
        if mode_key == "qcm" and len(df) < 4:
            st.info("Il te faut au moins 4 mots dans ta bibliothèque pour un QCM pertinent. Mode texte libre utilisé à la place.")
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

        top_row = st.columns([4, 1])
        with top_row[0]:
            st.subheader(f"{flag} {word[prompt_col]}")
            prompt_lang = "nl" if d == "nl_to_fr" else "fr"
            speak_button(word[prompt_col], lang=prompt_lang, label="🔊 Écouter le mot", key=f"speak_prompt_{word['ID']}")
        with top_row[1]:
            if st.button("⏭️ Passer"):
                new_question(df, direction)
                st.rerun()

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
            if st.session_state.is_correct:
                st.success("Correct ! 🎉")
            else:
                st.error(f"Faux. La bonne réponse était : **{word[correct_col]}**")

            if str(word.get("Contexte", "")).strip():
                st.info(f"Contexte : {word['Contexte']}")

            if correct_col == "Néerlandais":
                # Le néerlandais était la réponse à deviner : on propose d'en entendre la prononciation.
                speak_button(word["Néerlandais"], lang="nl", label="🔊 Écouter la prononciation", key=f"speak_answer_{word['ID']}")

            if st.session_state.is_correct:
                if mode_key == "texte":
                    st.write("Cette réponse t'a semblé...")
                    c1, c2, c3 = st.columns(3)
                    if c1.button("😓 Difficile"):
                        updated_df = update_word(df, word["ID"], 3)
                        new_question(updated_df, direction)
                        st.rerun()
                    if c2.button("🙂 Bien"):
                        updated_df = update_word(df, word["ID"], 4)
                        new_question(updated_df, direction)
                        st.rerun()
                    if c3.button("😎 Facile"):
                        updated_df = update_word(df, word["ID"], 5)
                        new_question(updated_df, direction)
                        st.rerun()
                else:
                    # En QCM (reconnaissance, pas rappel actif) on note directement
                    if st.button("Mot suivant ➔"):
                        updated_df = update_word(df, word["ID"], 4)
                        new_question(updated_df, direction)
                        st.rerun()
            else:
                if st.button("Continuer ➔"):
                    updated_df = update_word(df, word["ID"], 1)
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