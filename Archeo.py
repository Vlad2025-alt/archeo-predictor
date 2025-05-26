import streamlit as st
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

# Prediction logic as before

def build_dataset(games, window=2):
    X, y = [], []
    for game in games:
        for i in range(window, 10):
            X.append(game[i-window:i])
            y.append(game[i])
    return np.array(X), np.array(y)

def train_rf(games, window=2):
    X, y = build_dataset(games, window)
    if len(X) == 0:
        return None
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

def build_markov(games, order=2):
    tables = [defaultdict(lambda: [0, 0]) for _ in range(10)]
    for game in games:
        for i in range(order, 10):
            key = tuple(game[i-order:i])
            tables[i][key][game[i]] += 1
    return tables

def markov_predict(game_so_far, tables, order=2):
    preds = []
    for i in range(10):
        if i < order:
            preds.append(None)
            continue
        key = tuple(game_so_far[i-order:i])
        counts = tables[i][key]
        if sum(counts) == 0:
            preds.append(None)
        else:
            preds.append(int(counts[1] > counts[0]))
    return preds

def get_stats(games):
    left = np.zeros(10)
    right = np.zeros(10)
    for game in games:
        for i, x in enumerate(game):
            if x == 1:
                left[i] += 1
            else:
                right[i] += 1
    return left, right

def stat_predict(left, right):
    return [1 if left[i] > right[i] else 0 for i in range(10)]

def predict_next(game_so_far, rf, markov_tables, stats, window=2):
    markov_preds = markov_predict(game_so_far, markov_tables, order=window)
    left, right = stats
    stat_preds = stat_predict(left, right)
    X_pred = []
    for i in range(window, 10):
        X_pred.append(game_so_far[i-window:i])
    rf_preds = rf.predict(X_pred) if X_pred and rf is not None else []

    preds = []
    for i in range(10):
        votes = []
        if i >= window and rf is not None:
            votes.append(rf_preds[i - window])
        if markov_preds[i] is not None:
            votes.append(markov_preds[i])
        votes.append(stat_preds[i])
        pred = int(round(np.mean(votes)))
        preds.append(pred)
    return preds

# --- Streamlit UI ---

st.set_page_config(page_title="Archeo Predictor", layout="wide")
st.title("Archeo Game Predictor")

if "games" not in st.session_state:
    st.session_state.games = [
        [0,1,1,0,1,0,1,0,0,1],
        [1,0,1,0,0,1,0,1,0,1],
        [0,1,1,0,1,0,1,0,0,1],
        [0,1,0,1,0,0,0,0,0,0],
        [1,0,0,1,1,1,1,1,1,1],
        [1,0,0,1,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,1,0,0],
        [0,1,1,0,0,0,1,0,0,0],
        [0,1,0,0,1,1,0,0,0,1],
        [0,1,1,0,1,0,1,1,1,0],
        [0,1,1,0,0,1,0,1,0,0],
    ]
if "current_game" not in st.session_state or st.session_state.get("refresh_flag", False):
    st.session_state.current_game = [None]*10
    st.session_state.refresh_flag = False

window = 2
games = st.session_state.games
rf = train_rf(games, window)
markov_tables = build_markov(games, order=window)
stats = get_stats(games)
game_so_far = [x if x is not None else 0 for x in st.session_state.current_game]
pred = predict_next(game_so_far, rf, markov_tables, stats, window)

st.subheader("Game Prediction Table")

def manual_pick(idx, pick):
    st.session_state.current_game[idx] = pick

cols = st.columns([1,1,1])
cols[0].markdown("**Left**")
cols[1].markdown("**Right**")
cols[2].markdown("**Round**")

for i in range(10):
    left_color = "background-color: #98FB98;" if pred[i] == 1 else ""
    right_color = "background-color: #98FB98;" if pred[i] == 0 else ""
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("⬅️", key=f"left_{i}"):
        manual_pick(i, 1)
    c1.markdown(
        f"<div style='{left_color};padding:5px;border-radius:5px;text-align:center'>{'✔️' if st.session_state.current_game[i]==1 else 'Left'}</div>",
        unsafe_allow_html=True
    )
    if c2.button("➡️", key=f"right_{i}"):
        manual_pick(i, 0)
    c2.markdown(
        f"<div style='{right_color};padding:5px;border-radius:5px;text-align:center'>{'✔️' if st.session_state.current_game[i]==0 else 'Right'}</div>",
        unsafe_allow_html=True
    )
    c3.markdown(f"<div style='text-align:center'>Round {i+1}</div>", unsafe_allow_html=True)

if st.button("🔄 Refresh (New Game)"):
    st.session_state.current_game = [None] * 10
    st.session_state.refresh_flag = True
    st.experimental_rerun()

if st.button("✅ Save Game & Retrain"):
    if None not in st.session_state.current_game:
        st.session_state.games.append(st.session_state.current_game.copy())
        st.session_state.current_game = [None] * 10
        st.success("Game saved and model updated!")
        st.experimental_rerun()
    else:
        st.warning("Complete all rounds before saving.")

st.info("Green highlights show predicted picks for each round. Tap ⬅️ or ➡️ to enter the true result for each round, then 'Save Game & Retrain' to update the model. Use 'Refresh' for a new game.")
