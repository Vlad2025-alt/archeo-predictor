import streamlit as st
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

# --- Data Utilities ---
def encode_day(day):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days.index(day)

def build_dataset(games, window=2):
    X, y = [], []
    for game in games:
        results = game["results"]
        day = encode_day(game["day_of_week"])
        hour = game["hour"]
        for i in range(window, 10):
            features = results[i-window:i] + [day, hour]
            X.append(features)
            y.append(results[i])
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
        results = game["results"]
        for i in range(order, 10):
            key = tuple(results[i-order:i])
            tables[i][key][results[i]] += 1
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
        for i, x in enumerate(game["results"]):
            if x == 1:
                left[i] += 1
            else:
                right[i] += 1
    return left, right

def stat_predict(left, right):
    return [1 if left[i] > right[i] else 0 for i in range(10)]

def predict_next(game_so_far, rf, markov_tables, stats, day, hour, window=2):
    markov_preds = markov_predict(game_so_far, markov_tables, order=window)
    left, right = stats
    stat_preds = stat_predict(left, right)
    X_pred = []
    for i in range(window, 10):
        features = game_so_far[i-window:i] + [encode_day(day), hour]
        X_pred.append(features)
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
st.set_page_config(page_title="Archeo Predictor Mobile", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 700px;
    }
    /* Make columns stack on mobile */
    @media (max-width: 800px) {
        .stHorizontalBlock > div {
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }
    }
    /* Make button text larger and paddings more touch friendly */
    .stButton > button {
        font-size: 1.2rem;
        padding: 1rem 0.5rem;
        min-width: 60px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.title("🧠 Archeo Mobile Game Predictor")

with st.expander("ℹ️ How to use"):
    st.markdown(
        """
        - Tap **Left** or **Right** on each round to record your pick.
        - **Day of week** and **hour** are used to train the predictor!
        - Tap **Save Game & Retrain** to add your data.
        - Use **Refresh** only if you want to clear all choices.
        """
    )

# --- Session State Initialization ---
if "games" not in st.session_state:
    raw_games = [
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
    st.session_state.games = [
        {
            "results": g,
            "day_of_week": "Monday",
            "hour": 12
        } for g in raw_games
    ]
if "current_game" not in st.session_state:
    st.session_state.current_game = [None]*10

if "game_day" not in st.session_state:
    st.session_state.game_day = datetime.datetime.now().strftime("%A")
if "game_hour" not in st.session_state:
    st.session_state.game_hour = datetime.datetime.now().hour

st.session_state.game_day = st.selectbox(
    "Day of the week:",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(st.session_state.game_day),
    key="day_select"
)
st.session_state.game_hour = st.slider("Hour of day (0-23):", 0, 23, st.session_state.game_hour, key="hour_slider")

window = 2
games = st.session_state.games
rf = train_rf(games, window)
markov_tables = build_markov(games, window)
stats = get_stats(games)
game_so_far = [x if x is not None else 0 for x in st.session_state.current_game]
predictions = predict_next(game_so_far, rf, markov_tables, stats, st.session_state.game_day, st.session_state.game_hour, window)

st.subheader("🎯 Your Game (Tap to select Left/Right per round)")

progress = sum(x is not None for x in st.session_state.current_game) / 10
st.progress(progress)

# --- Mobile-Friendly Horizontal 10 Rounds ---
round_cols = st.columns(10)
for i, col in enumerate(round_cols):
    with col:
        for side, label in enumerate(["Left", "Right"]):
            is_pred = predictions[i] == side
            is_selected = st.session_state.current_game[i] == side
            box_color = "#98FB98" if is_pred else "#f0f0f0"
            border = "3px solid #2e8b57" if is_selected else "1px solid #ccc"
            if st.button(label, key=f"{label}_{i}"):
                st.session_state.current_game[i] = side
            st.markdown(
                f"<div style='background:{box_color};border:{border};padding:1.2em 0;text-align:center;border-radius:10px;font-size:1.2em;font-weight:bold;margin-bottom:4px;min-width:45px;max-width:80px'>{label} {'✔️' if is_selected else ''}</div>",
                unsafe_allow_html=True
            )
        st.markdown(f"<div style='text-align:center; color: #888;'>R{i+1}</div>", unsafe_allow_html=True)

# --- Action Buttons ---
disable_refresh = all(x is None for x in st.session_state.current_game)
colA, colB = st.columns([1, 1])
with colA:
    if st.button("🔄 Refresh (New Game)", disabled=disable_refresh):
        st.session_state.current_game = [None] * 10
        st.session_state.game_day = datetime.datetime.now().strftime("%A")
        st.session_state.game_hour = datetime.datetime.now().hour
        st.experimental_rerun()
with colB:
    if st.button("✅ Save Game & Retrain"):
        if None not in st.session_state.current_game:
            st.session_state.games.append({
                "results": st.session_state.current_game.copy(),
                "day_of_week": st.session_state.game_day,
                "hour": st.session_state.game_hour
            })
            st.session_state.current_game = [None] * 10
            st.success("Game saved and model updated!")
            st.experimental_rerun()
        else:
            st.warning("Complete all rounds before saving.")

with st.expander("📊 Model Stats"):
    total = len(st.session_state.games)
    st.write(f"Number of games learned: **{total}**")
    left, right = get_stats(st.session_state.games)
    st.write("Most common picks (per round):")
    st.write({f"Round {i+1}": "Left" if left[i] > right[i] else "Right" for i in range(10)})

st.caption("All 10 rounds fit in one row. Tap to select, works perfectly on mobile!")
