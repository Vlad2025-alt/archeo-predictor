import streamlit as st
import numpy as np
import random
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import datetime

# --- Utility Functions ---

def encode_day(day):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days.index(day)

def get_day_choices():
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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

# --- Slot/Treasure Visualization ---
def draw_columns(winning_cols, NUM_COLS=20, NUM_ROWS=10):
    symbols = ["7", "BAR", "X", "O", "★", "♠", "♥", "♦", "♣", "FUN"]
    table_html = '<style>td{font-size:18px; text-align:center;} .winner{background-color:#90ee90;font-weight:bold;}</style>'
    table_html += '<table style="border-collapse:collapse;">'
    table_html += '<tr>' + ''.join([f'<th style="padding:4px 8px;border:1px solid #888;">{i+1}</th>' for i in range(NUM_COLS)]) + '</tr>'
    for row in range(NUM_ROWS):
        table_html += '<tr>'
        for col in range(NUM_COLS):
            symbol = random.choice(symbols)
            highlight = 'winner' if col == winning_cols[row] else ''
            table_html += f'<td class="{highlight}" style="border:1px solid #888; padding:4px 8px;">{symbol}</td>'
        table_html += '</tr>'
    table_html += '</table>'
    st.markdown(table_html, unsafe_allow_html=True)

# --- Streamlit UI ---

st.set_page_config(page_title="Archeo Predictor", layout="wide")
st.title("🧠 Archeo Game Predictor – Day & Time Enhanced")

with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    - For each round, click **Left** or **Right** to record your pick as you play.
    - Set the **day of the week** and **hour** when you played.
    - The **predicted pick** is highlighted in green.
    - Click **Save Game & Retrain** to add your results to the data.
    - Use **Refresh** to start a new game.
    - The slot visualization below shows the prediction for each round as a highlighted column.
    """)

# --- Session State Initialization ---

if "games" not in st.session_state:
    # If you have old data as lists, convert to dicts with default day/hour
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

# Default to now for new games
if "game_day" not in st.session_state:
    st.session_state.game_day = datetime.datetime.now().strftime("%A")
if "game_hour" not in st.session_state:
    st.session_state.game_hour = datetime.datetime.now().hour

# --- Day/Hour Selectors ---
st.session_state.game_day = st.selectbox(
    "Day of the week:",
    get_day_choices(),
    index=get_day_choices().index(st.session_state.game_day)
)
st.session_state.game_hour = st.slider("Hour of day (0-23):", 0, 23, st.session_state.game_hour)

window = 2
games = st.session_state.games
rf = train_rf(games, window)
markov_tables = build_markov(games, window)
stats = get_stats(games)
game_so_far = [x if x is not None else 0 for x in st.session_state.current_game]
predictions = predict_next(game_so_far, rf, markov_tables, stats, st.session_state.game_day, st.session_state.game_hour, window)

# --- Game Entry Boxes ---
st.subheader("🎯 Your Game: Click to Select Each Round")
progress = sum(x is not None for x in st.session_state.current_game) / 10
st.progress(progress)

for i in range(10):
    cols = st.columns(2)
    for side, label in enumerate(["Left", "Right"]):
        # Prediction highlight: green if model predicts this side
        is_pred = predictions[i] == side
        is_selected = st.session_state.current_game[i] == side
        box_color = "#98FB98" if is_pred else "#f0f0f0"
        border = "3px solid #2e8b57" if is_selected else "1px solid #ccc"
        with cols[side]:
            if st.button(label, key=f"{label}_{i}"):
                st.session_state.current_game[i] = side
            st.markdown(
                f"<div style='background:{box_color};border:{border};padding:18px 0;text-align:center;border-radius:10px;font-size:20px;font-weight:bold;margin-bottom:4px;'>{label} {'✔️' if is_selected else ''}</div>",
                unsafe_allow_html=True
            )
    st.markdown(f"<div style='text-align:center; color: #888;'>Round {i+1}</div>", unsafe_allow_html=True)

# --- Action Buttons ---
colA, colB = st.columns([1, 1])
with colA:
    if st.button("🔄 Refresh (New Game)"):
        st.session_state.current_game = [None] * 10
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

# --- Model Stats Expander ---
with st.expander("📊 Model Stats"):
    total = len(st.session_state.games)
    st.write(f"Number of games learned: **{total}**")
    left, right = get_stats(st.session_state.games)
    st.write("Most common picks (per round):")
    st.write({f"Round {i+1}": "Left" if left[i] > right[i] else "Right" for i in range(10)})

# --- Slot/Treasure Visualization ---
st.subheader("Treasure Columns Visualization")
winning_cols = [5 if p == 1 else 15 for p in predictions]
draw_columns(winning_cols)
st.info("The green column is the predicted/treasure column for each round.")

st.caption("Use this app while you play. Enter your choices, day, and hour for each session. For automatic data matching, the game must provide an API or export function. Otherwise, enter results manually as you play.")
