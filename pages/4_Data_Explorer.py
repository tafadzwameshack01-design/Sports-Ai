import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import init_db
from utils.features import FEATURE_NAMES, build_state

st.set_page_config(
    page_title="Data Explorer · NBA AI",
    page_icon="🔍",
    layout="wide",
)

if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    _a = DQNAgent()
    _a.load()
    st.session_state.agent = _a

for key, default in [
    ("espn_raw",   None),
    ("sr_raw",     None),
    ("sr_stand",   None),
    ("espn_bpi_raw", None),
    ("espn_teams_raw", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent = st.session_state.agent

st.markdown("## 🔍 Data Explorer")
st.caption("Inspect raw API payloads, feature vectors, and model Q-values")
st.divider()

tab_espn, tab_sr, tab_fv, tab_qvals, tab_news = st.tabs(
    ["📺 ESPN", "📡 SportRadar", "🧮 Feature Vectors", "🧠 Q-Values", "📰 News"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: ESPN Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_espn:
    st.subheader("ESPN Scoreboard")

    ec1, ec2 = st.columns([2, 1])
    with ec2:
        fetch_espn = st.button("🔄 Fetch ESPN Scoreboard", key="btn_espn", use_container_width=True)
    with ec1:
        espn_view = st.radio(
            "View", ["Table", "Raw JSON"], horizontal=True, key="espn_view"
        )

    if fetch_espn:
        with st.spinner("Fetching ESPN scoreboard…"):
            from utils.espn_api import fetch_espn_scoreboard, fetch_espn_standings, fetch_espn_power_index
            try:
                games = fetch_espn_scoreboard()
                standings = fetch_espn_standings()
                bpi = fetch_espn_power_index()
                st.session_state.espn_raw = {
                    "games":     games,
                    "standings": standings,
                    "bpi":       bpi,
                }
                st.toast(f"ESPN: {len(games)} games fetched", icon="📺")
            except Exception as exc:
                st.error(f"ESPN fetch error: {exc}")

    if st.session_state.espn_raw:
        raw = st.session_state.espn_raw
        games = raw.get("games", [])

        st.caption(f"**{len(games)} games** on today's ESPN scoreboard")

        if espn_view == "Table" and games:
            df_g = pd.DataFrame(games)
            display_cols = [c for c in [
                "away_team", "home_team", "away_score", "home_score",
                "status", "odds_spread", "source"
            ] if c in df_g.columns]
            st.dataframe(df_g[display_cols], hide_index=True, use_container_width=True)

            st.subheader("Standings (Win %)")
            df_s = pd.DataFrame(
                list(raw["standings"].items()), columns=["Team", "Win %"]
            ).sort_values("Win %", ascending=False)
            st.dataframe(df_s, hide_index=True, use_container_width=True)

            if raw["bpi"]:
                st.subheader("ESPN Power Index (BPI)")
                df_bpi = pd.DataFrame(
                    list(raw["bpi"].items()), columns=["Team", "BPI"]
                ).sort_values("BPI", ascending=False)
                st.bar_chart(df_bpi.set_index("Team")["BPI"], use_container_width=True)

        else:
            st.json(raw)
    else:
        st.info("Click **Fetch ESPN Scoreboard** to load data.")

    # ESPN Teams reference
    st.subheader("ESPN Team Reference")
    if st.button("Load All NBA Teams", key="btn_teams"):
        with st.spinner("Loading teams…"):
            from utils.espn_api import fetch_espn_teams
            try:
                teams = fetch_espn_teams()
                st.session_state.espn_teams_raw = teams
            except Exception as exc:
                st.error(f"Teams fetch error: {exc}")

    if st.session_state.espn_teams_raw:
        df_teams = pd.DataFrame(st.session_state.espn_teams_raw)
        st.dataframe(df_teams, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: SportRadar Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_sr:
    st.subheader("SportRadar Schedule")
    st.caption("Uses trial API key — rate-limited to ~1 req/s. Errors are expected on heavy usage.")

    src1, src2 = st.columns([2, 1])
    with src2:
        fetch_sr = st.button("🔄 Fetch SR Schedule", key="btn_sr", use_container_width=True)
    with src1:
        sr_view = st.radio("View", ["Table", "Raw JSON"], horizontal=True, key="sr_view")

    if fetch_sr:
        with st.spinner("Fetching SportRadar schedule…"):
            from utils.sportradar import fetch_sr_todays_schedule, fetch_sr_standings
            try:
                sr_sched    = fetch_sr_todays_schedule()
                sr_standings = fetch_sr_standings()
                st.session_state.sr_raw   = sr_sched
                st.session_state.sr_stand = sr_standings
                st.toast(f"SR: {len(sr_sched)} games fetched", icon="📡")
            except Exception as exc:
                st.error(f"SportRadar fetch error: {exc}")

    if st.session_state.sr_raw is not None:
        sched = st.session_state.sr_raw
        st.caption(f"**{len(sched)} games** from SportRadar today")
        if sr_view == "Table" and sched:
            df_sr = pd.DataFrame(sched)
            st.dataframe(df_sr, hide_index=True, use_container_width=True)
        else:
            st.json(sched)

        if st.session_state.sr_stand:
            st.subheader("SR Standings (Win %)")
            df_sr_s = pd.DataFrame(
                list(st.session_state.sr_stand.items()), columns=["Abbr", "Win %"]
            ).sort_values("Win %", ascending=False)
            st.dataframe(df_sr_s, hide_index=True, use_container_width=True)
    else:
        st.info("Click **Fetch SR Schedule** to load data.")

    st.subheader("Manual Boxscore Lookup")
    game_id_input = st.text_input(
        "Enter game ID (from SR schedule above):", key="ti_game_id",
        placeholder="e.g. 91a4c9e4-ade9-4d1c-…"
    )
    if st.button("Fetch Boxscore", key="btn_box") and game_id_input:
        with st.spinner(f"Fetching boxscore for {game_id_input}…"):
            from utils.sportradar import fetch_sr_boxscore
            try:
                box = fetch_sr_boxscore(game_id_input.strip())
                st.json(box)
            except Exception as exc:
                st.error(f"Boxscore error: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Feature Vectors
# ══════════════════════════════════════════════════════════════════════════════
with tab_fv:
    st.subheader("Feature Vector Inspector")
    st.caption(
        "Select any game from the ESPN scoreboard to inspect the 12-D normalized "
        "state vector fed into the DQN."
    )

    if not st.session_state.espn_raw or not st.session_state.espn_raw.get("games"):
        st.info("Fetch ESPN data (in the **📺 ESPN** tab) to inspect feature vectors.")
    else:
        games     = st.session_state.espn_raw["games"]
        standings = st.session_state.espn_raw.get("standings", {})
        bpi_data  = st.session_state.espn_raw.get("bpi", {})
        options   = [f"{g['away_team']} @ {g['home_team']}" for g in games]

        sel = st.selectbox("Select game", options, key="sel_game_fv")
        idx = options.index(sel)
        game = games[idx]

        with st.spinner("Building state vector…"):
            try:
                state = build_state(game, standings, bpi_data)
                st.success("Feature vector computed successfully")

                fv_col1, fv_col2 = st.columns([1, 2])
                with fv_col1:
                    df_fv = pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Raw (norm.)": [round(float(v), 4) for v in state],
                    })
                    st.dataframe(df_fv, hide_index=True, use_container_width=True)
                with fv_col2:
                    st.bar_chart(
                        pd.DataFrame(
                            {"value": state}, index=FEATURE_NAMES
                        ),
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"Feature vector error: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: Q-Values
# ══════════════════════════════════════════════════════════════════════════════
with tab_qvals:
    st.subheader("DQN Q-Value Inspector")
    st.caption(
        "Inspect raw Q-values, softmax probabilities, and confidence for any "
        "custom or live state vector."
    )

    input_mode = st.radio(
        "State source", ["Use live game", "Custom state"], horizontal=True, key="qv_mode"
    )

    state_for_qv: np.ndarray | None = None

    if input_mode == "Use live game":
        if not st.session_state.espn_raw or not st.session_state.espn_raw.get("games"):
            st.info("Fetch ESPN data first.")
        else:
            games_q   = st.session_state.espn_raw["games"]
            standings = st.session_state.espn_raw.get("standings", {})
            bpi_data  = st.session_state.espn_raw.get("bpi", {})
            opt_q     = [f"{g['away_team']} @ {g['home_team']}" for g in games_q]
            sel_q     = st.selectbox("Select game", opt_q, key="sel_qv_game")
            idx_q     = opt_q.index(sel_q)
            try:
                state_for_qv = build_state(games_q[idx_q], standings, bpi_data)
            except Exception as exc:
                st.error(f"Feature build error: {exc}")
    else:
        st.caption("Adjust each feature (0.0 = min, 1.0 = max):")
        custom_vals = []
        qv_col1, qv_col2 = st.columns(2)
        for i, name in enumerate(FEATURE_NAMES):
            col = qv_col1 if i < 6 else qv_col2
            with col:
                v = st.slider(name, 0.0, 1.0, 0.5, step=0.01, key=f"sl_qv_{name}")
                custom_vals.append(v)
        state_for_qv = np.array(custom_vals, dtype=np.float32)

    if state_for_qv is not None:
        import tensorflow as tf
        q_vals  = agent.get_q_values(state_for_qv)
        probs   = tf.nn.softmax(q_vals).numpy()
        action  = int(np.argmax(q_vals))
        action_lbl = "🏠 Home wins" if action == 1 else "✈️ Away wins"

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            st.metric("Prediction",   action_lbl)
        with qc2:
            st.metric("Confidence",   f"{float(np.max(probs))*100:.1f}%")
        with qc3:
            st.metric("Q spread",     f"{abs(q_vals[0]-q_vals[1]):.4f}")

        df_qv = pd.DataFrame({
            "Action":       ["Away wins (0)", "Home wins (1)"],
            "Q-Value":      [round(float(q_vals[0]), 4), round(float(q_vals[1]), 4)],
            "Probability":  [round(float(probs[0]), 4), round(float(probs[1]), 4)],
        })
        st.dataframe(df_qv, hide_index=True, use_container_width=True)
        st.bar_chart(df_qv.set_index("Action")["Q-Value"], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5: News
# ══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.subheader("ESPN NBA News")

    if st.button("🗞️ Load Latest News", key="btn_news"):
        with st.spinner("Fetching ESPN news…"):
            from utils.espn_api import fetch_espn_news
            try:
                news = fetch_espn_news(limit=10)
                st.session_state.latest_news = news
            except Exception as exc:
                st.error(f"News fetch error: {exc}")

    if "latest_news" in st.session_state and st.session_state.latest_news:
        for article in st.session_state.latest_news:
            with st.container():
                st.markdown(f"**{article['headline']}**")
                if article.get("description"):
                    st.caption(article["description"])
                if article.get("published"):
                    st.caption(f"Published: {article['published'][:10]}")
                if article.get("url"):
                    st.markdown(f"[Read more]({article['url']})")
                st.markdown("---")
    else:
        st.info("Click **Load Latest News** to fetch ESPN NBA headlines.")
