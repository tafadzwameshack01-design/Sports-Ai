import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.espn_api import fetch_espn_scoreboard, fetch_espn_power_index, fetch_espn_standings
from utils.sportradar import fetch_sr_todays_schedule, fetch_sr_standings
from utils.features import build_state
from utils.database import save_prediction, get_recent_predictions, init_db

st.set_page_config(
    page_title="Live Predictions · NBA AI",
    page_icon="🔴",
    layout="wide",
)

# ── sys.path guard for agent ───────────────────────────────────────────────────
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    _a = DQNAgent()
    _a.load()
    st.session_state.agent = _a

# ── Session state for this page ────────────────────────────────────────────────
for key, default in [
    ("score_cache",   {}),
    ("cycle_count",   0),
    ("sr_standings",  {}),
    ("espn_bpi",      {}),
    ("espn_standings",{}),
    ("cached_games",  []),
    ("last_heavy_ts", 0.0),
    ("auto_refresh",  True),
    ("last_errors",   []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Auto-refresh every 60 seconds ─────────────────────────────────────────────
if st.session_state.auto_refresh:
    st_autorefresh(interval=60_000, key="live_autorefresh")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🔴 Live Predictions")
st.caption(f"Last rendered: {datetime.now().strftime('%H:%M:%S')}  ·  Cycle #{st.session_state.cycle_count + 1}")

# ── Controls ───────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
with ctrl1:
    st.session_state.auto_refresh = st.toggle(
        "⏱ Auto-refresh (60 s)", value=st.session_state.auto_refresh, key="toggle_ar"
    )
with ctrl2:
    train_live = st.toggle("🎓 Train on completed games", value=True, key="toggle_train")
with ctrl3:
    use_sr = st.toggle("📡 Include SportRadar games", value=True, key="toggle_sr")
with ctrl4:
    if st.button("🔄 Refresh Now", key="btn_manual_refresh"):
        st.rerun()

st.divider()

# ── Data fetching (heavy calls throttled to every 5 minutes) ──────────────────
now_ts  = time.time()
is_cold = (now_ts - st.session_state.last_heavy_ts) > 300.0

if is_cold:
    with st.spinner("Fetching standings & power index…"):
        try:
            sr_standings  = fetch_sr_standings()
            espn_standings = fetch_espn_standings()
            merged_standings = {**espn_standings, **sr_standings}  # SR wins on conflict
            espn_bpi      = fetch_espn_power_index()
            st.session_state.sr_standings   = merged_standings
            st.session_state.espn_bpi       = espn_bpi
            st.session_state.last_heavy_ts  = now_ts
        except Exception as exc:
            st.warning(f"Could not refresh standings: {exc}")
            merged_standings = st.session_state.sr_standings
            espn_bpi         = st.session_state.espn_bpi
else:
    merged_standings = st.session_state.sr_standings
    espn_bpi         = st.session_state.espn_bpi

# ── Fetch today's games ────────────────────────────────────────────────────────
espn_games: list[dict] = []
sr_games:   list[dict] = []

with st.spinner("Fetching ESPN scoreboard…"):
    try:
        espn_games = fetch_espn_scoreboard()
    except Exception as exc:
        st.error(f"ESPN scoreboard error: {exc}")

if use_sr:
    with st.spinner("Fetching SportRadar schedule…"):
        try:
            sr_games = fetch_sr_todays_schedule()
        except Exception as exc:
            st.warning(f"SportRadar schedule error: {exc}")

# Deduplicate: prefer ESPN entries (richer odds), supplement with SR
seen_matchups = {g["home_team"] + g["away_team"] for g in espn_games}
all_games     = espn_games + [
    g for g in sr_games
    if (g["home_team"] + g["away_team"]) not in seen_matchups
]

st.session_state.cycle_count += 1

# ── Score-change detection ─────────────────────────────────────────────────────
def detect_changes(games: list[dict]) -> list[dict]:
    cache   = st.session_state.score_cache
    changed = []
    for g in games:
        gid  = g.get("game_id", "")
        prev = cache.get(gid)
        curr = {
            "home": g.get("home_score", 0),
            "away": g.get("away_score", 0),
            "status": g.get("status", ""),
        }
        if prev is None:
            cache[gid] = curr
        elif prev != curr:
            g["_changed"]   = True
            g["_prev_home"] = prev["home"]
            g["_prev_away"] = prev["away"]
            cache[gid]      = curr
            changed.append(g)
        else:
            g["_changed"] = False
    st.session_state.score_cache = cache
    return changed


changed_games = detect_changes(all_games)

# ── Build predictions ──────────────────────────────────────────────────────────
agent         = st.session_state.agent
predictions   = []
training_logs = []

for game in all_games:
    try:
        state          = build_state(game, merged_standings, espn_bpi)
        action, conf   = agent.act_greedy(state)
        pred_label     = game["home_team"] if action == 1 else game["away_team"]
        h_score        = float(game.get("home_score", 0))
        a_score        = float(game.get("away_score", 0))
        spread         = float(game.get("odds_spread", 0.0) or 0.0)
        is_final       = game.get("is_final", False) or game.get("status") in (
            "STATUS_FINAL", "closed", "complete"
        )
        has_score      = h_score > 0 or a_score > 0
        changed        = game.get("_changed", False)

        row = {
            "matchup":    f"{game['away_team']}  @  {game['home_team']}",
            "predicted":  pred_label,
            "action":     action,
            "confidence": conf,
            "h_score":    h_score,
            "a_score":    a_score,
            "spread":     spread,
            "status":     game.get("status_desc", game.get("status", "")),
            "changed":    changed,
            "is_final":   is_final,
            "source":     game.get("source", "espn"),
            "game_id":    game.get("game_id", ""),
            "home_team":  game["home_team"],
            "away_team":  game["away_team"],
        }

        # ── Train on completed or score-changed games ──────────────────────────
        if train_live and has_score and (changed or is_final):
            actual        = 1 if h_score > a_score else 0
            actual_margin = h_score - a_score
            reward        = agent.calculate_reward(action, actual, spread, actual_margin)

            agent.remember(state, action, reward, state, done=is_final)
            loss = agent.replay(batch_size=64)

            save_prediction(
                game_id         = game.get("game_id", ""),
                source          = game.get("source", "espn"),
                home_team       = game["home_team"],
                away_team       = game["away_team"],
                prediction      = action,
                actual_result   = actual,
                predicted_margin= spread,
                actual_margin   = actual_margin,
                reward          = reward,
                confidence      = conf,
            )

            row["actual"]   = actual
            row["reward"]   = reward
            row["trained"]  = True
            training_logs.append({
                "game":   row["matchup"],
                "result": "✅ Correct" if action == actual else "❌ Wrong",
                "reward": reward,
                "loss":   loss,
            })
        else:
            row["trained"] = False

        predictions.append(row)

    except Exception as exc:
        st.session_state.last_errors.append(str(exc))


# ── Auto-save model if any training happened ───────────────────────────────────
if training_logs:
    try:
        agent.save()
    except Exception as exc:
        st.warning(f"Model save failed: {exc}")

# ── Summary bar ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Games Today",      len(all_games))
with m2:
    st.metric("Score Updates",    len(changed_games))
with m3:
    st.metric("Training Events",  len(training_logs))
with m4:
    st.metric("Agent ε",          f"{agent.epsilon:.4f}")

st.divider()

# ── Main predictions table ─────────────────────────────────────────────────────
if not predictions:
    st.info("📭 No games found for today. Check back during the NBA season, or refresh.")
else:
    for p in predictions:
        status_icon = "🔴" if p["changed"] else ("✅" if p["is_final"] else "⏳")
        score_str   = (
            f"{p['a_score']:.0f} – {p['h_score']:.0f}"
            if (p["h_score"] or p["a_score"])
            else "–"
        )
        trained_tag = "🎓 Trained" if p["trained"] else ""
        source_tag  = f"[{p['source'].upper()}]"

        with st.container():
            ca, cb, cc, cd, ce = st.columns([3, 2, 1.5, 1.5, 1.5])
            with ca:
                st.markdown(f"**{status_icon} {p['matchup']}** {source_tag}")
            with cb:
                colour = "green" if p["confidence"] >= 65 else "orange"
                st.markdown(f"🏆 :{colour}[**{p['predicted']}**]")
            with cc:
                st.markdown(f"`{p['confidence']:.1f}%` confidence")
            with cd:
                st.markdown(f"Score: **{score_str}**")
            with ce:
                st.markdown(f"{p['status']}  {trained_tag}")
        st.markdown("---")

# ── Training log expander ──────────────────────────────────────────────────────
if training_logs:
    with st.expander(f"🎓 Training Events This Cycle ({len(training_logs)})", expanded=True):
        for t in training_logs:
            loss_str = f"loss={t['loss']:.4f}" if t["loss"] is not None else "buffer warming up"
            st.markdown(
                f"- **{t['game']}** → {t['result']}  |  "
                f"reward={t['reward']:+.2f}  |  {loss_str}"
            )

# ── Agent diagnostics expander ─────────────────────────────────────────────────
with st.expander("🔧 Agent Diagnostics", expanded=False):
    dcol1, dcol2 = st.columns(2)
    sd = agent.get_state_dict()
    with dcol1:
        st.json(sd)
    with dcol2:
        st.caption("Replay buffer occupancy")
        buf_pct = len(agent.memory) / 5000
        st.progress(buf_pct, text=f"{len(agent.memory)} / 5000 experiences")
        if st.session_state.last_errors:
            st.warning("Recent errors:")
            for err in st.session_state.last_errors[-5:]:
                st.code(err)
            if st.button("Clear errors", key="btn_clear_err"):
                st.session_state.last_errors = []
                st.rerun()

# ── Recent prediction history ──────────────────────────────────────────────────
with st.expander("📋 Recent Prediction History (last 20)", expanded=False):
    import pandas as pd
    recent = get_recent_predictions(20)
    if recent:
        df = pd.DataFrame(recent)
        st.dataframe(
            df[["timestamp", "home_team", "away_team", "prediction",
                "actual_result", "reward", "confidence"]].rename(columns={
                "home_team": "Home",
                "away_team": "Away",
                "prediction": "Pred (1=Home)",
                "actual_result": "Actual",
                "reward": "Reward",
                "confidence": "Conf %",
                "timestamp": "Time",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No prediction history yet.")
