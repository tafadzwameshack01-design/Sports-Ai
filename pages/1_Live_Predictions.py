import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import init_db, save_prediction, get_recent_predictions
from utils.espn_api  import fetch_espn_scoreboard, fetch_espn_standings, fetch_espn_power_index
from utils.features  import build_state, FEATURE_NAMES

st.set_page_config(page_title="Live Predictions · NBA AI", page_icon="🔴", layout="wide")

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "db_ready" not in st.session_state:
    init_db(); st.session_state.db_ready = True

if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    a = DQNAgent(); a.load(); st.session_state.agent = a

if "improver" not in st.session_state:
    import anthropic
    from utils.self_improver import AutonomousImprover
    _key = os.environ.get("ANTHROPIC_API_KEY","")
    c    = anthropic.Anthropic(api_key=_key)
    st.session_state.improver = AutonomousImprover(st.session_state.agent, c)
    st.session_state.anthropic_client = c

for key, default in [
    ("score_cache",       {}),
    ("cycle_count",       0),
    ("standings",         {}),
    ("bpi",               {}),
    ("live_games",        []),
    ("last_heavy_ts",     0.0),
    ("auto_refresh_live", True),
    ("training_log",      []),
    ("page_losses",       []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent    = st.session_state.agent
improver = st.session_state.improver

# ── Auto-refresh ───────────────────────────────────────────────────────────────
from streamlit_autorefresh import st_autorefresh
if st.session_state.auto_refresh_live:
    st_autorefresh(interval=60_000, key="live_ar")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🔴 Live Predictions")
st.caption(f"Cycle #{st.session_state.cycle_count + 1}  ·  {datetime.now().strftime('%H:%M:%S')}")

# ── Controls row ──────────────────────────────────────────────────────────────
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1:
    st.session_state.auto_refresh_live = st.toggle(
        "⏱ Auto-refresh 60s", value=st.session_state.auto_refresh_live, key="tog_ar"
    )
with r1c2:
    train_live = st.toggle("🎓 Train on results", value=True, key="tog_train")
with r1c3:
    show_historical = st.toggle("📚 Include recent history", value=True, key="tog_hist")
with r1c4:
    days_back = st.selectbox("History days", [1, 3, 7], index=1, key="sel_hist_days")
with r1c5:
    if st.button("🔄 Refresh Now", key="btn_refresh", use_container_width=True):
        st.session_state.last_heavy_ts = 0.0
        st.rerun()

st.divider()

# ── Data fetch (always fresh on this page) ─────────────────────────────────────
now = time.time()
if (now - st.session_state.last_heavy_ts) > 55:   # refresh every ~60s matching autorefresh
    with st.spinner("📡 Fetching live ESPN data…"):
        try:
            today_games = fetch_espn_scoreboard()
            standings   = fetch_espn_standings()
            bpi         = fetch_espn_power_index()
            st.session_state.live_games    = today_games
            st.session_state.standings     = standings
            st.session_state.bpi           = bpi
            st.session_state.last_heavy_ts = now
        except Exception as e:
            st.error(f"ESPN fetch error: {e}")
            today_games = st.session_state.live_games
            standings   = st.session_state.standings
            bpi         = st.session_state.bpi
else:
    today_games = st.session_state.live_games
    standings   = st.session_state.standings
    bpi         = st.session_state.bpi

# Historical games for training buffer warm-up
historical: list[dict] = []
if show_historical:
    with st.spinner(f"📚 Loading {days_back}d historical data for training…"):
        try:
            from utils.espn_api import fetch_espn_scoreboard_range
            historical = fetch_espn_scoreboard_range(int(days_back))
        except Exception as e:
            st.warning(f"Historical fetch: {e}")

all_games = today_games + [
    g for g in historical
    if g["game_id"] not in {x["game_id"] for x in today_games}
]

st.session_state.cycle_count += 1

# ── Score-change detection ─────────────────────────────────────────────────────
def detect_changes(games: list[dict]) -> set[str]:
    changed = set()
    cache   = st.session_state.score_cache
    for g in games:
        gid  = g.get("game_id","")
        curr = {"h": g.get("home_score",0), "a": g.get("away_score",0), "s": g.get("status","")}
        prev = cache.get(gid)
        if prev and prev != curr:
            changed.add(gid)
        cache[gid] = curr
    st.session_state.score_cache = cache
    return changed

changed_ids = detect_changes(all_games)

# ── Build predictions + autonomous training ────────────────────────────────────
predictions:   list[dict] = []
training_logs: list[dict] = []
session_losses: list[float] = []

for game in all_games:
    try:
        state        = build_state(game, standings, bpi)
        action, conf = agent.act_greedy(state)
        pred_team    = game["home_team"] if action == 1 else game["away_team"]
        h_score      = float(game.get("home_score", 0))
        a_score      = float(game.get("away_score", 0))
        spread       = float(game.get("odds_spread", 0.0) or 0.0)
        is_final     = game.get("is_final", False)
        has_score    = h_score > 0 or a_score > 0
        changed      = game.get("game_id","") in changed_ids

        row = {
            "matchup":   game.get("matchup", f"{game['away_team']} @ {game['home_team']}"),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_abbr": game.get("home_abbr",""),
            "away_abbr": game.get("away_abbr",""),
            "predicted": pred_team,
            "action":    action,
            "conf":      conf,
            "h_score":   h_score,
            "a_score":   a_score,
            "spread":    spread,
            "ou":        game.get("over_under", 220.0),
            "status":    game.get("status_desc", game.get("status","")),
            "is_final":  is_final,
            "is_live":   game.get("is_live", False),
            "changed":   changed,
            "source":    game.get("source","espn"),
            "game_id":   game.get("game_id",""),
            "venue":     game.get("venue",""),
            "state":     state,
            "trained":   False,
            "correct":   None,
            "reward":    None,
            "lesson":    None,
        }

        if train_live and has_score and (changed or is_final):
            actual        = 1 if h_score > a_score else 0
            actual_margin = h_score - a_score
            reward        = agent.calculate_reward(action, actual, spread, actual_margin)

            # Route through autonomous improver (handles PER priority + analysis)
            result = improver.process_result(
                state      = state,
                action     = action,
                actual     = actual,
                game_info  = {"matchup": row["matchup"]},
                reward     = reward,
            )

            # Also run explicit replay
            loss = agent.replay(batch_size=64)
            if loss is not None:
                session_losses.append(loss)

            save_prediction(
                game_id          = game.get("game_id",""),
                source           = game.get("source","espn"),
                home_team        = game["home_team"],
                away_team        = game["away_team"],
                matchup          = row["matchup"],
                prediction       = action,
                actual_result    = actual,
                predicted_margin = spread,
                actual_margin    = actual_margin,
                reward           = reward,
                confidence       = conf,
            )

            row.update({
                "trained": True,
                "correct": action == actual,
                "reward":  reward,
                "lesson":  result.get("lesson"),
            })

            training_logs.append({
                "game":    row["matchup"],
                "correct": action == actual,
                "reward":  reward,
                "lesson":  result.get("lesson",""),
            })

        predictions.append(row)

    except Exception as ex:
        st.warning(f"Error on {game.get('home_team','?')}: {ex}")

# Auto-save if any training happened
if training_logs:
    try:
        agent.save()
    except Exception:
        pass

# Background retrain even without new results
if not training_logs and len(agent.replay_buffer) >= 64:
    bg_loss = improver.run_background_retrain(steps=30)
    if bg_loss:
        st.session_state.background_losses = (
            st.session_state.get("background_losses", []) + [bg_loss]
        )[-200:]

# ── Summary metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("Games Loaded",     len(all_games))
with m2: st.metric("Today's Games",    len(today_games))
with m3:
    live_n = sum(1 for p in predictions if p["is_live"])
    st.metric("🔴 Live",               live_n)
with m4: st.metric("Training Events",  len(training_logs))
with m5: st.metric("Agent ε",          f"{agent.epsilon:.4f}")
with m6:
    avg_loss = sum(session_losses)/len(session_losses) if session_losses else 0.0
    st.metric("Avg Loss",              f"{avg_loss:.4f}" if session_losses else "—")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_games, tab_training, tab_history, tab_diagnostics = st.tabs(
    ["🏀 Games & Predictions", "🎓 Training Log", "📋 History", "🔧 Diagnostics"]
)

with tab_games:
    if not predictions:
        st.info("📭 No games loaded. ESPN may not have today's schedule yet.")
    else:
        # Group: live → upcoming → final → historical
        live_preds  = [p for p in predictions if p["is_live"]]
        sched_preds = [p for p in predictions if not p["is_live"] and not p["is_final"] and p["h_score"] == 0]
        final_preds = [p for p in predictions if p["is_final"]]
        hist_preds  = [p for p in predictions if not p["is_live"] and not p["is_final"] and p["h_score"] > 0 and p not in sched_preds]

        def render_game_table(preds: list[dict]) -> None:
            for p in preds:
                status_icon = "🔴" if p["is_live"] else ("✅" if p["is_final"] else "⏳")
                score_str   = f"{p['a_score']:.0f}–{p['h_score']:.0f}" if (p["h_score"] or p["a_score"]) else "TBD"
                result_icon = ""
                if p["correct"] is True:   result_icon = " ✅"
                elif p["correct"] is False: result_icon = " ❌"
                trained_tag = " 🎓" if p["trained"] else ""

                with st.container():
                    c1, c2, c3, c4, c5, c6 = st.columns([3, 2.5, 1.5, 1.5, 1.5, 1.5])
                    with c1:
                        st.markdown(f"**{status_icon} {p['matchup']}**")
                        if p["venue"]:
                            st.caption(p["venue"])
                    with c2:
                        conf_col = "green" if p["conf"] >= 65 else "orange"
                        st.markdown(f"🏆 :{conf_col}[**{p['predicted']}**]{result_icon}{trained_tag}")
                    with c3:
                        st.metric("Confidence", f"{p['conf']:.1f}%", label_visibility="collapsed")
                        st.caption(f"{p['conf']:.1f}% conf")
                    with c4:
                        st.markdown(f"**{score_str}**")
                        st.caption(p["status"])
                    with c5:
                        st.caption(f"Spread: {p['spread']:.1f}")
                        st.caption(f"O/U: {p['ou']:.1f}")
                    with c6:
                        if p["reward"] is not None:
                            r_col = "green" if p["reward"] > 0 else "red"
                            st.markdown(f":{r_col}[{p['reward']:+.2f}]")
                st.markdown("---")

        if live_preds:
            st.subheader(f"🔴 Live Games ({len(live_preds)})")
            render_game_table(live_preds)

        if sched_preds:
            st.subheader(f"⏳ Upcoming Today ({len(sched_preds)})")
            render_game_table(sched_preds)

        if final_preds:
            st.subheader(f"✅ Final Today ({len(final_preds)})")
            render_game_table(final_preds)

        if hist_preds:
            with st.expander(f"📚 Historical ({days_back}d) — {len(hist_preds)} games"):
                render_game_table(hist_preds[:20])

with tab_training:
    if not training_logs:
        st.info("No training events this cycle. Training fires when game scores update or games finish.")
        if st.session_state.background_losses:
            st.subheader("Background Retrain Loss")
            st.line_chart(
                pd.DataFrame({"Loss": st.session_state.background_losses[-50:]}),
                use_container_width=True,
            )
    else:
        for t in training_logs:
            icon = "✅" if t["correct"] else "❌"
            st.markdown(
                f"{icon} **{t['game']}** — reward: `{t['reward']:+.2f}`"
                + (f" — *{t['lesson']}*" if t.get("lesson") else "")
            )

        if session_losses:
            st.subheader("Training Loss This Cycle")
            st.line_chart(
                pd.DataFrame({"Huber Loss": session_losses}),
                use_container_width=True,
            )

        st.subheader("Autonomous Improver Status")
        ic1, ic2, ic3 = st.columns(3)
        with ic1: st.metric("Total Misses",  improver.total_misses)
        with ic2: st.metric("Total Correct", improver.total_correct)
        with ic3: st.metric("Lessons",       len(improver.lessons))

        if improver.last_analysis:
            la = improver.last_analysis
            st.success(f"**Latest lesson:** {la.get('lesson','')}")
            st.caption(f"Bias: {la.get('bias_detected','')}  |  Expected Δ: +{la.get('predicted_improvement',0)*100:.1f}%")

with tab_history:
    limit = st.selectbox("Show records", [25, 50, 100, 200], key="sel_hist_limit")
    recent = get_recent_predictions(int(limit))
    if not recent:
        st.info("No prediction history yet.")
    else:
        df = pd.DataFrame(recent)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")
        df["result"]    = df.apply(
            lambda r: (
                "✅" if r.get("prediction") == r.get("actual_result") and r.get("actual_result") is not None
                else ("❌" if r.get("actual_result") is not None else "⏳")
            ), axis=1,
        )
        show_cols = ["timestamp","matchup","result","confidence","reward","source"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            df[show_cols].rename(columns={
                "timestamp":  "Time",
                "matchup":    "Game",
                "result":     "Result",
                "confidence": "Conf%",
                "reward":     "Reward",
                "source":     "Src",
            }),
            use_container_width=True, hide_index=True,
        )
        csv = df.to_csv(index=False)
        st.download_button("⬇️ CSV", csv, "predictions.csv", "text/csv", key="dl_pred_csv")

with tab_diagnostics:
    dc1, dc2 = st.columns(2)
    with dc1:
        st.subheader("Agent State")
        st.json(agent.get_state_dict())
    with dc2:
        st.subheader("Buffer Health")
        buf = len(agent.replay_buffer)
        st.progress(buf / 10_000, text=f"{buf:,} / 10 000 experiences")
        st.caption(f"Adam step t={agent.online.t}")
        if st.button("💾 Force Save Model", key="btn_force_save"):
            agent.save()
            st.toast("Model saved!", icon="💾")
        if st.button("🔄 Force Data Refresh", key="btn_force_refresh"):
            st.session_state.last_heavy_ts = 0.0
            st.rerun()
