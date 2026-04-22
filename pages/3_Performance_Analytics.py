import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import (
    init_db, get_recent_predictions, get_model_metrics,
    get_reward_series, get_training_history,
    get_monte_carlo_history, get_lessons,
)

st.set_page_config(page_title="Analytics · NBA AI", page_icon="📈", layout="wide")

if "db_ready" not in st.session_state:
    init_db(); st.session_state.db_ready = True
if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    a = DQNAgent(); a.load(); st.session_state.agent = a

agent   = st.session_state.agent
metrics = get_model_metrics()

st.markdown("## 📈 Performance Analytics")
st.caption("Accuracy curves · reward history · prediction log · AI lessons")
st.divider()

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Total Predictions",  metrics["total"])
with k2: st.metric("Accuracy",           f"{metrics['accuracy']:.1f}%",
                    delta=f"{metrics['accuracy_delta']:+.1f}%")
with k3: st.metric("Avg Reward",         f"{metrics['avg_reward']:+.4f}")
with k4: st.metric("Training Steps",     f"{agent.steps:,}")
with k5: st.metric("Today",              metrics["today"])

st.divider()

tab_acc, tab_rew, tab_log, tab_lessons, tab_mc = st.tabs([
    "🎯 Accuracy", "💰 Rewards", "📋 Prediction Log", "🧠 AI Lessons", "🎲 MC History"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_acc:
    preds = get_recent_predictions(1000)
    if not preds:
        st.info("No prediction data yet.")
    else:
        df = pd.DataFrame(preds)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["actual_result"]).sort_values("timestamp")

        if len(df) >= 5:
            win  = min(25, max(5, len(df)//5))
            df["rolling_acc"] = df.apply(
                lambda r: int(r["prediction"] == r["actual_result"]), axis=1
            ).rolling(win, min_periods=1).mean() * 100
            df["cum_acc"]     = df.apply(
                lambda r: int(r["prediction"] == r["actual_result"]), axis=1
            ).expanding().mean() * 100

            ca, cb = st.columns(2)
            with ca:
                st.subheader(f"Rolling Accuracy (window {win})")
                st.line_chart(df.set_index("timestamp")["rolling_acc"],
                              use_container_width=True, color="#6c63ff")
            with cb:
                st.subheader("Cumulative Accuracy")
                st.line_chart(df.set_index("timestamp")["cum_acc"],
                              use_container_width=True, color="#00d4ff")

            # Accuracy by source
            df["correct"] = df["prediction"] == df["actual_result"]
            src = df.groupby("source")["correct"].agg(["mean","count"])
            src["mean"] = (src["mean"] * 100).round(1)
            st.subheader("Accuracy by Source")
            st.dataframe(src.rename(columns={"mean":"Accuracy%","count":"Games"}),
                         use_container_width=True)

            # Confidence calibration
            if "confidence" in df.columns:
                st.subheader("Confidence Calibration")
                df["bucket"] = pd.cut(df["confidence"],
                                      bins=[0,55,65,75,85,101],
                                      labels=["<55%","55-65%","65-75%","75-85%",">85%"])
                cal = df.groupby("bucket", observed=False)["correct"].agg(["mean","count"]).reset_index()
                cal.columns = ["Confidence Range","Actual Acc%","N"]
                cal["Actual Acc%"] = (cal["Actual Acc%"] * 100).round(1)
                st.dataframe(cal, hide_index=True, use_container_width=True)
        else:
            st.info(f"Need ≥5 completed predictions (have {len(df)}).")

# ══════════════════════════════════════════════════════════════════════════════
with tab_rew:
    ts = get_reward_series()
    if not ts:
        st.info("No reward data yet.")
    else:
        df_r = pd.DataFrame(ts)
        df_r["timestamp"] = pd.to_datetime(df_r["timestamp"])
        df_r = df_r.sort_values("timestamp")
        win  = min(20, max(5, len(df_r)//5))
        df_r["rolling"]    = df_r["reward"].rolling(win, min_periods=1).mean()
        df_r["cumulative"] = df_r["reward"].expanding().mean()

        ra, rb = st.columns(2)
        with ra:
            st.subheader(f"Rolling Avg Reward (window {win})")
            st.line_chart(df_r.set_index("timestamp")["rolling"],
                          use_container_width=True, color="#00ff99")
        with rb:
            st.subheader("Cumulative Avg Reward")
            st.line_chart(df_r.set_index("timestamp")["cumulative"],
                          use_container_width=True, color="#ffcc00")

        st.subheader("Reward Distribution")
        rc = df_r["reward"].round(1).value_counts().sort_index()
        st.bar_chart(rc, use_container_width=True)

        rs1,rs2,rs3,rs4 = st.columns(4)
        with rs1: st.metric("Mean",  f"{df_r['reward'].mean():+.4f}")
        with rs2: st.metric("Std",   f"{df_r['reward'].std():.4f}")
        with rs3: st.metric("Max",   f"{df_r['reward'].max():+.2f}")
        with rs4: st.metric("Min",   f"{df_r['reward'].min():+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    show_done = st.checkbox("Completed only", value=True, key="cb_done_only")
    max_rows  = st.selectbox("Rows", [25,50,100,200], index=1, key="sel_max_rows")
    preds_all = get_recent_predictions(int(max_rows))
    if not preds_all:
        st.info("No predictions yet.")
    else:
        df_all = pd.DataFrame(preds_all)
        if show_done:
            df_all = df_all.dropna(subset=["actual_result"])
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.strftime("%m-%d %H:%M")
        df_all["result"]    = df_all.apply(
            lambda r: ("✅" if r.get("prediction")==r.get("actual_result") and pd.notna(r.get("actual_result"))
                       else ("❌" if pd.notna(r.get("actual_result")) else "⏳")), axis=1
        )
        if "confidence" in df_all.columns:
            df_all["confidence"] = df_all["confidence"].round(1)
        show = [c for c in ["timestamp","matchup","result","confidence","reward","source"]
                if c in df_all.columns]
        st.dataframe(
            df_all[show].rename(columns={
                "timestamp":"Time","matchup":"Game","result":"Result",
                "confidence":"Conf%","reward":"Reward","source":"Src"
            }),
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "⬇️ Download CSV",
            data=df_all.to_csv(index=False),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", key="dl_csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
with tab_lessons:
    st.subheader("AI Self-Improvement Lessons")
    lessons = get_lessons(50)
    if not lessons:
        st.info(
            "No lessons yet. The agent analyzes its mistakes after every 5 missed predictions "
            "and generates targeted improvement lessons via Claude."
        )
    else:
        for l in lessons[:20]:
            with st.expander(
                f"🧠 {l['timestamp'][:16]} — {l.get('bias_detected','lesson')}",
                expanded=False,
            ):
                lc1, lc2 = st.columns([3, 1])
                with lc1:
                    st.markdown(f"**Lesson:** {l.get('lesson','')}")
                    st.markdown(f"**Reasoning:** {l.get('reasoning','')}")
                with lc2:
                    st.metric("Expected Δacc", f"+{l.get('predicted_improvement',0)*100:.1f}%")
                    st.caption(f"Misses: {l.get('total_misses',0)}  Correct: {l.get('total_correct',0)}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    mc_hist = get_monte_carlo_history(30)
    if not mc_hist:
        st.info("No MC runs yet. Visit ⚗️ Training Lab.")
    else:
        df_mc = pd.DataFrame(mc_hist)
        df_mc["win_pct"] = (df_mc["win_rate"] * 100).round(1)
        df_mc["timestamp"] = pd.to_datetime(df_mc["timestamp"])
        st.line_chart(df_mc.set_index("timestamp")["win_pct"],
                      use_container_width=True, color="#6c63ff")
        df_mc["timestamp"] = df_mc["timestamp"].dt.strftime("%m-%d %H:%M")
        st.dataframe(
            df_mc[["timestamp","n_simulations","win_pct","avg_reward","std_reward","confidence_95"]],
            hide_index=True, use_container_width=True,
        )
