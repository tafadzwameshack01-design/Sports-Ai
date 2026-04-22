import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import (
    init_db, get_recent_predictions, get_model_metrics,
    get_reward_timeseries, get_training_history, get_monte_carlo_history,
)

st.set_page_config(
    page_title="Analytics · NBA AI",
    page_icon="📈",
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

agent   = st.session_state.agent
metrics = get_model_metrics()

st.markdown("## 📈 Performance Analytics")
st.caption("Historical accuracy · reward curves · prediction log · training metrics")
st.divider()

# ── Top KPIs ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Predictions",  metrics["total"])
with k2:
    st.metric("Overall Accuracy",   f"{metrics['accuracy']:.1f}%",
              delta=f"{metrics['accuracy_delta']:+.1f}% vs last week")
with k3:
    st.metric("Avg Reward",         f"{metrics['avg_reward']:+.4f}")
with k4:
    st.metric("Agent Steps",        f"{agent.steps:,}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_acc, tab_reward, tab_log, tab_mc_hist = st.tabs(
    ["🎯 Accuracy", "💰 Rewards", "📋 Prediction Log", "🎲 MC History"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Accuracy over time
# ══════════════════════════════════════════════════════════════════════════════
with tab_acc:
    preds = get_recent_predictions(500)
    if not preds:
        st.info("No prediction history yet.")
    else:
        df = pd.DataFrame(preds)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["actual_result"])
        df["correct"]   = (df["prediction"] == df["actual_result"]).astype(int)

        if len(df) >= 5:
            # Rolling accuracy
            df = df.sort_values("timestamp").reset_index(drop=True)
            window = min(20, max(5, len(df) // 5))
            df["rolling_acc"] = df["correct"].rolling(window, min_periods=1).mean() * 100
            df["cumulative_acc"] = df["correct"].expanding().mean() * 100

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"Rolling Accuracy (window={window})")
                st.line_chart(
                    df.set_index("timestamp")["rolling_acc"],
                    use_container_width=True,
                    color="#6c63ff",
                )
            with col_b:
                st.subheader("Cumulative Accuracy")
                st.line_chart(
                    df.set_index("timestamp")["cumulative_acc"],
                    use_container_width=True,
                    color="#00d4ff",
                )

            # Accuracy by source
            src_acc = (
                df.groupby("source")["correct"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "accuracy", "count": "games"})
            )
            src_acc["accuracy"] = (src_acc["accuracy"] * 100).round(1)
            st.subheader("Accuracy by Data Source")
            st.dataframe(src_acc, use_container_width=True)

            # Confidence calibration
            st.subheader("Confidence Calibration")
            if "confidence" in df.columns:
                df["conf_bucket"] = pd.cut(df["confidence"], bins=[0, 55, 65, 75, 85, 101],
                                           labels=["<55%", "55-65%", "65-75%", "75-85%", ">85%"])
                cal = df.groupby("conf_bucket", observed=False)["correct"].agg(["mean", "count"]).reset_index()
                cal.columns = ["Confidence Range", "Actual Accuracy", "Count"]
                cal["Actual Accuracy"] = (cal["Actual Accuracy"] * 100).round(1)
                st.dataframe(cal, hide_index=True, use_container_width=True)
        else:
            st.info(f"Need at least 5 completed predictions for charts (have {len(df)}).")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Rewards
# ══════════════════════════════════════════════════════════════════════════════
with tab_reward:
    ts_data = get_reward_timeseries()
    if not ts_data:
        st.info("No reward data yet.")
    else:
        df_r = pd.DataFrame(ts_data)
        df_r["timestamp"] = pd.to_datetime(df_r["timestamp"])
        df_r = df_r.sort_values("timestamp")

        # Rolling average reward
        win = min(20, max(5, len(df_r) // 5))
        df_r["rolling_reward"] = df_r["reward"].rolling(win, min_periods=1).mean()
        df_r["cum_reward"]     = df_r["reward"].expanding().mean()

        rc1, rc2 = st.columns(2)
        with rc1:
            st.subheader(f"Rolling Avg Reward (window={win})")
            chart_data = df_r.set_index("timestamp")["rolling_reward"]
            st.line_chart(chart_data, use_container_width=True, color="#00ff99")
        with rc2:
            st.subheader("Cumulative Avg Reward")
            st.line_chart(df_r.set_index("timestamp")["cum_reward"],
                          use_container_width=True, color="#ffcc00")

        # Reward distribution
        st.subheader("Reward Distribution")
        reward_counts = df_r["reward"].round(1).value_counts().sort_index()
        st.bar_chart(reward_counts, use_container_width=True)

        # Reward stats
        rs1, rs2, rs3, rs4 = st.columns(4)
        with rs1:
            st.metric("Mean Reward",    f"{df_r['reward'].mean():+.4f}")
        with rs2:
            st.metric("Std Dev",        f"{df_r['reward'].std():.4f}")
        with rs3:
            st.metric("Max Reward",     f"{df_r['reward'].max():+.2f}")
        with rs4:
            st.metric("Min Reward",     f"{df_r['reward'].min():+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Prediction Log
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.subheader("Full Prediction Log")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_only_completed = st.checkbox(
            "Show only completed games", value=True, key="cb_completed"
        )
    with filter_col2:
        max_rows = st.selectbox("Rows to show", [25, 50, 100, 200], index=1, key="sel_rows")

    all_preds = get_recent_predictions(int(max_rows))
    if not all_preds:
        st.info("No predictions yet.")
    else:
        df_all = pd.DataFrame(all_preds)
        if show_only_completed:
            df_all = df_all.dropna(subset=["actual_result"])

        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.strftime("%m-%d %H:%M")
        df_all["result"]    = df_all.apply(
            lambda r: (
                "✅ Correct" if r.get("prediction") == r.get("actual_result") and pd.notna(r.get("actual_result"))
                else ("❌ Wrong" if pd.notna(r.get("actual_result")) else "⏳ Pending")
            ),
            axis=1,
        )
        df_all["conf_fmt"]  = df_all["confidence"].round(1).astype(str) + "%" if "confidence" in df_all.columns else "–"

        display_cols = ["timestamp", "home_team", "away_team", "result",
                        "conf_fmt", "reward", "source"]
        display_cols = [c for c in display_cols if c in df_all.columns]
        st.dataframe(
            df_all[display_cols].rename(columns={
                "home_team": "Home",
                "away_team": "Away",
                "conf_fmt":  "Confidence",
                "reward":    "Reward",
                "source":    "Source",
                "result":    "Result",
                "timestamp": "Time",
            }),
            hide_index=True,
            use_container_width=True,
        )

        # Download button
        csv = df_all.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="btn_dl_csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: Monte Carlo History
# ══════════════════════════════════════════════════════════════════════════════
with tab_mc_hist:
    st.subheader("Monte Carlo Run History")
    mc_history = get_monte_carlo_history(20)
    if not mc_history:
        st.info("No Monte Carlo runs yet.  Visit **⚗️ Training Lab** to run simulations.")
    else:
        df_mc = pd.DataFrame(mc_history)
        df_mc["timestamp"] = pd.to_datetime(df_mc["timestamp"])
        df_mc["win_rate_pct"] = (df_mc["win_rate"] * 100).round(1)

        st.subheader("Win Rate over MC Runs")
        st.line_chart(
            df_mc.set_index("timestamp")["win_rate_pct"],
            use_container_width=True,
            color="#6c63ff",
        )

        df_mc["timestamp"] = df_mc["timestamp"].dt.strftime("%m-%d %H:%M")
        df_mc["ci_range"]  = (
            "[" + (df_mc["avg_reward"] - df_mc["confidence_95"]).round(4).astype(str)
            + ", "
            + (df_mc["avg_reward"] + df_mc["confidence_95"]).round(4).astype(str)
            + "]"
        )

        st.dataframe(
            df_mc[["timestamp", "n_simulations", "win_rate_pct",
                   "avg_reward", "std_reward", "ci_range"]].rename(columns={
                "win_rate_pct":  "Win Rate %",
                "n_simulations": "N Sims",
                "avg_reward":    "Avg Reward",
                "std_reward":    "Std Reward",
                "ci_range":      "95% CI",
                "timestamp":     "Time",
            }),
            hide_index=True,
            use_container_width=True,
        )
