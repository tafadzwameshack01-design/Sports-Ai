import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import (
    init_db, save_monte_carlo_run, get_monte_carlo_history,
    save_training_session, get_training_history,
)
from utils.features import run_monte_carlo, FEATURE_NAMES, get_feature_importance

st.set_page_config(
    page_title="Training Lab · NBA AI",
    page_icon="⚗️",
    layout="wide",
)

# ── Session init ───────────────────────────────────────────────────────────────
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    _a = DQNAgent()
    _a.load()
    st.session_state.agent = _a

for key, default in [
    ("mc_results",       None),
    ("mc_reward_series", []),
    ("train_loss_log",   []),
    ("stop_flag",        False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent = st.session_state.agent

st.markdown("## ⚗️ Training Lab")
st.caption("Monte Carlo validation · DQN hyperparameter control · experience replay")
st.divider()

tab_mc, tab_hp, tab_replay, tab_fi = st.tabs(
    ["🎲 Monte Carlo", "⚙️ Hyperparameters", "🔁 Replay Buffer", "📊 Feature Importance"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.subheader("Monte Carlo Agent Validation")
    st.markdown(
        "Runs N synthetic game simulations against a logistic ground-truth model "
        "to estimate the agent's expected win rate, average reward, and 95% CI."
    )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        n_sims = st.slider(
            "Simulations", min_value=500, max_value=5000, value=1000, step=500,
            key="sl_n_sims",
        )
    with mc2:
        mc_seed = st.number_input("Random seed", value=42, min_value=0, key="ni_seed")
    with mc3:
        st.write("")
        st.write("")
        run_mc = st.button("▶ Run Monte Carlo", key="btn_run_mc", use_container_width=True)

    if run_mc:
        with st.spinner(f"Running {n_sims:,} simulations…"):
            results = run_monte_carlo(agent, n_simulations=n_sims, seed=int(mc_seed))
            st.session_state.mc_results = results
            save_monte_carlo_run(
                n_simulations = results["n_simulations"],
                win_rate      = results["win_rate"],
                avg_reward    = results["avg_reward"],
                std_reward    = results["std_reward"],
                confidence_95 = results["confidence_95"],
            )
        st.toast(f"Monte Carlo complete — win rate {results['win_rate']*100:.1f}%", icon="🎲")

    if st.session_state.mc_results:
        r = st.session_state.mc_results
        ra, rb, rc, rd = st.columns(4)
        with ra:
            st.metric("Win Rate",    f"{r['win_rate']*100:.1f}%")
        with rb:
            st.metric("Avg Reward",  f"{r['avg_reward']:+.4f}")
        with rc:
            st.metric("Std Reward",  f"{r['std_reward']:.4f}")
        with rd:
            st.metric("95% CI Half", f"±{r['confidence_95']:.4f}")

        st.markdown(
            f"**95% Confidence Interval for mean reward:** "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        )

        # Performance assessment
        if r["win_rate"] >= 0.60:
            st.success(f"✅ Agent exceeds 60% win-rate threshold — performance ACCEPTABLE")
        elif r["win_rate"] >= 0.50:
            st.warning(f"⚠️  Agent above random baseline but below 60% — consider more training")
        else:
            st.error(f"❌ Agent below random baseline — hyperparameter tuning recommended")

    # History table
    mc_hist = get_monte_carlo_history(10)
    if mc_hist:
        st.subheader("MC Run History")
        df_mc = pd.DataFrame(mc_hist)
        df_mc["win_rate"]  = (df_mc["win_rate"]  * 100).round(1).astype(str) + "%"
        df_mc["timestamp"] = pd.to_datetime(df_mc["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            df_mc[["timestamp", "n_simulations", "win_rate", "avg_reward", "std_reward", "confidence_95"]],
            hide_index=True,
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
with tab_hp:
    st.subheader("DQN Hyperparameter Control")
    st.caption("Changes are applied immediately to the running agent (hot-update).")

    hp1, hp2 = st.columns(2)

    with hp1:
        new_lr = st.slider(
            "Learning rate", 1e-5, 1e-2, float(agent.lr),
            step=1e-5, format="%.5f", key="sl_lr",
        )
        new_gamma = st.slider(
            "Discount γ", 0.80, 0.999, float(agent.gamma),
            step=0.001, format="%.3f", key="sl_gamma",
        )

    with hp2:
        new_eps_decay = st.slider(
            "ε decay", 0.990, 0.9999, float(agent.epsilon_decay),
            step=0.0001, format="%.4f", key="sl_eps_decay",
        )
        new_eps_min = st.slider(
            "ε minimum", 0.01, 0.20, float(agent.epsilon_min),
            step=0.01, key="sl_eps_min",
        )

    apply_hp = st.button("✅ Apply Hyperparameters", key="btn_apply_hp")
    if apply_hp:
        agent.update_hyperparams(
            lr            = new_lr,
            gamma         = new_gamma,
            epsilon_decay = new_eps_decay,
            epsilon_min   = new_eps_min,
        )
        st.session_state.agent = agent
        st.toast("Hyperparameters updated!", icon="⚙️")
        st.rerun()

    st.divider()
    st.subheader("Synthetic Training Loop")
    st.caption(
        "Fills the replay buffer with synthetic experience using random states "
        "and a logistic ground-truth oracle, then runs experience replay."
    )

    train_col1, train_col2 = st.columns(2)
    with train_col1:
        n_train_steps = st.slider("Training steps", 50, 500, 100, step=50, key="sl_train_steps")
        batch_sz = st.selectbox("Batch size", [32, 64, 128], index=1, key="sel_batch")
    with train_col2:
        st.write("")
        st.write("")
        st.session_state.stop_flag = False
        stop_btn = st.button("⏹ Stop", key="btn_stop_train")
        start_train = st.button("▶ Run Synthetic Training", key="btn_start_train", use_container_width=True)

    if stop_btn:
        st.session_state.stop_flag = True

    if start_train:
        rng = np.random.default_rng(int(time.time()))
        progress_bar   = st.progress(0.0, text="Populating buffer…")
        loss_container = st.empty()
        loss_log: list[float] = []
        init_eps = agent.epsilon

        for step in range(n_train_steps):
            if st.session_state.stop_flag:
                st.warning("⏹ Training stopped by user.")
                break

            state  = rng.uniform(0, 1, 12).astype(np.float32)
            action = agent.act(state)

            bpi_d   = float(state[11]) * 200.0 - 100.0
            wp_d    = float(state[8]) - float(state[9])
            logit   = 0.03 * bpi_d + 2.0 * wp_d
            prob    = 1.0 / (1.0 + np.exp(-logit))
            actual  = int(rng.random() < prob)
            reward  = agent.calculate_reward(action, actual, 0.0, 0.0)
            next_s  = rng.uniform(0, 1, 12).astype(np.float32)

            agent.remember(state, action, reward, next_s, done=True)
            loss = agent.replay(batch_size=int(batch_sz))

            if loss is not None:
                loss_log.append(loss)

            progress_bar.progress(
                (step + 1) / n_train_steps,
                text=f"Step {step+1}/{n_train_steps}  ·  ε={agent.epsilon:.4f}  ·  "
                     f"loss={loss:.4f if loss else 'warming up'}",
            )

        st.session_state.train_loss_log = loss_log
        avg_r = sum([agent.calculate_reward(1, 1, 0, 0)] * max(len(loss_log), 1)) / max(len(loss_log), 1)
        save_training_session("synthetic", n_train_steps, init_eps, agent.epsilon, 1.0)
        agent.save()
        st.session_state.agent = agent
        st.toast(f"Training complete — {len(loss_log)} replay steps", icon="🎓")

    if st.session_state.train_loss_log:
        st.subheader("Training Loss Curve")
        st.line_chart(
            pd.DataFrame({"Huber Loss": st.session_state.train_loss_log}),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════
with tab_replay:
    st.subheader("Experience Replay Buffer")

    rb1, rb2, rb3 = st.columns(3)
    with rb1:
        st.metric("Buffer Size",   f"{len(agent.memory):,}")
    with rb2:
        st.metric("Capacity",      "5,000")
    with rb3:
        fill_pct = len(agent.memory) / 5000
        st.metric("Fill %",        f"{fill_pct*100:.1f}%")

    st.progress(fill_pct, text=f"Buffer: {len(agent.memory)} / 5 000")

    if st.button("🗑️ Clear Replay Buffer", key="btn_clear_buf"):
        agent.memory.clear()
        st.session_state.agent = agent
        st.toast("Replay buffer cleared.", icon="🗑️")
        st.rerun()

    if len(agent.memory) >= 5:
        st.subheader("Sample Experiences")
        import random
        samples = random.sample(list(agent.memory), min(5, len(agent.memory)))
        for i, (s, a, r, ns, done) in enumerate(samples):
            with st.expander(f"Experience {i+1}  |  action={a}  reward={r:+.2f}  done={done}"):
                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    st.caption("State vector")
                    st.json({n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, s)})
                with ecol2:
                    st.caption("Next state vector")
                    st.json({n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, ns)})

    st.subheader("Training Session History")
    history = get_training_history(15)
    if history:
        df_h = pd.DataFrame(history)
        df_h["timestamp"] = pd.to_datetime(df_h["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            df_h[["timestamp", "session_type", "iterations",
                  "initial_epsilon", "final_epsilon", "avg_reward", "notes"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No training sessions recorded yet.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.subheader("Feature Importance (Permutation Method)")
    st.caption(
        "Shuffles each feature independently across 200 random states and measures "
        "the resulting drop in Q-value variance.  Higher = more influential."
    )

    if st.button("📊 Compute Feature Importance", key="btn_fi"):
        with st.spinner("Computing permutation importance…"):
            importance = get_feature_importance(agent, n_samples=200)
            st.session_state.fi_results = importance
        st.toast("Feature importance computed!", icon="📊")

    if "fi_results" in st.session_state and st.session_state.fi_results:
        fi = st.session_state.fi_results
        df_fi = pd.DataFrame(
            {"Feature": list(fi.keys()), "Importance": list(fi.values())}
        ).sort_values("Importance", ascending=False)

        st.bar_chart(df_fi.set_index("Feature")["Importance"], use_container_width=True)

        st.dataframe(
            df_fi.style.format({"Importance": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )
