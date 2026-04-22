import os
import sys
import time
from pathlib import Path

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

st.set_page_config(page_title="Training Lab · NBA AI", page_icon="⚗️", layout="wide")

if "db_ready" not in st.session_state:
    init_db(); st.session_state.db_ready = True
if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    a = DQNAgent(); a.load(); st.session_state.agent = a

for key, default in [
    ("mc_results", None), ("fi_results", None),
    ("stop_flag", False), ("train_loss_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent = st.session_state.agent
st.markdown("## ⚗️ Training Lab")
st.caption("Monte Carlo validation · hyperparameter control · synthetic training · feature importance")
st.divider()

tab_mc, tab_hp, tab_train, tab_fi, tab_hist = st.tabs([
    "🎲 Monte Carlo", "⚙️ Hyperparameters", "🔁 Synthetic Training", "📊 Feature Importance", "📜 History"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.subheader("Monte Carlo Agent Validation")
    st.markdown(
        "Evaluates agent over **N synthetic games** using a logistic ground-truth model "
        "(bpi_delta + win_pct delta). Reports 95% CI on average reward."
    )
    mc1, mc2, mc3 = st.columns(3)
    with mc1: n_sims  = st.slider("Simulations", 500, 5000, 1000, 500, key="sl_nsims")
    with mc2: mc_seed = st.number_input("Seed", 0, 9999, 42, key="ni_seed")
    with mc3:
        st.write("")
        run_mc = st.button("▶ Run Monte Carlo", key="btn_mc", use_container_width=True)

    if run_mc:
        with st.spinner(f"Running {n_sims:,} simulations…"):
            res = run_monte_carlo(agent, n_simulations=n_sims, seed=int(mc_seed))
            st.session_state.mc_results = res
            save_monte_carlo_run(
                n_simulations=res["n_simulations"],
                win_rate=res["win_rate"],
                avg_reward=res["avg_reward"],
                std_reward=res["std_reward"],
                confidence_95=res["confidence_95"],
            )
        st.toast(f"Monte Carlo: {res['win_rate']*100:.1f}% win rate", icon="🎲")

    if st.session_state.mc_results:
        r = st.session_state.mc_results
        mc_a, mc_b, mc_c, mc_d = st.columns(4)
        with mc_a: st.metric("Win Rate",    f"{r['win_rate']*100:.1f}%")
        with mc_b: st.metric("Avg Reward",  f"{r['avg_reward']:+.4f}")
        with mc_c: st.metric("Std Dev",     f"{r['std_reward']:.4f}")
        with mc_d: st.metric("95% CI ±",    f"{r['confidence_95']:.4f}")

        st.markdown(
            f"**95% CI for mean reward:** `[{r['ci_low']:+.4f},  {r['ci_high']:+.4f}]`"
        )
        if r["win_rate"] >= 0.60:
            st.success("✅ Agent exceeds 60% win-rate threshold — ACCEPTABLE performance")
        elif r["win_rate"] >= 0.50:
            st.warning("⚠️ Above random baseline but below 60% target")
        else:
            st.error("❌ Below random baseline — tune hyperparameters or train more")

    mc_hist = get_monte_carlo_history(20)
    if mc_hist:
        st.subheader("MC History")
        df_mc = pd.DataFrame(mc_hist)
        df_mc["win_rate"] = (df_mc["win_rate"] * 100).round(1)
        df_mc["timestamp"] = pd.to_datetime(df_mc["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.line_chart(df_mc.set_index("id")["win_rate"], use_container_width=True)
        st.dataframe(
            df_mc[["timestamp","n_simulations","win_rate","avg_reward","std_reward","confidence_95"]],
            hide_index=True, use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
with tab_hp:
    st.subheader("DQN Hyperparameter Control")
    st.caption("Hot-applied to the running agent. Changes persist after model save.")

    h1, h2 = st.columns(2)
    with h1:
        new_lr    = st.slider("Learning rate",  1e-5, 1e-2, float(agent.online.lr),
                              step=1e-5, format="%.5f", key="sl_lr")
        new_gamma = st.slider("Discount γ",     0.80, 0.999, float(agent.gamma),
                              step=0.001, format="%.3f", key="sl_gamma")
    with h2:
        new_ed    = st.slider("ε decay",        0.990, 0.9999, float(agent.epsilon_decay),
                              step=0.0001, format="%.4f", key="sl_ed")
        new_emin  = st.slider("ε minimum",      0.01, 0.20,  float(agent.epsilon_min),
                              step=0.01, key="sl_emin")

    if st.button("✅ Apply Hyperparameters", key="btn_apply_hp"):
        agent.update_hyperparams(lr=new_lr, gamma=new_gamma,
                                 epsilon_decay=new_ed, epsilon_min=new_emin)
        st.session_state.agent = agent
        st.toast("Hyperparameters updated!", icon="⚙️")
        st.rerun()

    st.divider()
    st.subheader("Current Agent State")
    sd = agent.get_state_dict()
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: st.metric("ε",       f"{sd['epsilon']:.4f}")
    with sc2: st.metric("γ",       f"{sd['gamma']:.4f}")
    with sc3: st.metric("LR",      f"{sd['lr']:.5f}")
    with sc4: st.metric("Steps",   sd["steps"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Synthetic Training Loop")
    st.caption(
        "Fills the replay buffer with synthetic experiences drawn from a logistic "
        "ground-truth oracle, then runs Double DQN replay."
    )

    tc1, tc2, tc3 = st.columns(3)
    with tc1: n_steps  = st.slider("Training steps", 50, 1000, 200, 50, key="sl_nsteps")
    with tc2: batch_sz = st.selectbox("Batch size", [32, 64, 128], index=1, key="sel_batch")
    with tc3:
        st.write("")
        st.write("")
        if st.button("⏹ Stop", key="btn_stop"):
            st.session_state.stop_flag = True

    start_btn = st.button("▶ Start Synthetic Training", key="btn_start_train",
                          use_container_width=True)

    if start_btn:
        st.session_state.stop_flag = False
        rng         = np.random.default_rng(int(time.time()))
        progress    = st.progress(0.0, text="Warming up buffer…")
        loss_area   = st.empty()
        loss_log    = []
        init_eps    = agent.epsilon
        avg_rewards = []

        for step in range(n_steps):
            if st.session_state.stop_flag:
                st.warning("Stopped by user.")
                break

            state  = rng.uniform(0, 1, 12).astype(np.float32)
            action = agent.act(state)

            bpi_d  = float(state[11]) * 100.0 - 50.0
            wp_d   = float(state[8]) - float(state[9])
            logit  = 0.04 * bpi_d + 2.0 * wp_d
            prob   = 1.0 / (1.0 + np.exp(-logit))
            actual = int(rng.random() < prob)
            reward = agent.calculate_reward(action, actual, 0.0, 0.0)
            ns     = rng.uniform(0, 1, 12).astype(np.float32)

            # Missed prediction → high priority
            priority = 5.0 if action != actual else 1.0
            agent.remember(state, action, reward, ns, done=True, priority=priority)
            avg_rewards.append(reward)

            loss = agent.replay(batch_size=int(batch_sz))
            if loss is not None:
                loss_log.append(loss)

            pct = (step + 1) / n_steps
            loss_str = f"{loss:.4f}" if loss is not None else "warming"
            progress.progress(pct, text=f"Step {step+1}/{n_steps}  ε={agent.epsilon:.4f}  loss={loss_str}")

        avg_r = float(np.mean(avg_rewards)) if avg_rewards else 0.0
        avg_l = float(np.mean(loss_log))    if loss_log    else 0.0
        save_training_session(
            session_type="synthetic",
            iterations=n_steps,
            initial_epsilon=init_eps,
            final_epsilon=agent.epsilon,
            avg_reward=avg_r,
            avg_loss=avg_l,
            notes=f"batch={batch_sz}",
        )
        agent.save()
        st.session_state.agent     = agent
        st.session_state.train_loss_log = loss_log
        st.toast(f"Synthetic training complete — {len(loss_log)} replay steps", icon="🎓")

    if st.session_state.train_loss_log:
        st.subheader("Training Loss Curve")
        st.line_chart(
            pd.DataFrame({"Huber Loss": st.session_state.train_loss_log}),
            use_container_width=True,
        )

    # Replay buffer inspector
    st.subheader("Replay Buffer")
    buf_n = len(agent.replay_buffer)
    st.progress(buf_n / 10_000, text=f"{buf_n:,} / 10 000 ({buf_n/100:.1f}%)")
    if st.button("🗑️ Clear Buffer", key="btn_clr_buf"):
        agent.replay_buffer.buffer.clear()
        agent.replay_buffer.priorities.clear()
        agent.replay_buffer.pos = 0
        st.toast("Buffer cleared.", icon="🗑️")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.subheader("Feature Importance (Permutation Method)")
    if st.button("📊 Compute", key="btn_fi"):
        with st.spinner("Computing…"):
            st.session_state.fi_results = get_feature_importance(agent, 300)
        st.toast("Done!", icon="📊")

    if st.session_state.fi_results:
        fi = st.session_state.fi_results
        df_fi = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        st.bar_chart(df_fi.set_index("Feature")["Importance"], use_container_width=True)
        st.dataframe(df_fi.style.format({"Importance": "{:.4f}"}),
                     hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("Training Session History")
    hist = get_training_history(30)
    if hist:
        df_h = pd.DataFrame(hist)
        df_h["timestamp"] = pd.to_datetime(df_h["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            df_h[["timestamp","session_type","iterations","initial_epsilon",
                  "final_epsilon","avg_reward","avg_loss"]],
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No training sessions yet.")
