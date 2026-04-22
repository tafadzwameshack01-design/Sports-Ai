import sys
import os
import json
import difflib
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import anthropic

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.database import (
    init_db, save_optimizer_run, get_optimizer_history,
    get_model_metrics,
)
from utils.features import run_monte_carlo

st.set_page_config(
    page_title="AI Optimizer · NBA AI",
    page_icon="🤖",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME          = "claude-sonnet-4-20250514"
MAX_TOKENS_OPTIMIZER = 2048
PERFORMANCE_THRESHOLD = 0.55  # 55% win-rate is minimum acceptable

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
    ("stop_flag",           False),
    ("iteration_log",       []),
    ("optimizer_scores",    []),
    ("previous_agent_state", None),
    ("opt_running",         False),
    ("best_hyperparams",    None),
    ("best_score",          0.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Anthropic client ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY not set.")
    st.stop()

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Helper: build Claude optimizer prompt ─────────────────────────────────────
def build_optimizer_prompt(
    current_hyperparams: dict,
    current_score: float,
    iteration_history: list[dict],
    mc_stats: dict,
    db_metrics: dict,
) -> str:
    history_summary = "\n".join(
        f"  Iter {h['iteration']}: score={h['score']:.4f}  params={json.dumps(h['hyperparams'])}"
        for h in iteration_history[-5:]
    ) or "  (first iteration)"

    return f"""You are optimizing hyperparameters for a Deep Q-Network (DQN) that predicts NBA game outcomes.

CURRENT CONFIGURATION:
{json.dumps(current_hyperparams, indent=2)}

CURRENT PERFORMANCE:
- Monte Carlo win rate: {mc_stats.get('win_rate', 0)*100:.1f}%
- Avg reward: {mc_stats.get('avg_reward', 0):+.4f}
- 95% CI: [{mc_stats.get('ci_low', 0):+.4f}, {mc_stats.get('ci_high', 0):+.4f}]
- Historical accuracy: {db_metrics.get('accuracy', 0):.1f}%
- Current epsilon: {current_hyperparams.get('epsilon', 1.0):.4f}
- Training steps completed: {current_hyperparams.get('steps', 0)}
- Replay buffer: {current_hyperparams.get('memory_size', 0)} experiences

OPTIMIZATION HISTORY (last 5 iterations):
{history_summary}

OPTIMIZATION OBJECTIVE:
Maximize the composite score = 0.6 * win_rate + 0.4 * normalized_reward
Target: score > {PERFORMANCE_THRESHOLD}

CONSTRAINTS:
- learning_rate: [0.00001, 0.01]
- gamma: [0.80, 0.999]
- epsilon_decay: [0.990, 0.9999]
- epsilon_min: [0.01, 0.20]

TASK:
1. Analyze the current performance and history
2. Suggest improved hyperparameter values
3. Predict the expected composite score after these changes
4. Explain your reasoning

Return ONLY a valid JSON object (no markdown, no preamble) with this exact structure:
{{
  "suggested_hyperparams": {{
    "lr": <float>,
    "gamma": <float>,
    "epsilon_decay": <float>,
    "epsilon_min": <float>
  }},
  "predicted_score": <float between 0 and 1>,
  "reasoning": "<brief explanation under 100 words>",
  "iteration_score": <float between 0 and 1>
}}"""


# ── Helper: compute composite performance score ────────────────────────────────
def compute_composite_score(mc_stats: dict, db_metrics: dict) -> float:
    win_rate      = mc_stats.get("win_rate",   0.0)
    avg_reward    = mc_stats.get("avg_reward", 0.0)
    norm_reward   = (avg_reward + 1.0) / 2.5  # map [-1, 1.5] → [0, 1]
    return round(0.6 * win_rate + 0.4 * norm_reward, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🤖 AI Optimizer")
st.caption(
    "Claude-powered self-improvement loop · auto-calibrates DQN hyperparameters · "
    "rolls back if performance degrades"
)
st.divider()

agent      = st.session_state.agent
db_metrics = get_model_metrics()

# ── Current metrics display ────────────────────────────────────────────────────
opt_tab, hist_tab = st.tabs(["🔄 Optimization Loop", "📊 Optimization History"])

with opt_tab:
    km1, km2, km3, km4 = st.columns(4)
    with km1:
        st.metric("Current Accuracy",  f"{db_metrics.get('accuracy', 0):.1f}%")
    with km2:
        st.metric("Predictions Total", db_metrics.get("total", 0))
    with km3:
        st.metric("Agent ε",           f"{agent.epsilon:.4f}")
    with km4:
        st.metric("Training Steps",    f"{agent.steps:,}")

    st.divider()

    # ── Configuration ──────────────────────────────────────────────────────────
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        max_iterations = st.slider(
            "Max iterations", min_value=1, max_value=10, value=5, step=1, key="sl_max_iter"
        )
    with cfg2:
        mc_sims = st.slider(
            "MC sims per iteration", min_value=500, max_value=2000, value=1000, step=250,
            key="sl_mc_sims"
        )
    with cfg3:
        auto_save = st.checkbox("Auto-save best model", value=True, key="cb_auto_save")

    st.divider()

    # ── Start / Stop controls ──────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        start_btn = st.button(
            "🚀 Start Optimization Loop", key="btn_opt_start",
            disabled=st.session_state.opt_running,
            use_container_width=True,
        )
    with btn_col2:
        if st.button("⏹ Stop", key="btn_opt_stop", use_container_width=True):
            st.session_state.stop_flag = True
            st.toast("Stop requested — will halt after current iteration.", icon="⏹")
    with btn_col3:
        if st.button("↩ Rollback to Previous", key="btn_rollback",
                     disabled=st.session_state.previous_agent_state is None,
                     use_container_width=True):
            prev = st.session_state.previous_agent_state
            if prev:
                agent.update_hyperparams(
                    lr            = prev["lr"],
                    gamma         = prev["gamma"],
                    epsilon_decay = prev["epsilon_decay"],
                    epsilon_min   = prev["epsilon_min"],
                )
                st.session_state.agent = agent
                st.toast("Rolled back to previous hyperparameters.", icon="↩")
                st.rerun()

    # ── Main optimization loop ─────────────────────────────────────────────────
    if start_btn:
        st.session_state.opt_running    = True
        st.session_state.stop_flag      = False
        st.session_state.iteration_log  = []
        st.session_state.optimizer_scores = []

        progress_bar    = st.progress(0.0, text="Starting optimization loop…")
        status_area     = st.empty()
        log_placeholder = st.empty()

        for iteration in range(1, max_iterations + 1):
            if st.session_state.stop_flag:
                status_area.warning(f"⏹ Loop stopped at iteration {iteration - 1}.")
                break

            progress_bar.progress(
                iteration / max_iterations,
                text=f"Iteration {iteration}/{max_iterations}",
            )
            status_area.info(f"🔄 Iteration {iteration}: running Monte Carlo simulation…")

            # ── Step 1: evaluate current performance ──────────────────────────
            mc_stats   = run_monte_carlo(agent, n_simulations=mc_sims, seed=iteration * 13)
            cur_score  = compute_composite_score(mc_stats, db_metrics)

            # ── Step 2: save previous state for rollback ───────────────────────
            st.session_state.previous_agent_state = {
                "lr":            agent.lr,
                "gamma":         agent.gamma,
                "epsilon_decay": agent.epsilon_decay,
                "epsilon_min":   agent.epsilon_min,
            }

            # ── Step 3: ask Claude for better hyperparams ──────────────────────
            status_area.info(f"🤖 Iteration {iteration}: consulting Claude for hyperparameter suggestions…")
            current_hp = {
                **agent.get_state_dict(),
                "lr": agent.lr,
            }

            try:
                response = client.messages.create(
                    model       = MODEL_NAME,
                    max_tokens  = MAX_TOKENS_OPTIMIZER,
                    temperature = 0.3,
                    system      = (
                        "You are a machine learning optimization expert specializing in "
                        "reinforcement learning for sports prediction. Return ONLY valid JSON."
                    ),
                    messages    = [
                        {
                            "role": "user",
                            "content": build_optimizer_prompt(
                                current_hp,
                                cur_score,
                                st.session_state.iteration_log,
                                mc_stats,
                                db_metrics,
                            ),
                        }
                    ],
                )
                raw_text = response.content[0].text.strip()

                try:
                    suggestion = json.loads(raw_text)
                except json.JSONDecodeError:
                    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
                    suggestion = json.loads(cleaned)

                suggested_hp  = suggestion.get("suggested_hyperparams", {})
                predicted_sc  = float(suggestion.get("predicted_score",  cur_score))
                reasoning     = str(suggestion.get("reasoning",           "No reasoning provided"))
                iter_score    = float(suggestion.get("iteration_score",   cur_score))

            except Exception as exc:
                st.warning(f"Claude call failed (iter {iteration}): {exc} — keeping current params")
                suggested_hp  = {}
                predicted_sc  = cur_score
                reasoning     = f"Claude API error: {exc}"
                iter_score    = cur_score

            # ── Step 4: apply suggestions ──────────────────────────────────────
            if suggested_hp:
                agent.update_hyperparams(
                    lr            = suggested_hp.get("lr"),
                    gamma         = suggested_hp.get("gamma"),
                    epsilon_decay = suggested_hp.get("epsilon_decay"),
                    epsilon_min   = suggested_hp.get("epsilon_min"),
                )

            # ── Step 5: re-evaluate after applying ────────────────────────────
            mc_after   = run_monte_carlo(agent, n_simulations=mc_sims, seed=(iteration + 100) * 7)
            new_score  = compute_composite_score(mc_after, db_metrics)

            # ── Step 6: rollback if degraded ──────────────────────────────────
            if new_score < cur_score - 0.02:
                prev = st.session_state.previous_agent_state
                agent.update_hyperparams(
                    lr            = prev["lr"],
                    gamma         = prev["gamma"],
                    epsilon_decay = prev["epsilon_decay"],
                    epsilon_min   = prev["epsilon_min"],
                )
                rollback_note = f"⚠️ Rolled back (score dropped {cur_score:.4f} → {new_score:.4f})"
                new_score     = cur_score
            else:
                rollback_note = ""

            # ── Step 7: track best ─────────────────────────────────────────────
            if new_score > st.session_state.best_score:
                st.session_state.best_score      = new_score
                st.session_state.best_hyperparams = agent.get_state_dict()
                if auto_save:
                    try:
                        agent.save(Path("dqn_best_optimizer.keras"))
                    except Exception as exc:
                        pass

            # ── Step 8: log ────────────────────────────────────────────────────
            log_entry = {
                "iteration":    iteration,
                "score_before": round(cur_score,  4),
                "score_after":  round(new_score,  4),
                "score_delta":  round(new_score - cur_score, 4),
                "hyperparams":  dict(suggested_hp) or {"no_change": True},
                "reasoning":    reasoning,
                "mc_win_rate":  round(mc_after["win_rate"], 4),
                "rollback":     bool(rollback_note),
            }
            st.session_state.iteration_log.append(log_entry)
            st.session_state.optimizer_scores.append(new_score)

            save_optimizer_run(
                iteration   = iteration,
                hyperparams = json.dumps(suggested_hp),
                score       = new_score,
                notes       = f"{reasoning}  {rollback_note}",
            )

            # Update display
            with log_placeholder.container():
                with st.expander(
                    f"📋 Iteration Log ({len(st.session_state.iteration_log)} entries)",
                    expanded=True,
                ):
                    for e in reversed(st.session_state.iteration_log):
                        delta_color = "🟢" if e["score_delta"] >= 0 else "🔴"
                        rb_icon     = " ↩" if e["rollback"] else ""
                        st.markdown(
                            f"**Iter {e['iteration']}** {delta_color} "
                            f"score: {e['score_before']:.4f} → {e['score_after']:.4f} "
                            f"(Δ{e['score_delta']:+.4f}) "
                            f"win_rate={e['mc_win_rate']*100:.1f}%{rb_icon}"
                        )
                        st.caption(f"Reasoning: {e['reasoning']}")
                        st.markdown("---")

        # ── Loop complete ──────────────────────────────────────────────────────
        progress_bar.progress(1.0, text="Optimization complete!")
        st.session_state.opt_running = False
        st.session_state.agent       = agent
        agent.save()

        status_area.success(
            f"✅ Optimization complete — best score: {st.session_state.best_score:.4f}  "
            f"({max_iterations} iterations)"
        )
        st.toast("Optimization loop finished!", icon="🎉")

    # ── Score history chart ────────────────────────────────────────────────────
    if st.session_state.optimizer_scores:
        st.subheader("Composite Score per Iteration")
        st.line_chart(
            pd.DataFrame(
                {"Composite Score": st.session_state.optimizer_scores},
                index=list(range(1, len(st.session_state.optimizer_scores) + 1)),
            ),
            use_container_width=True,
            color="#6c63ff",
        )

    # ── Before/after diff ─────────────────────────────────────────────────────
    if (
        st.session_state.previous_agent_state
        and st.session_state.best_hyperparams
    ):
        st.subheader("Before / After Comparison")
        before_lines = json.dumps(st.session_state.previous_agent_state, indent=2).splitlines(keepends=True)
        after_lines  = json.dumps(
            {k: v for k, v in st.session_state.best_hyperparams.items()
             if k in ("lr", "gamma", "epsilon_decay", "epsilon_min")},
            indent=2
        ).splitlines(keepends=True)
        diff = list(difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="best"))
        if diff:
            st.code("".join(diff), language="diff")
        else:
            st.info("No parameter changes during this optimization run.")

    # ── Full iteration log ─────────────────────────────────────────────────────
    if st.session_state.iteration_log:
        with st.expander("📋 Full Iteration Log", expanded=False):
            for e in st.session_state.iteration_log:
                delta_color = "green" if e["score_delta"] >= 0 else "red"
                st.markdown(
                    f"**Iteration {e['iteration']}**  ·  "
                    f":{delta_color}[score Δ {e['score_delta']:+.4f}]  ·  "
                    f"win_rate={e['mc_win_rate']*100:.1f}%"
                )
                st.json(e["hyperparams"])
                st.caption(e["reasoning"])
                st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Optimization History
# ══════════════════════════════════════════════════════════════════════════════
with hist_tab:
    st.subheader("All Optimization Runs")

    opt_hist = get_optimizer_history(100)
    if not opt_hist:
        st.info("No optimization runs yet. Start the loop in **🔄 Optimization Loop** tab.")
    else:
        df_opt = pd.DataFrame(opt_hist)
        df_opt["timestamp"] = pd.to_datetime(df_opt["timestamp"]).dt.strftime("%m-%d %H:%M")

        st.subheader("Score History (all-time)")
        st.line_chart(
            df_opt.sort_values("id").set_index("id")["score"],
            use_container_width=True,
            color="#00d4ff",
        )

        st.dataframe(
            df_opt[["timestamp", "iteration", "score", "notes"]].rename(columns={
                "timestamp": "Time",
                "iteration": "Iter",
                "score":     "Score",
                "notes":     "Notes",
            }),
            hide_index=True,
            use_container_width=True,
        )

        # Stats
        hs1, hs2, hs3 = st.columns(3)
        with hs1:
            st.metric("Best Score",  f"{df_opt['score'].max():.4f}")
        with hs2:
            st.metric("Avg Score",   f"{df_opt['score'].mean():.4f}")
        with hs3:
            st.metric("Total Runs",  len(df_opt))

        # Download
        csv = df_opt.to_csv(index=False)
        st.download_button(
            "⬇️ Download optimization history CSV",
            data=csv,
            file_name=f"optimizer_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="btn_dl_opt",
        )
