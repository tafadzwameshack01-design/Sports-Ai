import os
import sys
import json
import difflib
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
    init_db, save_optimizer_run, get_optimizer_history, get_model_metrics,
    save_lesson, get_lessons,
)
from utils.features import run_monte_carlo

st.set_page_config(page_title="AI Optimizer · NBA AI", page_icon="🤖", layout="wide")

MODEL_NAME          = "claude-sonnet-4-20250514"
PERF_THRESHOLD      = 0.55

if "db_ready" not in st.session_state:
    init_db(); st.session_state.db_ready = True
if "agent" not in st.session_state:
    from utils.dqn_agent import DQNAgent
    a = DQNAgent(); a.load(); st.session_state.agent = a

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY","")
if not ANTHROPIC_KEY:
    st.error("ANTHROPIC_API_KEY missing"); st.stop()
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

for key, default in [
    ("stop_flag",       False),
    ("iteration_log",   []),
    ("opt_scores",      []),
    ("previous_hp",     None),
    ("best_hp",         None),
    ("best_score",      0.0),
    ("opt_running",     False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent      = st.session_state.agent
db_metrics = get_model_metrics()


def composite_score(mc: dict, dbm: dict) -> float:
    wr   = mc.get("win_rate",   0.0)
    ar   = mc.get("avg_reward", 0.0)
    nr   = (ar + 1.0) / 2.5
    return round(0.6 * wr + 0.4 * nr, 4)


def claude_optimize(agent, cur_score: float, history: list, mc: dict, dbm: dict) -> dict:
    history_txt = "\n".join(
        f"  iter {h['iteration']}: score={h['score_after']:.4f} params={json.dumps(h['hyperparams'])}"
        for h in history[-5:]
    ) or "  (first iteration)"

    prompt = f"""Optimize hyperparameters for a Double DQN NBA prediction agent.

CURRENT HYPERPARAMETERS:
{json.dumps(agent.get_state_dict(), indent=2)}

CURRENT PERFORMANCE:
- Composite score  : {cur_score:.4f}  (target > {PERF_THRESHOLD})
- MC win rate      : {mc.get('win_rate',0)*100:.1f}%
- Avg reward       : {mc.get('avg_reward',0):+.4f}
- 95% CI           : [{mc.get('ci_low',0):+.4f}, {mc.get('ci_high',0):+.4f}]
- Historical acc   : {dbm.get('accuracy',0):.1f}%
- Total predictions: {dbm.get('total',0)}

RECENT ITERATION HISTORY:
{history_txt}

CONSTRAINTS:
  lr: [0.00001, 0.01]
  gamma: [0.80, 0.999]
  epsilon_decay: [0.990, 0.9999]
  epsilon_min: [0.01, 0.20]

COMPOSITE SCORE FORMULA: 0.6 * win_rate + 0.4 * normalized_reward
(normalized_reward = (avg_reward + 1.0) / 2.5)

Analyze the history, identify what's holding performance back, and suggest improved values.

Return ONLY valid JSON — no markdown, no preamble:
{{
  "suggested_hyperparams": {{"lr": <float>, "gamma": <float>, "epsilon_decay": <float>, "epsilon_min": <float>}},
  "predicted_score": <float 0-1>,
  "iteration_score": <float 0-1>,
  "reasoning": "<2-3 sentences>",
  "bias_detected": "<one line — what pattern is hurting performance>"
}}"""

    resp = client.messages.create(
        model       = MODEL_NAME,
        max_tokens  = 1024,
        temperature = 0.25,
        system      = "You are a reinforcement-learning optimization expert. Return ONLY valid JSON.",
        messages    = [{"role":"user","content": prompt}],
    )
    raw  = resp.content[0].text.strip()
    return json.loads(raw.replace("```json","").replace("```","").strip())


# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 AI Optimizer")
st.caption("Claude-powered hyperparameter calibration · autonomous self-improvement loop · rollback support")
st.divider()

tab_loop, tab_lessons_tab, tab_hist = st.tabs(["🔄 Optimization Loop", "🧠 Lessons Log", "📜 History"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_loop:
    km1,km2,km3,km4 = st.columns(4)
    with km1: st.metric("Accuracy",       f"{db_metrics.get('accuracy',0):.1f}%")
    with km2: st.metric("Predictions",    db_metrics.get("total",0))
    with km3: st.metric("ε",              f"{agent.epsilon:.4f}")
    with km4: st.metric("Best Score",     f"{st.session_state.best_score:.4f}")

    st.divider()

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1: max_iters = st.slider("Max iterations",  1, 10, 5, key="sl_max_iter")
    with cfg2: mc_sims   = st.slider("MC sims / iter", 500, 2000, 1000, 250, key="sl_mc")
    with cfg3: auto_save = st.checkbox("Auto-save best model", value=True, key="cb_autosave")

    st.divider()

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        start = st.button("🚀 Start Optimization", key="btn_start",
                          disabled=st.session_state.opt_running,
                          use_container_width=True)
    with bc2:
        if st.button("⏹ Stop", key="btn_stop", use_container_width=True):
            st.session_state.stop_flag = True
            st.toast("Stop requested.", icon="⏹")
    with bc3:
        prev_hp = st.session_state.previous_hp
        if st.button("↩ Rollback", key="btn_rollback",
                     disabled=prev_hp is None,
                     use_container_width=True):
            agent.update_hyperparams(**{k:v for k,v in prev_hp.items()
                                        if k in ("lr","gamma","epsilon_decay","epsilon_min")})
            st.session_state.agent = agent
            st.toast("Rolled back.", icon="↩")
            st.rerun()

    if start:
        st.session_state.opt_running   = True
        st.session_state.stop_flag     = False
        st.session_state.iteration_log = []
        st.session_state.opt_scores    = []

        progress  = st.progress(0.0, text="Starting…")
        status    = st.empty()
        log_area  = st.empty()

        for it in range(1, max_iters + 1):
            if st.session_state.stop_flag:
                status.warning(f"Stopped at iteration {it-1}.")
                break

            progress.progress(it/max_iters, text=f"Iteration {it}/{max_iters}")
            status.info(f"🔄 Iter {it}: running Monte Carlo ({mc_sims:,} sims)…")

            mc_before = run_monte_carlo(agent, mc_sims, seed=it*13)
            score_before = composite_score(mc_before, db_metrics)

            st.session_state.previous_hp = {
                "lr":            agent.lr,
                "gamma":         agent.gamma,
                "epsilon_decay": agent.epsilon_decay,
                "epsilon_min":   agent.epsilon_min,
            }

            status.info(f"🤖 Iter {it}: consulting Claude…")
            try:
                suggestion    = claude_optimize(agent, score_before,
                                                st.session_state.iteration_log,
                                                mc_before, db_metrics)
                sug_hp        = suggestion.get("suggested_hyperparams", {})
                reasoning     = suggestion.get("reasoning","")
                bias_detected = suggestion.get("bias_detected","")
            except Exception as e:
                st.warning(f"Claude failed (iter {it}): {e}")
                sug_hp = {}; reasoning = f"Error: {e}"; bias_detected = ""

            if sug_hp:
                agent.update_hyperparams(
                    lr            = sug_hp.get("lr"),
                    gamma         = sug_hp.get("gamma"),
                    epsilon_decay = sug_hp.get("epsilon_decay"),
                    epsilon_min   = sug_hp.get("epsilon_min"),
                )

            status.info(f"🎲 Iter {it}: re-evaluating after changes…")
            mc_after     = run_monte_carlo(agent, mc_sims, seed=(it+100)*7)
            score_after  = composite_score(mc_after, db_metrics)

            # Rollback if degraded > 2%
            rollback_note = ""
            if score_after < score_before - 0.02:
                prev = st.session_state.previous_hp
                agent.update_hyperparams(**{k:v for k,v in prev.items()
                                           if k in ("lr","gamma","epsilon_decay","epsilon_min")})
                rollback_note = f"↩ Rolled back ({score_before:.4f}→{score_after:.4f})"
                score_after   = score_before

            # Track best
            if score_after > st.session_state.best_score:
                st.session_state.best_score = score_after
                st.session_state.best_hp    = agent.get_state_dict()
                if auto_save:
                    try: agent.save(Path("dqn_best_optimizer.npy"))
                    except Exception: pass

            entry = {
                "iteration":    it,
                "score_before": round(score_before, 4),
                "score_after":  round(score_after,  4),
                "score_delta":  round(score_after - score_before, 4),
                "hyperparams":  sug_hp or {"no_change": True},
                "reasoning":    reasoning,
                "bias_detected": bias_detected,
                "mc_win_rate":  round(mc_after["win_rate"], 4),
                "rollback":     bool(rollback_note),
            }
            st.session_state.iteration_log.append(entry)
            st.session_state.opt_scores.append(score_after)

            save_optimizer_run(
                iteration=it, hyperparams=json.dumps(sug_hp),
                score=score_after, notes=f"{reasoning}  {rollback_note}",
            )

            # Save lesson to DB
            if bias_detected:
                save_lesson({
                    "bias_detected":         bias_detected,
                    "lesson":                reasoning,
                    "reasoning":             reasoning,
                    "predicted_improvement": suggestion.get("predicted_score",0) - score_before,
                    "total_misses":          0,
                    "total_correct":         0,
                })

            with log_area.container():
                with st.expander(f"📋 Iteration Log ({it} entries)", expanded=True):
                    for e in reversed(st.session_state.iteration_log):
                        dc = "🟢" if e["score_delta"] >= 0 else "🔴"
                        rb = " ↩" if e["rollback"] else ""
                        st.markdown(
                            f"**Iter {e['iteration']}** {dc} "
                            f"{e['score_before']:.4f} → {e['score_after']:.4f} "
                            f"(Δ{e['score_delta']:+.4f}) "
                            f"win={e['mc_win_rate']*100:.1f}%{rb}"
                        )
                        if e.get("bias_detected"):
                            st.caption(f"🔍 {e['bias_detected']}")
                        st.caption(e['reasoning'])
                        st.markdown("---")

        progress.progress(1.0, text="Complete!")
        st.session_state.opt_running = False
        st.session_state.agent       = agent
        agent.save()
        status.success(
            f"✅ Optimization complete — best score: {st.session_state.best_score:.4f}"
        )
        st.toast("Optimization done!", icon="🎉")

    if st.session_state.opt_scores:
        st.subheader("Composite Score per Iteration")
        st.line_chart(
            pd.DataFrame(
                {"Score": st.session_state.opt_scores},
                index=list(range(1, len(st.session_state.opt_scores)+1)),
            ),
            use_container_width=True, color="#6c63ff",
        )

    if st.session_state.previous_hp and st.session_state.best_hp:
        st.subheader("Before / After Diff")
        before_lines = json.dumps(
            {k:v for k,v in st.session_state.previous_hp.items()
             if k in ("lr","gamma","epsilon_decay","epsilon_min")}, indent=2
        ).splitlines(keepends=True)
        after_lines  = json.dumps(
            {k:v for k,v in (st.session_state.best_hp or {}).items()
             if k in ("lr","gamma","epsilon_decay","epsilon_min","epsilon")}, indent=2
        ).splitlines(keepends=True)
        diff = list(difflib.unified_diff(before_lines, after_lines,
                                         fromfile="before", tofile="best"))
        if diff:
            st.code("".join(diff), language="diff")
        else:
            st.info("No parameter changes this run.")

# ══════════════════════════════════════════════════════════════════════════════
with tab_lessons_tab:
    st.subheader("All AI Lessons")
    lessons = get_lessons(50)
    if not lessons:
        st.info("Lessons appear here after the agent analyzes missed predictions (every 5 misses) or during optimizer runs.")
    else:
        lm1,lm2 = st.columns(2)
        with lm1: st.metric("Total Lessons", len(lessons))
        with lm2:
            avg_imp = sum(l.get("predicted_improvement",0) for l in lessons) / len(lessons)
            st.metric("Avg Expected Improvement", f"+{avg_imp*100:.2f}%")

        for l in lessons[:20]:
            with st.expander(f"🧠 {l['timestamp'][:16]}  ·  {l.get('bias_detected','')[:60]}"):
                st.markdown(f"**Lesson:** {l.get('lesson','')}")
                st.caption(f"Misses: {l.get('total_misses',0)}  ·  Correct: {l.get('total_correct',0)}")
                st.caption(f"Expected Δ: +{l.get('predicted_improvement',0)*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("All Optimization Runs")
    hist = get_optimizer_history(100)
    if not hist:
        st.info("No optimizer runs yet.")
    else:
        df_h = pd.DataFrame(hist)
        st.line_chart(df_h.set_index("id")["score"], use_container_width=True, color="#00d4ff")
        df_h["timestamp"] = pd.to_datetime(df_h["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            df_h[["timestamp","iteration","score","notes"]].rename(
                columns={"timestamp":"Time","iteration":"Iter","score":"Score","notes":"Notes"}
            ),
            hide_index=True, use_container_width=True,
        )
        hs1,hs2,hs3 = st.columns(3)
        with hs1: st.metric("Best Score", f"{df_h['score'].max():.4f}")
        with hs2: st.metric("Avg Score",  f"{df_h['score'].mean():.4f}")
        with hs3: st.metric("Runs",       len(df_h))
        st.download_button(
            "⬇️ Download history CSV",
            data=df_h.to_csv(index=False),
            file_name=f"optimizer_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", key="dl_opt",
        )
