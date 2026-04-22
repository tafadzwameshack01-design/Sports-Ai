"""
AutonomousImprover: the core self-improvement engine.

How it works
─────────────
1. Every time a game result arrives, call `process_result()`.
2. Correct predictions → normal-priority PER entry.
3. Missed predictions  → HIGH-priority PER entry (5× normal weight).
4. When ANALYSIS_TRIGGER misses accumulate, call Claude to:
   - Identify systematic bias (over-predicts home? ignores BPI?)
   - Suggest targeted hyperparameter adjustments
   - Prescribe a focused retraining recipe
5. Apply the prescription and run N extra replay steps.
6. Store the "lesson learned" in SQLite for the UI to display.
7. Reset the pending miss buffer (keep last 2 for continuity).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from utils.dqn_agent import DQNAgent

logger = logging.getLogger(__name__)

ANALYSIS_TRIGGER  = 5    # Claude analysis after every N misses
RETRAIN_STEPS     = 100  # extra replay steps after each analysis
HIGH_PRIORITY     = 6.0  # PER priority multiplier for misses
CORRECT_PRIORITY  = 1.0


class AutonomousImprover:
    def __init__(self, agent: "DQNAgent", anthropic_client):
        self.agent          = agent
        self.client         = anthropic_client
        self.pending_misses: list[dict] = []
        self.total_misses   = 0
        self.total_correct  = 0
        self.lessons:       list[dict] = []
        self.last_analysis: dict | None = None

    # ── Public API ─────────────────────────────────────────────────────────────
    def process_result(
        self,
        state:     np.ndarray,
        action:    int,
        actual:    int,
        game_info: dict,
        reward:    float,
    ) -> dict:
        """
        Register a completed game result.
        Returns a result dict consumed by the UI.
        """
        correct = action == actual
        result  = {
            "correct":    correct,
            "action":     action,
            "actual":     actual,
            "reward":     reward,
            "game":       game_info,
            "analysis":   None,
            "lesson":     None,
            "retrained":  False,
        }

        if correct:
            self.total_correct += 1
            self.agent.remember(state, action, reward, state, done=True,
                                priority=CORRECT_PRIORITY)
        else:
            self.total_misses += 1
            self.agent.remember(state, action, reward, state, done=True,
                                priority=HIGH_PRIORITY)
            miss = {
                "state":     state.tolist(),
                "action":    action,
                "actual":    actual,
                "game":      game_info,
                "reward":    reward,
                "timestamp": datetime.now().isoformat(),
            }
            self.pending_misses.append(miss)
            result["retrained"] = True

            if len(self.pending_misses) >= ANALYSIS_TRIGGER:
                analysis = self._run_full_analysis()
                result["analysis"] = analysis
                if analysis:
                    result["lesson"] = analysis.get("lesson", "")
                    lesson_rec = {
                        "timestamp": datetime.now().isoformat(),
                        "total_misses": self.total_misses,
                        "total_correct": self.total_correct,
                        **analysis,
                    }
                    self.lessons.append(lesson_rec)
                    self.last_analysis = lesson_rec
                # Keep only last 2 misses for continuity
                self.pending_misses = self.pending_misses[-2:]

        # Always run a few replay steps on each result
        self._run_replay(steps=20)
        return result

    def run_background_retrain(self, steps: int = 50) -> float | None:
        """
        Call periodically (e.g. after fetching new data) to keep the
        agent improving even when no new game results arrive.
        """
        losses = []
        for _ in range(steps):
            loss = self.agent.replay(batch_size=64)
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else None

    def accuracy_rate(self) -> float:
        total = self.total_misses + self.total_correct
        return (self.total_correct / total) if total > 0 else 0.0

    # ── Private helpers ────────────────────────────────────────────────────────
    def _run_replay(self, steps: int = 20) -> None:
        for _ in range(steps):
            self.agent.replay(batch_size=64)

    def _run_full_analysis(self) -> dict | None:
        try:
            prompt   = self._build_analysis_prompt()
            response = self.client.messages.create(
                model       = "claude-sonnet-4-20250514",
                max_tokens  = 1024,
                temperature = 0.25,
                system      = (
                    "You are a reinforcement-learning expert specializing in sports prediction. "
                    "Analyze missed predictions, identify systematic bias, and prescribe fixes. "
                    "Return ONLY valid JSON — no markdown, no preamble."
                ),
                messages    = [{"role": "user", "content": prompt}],
            )
            raw  = response.content[0].text.strip()
            data = json.loads(raw.replace("```json", "").replace("```", "").strip())

            # Apply hyperparameter prescription if provided
            hp = data.get("hyperparams", {})
            if hp:
                self.agent.update_hyperparams(
                    lr            = hp.get("lr"),
                    gamma         = hp.get("gamma"),
                    epsilon_decay = hp.get("epsilon_decay"),
                    epsilon_min   = hp.get("epsilon_min"),
                )

            # Run focused retraining
            for _ in range(RETRAIN_STEPS):
                self.agent.replay(batch_size=64)

            logger.info(f"Self-improvement cycle complete. Lesson: {data.get('lesson','')}")
            return data

        except Exception as exc:
            logger.warning(f"Self-improvement analysis failed: {exc}")
            return {
                "lesson":       "Claude analysis unavailable — continuing with priority replay only.",
                "hyperparams":  {},
                "bias_detected": "unknown",
                "predicted_improvement": 0.0,
            }

    def _build_analysis_prompt(self) -> str:
        acc_rate = self.accuracy_rate()
        misses   = self.pending_misses

        # Summarize the missed-prediction states
        state_summaries = []
        for m in misses:
            s = m["state"]
            state_summaries.append({
                "game":          m["game"].get("matchup", "unknown"),
                "predicted":     "home" if m["action"] == 1 else "away",
                "actual":        "home" if m["actual"] == 1 else "away",
                "home_win_pct":  round(s[8],  3),
                "away_win_pct":  round(s[9],  3),
                "odds_spread":   round(s[10], 3),
                "bpi_delta":     round(s[11], 3),
                "home_pts_norm": round(s[0],  3),
                "away_pts_norm": round(s[4],  3),
            })

        history_summary = json.dumps(state_summaries, indent=2)

        return f"""Analyze {len(misses)} consecutive missed NBA game predictions for a DQN agent.

CURRENT AGENT STATE:
- Accuracy rate   : {acc_rate*100:.1f}%
- Total misses    : {self.total_misses}
- Total correct   : {self.total_correct}
- Epsilon         : {self.agent.epsilon:.4f}
- Gamma           : {self.agent.gamma:.4f}
- LR              : {self.agent.lr}
- Buffer size     : {len(self.agent.replay_buffer)}

MISSED PREDICTIONS (state features normalized 0–1):
{history_summary}

STATE FEATURE LEGEND:
  [0-3]  home pts/ast/reb/to  (higher = stronger offense/defense)
  [4-7]  away pts/ast/reb/to
  [8]    home win% (0-1)
  [9]    away win% (0-1)
  [10]   |odds_spread| (0-1, 0=even, 1=30pt spread)
  [11]   BPI_delta home-away (0=−100, 0.5=even, 1=+100)

TASK:
1. Identify the dominant failure pattern (e.g. over-predicts home team, ignores BPI delta, etc.)
2. Suggest specific hyperparameter adjustments to fix the bias
3. Estimate expected improvement

Return ONLY this JSON structure:
{{
  "bias_detected": "<one-line description of the main bias>",
  "lesson": "<one actionable sentence — what the agent should learn>",
  "hyperparams": {{
    "lr": <float or null if no change needed>,
    "gamma": <float or null>,
    "epsilon_decay": <float or null>,
    "epsilon_min": <float or null>
  }},
  "predicted_improvement": <float 0-0.2, expected accuracy gain>,
  "reasoning": "<2-3 sentence explanation>"
}}"""
