from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "home_pts", "home_ast", "home_reb", "home_to",
    "away_pts", "away_ast", "away_reb", "away_to",
    "home_win_pct", "away_win_pct",
    "odds_spread",  "bpi_delta",
]

FEATURE_MAX = np.array(
    [130., 35., 60., 25., 130., 35., 60., 25., 1., 1., 30., 100.],
    dtype=np.float32,
)

NBA_DEFAULTS = {
    "home_pts": 112., "home_ast": 26., "home_reb": 44., "home_to": 14.,
    "away_pts": 110., "away_ast": 25., "away_reb": 43., "away_to": 14.,
}


def build_state(game: dict, standings: dict, bpi: dict) -> np.ndarray:
    box = _get_box(game)

    home_abbr = str(game.get("home_abbr", game.get("home_team", "")[:3])).upper()
    away_abbr = str(game.get("away_abbr", game.get("away_team", "")[:3])).upper()

    # Win% — prefer standings dict, fallback to game record embedded in ESPN payload
    home_wp  = float(standings.get(home_abbr,
                     game.get("home_win_pct", 0.5)) or 0.5)
    away_wp  = float(standings.get(away_abbr,
                     game.get("away_win_pct", 0.5)) or 0.5)

    spread   = float(game.get("odds_spread", 0.0) or 0.0)
    bpi_h    = float(bpi.get(home_abbr, 50.0))
    bpi_a    = float(bpi.get(away_abbr, 50.0))
    bpi_d    = bpi_h - bpi_a  # range ≈ −100 to +100

    raw = np.array([
        box["home_pts"], box["home_ast"], box["home_reb"], box["home_to"],
        box["away_pts"], box["away_ast"], box["away_reb"], box["away_to"],
        home_wp, away_wp, abs(spread), bpi_d + 50.0,  # shift bpi_d to ≥0
    ], dtype=np.float32)

    return np.clip(raw / FEATURE_MAX, 0.0, 1.0)


def _get_box(game: dict) -> dict:
    from utils.sportradar import fetch_sr_boxscore
    from utils.espn_api   import fetch_espn_team_stats

    if game.get("source") == "sportradar" and game.get("game_id"):
        try:
            box = fetch_sr_boxscore(game["game_id"])
            if box.get("home_pts", 0) > 0:
                return box
        except Exception as e:
            logger.debug(f"SR boxscore skipped: {e}")

    h_id = game.get("home_id", "")
    a_id = game.get("away_id", "")
    if h_id or a_id:
        try:
            h = fetch_espn_team_stats(h_id)
            a = fetch_espn_team_stats(a_id)
            result = {
                "home_pts": h.get("pts", NBA_DEFAULTS["home_pts"]),
                "home_ast": h.get("ast", NBA_DEFAULTS["home_ast"]),
                "home_reb": h.get("reb", NBA_DEFAULTS["home_reb"]),
                "home_to":  h.get("to",  NBA_DEFAULTS["home_to"]),
                "away_pts": a.get("pts", NBA_DEFAULTS["away_pts"]),
                "away_ast": a.get("ast", NBA_DEFAULTS["away_ast"]),
                "away_reb": a.get("reb", NBA_DEFAULTS["away_reb"]),
                "away_to":  a.get("to",  NBA_DEFAULTS["away_to"]),
            }
            if result["home_pts"] > 0:
                return result
        except Exception as e:
            logger.debug(f"ESPN team stats skipped: {e}")

    return dict(NBA_DEFAULTS)


def run_monte_carlo(agent, n_simulations: int = 1000, seed: int = 42) -> dict:
    """
    Simulate agent over n_simulations synthetic game states.
    Ground-truth: logistic model on bpi_delta + win_pct delta.
    """
    rng     = np.random.default_rng(seed)
    rewards = np.zeros(n_simulations, dtype=np.float32)
    correct = np.zeros(n_simulations, dtype=np.int32)

    for i in range(n_simulations):
        state       = rng.uniform(0, 1, len(FEATURE_NAMES)).astype(np.float32)
        action, _   = agent.act_greedy(state)

        bpi_d       = float(state[11]) * 100.0 - 50.0   # un-shift
        wp_d        = float(state[8])  - float(state[9])
        logit       = 0.04 * bpi_d + 2.0 * wp_d
        home_p      = 1.0 / (1.0 + np.exp(-logit))
        actual      = int(rng.random() < home_p)

        pm          = float(state[10]) * 30.0
        am          = float(rng.normal(pm, 8.0))
        reward      = agent.calculate_reward(action, actual, pm, am)
        rewards[i]  = reward
        correct[i]  = int(action == actual)

    wr    = float(correct.mean())
    avg_r = float(rewards.mean())
    std_r = float(rewards.std())
    ci    = 1.96 * std_r / np.sqrt(n_simulations)

    return {
        "n_simulations": n_simulations,
        "win_rate":      round(wr,    4),
        "avg_reward":    round(avg_r, 4),
        "std_reward":    round(std_r, 4),
        "confidence_95": round(ci,    4),
        "ci_low":        round(avg_r - ci, 4),
        "ci_high":       round(avg_r + ci, 4),
    }


def get_feature_importance(agent, n_samples: int = 300) -> dict[str, float]:
    rng        = np.random.default_rng(7)
    states     = rng.uniform(0, 1, (n_samples, len(FEATURE_NAMES))).astype(np.float32)
    base_q     = agent.online.predict(states)
    base_var   = float(np.var(base_q))
    importance = {}
    for i, name in enumerate(FEATURE_NAMES):
        p       = states.copy()
        p[:, i] = rng.permutation(p[:, i])
        pq      = agent.online.predict(p)
        importance[name] = abs(base_var - float(np.var(pq)))
    total = sum(importance.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}
