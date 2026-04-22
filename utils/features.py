import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Per-feature normalization denominators ────────────────────────────────────
FEATURE_MAX = np.array(
    [130.0, 35.0, 60.0, 25.0,   # home pts/ast/reb/to
     130.0, 35.0, 60.0, 25.0,   # away pts/ast/reb/to
     1.0,   1.0,                 # home/away win_pct
     30.0,                       # |odds_spread|
     100.0],                     # bpi_delta  (range −100 → +100)
    dtype=np.float32,
)

FEATURE_NAMES = [
    "home_pts", "home_ast", "home_reb", "home_to",
    "away_pts", "away_ast", "away_reb", "away_to",
    "home_win_pct", "away_win_pct",
    "odds_spread", "bpi_delta",
]

# ── NBA-average defaults used when API data is unavailable ───────────────────
_DEFAULT_BOX = {
    "home_pts": 110.0, "home_ast": 25.0,
    "home_reb":  44.0, "home_to":  14.0,
    "away_pts": 108.0, "away_ast": 24.0,
    "away_reb":  43.0, "away_to":  14.0,
}


def build_state(game: dict, standings: dict, bpi: dict) -> np.ndarray:
    """
    Construct a normalized 12-D float32 state vector for the DQN.

    Feature assembly priority:
      pts/ast/reb/to  → SportRadar boxscore (if source='sportradar' and game complete)
                        else ESPN season averages for home/away team IDs
                        else NBA-average defaults
      win_pct         → standings dict (SR preferred, ESPN fallback keyed by abbr)
      odds_spread     → ESPN competition odds (absolute value)
      bpi_delta       → ESPN BPI home minus away (positive favours home)

    All 12 features are divided by FEATURE_MAX then clipped to [0, 1].
    """
    box = _get_box_stats(game)

    home_abbr = str(game.get("home_abbr", game.get("home_team", "")[:3])).upper()
    away_abbr = str(game.get("away_abbr", game.get("away_team", "")[:3])).upper()

    home_wp   = float(standings.get(home_abbr, 0.5))
    away_wp   = float(standings.get(away_abbr, 0.5))

    spread    = float(game.get("odds_spread", 0.0) or 0.0)
    bpi_h     = float(bpi.get(home_abbr, 50.0))
    bpi_a     = float(bpi.get(away_abbr, 50.0))
    bpi_delta = bpi_h - bpi_a

    raw = np.array(
        [
            box["home_pts"], box["home_ast"], box["home_reb"], box["home_to"],
            box["away_pts"], box["away_ast"], box["away_reb"], box["away_to"],
            home_wp, away_wp,
            abs(spread),
            bpi_delta,
        ],
        dtype=np.float32,
    )
    return np.clip(raw / FEATURE_MAX, 0.0, 1.0)


def _get_box_stats(game: dict) -> dict:
    """
    Try SportRadar boxscore first (completed SR games),
    then ESPN season averages, then NBA-average defaults.
    """
    from utils.sportradar import fetch_sr_boxscore
    from utils.espn_api import fetch_espn_team_stats

    if game.get("source") == "sportradar" and game.get("game_id"):
        try:
            box = fetch_sr_boxscore(game["game_id"])
            if box.get("home_pts", 0) > 0 or box.get("away_pts", 0) > 0:
                return box
        except Exception as exc:
            logger.warning(f"SR boxscore failed for {game.get('game_id')}: {exc}")

    home_id = game.get("home_id", "")
    away_id = game.get("away_id", "")
    if home_id or away_id:
        try:
            h = fetch_espn_team_stats(home_id)
            a = fetch_espn_team_stats(away_id)
            result = {
                "home_pts": h.get("pts", _DEFAULT_BOX["home_pts"]),
                "home_ast": h.get("ast", _DEFAULT_BOX["home_ast"]),
                "home_reb": h.get("reb", _DEFAULT_BOX["home_reb"]),
                "home_to":  h.get("to",  _DEFAULT_BOX["home_to"]),
                "away_pts": a.get("pts", _DEFAULT_BOX["away_pts"]),
                "away_ast": a.get("ast", _DEFAULT_BOX["away_ast"]),
                "away_reb": a.get("reb", _DEFAULT_BOX["away_reb"]),
                "away_to":  a.get("to",  _DEFAULT_BOX["away_to"]),
            }
            if result["home_pts"] > 0 or result["away_pts"] > 0:
                return result
        except Exception as exc:
            logger.warning(f"ESPN team stats failed: {exc}")

    return dict(_DEFAULT_BOX)


def get_feature_importance(agent, n_samples: int = 200) -> dict[str, float]:
    """
    Permutation-based feature importance:
    For each feature, shuffle its column across n_samples random states and
    measure the reduction in Q-value variance caused by losing that feature.

    Higher value → feature contributes more to the agent's decision boundary.
    Result is normalized to sum to 1.0.
    """
    rng         = np.random.default_rng(42)
    base_states = rng.uniform(0, 1, (n_samples, len(FEATURE_NAMES))).astype(np.float32)
    base_q      = agent.model.predict(base_states, verbose=0)
    base_var    = float(np.var(base_q))

    importance: dict[str, float] = {}
    for i, name in enumerate(FEATURE_NAMES):
        perturbed          = base_states.copy()
        perturbed[:, i]    = rng.permutation(perturbed[:, i])
        perm_q             = agent.model.predict(perturbed, verbose=0)
        importance[name]   = abs(base_var - float(np.var(perm_q)))

    total = sum(importance.values()) or 1.0
    return {k: round(v / total, 4) for k, v in importance.items()}


def run_monte_carlo(
    agent, n_simulations: int = 1000, seed: int = 42
) -> dict:
    """
    Simulate agent performance over n_simulations random game states.

    Each simulation:
      - Draws a random normalized state vector
      - Generates a synthetic true outcome via logistic model on bpi_delta + win_pct delta
      - Evaluates agent's greedy action and computes shaped reward

    Returns aggregated statistics used for performance validation.
    """
    rng     = np.random.default_rng(seed)
    rewards = np.zeros(n_simulations, dtype=np.float32)
    correct = np.zeros(n_simulations, dtype=np.int32)

    for i in range(n_simulations):
        state      = rng.uniform(0, 1, len(FEATURE_NAMES)).astype(np.float32)
        action, _  = agent.act_greedy(state)

        # Synthetic ground truth: logistic model on bpi_delta + win_pct advantage
        bpi_delta  = float(state[11]) * 200.0 - 100.0   # un-normalise
        wp_delta   = float(state[8]) - float(state[9])  # home_wp - away_wp
        logit      = 0.03 * bpi_delta + 2.0 * wp_delta
        home_prob  = 1.0 / (1.0 + np.exp(-logit))
        actual     = int(rng.random() < home_prob)

        pred_margin   = float(state[10]) * 30.0
        actual_margin = float(rng.normal(pred_margin, 8.0))

        reward       = agent.calculate_reward(action, actual, pred_margin, actual_margin)
        rewards[i]   = reward
        correct[i]   = int(action == actual)

    win_rate       = float(correct.mean())
    avg_reward     = float(rewards.mean())
    std_reward     = float(rewards.std())
    z95            = 1.96
    ci_half        = z95 * std_reward / np.sqrt(n_simulations)

    return {
        "n_simulations": n_simulations,
        "win_rate":      round(win_rate,   4),
        "avg_reward":    round(avg_reward, 4),
        "std_reward":    round(std_reward, 4),
        "confidence_95": round(ci_half,    4),
        "ci_low":        round(avg_reward - ci_half, 4),
        "ci_high":       round(avg_reward + ci_half, 4),
    }
