import os
import time
import logging
from datetime import date

import requests

logger = logging.getLogger(__name__)

SPORTRADAR_API_KEY = os.environ.get(
    "SPORTRADAR_API_KEY", "GjLrUDGx4IOu6AhELBZUlnZD1R23jfOrcN21EcQ7"
)
SPORTRADAR_BASE = "https://api.sportradar.com/nba/trial/v8/en"
SR_HEADERS      = {"accept": "application/json"}


def sr_get(endpoint: str, max_retries: int = 3) -> dict:
    """
    Generic SportRadar GET with exponential back-off on 429s.
    Returns empty dict on all unrecoverable errors so callers can safely use .get().
    """
    url = f"{SPORTRADAR_BASE}/{endpoint}?api_key={SPORTRADAR_API_KEY}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=SR_HEADERS, timeout=10)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"SportRadar 429 — sleeping {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                logger.error("SportRadar 403: invalid key or unauthorized endpoint")
                return {}
            if resp.status_code == 404:
                logger.warning(f"SportRadar 404: {endpoint}")
                return {}
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            logger.warning(f"SportRadar timeout (attempt {attempt+1}/{max_retries}): {endpoint}")
            time.sleep(2 ** attempt)
        except requests.RequestException as exc:
            logger.error(f"SportRadar error (attempt {attempt+1}/{max_retries}): {exc}")
            time.sleep(2 ** attempt)
    return {}


def fetch_sr_todays_schedule() -> list[dict]:
    """Return today's NBA game stubs from SportRadar."""
    today = date.today().strftime("%Y/%m/%d")
    data  = sr_get(f"games/{today}/schedule.json")
    games = []
    for game in data.get("games", []):
        games.append({
            "game_id":   str(game.get("id",                       "")),
            "home_team": game.get("home", {}).get("name",       "Unknown"),
            "away_team": game.get("away", {}).get("name",       "Unknown"),
            "home_id":   str(game.get("home", {}).get("id",     "")),
            "away_id":   str(game.get("away", {}).get("id",     "")),
            "home_abbr": game.get("home", {}).get("alias",      ""),
            "away_abbr": game.get("away", {}).get("alias",      ""),
            "status":    game.get("status",                      ""),
            "source":    "sportradar",
        })
    logger.info(f"SportRadar schedule: {len(games)} games today")
    return games


def fetch_sr_boxscore(game_id: str) -> dict:
    """
    Return team-level boxscore for a completed or in-progress game.
    Defaults to 0 for any missing stat.
    """
    data = sr_get(f"games/{game_id}/boxscore.json")
    home = data.get("home", {}).get("statistics", {})
    away = data.get("away", {}).get("statistics", {})
    return {
        "home_pts": float(home.get("points",    0) or 0),
        "home_ast": float(home.get("assists",   0) or 0),
        "home_reb": float(home.get("rebounds",  0) or 0),
        "home_to":  float(home.get("turnovers", 0) or 0),
        "away_pts": float(away.get("points",    0) or 0),
        "away_ast": float(away.get("assists",   0) or 0),
        "away_reb": float(away.get("rebounds",  0) or 0),
        "away_to":  float(away.get("turnovers", 0) or 0),
    }


def fetch_sr_standings() -> dict[str, float]:
    """
    Return a mapping {team_alias: win_percentage} from the current season standings.
    Win percentage is in [0.0, 1.0].
    """
    data    = sr_get("seasons/2024/REG/standings.json")
    win_pct: dict[str, float] = {}
    for conf in data.get("conferences", []):
        for div in conf.get("divisions", []):
            for team in div.get("teams", []):
                abbr   = team.get("alias", "")
                wins   = int(team.get("wins",   0) or 0)
                losses = int(team.get("losses", 0) or 0)
                total  = wins + losses
                if abbr:
                    win_pct[abbr] = (wins / total) if total > 0 else 0.5
    return win_pct


def fetch_sr_injuries(team_id: str) -> int:
    """Return the number of injured players reported for a team."""
    if not team_id:
        return 0
    data = sr_get(f"teams/{team_id}/injuries.json")
    return len(data.get("players", []))


def fetch_sr_game_probabilities(game_id: str) -> dict[str, float]:
    """Return pre-game win probabilities from SportRadar."""
    data  = sr_get(f"games/{game_id}/probabilities.json")
    probs = data.get("probabilities", {})
    return {
        "home_win_prob": float(probs.get("home_team_probability", 0.5) or 0.5),
        "away_win_prob": float(probs.get("away_team_probability", 0.5) or 0.5),
    }


def fetch_sr_recent_results(days_back: int = 3) -> list[dict]:
    """
    Fallback: fetch results from the past N days when no games are scheduled today.
    Returns a flat list of completed game stubs.
    """
    from datetime import timedelta
    today  = date.today()
    result = []
    for delta in range(1, days_back + 1):
        day_str = (today - timedelta(days=delta)).strftime("%Y/%m/%d")
        data    = sr_get(f"games/{day_str}/schedule.json")
        for game in data.get("games", []):
            if game.get("status") in ("closed", "complete"):
                result.append({
                    "game_id":   str(game.get("id",                   "")),
                    "home_team": game.get("home", {}).get("name",   "Unknown"),
                    "away_team": game.get("away", {}).get("name",   "Unknown"),
                    "home_id":   str(game.get("home", {}).get("id", "")),
                    "away_id":   str(game.get("away", {}).get("id", "")),
                    "home_abbr": game.get("home", {}).get("alias",  ""),
                    "away_abbr": game.get("away", {}).get("alias",  ""),
                    "status":    game.get("status",                  ""),
                    "source":    "sportradar",
                })
    return result
