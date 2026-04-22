from __future__ import annotations

import os
import time
import logging
from datetime import date, timedelta

import requests

logger = logging.getLogger(__name__)

SPORTRADAR_API_KEY = os.environ.get(
    "SPORTRADAR_API_KEY", "GjLrUDGx4IOu6AhELBZUlnZD1R23jfOrcN21EcQ7"
)
SR_BASE    = "https://api.sportradar.com/nba/trial/v8/en"
SR_HEADERS = {"accept": "application/json"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def sr_get(endpoint: str, retries: int = 3) -> dict:
    url = f"{SR_BASE}/{endpoint}?api_key={SPORTRADAR_API_KEY}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=SR_HEADERS, timeout=12)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"SR 429 — sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code in (403, 401):
                logger.error("SR auth error — check API key")
                return {}
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        except requests.Timeout:
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.error(f"SR request error ({attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return {}


def fetch_sr_todays_schedule() -> list[dict]:
    today = date.today().strftime("%Y/%m/%d")
    data  = sr_get(f"games/{today}/schedule.json")
    return _parse_schedule(data)


def fetch_sr_schedule_range(days_back: int = 7) -> list[dict]:
    games = []
    for d in range(days_back, 0, -1):
        day = (date.today() - timedelta(days=d)).strftime("%Y/%m/%d")
        data = sr_get(f"games/{day}/schedule.json")
        parsed = _parse_schedule(data)
        games.extend([g for g in parsed if g["status"] in ("closed", "complete")])
        time.sleep(1.1)  # respect rate limit
    return games


def _parse_schedule(data: dict) -> list[dict]:
    games = []
    for g in data.get("games", []):
        games.append({
            "game_id":   str(g.get("id", "")),
            "home_team": g.get("home", {}).get("name",  "Unknown"),
            "away_team": g.get("away", {}).get("name",  "Unknown"),
            "home_abbr": g.get("home", {}).get("alias", ""),
            "away_abbr": g.get("away", {}).get("alias", ""),
            "home_id":   str(g.get("home", {}).get("id", "")),
            "away_id":   str(g.get("away", {}).get("id", "")),
            "status":    g.get("status", ""),
            "source":    "sportradar",
        })
    return games


def fetch_sr_boxscore(game_id: str) -> dict:
    data = sr_get(f"games/{game_id}/boxscore.json")
    home = data.get("home", {}).get("statistics", {})
    away = data.get("away", {}).get("statistics", {})
    return {
        "home_pts": _safe_float(home.get("points")),
        "home_ast": _safe_float(home.get("assists")),
        "home_reb": _safe_float(home.get("rebounds")),
        "home_to":  _safe_float(home.get("turnovers")),
        "away_pts": _safe_float(away.get("points")),
        "away_ast": _safe_float(away.get("assists")),
        "away_reb": _safe_float(away.get("rebounds")),
        "away_to":  _safe_float(away.get("turnovers")),
    }


def fetch_sr_standings() -> dict[str, float]:
    data    = sr_get("seasons/2024/REG/standings.json")
    win_pct: dict[str, float] = {}
    for conf in data.get("conferences", []):
        for div in conf.get("divisions", []):
            for team in div.get("teams", []):
                abbr = team.get("alias", "")
                w    = int(team.get("wins",   0) or 0)
                l    = int(team.get("losses", 0) or 0)
                if abbr:
                    win_pct[abbr] = (w / (w + l)) if (w + l) > 0 else 0.5
    return win_pct


def fetch_sr_injuries(team_id: str) -> int:
    if not team_id:
        return 0
    data = sr_get(f"teams/{team_id}/injuries.json")
    return len(data.get("players", []))
