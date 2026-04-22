"""
ESPN Public API layer.
All endpoints are unauthenticated — no API key required.

The scoreboard parser handles the full 2025-26 API structure:
  - score as string ("112") → cast to float
  - nested odds providers
  - competitor records for win-pct extraction
  - live status detection
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

ESPN_SITE_V2 = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE_V2 = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _get(url: str, params: dict | None = None, timeout: int = 12) -> dict:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.Timeout:
        logger.warning(f"ESPN timeout: {url}")
    except requests.HTTPError as e:
        logger.warning(f"ESPN HTTP {e.response.status_code}: {url}")
    except Exception as e:
        logger.error(f"ESPN error ({url}): {e}")
    return {}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and val != "" else default
    except (ValueError, TypeError):
        return default


def _parse_record(competitor: dict) -> float:
    """Extract win% from a competitor's records list."""
    for rec in competitor.get("records", []):
        if rec.get("name") == "overall":
            summary = rec.get("summary", "")  # e.g. "46-36"
            parts   = summary.split("-")
            if len(parts) == 2:
                w = _safe_float(parts[0])
                l = _safe_float(parts[1])
                return w / (w + l) if (w + l) > 0 else 0.5
    return 0.5


def fetch_espn_scoreboard(date_str: str | None = None) -> list[dict]:
    """
    Return game list from ESPN scoreboard.
    date_str: 'YYYYMMDD' — defaults to today.
    """
    params = {}
    if date_str:
        params["dates"] = date_str

    data  = _get(f"{ESPN_SITE_V2}/scoreboard", params=params)
    games = []

    for event in data.get("events", []):
        try:
            comps = event.get("competitions", [{}])
            comp  = comps[0] if comps else {}

            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            if not home or not away:
                continue

            h_team = home.get("team", {})
            a_team = away.get("team",  {})

            # Scores are strings in the API
            h_score = _safe_float(home.get("score", "0"))
            a_score = _safe_float(away.get("score", "0"))

            # Odds — may be missing
            odds_list  = comp.get("odds", [])
            spread_raw = 0.0
            ou_raw     = 220.0
            if odds_list:
                o = odds_list[0]
                # spread field or parse from 'details' string e.g. "BOS -7.5"
                spread_raw = _safe_float(o.get("spread", 0.0))
                ou_raw     = _safe_float(o.get("overUnder", 220.0), default=220.0)
                if spread_raw == 0.0 and o.get("details"):
                    parts = o["details"].split()
                    for p in parts:
                        try:
                            spread_raw = abs(float(p))
                            break
                        except ValueError:
                            continue

            # Status
            status_obj  = event.get("status", {})
            status_type = status_obj.get("type", {})
            is_final    = status_type.get("completed", False)
            status_name = status_type.get("name", "")
            status_desc = status_type.get("description", "Scheduled")
            period      = _safe_float(status_obj.get("period", 0))
            clock       = status_obj.get("displayClock", "")

            # Win probabilities from predictor
            predictor = comp.get("predictor", {})
            h_wp = _safe_float(
                predictor.get("homeTeam", {}).get("gameProjection", 50.0), 50.0
            ) / 100.0
            a_wp = 1.0 - h_wp

            games.append({
                "game_id":       str(event.get("id", "")),
                "matchup":       event.get("shortName", f"{a_team.get('abbreviation','')} @ {h_team.get('abbreviation','')}"),
                "home_team":     h_team.get("displayName",   "Unknown"),
                "away_team":     a_team.get("displayName",   "Unknown"),
                "home_abbr":     h_team.get("abbreviation",  "???"),
                "away_abbr":     a_team.get("abbreviation",  "???"),
                "home_id":       str(h_team.get("id", "")),
                "away_id":       str(a_team.get("id", "")),
                "home_color":    h_team.get("color", "555555"),
                "away_color":    a_team.get("color", "555555"),
                "home_score":    h_score,
                "away_score":    a_score,
                "home_win_pct":  _parse_record(home),
                "away_win_pct":  _parse_record(away),
                "status":        status_name,
                "status_desc":   status_desc,
                "is_final":      is_final,
                "is_live":       status_type.get("state") == "in",
                "period":        int(period),
                "clock":         clock,
                "odds_spread":   spread_raw,
                "over_under":    ou_raw,
                "home_win_prob": h_wp,
                "away_win_prob": a_wp,
                "venue":         comp.get("venue", {}).get("fullName", ""),
                "source":        "espn",
            })
        except Exception as e:
            logger.warning(f"ESPN scoreboard parse error for event {event.get('id','?')}: {e}")
            continue

    logger.info(f"ESPN scoreboard: {len(games)} games")
    return games


def fetch_espn_scoreboard_range(days_back: int = 7) -> list[dict]:
    """Fetch scoreboard for the past N days — used to warm up the training buffer."""
    all_games = []
    for d in range(days_back, 0, -1):
        date_str = (datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
        games    = fetch_espn_scoreboard(date_str)
        # Only include completed games for training
        completed = [g for g in games if g["is_final"]]
        all_games.extend(completed)
    logger.info(f"Historical ESPN games (last {days_back} days): {len(all_games)}")
    return all_games


def fetch_espn_standings() -> dict[str, float]:
    """Return {team_abbr: win_pct} from ESPN standings."""
    data    = _get(f"{ESPN_SITE_V2}/standings")
    win_pct: dict[str, float] = {}

    for group in data.get("groups", []):
        entries = group.get("standings", {}).get("entries", [])
        for entry in entries:
            try:
                abbr  = entry.get("team", {}).get("abbreviation", "")
                stats = {s["name"]: _safe_float(s.get("value")) for s in entry.get("stats", [])}
                pct   = stats.get("winPercent", stats.get("wins", 0) / max(stats.get("wins", 0) + stats.get("losses", 1), 1))
                if abbr:
                    win_pct[abbr] = pct
            except Exception:
                continue

    logger.info(f"ESPN standings: {len(win_pct)} teams")
    return win_pct


def fetch_espn_team_stats(team_id: str) -> dict:
    """Return season-average stats for a team via ESPN Core API."""
    if not team_id:
        return {"pts": 110.0, "ast": 25.0, "reb": 44.0, "to": 14.0}

    data     = _get(f"{ESPN_CORE_V2}/teams/{team_id}/statistics")
    stat_map: dict[str, float] = {}
    for cat in data.get("splits", {}).get("categories", []):
        for stat in cat.get("stats", []):
            name = stat.get("name", "")
            val  = stat.get("value")
            if name and val is not None:
                stat_map[name] = _safe_float(val)

    return {
        "pts": stat_map.get("avgPoints",    110.0),
        "ast": stat_map.get("avgAssists",    25.0),
        "reb": stat_map.get("avgRebounds",   44.0),
        "to":  stat_map.get("avgTurnovers",  14.0),
        "fg_pct": stat_map.get("fieldGoalPct", 0.46),
        "3pt_pct": stat_map.get("threePointPct", 0.36),
        "ft_pct":  stat_map.get("freeThrowPct",  0.78),
    }


def fetch_espn_power_index() -> dict[str, float]:
    """Return ESPN Basketball Power Index keyed by team abbreviation."""
    data = _get(f"{ESPN_CORE_V2}/powerindex")
    bpi: dict[str, float] = {}
    for item in data.get("items", []):
        abbr = item.get("team", {}).get("abbreviation", "")
        val  = _safe_float(item.get("value", 50.0), 50.0)
        if abbr:
            bpi[abbr] = val
    logger.info(f"ESPN BPI: {len(bpi)} entries")
    return bpi


def fetch_espn_injuries(team_id: str) -> list[dict]:
    """Return injury report for a team."""
    if not team_id:
        return []
    data    = _get(f"{ESPN_CORE_V2}/teams/{team_id}/injuries")
    results = []
    for item in data.get("items", []):
        try:
            athlete = item.get("athlete", {})
            results.append({
                "name":   athlete.get("displayName", "Unknown"),
                "status": item.get("status", ""),
                "detail": item.get("shortComment", ""),
            })
        except Exception:
            continue
    return results


def fetch_espn_teams() -> list[dict]:
    """Return all 30 NBA team stubs."""
    data  = _get(f"{ESPN_SITE_V2}/teams")
    teams = []
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for item in league.get("teams", []):
                t = item.get("team", {})
                teams.append({
                    "id":          str(t.get("id", "")),
                    "abbreviation": t.get("abbreviation", ""),
                    "displayName":  t.get("displayName", ""),
                    "location":     t.get("location", ""),
                    "color":        t.get("color", "555555"),
                })
    return teams


def fetch_espn_news(limit: int = 15) -> list[dict]:
    """Return recent ESPN NBA news headlines."""
    data     = _get(f"{ESPN_SITE_V2}/news", params={"limit": limit})
    articles = []
    for a in data.get("articles", [])[:limit]:
        articles.append({
            "headline":    a.get("headline",    ""),
            "description": a.get("description", ""),
            "published":   a.get("published",   ""),
            "url":         a.get("links", {}).get("web", {}).get("href", ""),
            "source":      a.get("byline", "ESPN"),
        })
    return articles


def fetch_espn_win_probability(game_id: str) -> dict:
    """Return win probability for a live or recent game."""
    data = _get(f"{ESPN_CORE_V2}/events/{game_id}/competitions/{game_id}/probabilities")
    items = data.get("items", [])
    if items:
        last = items[-1]
        return {
            "home_win_prob": _safe_float(last.get("homeTeam", {}).get("winPercentage", 50.0)) / 100.0,
            "away_win_prob": _safe_float(last.get("awayTeam", {}).get("winPercentage", 50.0)) / 100.0,
            "seconds_left":  _safe_float(last.get("secondsLeft", 0)),
        }
    return {"home_win_prob": 0.5, "away_win_prob": 0.5, "seconds_left": 0.0}
