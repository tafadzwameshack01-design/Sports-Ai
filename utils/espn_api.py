import logging

import requests

logger = logging.getLogger(__name__)

ESPN_CORE_V2 = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ESPN_SITE_V2 = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_HEADERS = {
    "accept":     "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; SportsAI/1.0; +https://github.com)",
}


def espn_get(url: str, params: dict | None = None, timeout: int = 10) -> dict:
    """
    Generic ESPN GET.  Returns empty dict on any failure so callers can safely
    use .get() without extra guards.
    """
    try:
        resp = requests.get(url, headers=ESPN_HEADERS, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        logger.warning(f"ESPN timeout: {url}")
    except requests.HTTPError as exc:
        logger.warning(f"ESPN HTTP {exc.response.status_code}: {url}")
    except requests.RequestException as exc:
        logger.error(f"ESPN request error: {exc}")
    return {}


def fetch_espn_scoreboard() -> list[dict]:
    """
    Return today's NBA game list from the ESPN Site API scoreboard.
    Includes live scores and odds spread when available.
    """
    data  = espn_get(f"{ESPN_SITE_V2}/scoreboard")
    games = []
    for event in data.get("events", []):
        try:
            comp        = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((t for t in competitors if t.get("homeAway") == "home"), {})
            away = next((t for t in competitors if t.get("homeAway") == "away"), {})

            odds_list  = comp.get("odds", [])
            raw_spread = odds_list[0].get("spread", 0.0) if odds_list else 0.0
            spread     = float(raw_spread) if raw_spread is not None else 0.0

            status_type = event.get("status", {}).get("type", {})
            predictor   = comp.get("predictor", {})

            games.append({
                "game_id":        str(event.get("id", "")),
                "home_team":      home.get("team", {}).get("displayName",   "Unknown"),
                "away_team":      away.get("team", {}).get("displayName",   "Unknown"),
                "home_id":        str(home.get("team", {}).get("id",        "")),
                "away_id":        str(away.get("team", {}).get("id",        "")),
                "home_abbr":      home.get("team", {}).get("abbreviation",  ""),
                "away_abbr":      away.get("team", {}).get("abbreviation",  ""),
                "home_score":     float(home.get("score", 0) or 0),
                "away_score":     float(away.get("score", 0) or 0),
                "status":         status_type.get("name",        ""),
                "status_desc":    status_type.get("description", ""),
                "is_final":       status_type.get("completed",   False),
                "odds_spread":    spread,
                "home_win_prob":  float(
                    predictor.get("homeTeam", {}).get("statistics", [{}])[0]
                    .get("value", 0.5) or 0.5
                ) if predictor.get("homeTeam", {}).get("statistics") else 0.5,
                "source":         "espn",
            })
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    logger.info(f"ESPN scoreboard: {len(games)} events")
    return games


def fetch_espn_team_stats(team_id: str) -> dict:
    """Return ESPN season-average stats for a team via Core API."""
    if not team_id:
        return {"pts": 0.0, "ast": 0.0, "reb": 0.0, "to": 0.0}

    url  = f"{ESPN_CORE_V2}/teams/{team_id}/statistics"
    data = espn_get(url)

    stat_map: dict[str, float] = {}
    for category in data.get("splits", {}).get("categories", []):
        for stat in category.get("stats", []):
            name = stat.get("name", "")
            val  = stat.get("value", 0.0)
            if name:
                stat_map[name] = float(val) if val is not None else 0.0

    return {
        "pts": stat_map.get("avgPoints",    0.0),
        "ast": stat_map.get("avgAssists",   0.0),
        "reb": stat_map.get("avgRebounds",  0.0),
        "to":  stat_map.get("avgTurnovers", 0.0),
    }


def fetch_espn_power_index() -> dict[str, float]:
    """Return ESPN Basketball Power Index (BPI) keyed by team abbreviation."""
    url  = f"{ESPN_CORE_V2}/powerindex"
    data = espn_get(url)
    bpi: dict[str, float] = {}
    for item in data.get("items", []):
        abbr = item.get("team", {}).get("abbreviation", "")
        val  = item.get("value", 50.0)
        if abbr:
            bpi[abbr] = float(val) if val is not None else 50.0
    return bpi


def fetch_espn_injuries(team_id: str) -> int:
    """Return the ESPN injury count for a team."""
    if not team_id:
        return 0
    url  = f"{ESPN_CORE_V2}/teams/{team_id}/injuries"
    data = espn_get(url)
    return len(data.get("items", []))


def fetch_espn_standings() -> dict[str, float]:
    """Return win percentages from ESPN standings keyed by team abbreviation."""
    url  = f"{ESPN_SITE_V2}/standings"
    data = espn_get(url)
    win_pct: dict[str, float] = {}
    for group in data.get("groups", []):
        for entry in group.get("standings", {}).get("entries", []):
            try:
                abbr  = entry.get("team", {}).get("abbreviation", "")
                stats = {s["name"]: s["value"] for s in entry.get("stats", [])}
                pct   = float(stats.get("winPercent", 0.5) or 0.5)
                if abbr:
                    win_pct[abbr] = pct
            except (KeyError, TypeError, ValueError):
                continue
    return win_pct


def fetch_espn_news(team_id: str = "", limit: int = 10) -> list[dict]:
    """Return recent ESPN NBA news headlines."""
    if team_id:
        url = f"{ESPN_SITE_V2}/news?team={team_id}&limit={limit}"
    else:
        url = f"{ESPN_SITE_V2}/news?limit={limit}"

    data     = espn_get(url)
    articles = []
    for article in data.get("articles", [])[:limit]:
        articles.append({
            "headline":    article.get("headline",    ""),
            "description": article.get("description", ""),
            "published":   article.get("published",   ""),
            "url": article.get("links", {}).get("web", {}).get("href", ""),
        })
    return articles


def fetch_espn_teams() -> list[dict]:
    """Return all NBA team stubs (id, abbreviation, displayName)."""
    url  = f"{ESPN_SITE_V2}/teams"
    data = espn_get(url)
    teams = []
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for team in league.get("teams", []):
                t = team.get("team", {})
                teams.append({
                    "id":           str(t.get("id",          "")),
                    "abbreviation": t.get("abbreviation",    ""),
                    "displayName":  t.get("displayName",     ""),
                    "location":     t.get("location",        ""),
                    "color":        t.get("color",           "000000"),
                })
    return teams
