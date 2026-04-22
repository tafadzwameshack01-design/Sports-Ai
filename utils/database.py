import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_PATH = Path("sports_ai.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                source TEXT,
                home_team TEXT,
                away_team TEXT,
                prediction INTEGER,
                actual_result INTEGER,
                predicted_margin REAL,
                actual_margin REAL,
                reward REAL,
                confidence REAL,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS game_features (
                game_id TEXT PRIMARY KEY,
                home_pts REAL, home_ast REAL, home_reb REAL, home_to REAL,
                away_pts REAL, away_ast REAL, away_reb REAL, away_to REAL,
                home_win_pct REAL, away_win_pct REAL,
                home_injuries INTEGER, away_injuries INTEGER,
                odds_spread REAL,
                source TEXT,
                fetched_at TEXT
            );

            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_type TEXT,
                iterations INTEGER,
                initial_epsilon REAL,
                final_epsilon REAL,
                avg_reward REAL,
                notes TEXT,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS monte_carlo_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                n_simulations INTEGER,
                win_rate REAL,
                avg_reward REAL,
                std_reward REAL,
                confidence_95 REAL,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS optimizer_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                hyperparams TEXT,
                score REAL,
                notes TEXT,
                timestamp TEXT
            );
        """)


def save_prediction(
    game_id: str,
    source: str,
    home_team: str,
    away_team: str,
    prediction: int,
    actual_result: int,
    predicted_margin: float,
    actual_margin: float,
    reward: float,
    confidence: float,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (game_id, source, home_team, away_team, prediction, actual_result,
             predicted_margin, actual_margin, reward, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id, source, home_team, away_team, prediction,
                actual_result, predicted_margin, actual_margin,
                reward, confidence, datetime.now().isoformat(),
            ),
        )


def get_recent_predictions(limit: int = 200) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_metrics() -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if total == 0:
            return {
                "total": 0, "today": 0, "accuracy": 0.0,
                "avg_reward": 0.0, "accuracy_delta": 0.0,
                "correct": 0,
            }

        correct = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = actual_result AND actual_result IS NOT NULL"
        ).fetchone()[0]

        today_str = datetime.now().strftime("%Y-%m-%d")
        today_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ?",
            (f"{today_str}%",),
        ).fetchone()[0]

        avg_reward = conn.execute(
            "SELECT AVG(reward) FROM predictions"
        ).fetchone()[0] or 0.0

        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        old_total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp < ?", (week_ago,)
        ).fetchone()[0]
        old_correct = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = actual_result AND timestamp < ?",
            (week_ago,),
        ).fetchone()[0]

        old_acc = (old_correct / old_total * 100) if old_total > 0 else 0.0
        current_acc = (correct / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "today": today_count,
        "accuracy": current_acc,
        "avg_reward": float(avg_reward),
        "accuracy_delta": current_acc - old_acc,
        "correct": correct,
    }


def save_training_session(
    session_type: str,
    iterations: int,
    initial_epsilon: float,
    final_epsilon: float,
    avg_reward: float,
    notes: str = "",
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO training_sessions
            (session_type, iterations, initial_epsilon, final_epsilon, avg_reward, notes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_type, iterations, initial_epsilon,
                final_epsilon, avg_reward, notes, datetime.now().isoformat(),
            ),
        )


def get_training_history(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM training_sessions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_monte_carlo_run(
    n_simulations: int,
    win_rate: float,
    avg_reward: float,
    std_reward: float,
    confidence_95: float,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO monte_carlo_runs
            (n_simulations, win_rate, avg_reward, std_reward, confidence_95, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                n_simulations, win_rate, avg_reward, std_reward,
                confidence_95, datetime.now().isoformat(),
            ),
        )


def get_monte_carlo_history(limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM monte_carlo_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_optimizer_run(
    iteration: int, hyperparams: str, score: float, notes: str = ""
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO optimizer_runs (iteration, hyperparams, score, notes, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (iteration, hyperparams, score, notes, datetime.now().isoformat()),
        )


def get_optimizer_history(limit: int = 100) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM optimizer_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_reward_timeseries() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, reward, prediction = actual_result AS correct
            FROM predictions
            WHERE actual_result IS NOT NULL
            ORDER BY timestamp ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]
