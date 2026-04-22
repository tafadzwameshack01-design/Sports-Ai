from __future__ import annotations

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("sports_ai.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, source TEXT,
            home_team TEXT, away_team TEXT,
            matchup TEXT,
            prediction INTEGER, actual_result INTEGER,
            predicted_margin REAL, actual_margin REAL,
            reward REAL, confidence REAL,
            timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_type TEXT, iterations INTEGER,
            initial_epsilon REAL, final_epsilon REAL,
            avg_reward REAL, avg_loss REAL, notes TEXT,
            timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS monte_carlo_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            n_simulations INTEGER,
            win_rate REAL, avg_reward REAL,
            std_reward REAL, confidence_95 REAL,
            timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS optimizer_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER, hyperparams TEXT,
            score REAL, notes TEXT, timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bias_detected TEXT, lesson TEXT,
            reasoning TEXT, predicted_improvement REAL,
            total_misses INTEGER, total_correct INTEGER,
            timestamp TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_pred_result ON predictions(actual_result);
        """)


def save_prediction(**kw) -> None:
    with get_conn() as c:
        c.execute("""
            INSERT INTO predictions
            (game_id,source,home_team,away_team,matchup,prediction,actual_result,
             predicted_margin,actual_margin,reward,confidence,timestamp)
            VALUES(:game_id,:source,:home_team,:away_team,:matchup,:prediction,
                   :actual_result,:predicted_margin,:actual_margin,:reward,:confidence,:timestamp)
        """, {**kw, "timestamp": kw.get("timestamp", datetime.now().isoformat())})


def save_lesson(data: dict) -> None:
    with get_conn() as c:
        c.execute("""
            INSERT INTO lessons (bias_detected,lesson,reasoning,predicted_improvement,
                                 total_misses,total_correct,timestamp)
            VALUES(?,?,?,?,?,?,?)
        """, (
            data.get("bias_detected",""),
            data.get("lesson",""),
            data.get("reasoning",""),
            float(data.get("predicted_improvement", 0)),
            int(data.get("total_misses", 0)),
            int(data.get("total_correct", 0)),
            datetime.now().isoformat(),
        ))


def get_lessons(limit: int = 20) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM lessons ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_predictions(limit: int = 500) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_metrics() -> dict:
    with get_conn() as c:
        total   = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if total == 0:
            return {"total":0,"today":0,"accuracy":0.0,"avg_reward":0.0,"accuracy_delta":0.0,"correct":0}

        correct = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction=actual_result AND actual_result IS NOT NULL"
        ).fetchone()[0]

        today   = datetime.now().strftime("%Y-%m-%d")
        today_n = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()[0]

        avg_r   = c.execute("SELECT AVG(reward) FROM predictions").fetchone()[0] or 0.0

        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        ot       = c.execute("SELECT COUNT(*) FROM predictions WHERE timestamp < ?", (week_ago,)).fetchone()[0]
        oc       = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction=actual_result AND timestamp < ?", (week_ago,)
        ).fetchone()[0]

        old_acc = (oc / ot * 100) if ot > 0 else 0.0
        cur_acc = (correct / total * 100) if total > 0 else 0.0

    return {
        "total":          total,
        "today":          today_n,
        "accuracy":       cur_acc,
        "avg_reward":     float(avg_r),
        "accuracy_delta": cur_acc - old_acc,
        "correct":        correct,
    }


def get_reward_series() -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            """SELECT timestamp, reward,
               CAST(prediction = actual_result AS INTEGER) AS correct
               FROM predictions WHERE actual_result IS NOT NULL
               ORDER BY timestamp ASC""",
        ).fetchall()
    return [dict(r) for r in rows]


def save_training_session(**kw) -> None:
    with get_conn() as c:
        c.execute("""
            INSERT INTO training_sessions
            (session_type,iterations,initial_epsilon,final_epsilon,avg_reward,avg_loss,notes,timestamp)
            VALUES(:session_type,:iterations,:initial_epsilon,:final_epsilon,
                   :avg_reward,:avg_loss,:notes,:timestamp)
        """, {**kw, "timestamp": kw.get("timestamp", datetime.now().isoformat())})


def get_training_history(limit: int = 50) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM training_sessions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_monte_carlo_run(**kw) -> None:
    with get_conn() as c:
        c.execute("""
            INSERT INTO monte_carlo_runs
            (n_simulations,win_rate,avg_reward,std_reward,confidence_95,timestamp)
            VALUES(:n_simulations,:win_rate,:avg_reward,:std_reward,:confidence_95,:timestamp)
        """, {**kw, "timestamp": kw.get("timestamp", datetime.now().isoformat())})


def get_monte_carlo_history(limit: int = 30) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM monte_carlo_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_optimizer_run(**kw) -> None:
    with get_conn() as c:
        c.execute("""
            INSERT INTO optimizer_runs (iteration,hyperparams,score,notes,timestamp)
            VALUES(:iteration,:hyperparams,:score,:notes,:timestamp)
        """, {**kw, "timestamp": kw.get("timestamp", datetime.now().isoformat())})


def get_optimizer_history(limit: int = 100) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM optimizer_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
