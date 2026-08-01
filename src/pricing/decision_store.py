import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "pricing_decisions.db"


class DecisionStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pricing_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quote_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    recommended_offer REAL NOT NULL,
                    final_offer REAL,
                    reason TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record_decision(self, payload):
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pricing_decisions (
                    quote_id,
                    user_id,
                    decision,
                    recommended_offer,
                    final_offer,
                    reason,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["quote_id"],
                    payload["user_id"],
                    payload["decision"],
                    payload["recommended_offer"],
                    payload.get("final_offer"),
                    payload.get("reason"),
                    created_at,
                ),
            )
            decision_id = cursor.lastrowid

        return {
            "id": decision_id,
            "quote_id": payload["quote_id"],
            "status": "recorded",
            "created_at": created_at,
        }

    def list_decisions(self, limit=25):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id,
                    quote_id,
                    user_id,
                    decision,
                    recommended_offer,
                    final_offer,
                    reason,
                    created_at
                FROM pricing_decisions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
