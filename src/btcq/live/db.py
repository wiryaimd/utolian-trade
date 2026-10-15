"""SQLite persistence: bot state (for restarts), closed trades, events and daily equity."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, book TEXT, symbol TEXT, dir INTEGER,
  entry_time INTEGER, exit_time INTEGER, entry_px REAL, exit_px REAL, qty REAL, init_risk REAL,
  gross REAL, fees REAL, funding REAL, net REAL, R REAL, reason TEXT, eq_before REAL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, book TEXT, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS equity (book TEXT, day_ms INTEGER, equity REAL, PRIMARY KEY (book, day_ms));
"""
TRADE_COLS = ["book", "symbol", "dir", "entry_time", "exit_time", "entry_px", "exit_px", "qty", "init_risk",
              "gross", "fees", "funding", "net", "R", "reason", "eq_before"]


class StateDB:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save_state(self, state: dict) -> None:
        self.conn.execute("INSERT INTO kv(key, value) VALUES('state', ?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(state),))
        self.conn.commit()

    def load_state(self) -> dict | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key='state'").fetchone()
        return json.loads(row[0]) if row else None

    def add_trade(self, t: dict) -> None:
        self.conn.execute(f"INSERT INTO trades({','.join(TRADE_COLS)}) VALUES({','.join('?' * len(TRADE_COLS))})",
                          [t.get(c) for c in TRADE_COLS])
        self.conn.commit()

    def add_event(self, ts: int, book: str, kind: str, data: dict) -> None:
        self.conn.execute("INSERT INTO events(ts, book, kind, data) VALUES(?,?,?,?)",
                          (ts, book, kind, json.dumps(data, default=str)))
        self.conn.commit()

    def set_equity(self, book: str, day_ms: int, equity: float) -> None:
        self.conn.execute("INSERT INTO equity(book, day_ms, equity) VALUES(?,?,?) "
                          "ON CONFLICT(book, day_ms) DO UPDATE SET equity=excluded.equity", (book, day_ms, equity))
        self.conn.commit()

    def trades(self, book: str | None = None, limit: int = 1_000_000) -> list[dict]:
        q, args = "SELECT " + ",".join(TRADE_COLS) + " FROM trades", []
        if book:
            q += " WHERE book=?"
            args.append(book)
        q += " ORDER BY exit_time DESC, id DESC LIMIT ?"
        args.append(limit)
        return [dict(zip(TRADE_COLS, r)) for r in self.conn.execute(q, args).fetchall()]

    def equity_curve(self, book: str) -> list[tuple[int, float]]:
        return self.conn.execute("SELECT day_ms, equity FROM equity WHERE book=? ORDER BY day_ms",
                                 (book,)).fetchall()

    def close(self) -> None:
        self.conn.close()
