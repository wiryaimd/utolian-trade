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
CREATE TABLE IF NOT EXISTS snapshots (
  book TEXT, ts INTEGER, equity REAL, drawdown REAL, n_open INTEGER, PRIMARY KEY (book, ts));
CREATE TABLE IF NOT EXISTS daily_closes (symbol TEXT, day_ms INTEGER, close REAL, PRIMARY KEY (symbol, day_ms));
"""
TRADE_COLS = ["book", "symbol", "dir", "entry_time", "exit_time", "entry_px", "exit_px", "qty", "init_risk",
              "gross", "fees", "funding", "net", "R", "reason", "eq_before", "lookback"]


class StateDB:
    def __init__(self, path: Path | str, readonly: bool = False):
        if readonly:  # dashboard: never writes, never creates
            self.conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True,
                                        check_same_thread=False, timeout=1.0)
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)
        if "lookback" not in self._trade_columns():     # databases from before the D1 book
            self.conn.execute("ALTER TABLE trades ADD COLUMN lookback INTEGER")
        # exit events used to be stored without a time (ts 0); their data carries exit_time
        for rid, data in self.conn.execute("SELECT id, data FROM events WHERE ts=0 AND kind='exit'").fetchall():
            t = json.loads(data).get("exit_time")
            if t:
                self.conn.execute("UPDATE events SET ts=? WHERE id=?", (int(t), rid))
        self.conn.commit()

    def _trade_columns(self) -> list[str]:
        return [r[1] for r in self.conn.execute("PRAGMA table_info(trades)").fetchall()]

    def add_daily_close(self, symbol: str, day_ms: int, close: float) -> None:
        """Buffered: committed with the next state save."""
        self.conn.execute("INSERT OR REPLACE INTO daily_closes(symbol, day_ms, close) VALUES(?,?,?)",
                          (symbol, day_ms, close))

    def daily_closes(self) -> list[tuple]:
        try:
            return self.conn.execute("SELECT symbol, day_ms, close FROM daily_closes ORDER BY day_ms").fetchall()
        except sqlite3.OperationalError as e:         # an older database opened read-only has no such table
            if "no such table" in str(e):
                return []
            raise

    def set_meta(self, key: str, value) -> None:
        self.conn.execute("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (f"meta:{key}", json.dumps(value)))
        self.conn.commit()

    def meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (f"meta:{key}",)).fetchone()
        return json.loads(row[0]) if row else default

    def add_snapshot(self, book: str, ts: int, equity: float, drawdown: float, n_open: int) -> None:
        """Buffered: committed with the next state save."""
        self.conn.execute("INSERT OR REPLACE INTO snapshots(book, ts, equity, drawdown, n_open) VALUES(?,?,?,?,?)",
                          (book, ts, equity, drawdown, n_open))

    def snapshots(self, book: str | None = None) -> list[tuple]:
        q = "SELECT book, ts, equity, drawdown, n_open FROM snapshots"
        args = []
        if book:
            q += " WHERE book=?"
            args.append(book)
        return self.conn.execute(q + " ORDER BY ts", args).fetchall()

    def events(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute("SELECT ts, book, kind, data FROM events ORDER BY id DESC LIMIT ?",
                                 (limit,)).fetchall()
        return [{"ts": r[0], "book": r[1], "kind": r[2], **json.loads(r[3])} for r in rows]

    def save_state(self, state: dict) -> None:
        self.conn.execute("INSERT INTO kv(key, value) VALUES('state', ?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(state),))
        self.conn.commit()

    def load_state(self) -> dict | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key='state'").fetchone()
        return json.loads(row[0]) if row else None

    # Paper trades, events and daily equity are buffered and committed with the next state save, so a crash before
    # that save rolls them back together with the state, and the replayed minutes write them exactly once.
    # Real-account rows (live fills) also commit at once; the app saves the state right after each of them.

    def add_trade(self, t: dict) -> None:
        self.conn.execute(f"INSERT INTO trades({','.join(TRADE_COLS)}) VALUES({','.join('?' * len(TRADE_COLS))})",
                          [t.get(c) for c in TRADE_COLS])
        if str(t.get("book", "")).endswith("-live"):
            self.conn.commit()

    def add_event(self, ts: int, book: str, kind: str, data: dict) -> None:
        self.conn.execute("INSERT INTO events(ts, book, kind, data) VALUES(?,?,?,?)",
                          (ts, book, kind, json.dumps(data, default=str)))
        if kind.startswith("live_"):
            self.conn.commit()

    def set_equity(self, book: str, day_ms: int, equity: float) -> None:
        """Buffered: committed with the next state save."""
        self.conn.execute("INSERT INTO equity(book, day_ms, equity) VALUES(?,?,?) "
                          "ON CONFLICT(book, day_ms) DO UPDATE SET equity=excluded.equity", (book, day_ms, equity))

    def trades(self, book: str | None = None, limit: int = 1_000_000) -> list[dict]:
        cols = [c for c in TRADE_COLS if c in set(self._trade_columns())]
        q, args = "SELECT " + ",".join(cols) + " FROM trades", []
        if book:
            q += " WHERE book=?"
            args.append(book)
        q += " ORDER BY exit_time DESC, id DESC LIMIT ?"
        args.append(limit)
        return [dict(zip(cols, r)) for r in self.conn.execute(q, args).fetchall()]

    def equity_curve(self, book: str) -> list[tuple[int, float]]:
        return self.conn.execute("SELECT day_ms, equity FROM equity WHERE book=? ORDER BY day_ms",
                                 (book,)).fetchall()

    def close(self) -> None:
        self.conn.close()
