"""Telegram Bot API client (plain HTTPS long polling) and a log-only fallback.

Only messages from the configured chat id are answered; everything else is ignored.
Send errors are logged and never interrupt trading.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("btcq.telegram")
MAX_LEN = 4000
_TOKEN = re.compile(r"\d{5,}:[A-Za-z0-9_-]{20,}")


def redact(text) -> str:
    """Hide bot tokens: httpx errors quote the request URL, which contains the token."""
    return _TOKEN.sub("<token>", str(text))


def _now() -> int:
    return int(time.time() * 1000)


class LogNotifier:
    """Used when no Telegram token is configured (and in replays)."""

    def __init__(self):
        self.sent: list[str] = []
        self.health = {"configured": False}

    async def send(self, text: str) -> None:
        self.sent.append(text)
        log.info("notify: %s", text.replace("\n", " | "))

    async def poll_commands(self, handler) -> None:
        await asyncio.Event().wait()


class Telegram:
    def __init__(self, token: str, chat_id: str, client: httpx.AsyncClient | None = None):
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = str(chat_id)
        self.client = client or httpx.AsyncClient(timeout=70)
        self.offset = 0
        # Saved with the bot's state for the dashboard's System health tab (never contains the token).
        self.health = {"configured": True, "last_poll_ok": None, "last_send_ok": None,
                       "poll_error": None, "poll_error_at": None, "poll_fail_streak": 0,
                       "send_error": None, "send_error_at": None, "send_fail_streak": 0}

    def _failed(self, kind: str, what: str) -> None:
        self.health[f"{kind}_error"], self.health[f"{kind}_error_at"] = redact(what)[:200], _now()
        self.health[f"{kind}_fail_streak"] += 1

    async def send(self, text: str) -> None:
        for i in range(0, max(len(text), 1), MAX_LEN):
            try:
                r = await self.client.post(f"{self.base}/sendMessage",
                                           json={"chat_id": self.chat_id, "text": text[i:i + MAX_LEN],
                                                 "parse_mode": "HTML", "disable_web_page_preview": True})
                if r.status_code != 200:
                    log.warning("telegram send failed: %s %s", r.status_code, redact(r.text[:200]))
                    self._failed("send", f"HTTP {r.status_code}: {r.text[:120]}")
                else:
                    self.health["last_send_ok"], self.health["send_fail_streak"] = _now(), 0
            except httpx.HTTPError as e:
                log.warning("telegram send error: %s", redact(e))
                self._failed("send", f"{type(e).__name__}: {e}")

    async def get_me(self) -> dict:
        r = await self.client.get(f"{self.base}/getMe")
        r.raise_for_status()
        return r.json()

    async def handle_updates(self, updates: list[dict],
                             handler: Callable[[str], Awaitable[str | None]]) -> None:
        for u in updates:
            self.offset = max(self.offset, int(u["update_id"]) + 1)
            msg = u.get("message") or u.get("edited_message") or {}
            chat = str(msg.get("chat", {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            if chat != self.chat_id:
                log.warning("ignored message from unauthorised chat %s", chat)
                continue
            try:
                reply = await handler(text)
            except Exception as e:  # noqa: BLE001
                log.exception("command failed")
                reply = f"Command failed: {e}"
            if reply:
                await self.send(reply)

    async def poll_commands(self, handler: Callable[[str], Awaitable[str | None]]) -> None:
        while True:
            try:
                r = await self.client.get(f"{self.base}/getUpdates",
                                          params={"timeout": 50, "offset": self.offset,
                                                  "allowed_updates": '["message"]'})
                r.raise_for_status()
                self.health["last_poll_ok"], self.health["poll_fail_streak"] = _now(), 0
                await self.handle_updates(r.json().get("result", []), handler)
            except (httpx.HTTPError, ValueError) as e:
                log.warning("telegram poll error: %s", redact(e))
                self._failed("poll", f"{type(e).__name__}: {e}")
                await asyncio.sleep(5)
