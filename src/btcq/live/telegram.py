"""Telegram Bot API client (plain HTTPS long polling) and a log-only fallback.

Only messages from the configured chat id are answered; everything else is ignored.
Send errors are logged and never interrupt trading.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("btcq.telegram")
MAX_LEN = 4000


class LogNotifier:
    """Used when no Telegram token is configured (and in replays)."""

    def __init__(self):
        self.sent: list[str] = []

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

    async def send(self, text: str) -> None:
        for i in range(0, max(len(text), 1), MAX_LEN):
            try:
                r = await self.client.post(f"{self.base}/sendMessage",
                                           json={"chat_id": self.chat_id, "text": text[i:i + MAX_LEN],
                                                 "parse_mode": "HTML", "disable_web_page_preview": True})
                if r.status_code != 200:
                    log.warning("telegram send failed: %s %s", r.status_code, r.text[:200])
            except httpx.HTTPError as e:
                log.warning("telegram send error: %s", e)

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
                await self.handle_updates(r.json().get("result", []), handler)
            except (httpx.HTTPError, ValueError) as e:
                log.warning("telegram poll error: %s", e)
                await asyncio.sleep(5)
