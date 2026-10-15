"""Higher-timeframe values projected onto a lower timeframe without look-ahead."""
from __future__ import annotations

import numpy as np

from btcq.config import TF_MS


def align_htf(ltf_ts, ltf: str, htf_ts, htf: str, *values):
    """For each LTF bar, the values of the latest HTF bar that had CLOSED by the LTF bar's close.

    A 4h bar opening at 08:00 closes at 12:00, so its values first appear on the
    15m bar 11:45-12:00 (whose close is 12:00), never earlier.
    """
    ltf_close = np.asarray(ltf_ts, dtype=np.int64) + TF_MS[ltf]
    htf_close = np.asarray(htf_ts, dtype=np.int64) + TF_MS[htf]
    idx = np.searchsorted(htf_close, ltf_close, side="right") - 1
    ok = idx >= 0
    out = []
    for v in values:
        v = np.asarray(v)
        res = np.full(len(ltf_close), np.nan if v.dtype.kind == "f" else 0, dtype=v.dtype)
        res[ok] = v[idx[ok]]
        out.append(res)
    return out[0] if len(out) == 1 else out


def align_events(ltf_ts, ltf: str, event_ts, *values):
    """Point-in-time values (e.g. funding settlements) known at event_ts, as of each LTF close."""
    ltf_close = np.asarray(ltf_ts, dtype=np.int64) + TF_MS[ltf]
    idx = np.searchsorted(np.asarray(event_ts, dtype=np.int64), ltf_close, side="right") - 1
    ok = idx >= 0
    out = []
    for v in values:
        v = np.asarray(v, dtype=np.float64)
        res = np.full(len(ltf_close), np.nan)
        res[ok] = v[idx[ok]]
        out.append(res)
    return out[0] if len(out) == 1 else out
