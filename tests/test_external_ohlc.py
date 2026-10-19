"""Round 8 OHLC re-parse, the Round 6 parity check, the manifest and the Round 8 holdout lock. Synthetic Yahoo JSON in a
temporary folder only: the real raw files are never read here."""
import hashlib
import io
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from btcq.data import external as E
from btcq.data import external_ohlc as X
from btcq.data import store
from btcq.research import guards

NY = ZoneInfo("America/New_York")
DOWNLOAD = datetime(2016, 1, 15, 22, 0, tzinfo=timezone.utc)


def yahoo_json(seed=0) -> bytes:
    """A chart response for 2015-12-14 .. 2016-01-15 with an evening bar, a duplicate date, a null close, a null
    high, and the unfinished bar of the download day."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-12-14", "2016-01-15")
    ts, o, h, lo, c = [], [], [], [], []
    px = 200.0
    for d in days:
        px *= np.exp(rng.normal(0, 0.01))
        t = datetime(d.year, d.month, d.day, 9, 30, tzinfo=NY)
        ts.append(int(t.timestamp()))
        o.append(px * 0.999)
        h.append(px * 1.01)
        lo.append(px * 0.99)
        c.append(px)
    # an evening bar stamped Friday 2015-12-18 17:00 local: it belongs to Monday 2015-12-21 (replaces nothing:
    # Monday's own bar comes later and, as a duplicate date, is kept as the last)
    ev = int(datetime(2015, 12, 18, 17, 0, tzinfo=NY).timestamp())
    i = days.get_loc(pd.Timestamp("2015-12-21"))
    ts.insert(i, ev)
    o.insert(i, 1.0)
    h.insert(i, 2.0)
    lo.insert(i, 0.5)
    c.insert(i, 1.5)
    j = days.get_loc(pd.Timestamp("2015-12-28")) + 1
    c[j] = None                                              # a null close: the row is dropped
    k = days.get_loc(pd.Timestamp("2016-01-06")) + 1
    h[k] = None                                              # a null high: kept with NaN
    res = {"meta": {"exchangeTimezoneName": "America/New_York", "dataGranularity": "1d"}, "timestamp": ts,
           "indicators": {"quote": [{"open": o, "high": h, "low": lo, "close": c, "volume": [1] * len(ts)}],
                          "adjclose": [{"adjclose": [x if x is None else x * 0.98 for x in c]}]},
           "events": {}}
    return json.dumps({"chart": {"result": [res]}}).encode()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Round 6's raw files, parquets and manifest for the 3 markets, built the way external.download_all builds
    them, in a temporary folder; Round 8's paths and the holdout log there too."""
    ext = tmp_path / "external"
    monkeypatch.setattr(E, "EXT_DIR", ext)
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "r6" / "data_manifest.csv")
    monkeypatch.setattr(X, "R6_MANIFEST", tmp_path / "r6" / "data_manifest.csv")
    monkeypatch.setattr(X, "R8_DIR", tmp_path / "r8")
    monkeypatch.setattr(X, "MANIFEST", tmp_path / "r8" / "data_manifest.csv")
    monkeypatch.setattr(X, "OHLC_DIR", ext / "ohlc")
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(X, "_R8_UNLOCKED", False)
    rows = []
    for n, m in enumerate(X.MARKETS):
        src = E.SOURCES[m]
        raw = yahoo_json(n)
        (ext / "raw" / "yahoo").mkdir(parents=True, exist_ok=True)
        name = f"{src.stem}_20160115T220000Z.json"
        (ext / "raw" / "yahoo" / name).write_bytes(raw)
        df, info = E.parse_yahoo(raw, DOWNLOAD)
        df, info2 = E.clean(df, m)
        out = pd.DataFrame({"ts": E.date_ms(df["date"]), "close": df["close"], "adjclose": df["adjclose"]})
        buf = io.BytesIO()
        out.to_parquet(buf, index=False)
        E.parquet_path(src).parent.mkdir(parents=True, exist_ok=True)
        E.parquet_path(src).write_bytes(buf.getvalue())
        rows.append({"market": m, "source": "yahoo", "ticker": src.ticker, "download_utc": DOWNLOAD.isoformat(),
                     "raw_file": name, "raw_sha256": sha(raw), "parquet_sha256": sha(buf.getvalue()),
                     "rows": len(out)})
    (tmp_path / "r6").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "r6" / "data_manifest.csv", index=False)
    return tmp_path


def test_parse_keeps_ohlc_with_parse_yahoos_date_rules():
    raw = yahoo_json()
    df, _ = X.parse_yahoo_ohlc(raw, DOWNLOAD)
    ref, info = E.parse_yahoo(raw, DOWNLOAD)
    assert (df["date"].to_numpy() == ref["date"].to_numpy()).all()
    np.testing.assert_array_equal(df["close"].to_numpy(), ref["close"].to_numpy())
    assert info["nulls_dropped"] == 1 and info["dups_dropped"] == 1 and info["unfinished_dropped"] == 1
    mon = df[df["date"] == pd.Timestamp("2015-12-21", tz="UTC")].iloc[0]
    assert mon["close"] != 1.5 and mon["high"] != 2.0                  # of duplicate dates the last is kept
    assert df["date"].max() == pd.Timestamp("2016-01-14", tz="UTC")     # the download day's bar is unfinished
    assert df["high"].isna().sum() == 1 and df["open"].notna().all()
    assert (df["low"] <= df["close"]).all()


def test_parse_refuses_when_it_differs_from_parse_yahoo(monkeypatch):
    raw = yahoo_json()
    real = E.parse_yahoo

    def shifted(r, d):
        df, info = real(r, d)
        df.loc[3, "close"] += 0.01
        return df, info
    monkeypatch.setattr(E, "parse_yahoo", shifted)
    with pytest.raises(X.DataError):
        X.parse_yahoo_ohlc(raw, DOWNLOAD)


def test_build_matches_round6_and_writes_the_manifest(env):
    man = X.build_all()
    assert list(man["market"]) == X.MARKETS and (man["rows"] == man["rows_dev"] + 10).all()   # 2016-01-01 .. 01-14
    for m in X.MARKETS:
        df = X._verified_frame(m)
        r6 = E._verified_frame(m)
        np.testing.assert_array_equal(df["ts"].to_numpy(), r6["ts"].to_numpy())
        np.testing.assert_array_equal(df["close"].to_numpy(), r6["close"].to_numpy())
        assert {"open", "high", "low"} <= set(df.columns)
    with pytest.raises(FileExistsError):
        X.build_all()                                                   # r8-data runs once


def test_a_close_or_raw_mismatch_with_round6_is_a_hard_error(env):
    src = E.SOURCES["QQQUSDT"]
    p = E.parquet_path(src)
    df = pd.read_parquet(p)
    df.loc[5, "close"] *= 1.0001
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    p.write_bytes(buf.getvalue())
    man = pd.read_csv(E.MANIFEST)
    man.loc[man["market"] == "QQQUSDT", "parquet_sha256"] = sha(buf.getvalue())
    man.to_csv(E.MANIFEST, index=False)
    with pytest.raises(X.DataError, match="differ from Round 6"):
        X._build_market("QQQUSDT")
    raw = env / "external" / "raw" / "yahoo" / f"{E.SOURCES['SPYUSDT'].stem}_20160115T220000Z.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(X.DataError, match="does not match Round 6"):
        X._build_market("SPYUSDT")


def test_loads_check_the_manifest_and_only_three_markets(env):
    X.build_all()
    p = X.ohlc_path("EWYUSDT")
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(X.DataError):
        X.load_ohlc("EWYUSDT", "dev")
    with pytest.raises(PermissionError):
        X.load_ohlc("XAUUSDT", "dev")
    with pytest.raises(PermissionError):
        X._verified_frame("CLUSDT")


def test_the_holdout_lock(env):
    X.build_all()
    dev = X.load_ohlc("SPYUSDT", "dev")
    assert int(dev["ts"].max()) < int(X.HOLDOUT_START.value // 10**6)
    with pytest.raises(PermissionError, match="locked"):
        X.load_ohlc("SPYUSDT", "full")
    with pytest.raises(ValueError):
        X.unlock_r8("x", kind="resume", version=1)                      # Round 8 has no resume
    with pytest.raises(PermissionError, match="needs the earlier holdout unlock"):
        X.unlock_r8("early peek [R8 verify x]", kind="verify")          # no load before the one unlock
    with pytest.raises(ValueError, match="round tag"):
        X.unlock_r8("x [R5: XS]", version=1)                            # would be miscounted by Round 5's parser
    assert not X._R8_UNLOCKED and not store.HOLDOUT_LOG.exists()
    X.unlock_r8("Round 8 holdout (pre-registered)", version=1)
    full = X.load_ohlc("SPYUSDT", "full")
    assert int(full["ts"].max()) >= int(X.HOLDOUT_START.value // 10**6)
    rec = [json.loads(x) for x in store.HOLDOUT_LOG.read_text().splitlines()]
    assert rec[0]["round"] == 8 and rec[0]["kind"] == "unlock" and rec[0]["reason"].endswith("[R8: IBS v1]")
    assert not store._HOLDOUT_UNLOCKED                                   # never the crypto holdout
    assert E.r6_unlock_records() == []                                   # nor counted as a Round 6 unlock
    assert [r["kind"] for r in X.r8_unlock_records()] == ["unlock"]
    with pytest.raises(PermissionError, match="already unlocked"):
        X.unlock_r8("again", version=1)                                  # one unlock only
    X.unlock_r8("check [R8 verify resim]", kind="verify")              # an independent check after it
    assert [r["kind"] for r in X.r8_unlock_records()] == ["unlock", "verify"]


def test_unlock_records_count_tags_and_committed_versions(env, monkeypatch):
    lines = [{"utc": "a", "reason": "something [R8: IBS v1]"}, {"utc": "b", "reason": "x [R8 resume IBS v1]"},
             {"utc": "c", "reason": "x [R7: RT v1]"}, {"utc": "d", "round": 8, "kind": "verify", "reason": "y"}]
    committed = "".join(json.dumps(x) + "\n" for x in lines[:2])
    store.HOLDOUT_LOG.write_text(json.dumps(lines[2]) + "\n" + json.dumps(lines[3]) + "\n")
    monkeypatch.setattr(guards, "committed_versions", lambda p: [committed] if p == store.HOLDOUT_LOG else [])
    kinds = sorted(r["kind"] for r in X.r8_unlock_records())
    assert kinds == ["unlock", "unlock", "verify"]                       # a restored working copy can't hide one
    with pytest.raises(ValueError):
        X.unlock_r8("no tag", kind="verify")
