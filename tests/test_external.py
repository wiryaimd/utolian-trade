"""Round 6 external data: Yahoo/EIA parsing, cleaning, engine rows, the Round 6 holdout lock, the manifest."""
import io
import json
import sys
import types
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from btcq.backtest import engine
from btcq.data import external as E
from btcq.data import store

NY = ZoneInfo("America/New_York")


def _ts(y, m, d, h=9, mi=30, tz=NY) -> int:
    return int(datetime(y, m, d, h, mi, tzinfo=tz).timestamp())


def yahoo_json(stamps, closes, adj=None, splits=None, tz="America/New_York") -> bytes:
    res = {"meta": {"exchangeTimezoneName": tz}, "timestamp": stamps,
           "indicators": {"quote": [{"close": closes}], "adjclose": [{"adjclose": adj or closes}]}}
    if splits:
        res["events"] = {"splits": {str(s): {"date": s, "numerator": 2, "denominator": 1} for s in splits}}
    return json.dumps({"chart": {"result": [res]}}).encode()


def test_parse_yahoo_dates_evening_unfinished_nulls_dups():
    stamps = [_ts(2024, 1, 2), _ts(2024, 1, 3), _ts(2024, 1, 4), _ts(2024, 1, 4, 15, 0),
              _ts(2024, 1, 5, 18, 0),                       # evening session on Friday -> Monday 2024-01-08
              _ts(2024, 1, 9), _ts(2024, 1, 10)]             # 2024-01-10 is the download day: unfinished
    closes = [10.0, None, 11.0, 11.5, 12.0, 12.5, 13.0]
    dl = datetime(2024, 1, 10, 20, 0, tzinfo=timezone.utc)
    df, info = E.parse_yahoo(yahoo_json(stamps, closes, splits=[_ts(2024, 1, 9)]), dl)
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-04", "2024-01-08", "2024-01-09"]
    assert df["close"].tolist() == [10.0, 11.5, 12.0, 12.5]          # duplicate date keeps the last
    assert info["nulls_dropped"] == 1 and info["dups_dropped"] == 1 and info["unfinished_dropped"] == 1
    assert info["tz"] == "America/New_York" and info["splits"] == 1 and info["split_dates"] == "2024-01-09"
    assert (df["date"].dt.hour == 0).all() and str(df["date"].dt.tz) == "UTC"


def test_parse_yahoo_uses_exchange_timezone():
    # 2024-03-04 23:30 UTC is still 2024-03-04 in Chicago before 17:00 local? 17:30 CST -> evening -> next weekday
    t = int(datetime(2024, 3, 4, 23, 30, tzinfo=timezone.utc).timestamp())
    df, _ = E.parse_yahoo(yahoo_json([t], [1.0], tz="America/Chicago"), datetime(2024, 4, 1, tzinfo=timezone.utc))
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-05"]
    t2 = int(datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc).timestamp())       # 08:30 CST
    df2, _ = E.parse_yahoo(yahoo_json([t2], [1.0], tz="America/Chicago"), datetime(2024, 4, 1, tzinfo=timezone.utc))
    assert df2["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-04"]


def _fake_xlrd(monkeypatch, key, rows):
    """xlrd stand-in: sheet 'Data 1', series key at (1,1), data from row 3 (dates as Excel serials)."""
    base = datetime(1899, 12, 30)

    class Sheet:
        nrows = 3 + len(rows)

        def cell_value(self, r, c):
            if r == 1 and c == 1:
                return key
            if r < 3:
                return "hdr"
            d, v = rows[r - 3]
            if c == 0:
                return "" if d is None else float((pd.Timestamp(d) - pd.Timestamp(base)).days)
            return v

    class Book:
        datemode = 0

        def sheet_by_name(self, name):
            assert name == "Data 1"
            return Sheet()

    mod = types.SimpleNamespace(open_workbook=lambda file_contents: Book(),
                                xldate_as_datetime=lambda v, mode: base + pd.Timedelta(days=v))
    monkeypatch.setitem(sys.modules, "xlrd", mod)


def test_parse_eia_rows_key_unfinished_and_nulls(monkeypatch):
    rows = [("1986-01-02", 25.56), ("1986-01-03", ""), ("1986-01-06", 26.0), (None, ""), ("1986-01-07", 26.5)]
    _fake_xlrd(monkeypatch, "RWTC", rows)
    df, info = E.parse_eia_xls(b"x", "RWTC", datetime(1986, 1, 7, 12, tzinfo=timezone.utc))
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["1986-01-02", "1986-01-06"]
    assert df["close"].tolist() == [25.56, 26.0] and info["nulls_dropped"] == 1
    _fake_xlrd(monkeypatch, "RBRTE", rows)
    with pytest.raises(ValueError):
        E.parse_eia_xls(b"x", "RWTC", datetime(1990, 1, 1, tzinfo=timezone.utc))


def test_nonpositive_rule_allowlist():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21"], utc=True),
                       "close": [18.3, -36.98, 8.9], "adjclose": np.nan})
    out, info = E.clean(df, "CLUSDT")
    assert len(out) == 2 and info["nonpositive_dropped"] == "2020-04-20" and info["nonpositive_unexpected"] == ""
    out, info = E.clean(df, "BZUSDT")
    assert info["nonpositive_unexpected"] == "2020-04-20"


def test_source_urls_and_stems():
    s = E.SOURCES["COPPERUSDT"]
    assert "HG%3DF" in s.url and s.stem == "HG_F"
    assert "CL%3DF" in E.SOURCES["CLUSDT"].url and "NG%3DF" in E.SOURCES["NATGASUSDT"].url   # Amendment 1
    assert E.SOURCES["BZUSDT"].url.endswith("/pet/hist_xls/RBRTEd.xls") and E.SOURCES["BZUSDT"].start == "2007-01-01"
    assert len(E.MARKETS) == 11 and set(E.MARKETS) == set(E.SOURCES)


def _closes(dates, vals=None):
    d = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("UTC").as_unit("ms")
    v = np.arange(1.0, len(d) + 1) if vals is None else np.asarray(vals, float)
    return pd.Series(v, index=d.asi8)


def test_rows_from_closes_last_row_kept_and_friday_fills_monday():
    c = _closes(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    rows = E.rows_from_closes(c)
    assert (rows["ts"] - c.index.to_numpy() == E.ROW_OFFSET_MS).all()
    mk = engine.Market(rows, None, None, synthetic_funding=0.0)
    bars = mk.bars("1d")
    assert bars["ts"].tolist() == c.index.tolist() and bars["close"].tolist() == [1.0, 2.0, 3.0, 4.0]
    ex = mk.exec_idx("1d", 0)
    assert ex.tolist() == [1, 2, 3, 4]                  # Friday (bar 1) executes at the Monday row (2)
    assert not mk.fund.any()


@pytest.fixture
def ext_store(tmp_path, monkeypatch):
    """One market (SPYUSDT) on disk with a manifest, 2015-06 .. 2027-01, in a temporary data dir."""
    monkeypatch.setattr(E, "EXT_DIR", tmp_path / "external")
    monkeypatch.setattr(E, "R6_DIR", tmp_path / "round6")
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "round6" / "data_manifest.csv")
    monkeypatch.setattr(E, "_R6_UNLOCKED", False)
    monkeypatch.setattr(store, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(store, "_HOLDOUT_UNLOCKED", False)
    d = pd.bdate_range("2015-06-01", "2027-01-29", tz="UTC")
    out = pd.DataFrame({"ts": d.as_unit("ms").asi8, "close": np.linspace(100, 200, len(d)), "adjclose": np.nan})
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    p = E.parquet_path(E.SOURCES["SPYUSDT"])
    p.parent.mkdir(parents=True)
    p.write_bytes(buf.getvalue())
    E._write_manifest([{"market": "SPYUSDT", "parquet_sha256": E._sha(buf.getvalue())}])
    return p


def test_lock_dev_full_and_unlock(ext_store):
    dev = E.load_closes("SPYUSDT", "dev")
    assert pd.Timestamp(int(dev.index.max()), unit="ms", tz="UTC") < E.R6_HOLDOUT_START
    with pytest.raises(PermissionError):
        E.load_closes("SPYUSDT", "full")
    E.unlock_r6("plain reason", version=1)
    full = E.load_closes("SPYUSDT", "full")
    assert pd.Timestamp(int(full.index.max()), unit="ms", tz="UTC") < E.R6_END
    assert pd.Timestamp(int(full.index.max()), unit="ms", tz="UTC") >= pd.Timestamp("2026-09-30", tz="UTC")
    rec = json.loads(store.HOLDOUT_LOG.read_text().splitlines()[-1])
    assert rec["round"] == 6 and rec["kind"] == "unlock" and rec["version"] == 1
    assert rec["reason"] == "plain reason [R6: M1 v1]"           # the tag is added by unlock_r6 itself
    assert store._HOLDOUT_UNLOCKED is False                 # the crypto 2024+ lock is untouched
    assert len(E.load_dates("SPYUSDT")) > len(full)          # dates only, whole file


def test_manifest_sha_mismatch_raises(ext_store):
    b = ext_store.read_bytes()
    ext_store.write_bytes(b[:-1] + bytes([b[-1] ^ 1]))
    with pytest.raises(RuntimeError):
        E.load_closes("SPYUSDT", "dev")


def test_r6_unlock_records(ext_store):
    store.HOLDOUT_LOG.write_text("".join(json.dumps(x) + "\n" for x in [
        {"reason": "crypto final holdout"}, {"reason": "old style [R6: M1]"},
        {"reason": "old style [R6 resume M1]"}]))
    E.unlock_r6("r", kind="resume", version=2)
    recs = E.r6_unlock_records()
    assert [r["kind"] for r in recs] == ["unlock", "resume", "resume"] and recs[-1]["version"] == 2
    with pytest.raises(ValueError):
        E.unlock_r6("r", kind="peek")


def test_stored_dates_must_be_utc_midnight_ms(ext_store):
    out = pd.DataFrame({"ts": np.array([1451520000, 1451606400], np.int64), "close": [1.0, 2.0], "adjclose": np.nan})
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    ext_store.write_bytes(buf.getvalue())
    E._write_manifest([{"market": "SPYUSDT", "parquet_sha256": E._sha(buf.getvalue())}])
    with pytest.raises(ValueError):
        E.load_closes("SPYUSDT", "dev")                         # seconds, not ms: refused on every load


def test_yahoo_must_be_daily_bars():
    raw = json.loads(yahoo_json([_ts(2024, 1, 2)], [1.0]))
    raw["chart"]["result"][0]["meta"]["dataGranularity"] = "1mo"
    with pytest.raises(ValueError):
        E.parse_yahoo(json.dumps(raw).encode(), datetime(2024, 3, 1, tzinfo=timezone.utc))


def test_download_never_overwrites(ext_store, monkeypatch):
    monkeypatch.setattr(E, "fetch", lambda url, **k: (_ for _ in ()).throw(AssertionError("no fetch")))
    with pytest.raises(FileExistsError):
        E.download_all()                                    # a manifest exists: only a resume may run
    orphan = E.parquet_path(E.SOURCES["QQQUSDT"])
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        E.download_all(resume=True)                         # parquet without a manifest row


def test_download_resume_fills_missing_market_only(ext_store, monkeypatch):
    stamps = [_ts(2024, 1, 2), _ts(2024, 1, 3)]
    calls = []
    monkeypatch.setattr(E, "fetch", lambda url, **k: calls.append(url) or yahoo_json(stamps, [1.0, 2.0]))
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    before = ext_store.read_bytes()
    man = E.download_all(markets=["SPYUSDT", "QQQUSDT"], resume=True)
    assert len(calls) == 1 and "QQQ" in calls[0]
    assert ext_store.read_bytes() == before
    assert man["market"].tolist() == ["SPYUSDT", "QQQUSDT"]
    q = E._verified_frame("QQQUSDT")
    assert q["close"].tolist() == [1.0, 2.0]
    assert q["ts"].tolist() == [1704153600000, 1704240000000]       # 2024-01-02 and 2024-01-03, 00:00 UTC in ms
    raw = list((E.EXT_DIR / "raw" / "yahoo").glob("QQQ_*.json"))
    assert len(raw) == 1


def test_download_writes_exact_ms_dates_and_never_reaches_the_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "EXT_DIR", tmp_path / "external")
    monkeypatch.setattr(E, "R6_DIR", tmp_path / "round6")
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "round6" / "data_manifest.csv")
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    days = pd.bdate_range("2015-12-28", "2016-01-08")
    stamps = [_ts(d.year, d.month, d.day) for d in days]
    early = [("2006-12-28", 20.0), ("2006-12-29", 21.0)]               # before Brent's 2007 start: dropped
    _fake_xlrd(monkeypatch, "RBRTE", early + [(d.strftime("%Y-%m-%d"), 30.0 + i) for i, d in enumerate(days)])
    monkeypatch.setattr(E, "fetch", lambda url, **k: b"x" if "eia" in url else yahoo_json(stamps, list(range(1, 11))))
    man = E.download_all(markets=["SPYUSDT", "BZUSDT"])
    assert man.set_index("market").loc["BZUSDT", "dropped_before_start"] == 2
    want = days.tz_localize("UTC").as_unit("ms").asi8.tolist()
    for m in ("SPYUSDT", "BZUSDT"):
        assert E._verified_frame(m)["ts"].tolist() == want
        dev = E.load_closes(m, "dev")
        assert len(dev) == 4 and int(dev.index[-1]) == want[3]          # ends 2015-12-31
    from btcq.research import round6 as R
    assert len(R.market_from_closes("SPYUSDT", E.load_closes("SPYUSDT", "dev")).bar_ts) == 4


def test_a_raw_file_blocks_a_fresh_download(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "EXT_DIR", tmp_path / "external")
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "round6" / "data_manifest.csv")
    d = E.EXT_DIR / "raw" / "yahoo"
    d.mkdir(parents=True)
    (d / "GLD_20261017T101010Z.json").write_bytes(b"{}")            # a failed first attempt left its raw file
    monkeypatch.setattr(E, "fetch", lambda url, **k: (_ for _ in ()).throw(AssertionError("no fetch")))
    with pytest.raises(FileExistsError):
        E.download_all()


def test_null_dates_are_reparsed_from_the_verified_raw_file(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "EXT_DIR", tmp_path / "external")
    monkeypatch.setattr(E, "R6_DIR", tmp_path / "round6")
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "round6" / "data_manifest.csv")
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    stamps = [_ts(2024, 1, 2), _ts(2024, 1, 15), _ts(2024, 1, 16), _ts(2024, 1, 19, 18, 0)]
    monkeypatch.setattr(E, "fetch", lambda url, **k: yahoo_json(stamps, [1.0, None, 2.0, None]))
    E.download_all(markets=["COPPERUSDT"])
    assert E.null_dates("COPPERUSDT") == ["2024-01-15", "2024-01-22"]      # evening row -> next weekday
    raw = list((E.EXT_DIR / "raw" / "yahoo").glob("HG_F_*.json"))[0]
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError):
        E.null_dates("COPPERUSDT")
