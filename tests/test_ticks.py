"""X1 tick archives: dev-only keys and windows, streaming extraction, checksums, resumable builds.
No network: archives are fake zips built in tmp_path and the HTTP getter is injected."""
import hashlib
import io
import json
import socket
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from btcq.config import HOLDOUT_START, ROOT
from btcq.data import binance_vision as bv
from btcq.data import ticks

DAY = 86_400_000
HOLD = int(HOLDOUT_START.value // 10**6)
D = "2023-06-01"
D0 = int(pd.Timestamp(D, tz="UTC").value // 10**6)
NS = "http://s3.amazonaws.com/doc/2006-03-01/"
REAL_HTTP_GET = ticks._http_get


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A test that forgot to inject its getter or lister fails instead of reaching the archive: the default
    getter, every real httpx transport (MockTransport is a separate class) and, below them, sockets refuse."""
    def refuse(*a, **k):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(ticks, "_http_get", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 10**6)


def csv_rows(dataset, ts):
    if dataset == "aggTrades":
        return [f"{i},{100 + i}.5,{0.25 + i},{10 * i},{10 * i + 1},{t},{'true' if i % 2 else 'false'}"
                for i, t in enumerate(ts)]
    return [f"{i},{100 + i}.5,{1 + i},{101 + i}.5,{2 + i},{t},{t + 3}" for i, t in enumerate(ts)]


def zip_bytes(dataset, ts, header=False, name="x.csv") -> bytes:
    text = (",".join(ticks.COLS[dataset]) + "\n" if header else "") + "\n".join(csv_rows(dataset, ts)) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def write_zip(path, dataset, ts, header=False):
    path.write_bytes(zip_bytes(dataset, ts, header))
    return path


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(ticks, "TICK_DIR", tmp_path / "ticks")
    monkeypatch.setattr(ticks, "SCRATCH", tmp_path / "ticks" / "_tmp")
    return tmp_path


class FakeArchive:
    """getter(url, dest) over an in-memory archive; counts calls and can corrupt one key's checksum."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []

    def add(self, key, body: bytes, checksum: str | None = None):
        self.files[bv.BASE_URL + key] = body
        sha = checksum or hashlib.sha256(body).hexdigest()
        self.files[bv.BASE_URL + key + ".CHECKSUM"] = f"{sha}  {key.rsplit('/', 1)[1]}\n".encode()

    def __call__(self, url, dest):
        self.calls.append(url)
        if url not in self.files:
            raise httpx.HTTPStatusError("404", request=httpx.Request("GET", url),
                                        response=httpx.Response(404))
        dest.write_bytes(self.files[url])


# ---------------------------------------------------------------- keys

def test_dev_last_day_is_the_day_before_the_holdout():
    assert ticks.DEV_LAST_DAY == (HOLDOUT_START - pd.Timedelta(days=1)).date()
    assert ticks.HOLDOUT_MS == HOLD


def test_key_last_day_monthly_and_daily():
    p = "data/futures/um/monthly/klines/BTCUSDT/1m/"
    # the leak this guards against: as strings, a January 2024 monthly period sorts before the holdout day
    assert "2024-01" < "2024-01-01"
    assert ticks.key_last_day(p + "BTCUSDT-1m-2024-01.zip") == date(2024, 1, 31)
    assert not ticks.is_dev_key(p + "BTCUSDT-1m-2024-01.zip")
    assert ticks.key_last_day(p + "BTCUSDT-1m-2023-12.zip") == date(2023, 12, 31)
    assert ticks.is_dev_key(p + "BTCUSDT-1m-2023-12.zip")
    assert ticks.key_last_day(p + "BTCUSDT-1m-2023-02.zip") == date(2023, 2, 28)
    assert ticks.key_last_day(p + "BTCUSDT-1m-2020-02.zip") == date(2020, 2, 29)
    k = ticks.daily_key("aggTrades", "1000SHIBUSDT", "2023-12-31")
    assert k == "data/futures/um/daily/aggTrades/1000SHIBUSDT/1000SHIBUSDT-aggTrades-2023-12-31.zip"
    assert ticks.key_last_day(k) == date(2023, 12, 31) and ticks.is_dev_key(k)
    k24 = ticks.daily_key("bookTicker", "BTCUSDT", "2024-01-01")
    assert ticks.key_last_day(k24) == date(2024, 1, 1) and not ticks.is_dev_key(k24)
    for bad in ("x/BTCUSDT-aggTrades.zip", "x/BTCUSDT-1m-2023-13.zip", "x/A-2023-02-30.zip"):
        with pytest.raises(ValueError):
            ticks.key_last_day(bad)
        assert not ticks.is_dev_key(bad)


def test_require_dev_keys_lists_every_bad_key():
    good = [ticks.daily_key("aggTrades", "BTCUSDT", "2023-12-31"), "m/BTCUSDT-1m-2023-12.zip"]
    ticks.require_dev_keys(good)
    bad = ["m/BTCUSDT-1m-2024-01.zip", ticks.daily_key("aggTrades", "ETHUSDT", "2024-02-03"), "junk.zip"]
    with pytest.raises(PermissionError) as e:
        ticks.require_dev_keys(good + bad)
    assert all(b in str(e.value) for b in bad) and good[0] not in str(e.value)


def test_list_dev_daily_keeps_dev_daily_keys_only():
    pre = ticks.DATASETS["bookTicker"].format(sym="ETHUSDT")
    k1 = pre + "ETHUSDT-bookTicker-2023-05-16.zip"
    k2 = pre + "ETHUSDT-bookTicker-2023-12-31.zip"
    listing = [(k2, 20), (k1, 10), (k1 + ".CHECKSUM", 1), (pre + "ETHUSDT-bookTicker-2024-01-01.zip", 30),
               (pre + "ETHUSDT-bookTicker-2023-12.zip", 40), (pre + "BTCUSDT-bookTicker-2023-06-01.zip", 50)]
    seen = []

    def lister(prefix):
        seen.append(prefix)
        return listing

    assert ticks.list_dev_daily("bookTicker", "ETHUSDT", lister) == [(k1, 10), (k2, 20)]
    assert seen == [pre]


def test_list_sizes_pages_through_the_s3_listing():
    pre = "data/futures/um/daily/aggTrades/BTCUSDT/"

    def contents(keys):
        return "".join(f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in keys)

    def handler(req):
        if req.url.params.get("marker", "") == "":
            body = (f'<ListBucketResult xmlns="{NS}"><IsTruncated>true</IsTruncated>'
                    f"{contents([(pre + 'a.zip', 5), (pre + 'b.zip', 7)])}</ListBucketResult>")
        else:
            assert req.url.params["marker"] == pre + "b.zip"
            body = f'<ListBucketResult xmlns="{NS}"><IsTruncated>false</IsTruncated>{contents([(pre + "c.zip", 9)])}' \
                   "</ListBucketResult>"
        return httpx.Response(200, text=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert ticks._list_sizes(pre, client) == [(pre + "a.zip", 5), (pre + "b.zip", 7), (pre + "c.zip", 9)]


def test_the_network_guard_covers_the_default_lister_and_getter(tmp_path):
    with pytest.raises(AssertionError, match="network"):
        ticks.list_dev_daily("aggTrades", "BTCUSDT")  # lister=None: a real httpx client
    with pytest.raises(AssertionError, match="network"):
        REAL_HTTP_GET("https://example.invalid/a.zip", tmp_path / "a.zip")
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("example.invalid", 443))
    assert not (tmp_path / "a.zip").exists()


def test_dev_days_keeps_days_present_in_every_dataset():
    def keys(ds, sym, days):
        return [(ticks.daily_key(ds, sym, d), 1) for d in days]

    listing = {ticks.DATASETS["aggTrades"].format(sym="BTCUSDT"):
               keys("aggTrades", "BTCUSDT", ["2023-05-14", "2023-05-15", "2023-05-16", "2023-05-17", "2024-01-01"]),
               ticks.DATASETS["bookTicker"].format(sym="BTCUSDT"):
               keys("bookTicker", "BTCUSDT", ["2023-05-16", "2023-05-17", "2023-05-18", "2023-12-31", "2024-01-01"]),
               ticks.DATASETS["aggTrades"].format(sym="ETHUSDT"): keys("aggTrades", "ETHUSDT", ["2023-06-01"]),
               ticks.DATASETS["bookTicker"].format(sym="ETHUSDT"): []}
    got = ticks.dev_days(["BTCUSDT", "ETHUSDT"], lister=lambda prefix: listing[prefix])
    assert got == {"BTCUSDT": ["2023-05-16", "2023-05-17"], "ETHUSDT": []}
    agg = ticks.dev_days(["ETHUSDT"], datasets=["aggTrades"], lister=lambda prefix: listing[prefix])
    assert agg == {"ETHUSDT": ["2023-06-01"]}


# ---------------------------------------------------------------- B1 days and windows

def _avail():
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-01-01", "2024-02-10", freq="D")]
    return {"ETHUSDT": days, "BTCUSDT": days[::2], "XRPUSDT": days[3:]}


def test_select_b1_days_is_deterministic_and_within_bounds():
    a = ticks.select_b1_days(_avail())
    assert a == ticks.select_b1_days(_avail())
    # independent of dict order
    assert a == ticks.select_b1_days(dict(reversed(list(_avail().items()))))
    df = pd.DataFrame(a, columns=["symbol", "day"])
    assert df.groupby("symbol").size().to_dict() == {"BTCUSDT": 15, "ETHUSDT": 15, "XRPUSDT": 15}
    assert df["day"].between("2023-05-16", "2023-12-30").all() and not df.duplicated().any()
    assert set(df.loc[df.symbol == "BTCUSDT", "day"]) <= set(_avail()["BTCUSDT"])
    assert list(df["symbol"]) == sorted(df["symbol"])
    assert a != ticks.select_b1_days(_avail(), seed=7)


def test_select_b1_days_refuses_holdout_and_short_availability():
    with pytest.raises(PermissionError):
        ticks.select_b1_days(_avail(), hi="2024-01-01")
    with pytest.raises(ValueError, match="only 3 days"):
        ticks.select_b1_days({"BTCUSDT": ["2023-06-01", "2023-06-02", "2023-06-03", "2024-01-02"]})


def test_leg_windows_split_at_midnight_and_drop_at_holdout():
    legs = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
                         "t_live_ms": [D0 + 23 * 3_600_000 + 55 * 60_000,   # crosses into the next day
                                       D0 + 5 * 60_000,                     # pre-window reaches the previous day
                                       HOLD - 7_200_000,                    # end == HOLDOUT_START: dropped
                                       HOLD - 7_200_001]},                  # ends 1 ms before: kept
                        index=[10, 11, 12, 13])
    w = ticks.leg_windows(legs)
    btc = w[w.symbol == "BTCUSDT"]
    assert btc["day"].tolist() == ["2023-06-01", "2023-06-02"]
    assert btc["start_ms"].tolist() == [D0 + 23 * 3_600_000 + 45 * 60_000, D0 + DAY]
    assert btc["end_ms"].tolist() == [D0 + DAY, D0 + DAY + 1 * 3_600_000 + 55 * 60_000]
    eth = w[w.symbol == "ETHUSDT"]
    assert eth["day"].tolist() == ["2023-05-31", "2023-06-01"]
    assert eth["start_ms"].tolist() == [D0 - 5 * 60_000, D0]
    assert eth["end_ms"].tolist() == [D0, D0 + 2 * 3_600_000 + 5 * 60_000]
    assert "SOLUSDT" not in set(w.symbol)
    xrp = w[w.symbol == "XRPUSDT"]
    assert xrp["day"].tolist() == ["2023-12-31"] and xrp["end_ms"].iloc[0] == HOLD - 1
    assert w["leg"].tolist() == [10, 10, 11, 11, 13]
    assert w.attrs["dropped"] == [{"leg": 12, "symbol": "SOLUSDT", "t_live_ms": HOLD - 7_200_000, "end_ms": HOLD}]
    assert (w["end_ms"] < HOLD).all() and (w["end_ms"] > w["start_ms"]).all()
    pd.concat([w, w])  # attrs hold plain data, so concat does not choke on them
    empty = ticks.leg_windows(legs.iloc[:0])
    assert empty.empty and list(empty.columns) == ["symbol", "day", "start_ms", "end_ms", "leg"]
    assert ticks.leg_windows(pd.DataFrame(columns=["symbol", "t_live_ms"])).empty  # object dtype, no rows


def test_leg_windows_refuses_t_live_ms_that_is_not_epoch_ms():
    # a NaN used to cast to INT64_MIN and the leg vanished from both the windows and attrs["dropped"]
    for bad in ([float(D0), np.nan], [float(D0), D0 + 0.5], [D0, D0 * 10**6]):
        legs = pd.DataFrame({"symbol": ["A", "B"], "t_live_ms": bad}, index=[7, 8])
        with pytest.raises(ValueError, match=r"legs \[8\]"):
            ticks.leg_windows(legs)
    na = pd.DataFrame({"symbol": ["A", "B"], "t_live_ms": pd.array([D0, None], dtype="Int64")})
    with pytest.raises(ValueError, match=r"legs \[1\]"):
        ticks.leg_windows(na)
    dt = pd.DataFrame({"symbol": ["A"], "t_live_ms": [pd.Timestamp(D, tz="UTC")]})
    with pytest.raises(ValueError, match="dtype"):
        ticks.leg_windows(dt)
    as_float = pd.DataFrame({"symbol": ["A"], "t_live_ms": [float(D0 + 60_000)]})
    as_int = pd.DataFrame({"symbol": ["A"], "t_live_ms": [D0 + 60_000]})
    assert ticks.leg_windows(as_float).equals(ticks.leg_windows(as_int))


def test_leg_windows_drops_legs_with_a_day_missing_from_avail():
    d16 = ms("2023-05-16")
    legs = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "BTCUSDT"],
                         "t_live_ms": [d16 + 60_000,         # pre-window reaches 2023-05-15: no bookTicker day
                                       d16 + 3_600_000,      # inside 2023-05-16
                                       d16 + 3_600_000,      # symbol absent from avail
                                       HOLD - 7_200_000]},   # holdout drop comes first
                        index=[1, 2, 3, 4])
    avail = {"BTCUSDT": ["2023-05-16", date(2023, 5, 17)]}
    w = ticks.leg_windows(legs, avail=avail)
    assert w["leg"].tolist() == [2] and w["day"].tolist() == ["2023-05-16"]
    assert w.attrs["missing"] == [{"leg": 1, "symbol": "BTCUSDT", "t_live_ms": d16 + 60_000, "days": ["2023-05-15"]},
                                  {"leg": 3, "symbol": "ETHUSDT", "t_live_ms": d16 + 3_600_000, "days": ["2023-05-16"]}]
    assert [r["leg"] for r in w.attrs["dropped"]] == [4]
    plain = ticks.leg_windows(legs)
    assert plain.attrs["missing"] == [] and set(plain["leg"]) == {1, 2, 3}


def test_jobs_from_windows_and_full_days():
    w = pd.DataFrame({"symbol": ["A", "A", "B"], "day": [D, D, D],
                      "start_ms": [D0 + 50, D0 + 10, D0], "end_ms": [D0 + 60, D0 + 20, D0 + 5]})
    jobs = ticks.jobs_from_windows(w)
    assert jobs[["dataset", "symbol"]].values.tolist() == [["aggTrades", "A"], ["bookTicker", "A"],
                                                           ["aggTrades", "B"], ["bookTicker", "B"]]
    assert jobs["windows"].iloc[0] == [(D0 + 10, D0 + 20), (D0 + 50, D0 + 60)]
    f = ticks.full_day_windows([("A", "2023-06-01"), ("B", date(2023, 12, 30))])
    assert f.values.tolist() == [["A", "2023-06-01", D0, D0 + DAY],
                                 ["B", "2023-12-30", ms("2023-12-30"), ms("2023-12-31")]]


def test_windows_are_validated_and_merged():
    assert ticks._windows([(5, 9), (1, 3), (3, 4), (8, 12)]).tolist() == [[1, 4], [5, 12]]
    with pytest.raises(PermissionError):
        ticks._windows([(HOLD - 10, HOLD + 1)])
    assert ticks._windows([(HOLD - 10, HOLD)]).tolist() == [[HOLD - 10, HOLD]]  # half-open: last ms is 2023
    with pytest.raises(ValueError):
        ticks._windows([(5, 5)])
    with pytest.raises(ValueError):
        ticks._windows([])
    with pytest.raises(ValueError, match="outside"):
        ticks._windows([(D0 - 1, D0 + 10)], D)


# ---------------------------------------------------------------- reading and extraction

@pytest.mark.parametrize("dataset", ["aggTrades", "bookTicker"])
def test_header_and_headerless_csvs_read_the_same(tmp_path, dataset):
    ts = [D0 + 10 * i for i in range(50)]
    a = write_zip(tmp_path / "a.zip", dataset, ts, header=True)
    b = write_zip(tmp_path / "b.zip", dataset, ts, header=False)
    ta = [x for x in ticks.read_zip_batches(a, dataset, block_size=256)]
    tb = [x for x in ticks.read_zip_batches(b, dataset, block_size=256)]
    assert len(ta) > 1  # streamed in several batches
    A, B = pa.Table.from_batches(ta), pa.Table.from_batches(tb)
    assert A.equals(B) and A.num_rows == 50 and A.schema == ticks.SCHEMAS[dataset]
    assert A.column(ticks.TS_COL[dataset]).to_pylist() == ts
    if dataset == "aggTrades":
        assert A.column("is_buyer_maker").to_pylist()[:3] == [False, True, False]


def test_read_zip_rejects_a_foreign_header_and_multi_member_zips(tmp_path):
    p = tmp_path / "h.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("x.csv", "id,price,qty,a,b,time,maker\n1,2,3,4,5,6,true\n")
    with pytest.raises(ValueError, match="header"):
        list(ticks.read_zip_batches(p, "aggTrades"))
    m = tmp_path / "m.zip"
    with zipfile.ZipFile(m, "w") as zf:
        zf.writestr("a.csv", "1")
        zf.writestr("b.csv", "2")
    with pytest.raises(ValueError, match="one CSV"):
        list(ticks.read_zip_batches(m, "aggTrades"))
    e = tmp_path / "e.zip"
    with zipfile.ZipFile(e, "w") as zf:
        zf.writestr("a.csv", "")
    assert list(ticks.read_zip_batches(e, "aggTrades")) == []


def _zip_text(path, text):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x.csv", text)
    return path


@pytest.mark.parametrize("lead", ["\n", " \n", "\r\n", "\n\n \r\n"])
@pytest.mark.parametrize("header", [False, True])
def test_leading_blank_lines_are_skipped_not_read_as_an_empty_file(tmp_path, lead, header):
    # the first line alone used to decide "empty": 0 rows, recorded as done and never refetched
    ts = [D0 + 10 * i for i in range(5)]
    text = lead + (",".join(ticks.COLS["aggTrades"]) + "\n" if header else "") + "\n".join(csv_rows("aggTrades", ts))
    z = _zip_text(tmp_path / "x.zip", text + "\n")
    t = pa.Table.from_batches(list(ticks.read_zip_batches(z, "aggTrades")))
    assert t.column("transact_time").to_pylist() == ts and t.column("agg_trade_id").to_pylist() == list(range(5))
    st = ticks.extract(z, "aggTrades", [(D0, D0 + DAY)], tmp_path / "o.parquet")
    assert st["rows_total"] == 5 and st["rows_kept"] == 5


def test_blank_and_header_only_members_read_as_empty(tmp_path):
    hdr = ",".join(ticks.COLS["bookTicker"])
    for i, text in enumerate(["\n", " \r\n\n", hdr, hdr + "\n", "\n" + hdr + "\r\n"]):
        z = _zip_text(tmp_path / f"{i}.zip", text)
        assert sum(b.num_rows for b in ticks.read_zip_batches(z, "bookTicker")) == 0
        assert ticks.extract(z, "bookTicker", [(D0, D0 + 10)], tmp_path / f"{i}.parquet")["rows_total"] == 0


@pytest.mark.parametrize("dataset", ["aggTrades", "bookTicker"])
def test_extract_keeps_exactly_the_window_rows(tmp_path, dataset):
    offs = [0, 999, 1000, 1500, 1999, 2000, 2500, 4999, 5000, 5999, 6000, 9000]
    z = write_zip(tmp_path / "x.zip", dataset, [D0 + o for o in offs], header=dataset == "bookTicker")
    out = tmp_path / "out" / "x.parquet"
    st = ticks.extract(z, dataset, [(D0 + 5000, D0 + 6000), (D0 + 1000, D0 + 2000)], out, block_size=128)
    t = pq.read_table(out)
    kept = [1000, 1500, 1999, 5000, 5999]
    assert t.column(ticks.TS_COL[dataset]).to_pylist() == [D0 + o for o in kept]
    idx = [offs.index(o) for o in kept]
    first_val = ticks.COLS[dataset][1]
    assert t.column(first_val).to_pylist() == [100 + i + 0.5 for i in idx]  # the rows themselves, not just ts
    assert t.column(ticks.COLS[dataset][0]).to_pylist() == idx
    if dataset == "bookTicker":
        assert t.column("event_time").to_pylist() == [D0 + o + 3 for o in kept]
    assert st["rows_total"] == len(offs) and st["rows_kept"] == 5
    assert st["first_ts"] == D0 + 1000 and st["last_ts"] == D0 + 5999 and st["ts_decreases"] == 0
    assert not out.with_name(out.name + ".part").exists()
    assert json.loads(pq.read_schema(out).metadata[b"x1_windows"]) == [[D0 + 1000, D0 + 2000], [D0 + 5000, D0 + 6000]]


def test_extract_converts_microsecond_timestamps(tmp_path):
    ts = [D0 + 100, (D0 + 200) * 1000, (D0 + 300) * 1000 + 999, D0 + 400]
    z = write_zip(tmp_path / "x.zip", "bookTicker", ts)
    out = tmp_path / "x.parquet"
    st = ticks.extract(z, "bookTicker", [(D0, D0 + 1000)], out)
    t = pq.read_table(out)
    assert t.column("transaction_time").to_pylist() == [D0 + 100, D0 + 200, D0 + 300, D0 + 400]
    # event_time = ts + 3 in the same unit, converted too
    assert t.column("event_time").to_pylist() == [D0 + 103, D0 + 200, D0 + 301, D0 + 403]
    assert st["us_rows"] == 2 and st["rows_kept"] == 4


def test_extract_with_no_rows_in_the_windows_writes_an_empty_file(tmp_path):
    z = write_zip(tmp_path / "x.zip", "aggTrades", [D0 + 1, D0 + 2])
    out = tmp_path / "x.parquet"
    st = ticks.extract(z, "aggTrades", [(D0 + 100, D0 + 200)], out)
    assert st["rows_kept"] == 0 and st["first_ts"] is None and pq.read_table(out).num_rows == 0


def test_extract_reports_decreasing_timestamps(tmp_path):
    z = write_zip(tmp_path / "x.zip", "bookTicker", [D0 + 1, D0 + 5, D0 + 3, D0 + 7, D0 + 6])
    st = ticks.extract(z, "bookTicker", [(D0, D0 + 100)], tmp_path / "x.parquet", block_size=64)
    assert st["ts_decreases"] == 2 and st["first_ts"] == D0 + 1 and st["last_ts"] == D0 + 7


def test_extract_refuses_windows_past_the_holdout(tmp_path):
    z = write_zip(tmp_path / "x.zip", "aggTrades", [HOLD - 5, HOLD + 5])
    out = tmp_path / "x.parquet"
    with pytest.raises(PermissionError):
        ticks.extract(z, "aggTrades", [(HOLD - 10, HOLD + 10)], out)
    assert not out.exists()


def test_extract_refuses_an_archive_holding_rows_past_the_holdout(tmp_path):
    # such rows used to be parsed and dropped by the window filter without a trace
    out = tmp_path / "x.parquet"
    z = write_zip(tmp_path / "a.zip", "aggTrades", [HOLD - 5, HOLD + 5])
    with pytest.raises(PermissionError, match="1 row.*HOLDOUT_START"):
        ticks.extract(z, "aggTrades", [(HOLD - 10, HOLD)], out)
    assert not out.exists() and not out.with_name(out.name + ".part").exists()
    # bookTicker: transaction_time in 2023 but event_time in 2024 (the row is inside the window)
    b = _zip_text(tmp_path / "b.zip", f"1,100.5,1,101.5,2,{HOLD - 1},{HOLD + 5}\n")
    with pytest.raises(PermissionError, match="HOLDOUT_START"):
        ticks.extract(b, "bookTicker", [(HOLD - DAY, HOLD)], out)
    assert not out.exists()
    # microsecond values are converted before the check: the last µs of 2023 is not a 2024 row
    us = write_zip(tmp_path / "c.zip", "aggTrades", [(HOLD - 1) * 1000 + 999])
    st = ticks.extract(us, "aggTrades", [(HOLD - 10, HOLD)], out)
    assert st["rows_kept"] == 1 and pq.read_table(out).column("transact_time").to_pylist() == [HOLD - 1]


def test_post_check_catches_a_leaking_filter(tmp_path, monkeypatch):
    # if the window filter were broken, the independent re-read must refuse the file
    monkeypatch.setattr(ticks, "_inside", lambda ts, w: np.ones(len(ts), bool))
    out = tmp_path / "x.parquet"
    z2 = write_zip(tmp_path / "y.zip", "aggTrades", [D0 + 1, D0 + 50])
    with pytest.raises(PermissionError, match="outside"):
        ticks.extract(z2, "aggTrades", [(D0, D0 + 10)], out)
    assert not out.exists() and not out.with_name(out.name + ".part").exists()


@pytest.mark.parametrize("col", ["transaction_time", "event_time"])
def test_post_check_reads_every_time_column(tmp_path, col):
    t = {"transaction_time": [HOLD - 2, HOLD - 1], "event_time": [HOLD - 1, HOLD - 1]}
    t[col][1] = HOLD
    vals = {c: [1, 2] if c == "update_id" else [1.0, 2.0] for c in ticks.COLS["bookTicker"]} | t
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table(vals, schema=ticks.SCHEMAS["bookTicker"]), p)
    w = np.array([[HOLD - DAY, HOLD]], dtype=np.int64)
    with pytest.raises(PermissionError, match=f"kept {col}"):
        ticks._verify(p, "bookTicker", w, 2)
    t[col][1] = HOLD - 1
    pq.write_table(pa.table({**vals, **t}, schema=ticks.SCHEMAS["bookTicker"]), p)
    ticks._verify(p, "bookTicker", w, 2)


# ---------------------------------------------------------------- fetch

def _key(ds="aggTrades", sym="BTCUSDT", day=D):
    return ticks.daily_key(ds, sym, day)


def test_fetch_extract_verifies_and_deletes_the_zip(dirs):
    arc = FakeArchive()
    key = _key()
    arc.add(key, zip_bytes("aggTrades", [D0 + 1, D0 + 20, D0 + 30]))
    out = dirs / "o" / "x.parquet"
    st = ticks.fetch_extract(key, [(D0, D0 + 25)], out, getter=arc)
    assert pq.read_table(out).num_rows == 2 and st["rows_kept"] == 2 and st["rows_total"] == 3
    assert st["sha256"] == hashlib.sha256(arc.files[bv.BASE_URL + key]).hexdigest()
    assert st["bytes"] == len(arc.files[bv.BASE_URL + key])
    assert arc.calls == [bv.BASE_URL + key, bv.BASE_URL + key + ".CHECKSUM"]
    assert list(ticks.SCRATCH.iterdir()) == []
    md = pq.read_schema(out).metadata
    assert md[b"x1_source_key"] == key.encode() and md[b"x1_source_sha256"] == st["sha256"].encode()


def test_checksum_mismatch_raises_and_leaves_no_file(dirs):
    arc = FakeArchive()
    key = _key()
    arc.add(key, zip_bytes("aggTrades", [D0 + 1]), checksum="0" * 64)
    out = dirs / "o" / "x.parquet"
    with pytest.raises(ValueError, match="checksum mismatch"):
        ticks.fetch_extract(key, [(D0, D0 + 25)], out, getter=arc)
    assert not out.exists() and not out.with_name(out.name + ".part").exists()
    assert list(ticks.SCRATCH.iterdir()) == []


def test_zip_is_deleted_after_a_failed_extraction_and_a_failed_download(dirs):
    arc = FakeArchive()
    key = _key()
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("x.csv", "wrong,header\n1,2\n")
    arc.add(key, bad.getvalue())
    out = dirs / "o" / "x.parquet"
    with pytest.raises(ValueError, match="header"):
        ticks.fetch_extract(key, [(D0, D0 + 25)], out, getter=arc)
    assert list(ticks.SCRATCH.iterdir()) == [] and not out.exists()

    arc2 = FakeArchive()
    arc2.files[bv.BASE_URL + key] = zip_bytes("aggTrades", [D0 + 1])  # no .CHECKSUM: the second GET fails
    with pytest.raises(httpx.HTTPStatusError):
        ticks.fetch_extract(key, [(D0, D0 + 25)], out, getter=arc2)
    assert len(arc2.calls) == 2 and list(ticks.SCRATCH.iterdir()) == [] and not out.exists()


def test_fetch_extract_refuses_before_downloading(dirs):
    arc = FakeArchive()
    with pytest.raises(PermissionError):
        ticks.fetch_extract(_key(day="2024-01-01"), [(HOLD, HOLD + 10)], dirs / "x.parquet", getter=arc)
    with pytest.raises(PermissionError):
        ticks.fetch_extract("data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip",
                            [(D0, D0 + 1)], dirs / "x.parquet", getter=arc)
    with pytest.raises(ValueError, match="daily"):  # a dev monthly key is still not a daily archive
        ticks.fetch_extract("data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-12.zip",
                            [(D0, D0 + 1)], dirs / "x.parquet", getter=arc)
    with pytest.raises(ValueError, match="outside"):
        ticks.fetch_extract(_key(), [(D0 + DAY - 5, D0 + DAY + 5)], dirs / "x.parquet", getter=arc)
    assert arc.calls == []


def test_http_get_retries_5xx_and_streams_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(ticks.time, "sleep", lambda s: None)
    codes = iter([503, 429, 200])
    body = b"z" * 3_000_000

    def handler(req):
        c = next(codes)
        return httpx.Response(c, content=body if c == 200 else b"")

    REAL_HTTP_GET("https://example.invalid/a.zip", tmp_path / "a.zip",
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert (tmp_path / "a.zip").read_bytes() == body

    calls = []

    def h404(req):
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        REAL_HTTP_GET("https://example.invalid/b.zip", tmp_path / "b.zip",
                        client=httpx.Client(transport=httpx.MockTransport(h404)))
    assert calls == [1] and not (tmp_path / "b.zip").exists()


# ---------------------------------------------------------------- build

def _jobs():
    w = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"], "day": [D, "2023-06-02", D],
                      "start_ms": [D0 + 10, ms("2023-06-02"), D0], "end_ms": [D0 + 30, ms("2023-06-02") + 50, D0 + 15]})
    return ticks.jobs_from_windows(w)


def _archive(jobs):
    arc = FakeArchive()
    for ds, sym, day in jobs[["dataset", "symbol", "day"]].itertuples(index=False):
        t0 = ms(day)
        arc.add(ticks.daily_key(ds, sym, day), zip_bytes(ds, [t0 + 5 * i for i in range(20)], header=ds == "aggTrades"))
    return arc


def test_build_writes_the_manifest_and_resumes(dirs):
    jobs = _jobs()
    arc = _archive(jobs)
    mpath = dirs / "manifest.csv"
    man = ticks.build(jobs, mpath, getter=arc, workers=2)
    assert list(man.columns) == ticks.MANIFEST_COLS and len(man) == 6 and len(arc.calls) == 12
    assert list(ticks.SCRATCH.iterdir()) == []
    r = man.set_index(["dataset", "symbol", "day"]).loc[("aggTrades", "BTCUSDT", D)]
    assert r["rows_total"] == 20 and r["rows_kept"] == 4 and r["first_ts"] == D0 + 10 and r["last_ts"] == D0 + 25
    assert r["key"] == ticks.daily_key("aggTrades", "BTCUSDT", D) and len(r["sha256"]) == 64
    for f in man["out_file"]:
        assert ticks._resolve(f).exists()
    # second run: nothing is fetched, the manifest is unchanged
    arc.calls.clear()
    again = ticks.build(jobs, mpath, getter=arc)
    assert arc.calls == [] and again.equals(man)
    # a missing out file and a manifest whose sha no longer matches the file are both redone
    ticks._resolve(man["out_file"].iloc[0]).unlink()
    m2 = ticks.read_manifest(mpath)
    m2.loc[1, "sha256"] = "f" * 64
    m2.to_csv(mpath, index=False)
    ticks.build(jobs, mpath, getter=arc)
    redone = {u.removeprefix(bv.BASE_URL) for u in arc.calls if u.endswith(".zip")}
    assert redone == {man["key"].iloc[0], man["key"].iloc[1]}
    assert ticks.read_manifest(mpath)["sha256"].tolist() == man["sha256"].tolist()
    # changed windows -> a new out file, fetched again
    arc.calls.clear()
    j2 = jobs.copy()
    j2.at[0, "windows"] = [(D0 + 10, D0 + 40)]
    m3 = ticks.build(j2, mpath, getter=arc)
    assert len(arc.calls) == 2 and len(m3) == 6
    k0 = ticks.daily_key(*jobs.loc[0, ["dataset", "symbol", "day"]])
    assert m3.set_index("key").loc[k0, "rows_kept"] == 6


def test_build_refuses_holdout_jobs_before_any_download(dirs):
    arc = FakeArchive()
    jobs = pd.DataFrame([{"dataset": "aggTrades", "symbol": "BTCUSDT", "day": "2024-01-01",
                          "windows": [(HOLD, HOLD + 10)]}])
    with pytest.raises(PermissionError):
        ticks.build(jobs, dirs / "m.csv", getter=arc)
    bad = _jobs()
    bad.at[0, "windows"] = [(D0 - 10, D0 + 10)]  # crosses into the previous day's archive
    with pytest.raises(ValueError, match="outside"):
        ticks.build(bad, dirs / "m.csv", getter=arc)
    with pytest.raises(ValueError, match="duplicate"):
        ticks.build(pd.concat([_jobs(), _jobs().iloc[[0]]], ignore_index=True), dirs / "m.csv", getter=arc)
    assert arc.calls == [] and not (dirs / "m.csv").exists()


def test_build_collects_failures_and_keeps_the_good_rows(dirs):
    jobs = _jobs()
    arc = _archive(jobs)
    k = ticks.daily_key("bookTicker", "ETHUSDT", D)
    arc.add(k, arc.files[bv.BASE_URL + k], checksum="0" * 64)
    mpath = dirs / "m.csv"
    with pytest.raises(RuntimeError, match="1 of 6 jobs failed.*checksum mismatch"):
        ticks.build(jobs, mpath, getter=arc)
    man = ticks.read_manifest(mpath)
    assert len(man) == 5 and k not in set(man["key"]) and list(ticks.SCRATCH.iterdir()) == []
    # once the archive is fixed only the failed job runs
    arc.add(k, arc.files[bv.BASE_URL + k])
    arc.calls.clear()
    assert len(ticks.build(jobs, mpath, getter=arc)) == 6 and len(arc.calls) == 2


def test_build_clears_leftover_scratch_zips(dirs):
    ticks.SCRATCH.mkdir(parents=True)
    (ticks.SCRATCH / "old.zip").write_bytes(b"x")
    (ticks.SCRATCH / "old.zip.CHECKSUM").write_bytes(b"x")
    (ticks.SCRATCH / "keep.txt").write_bytes(b"x")
    jobs = _jobs().iloc[:1]
    ticks.build(jobs, dirs / "m.csv", getter=_archive(jobs), workers=1)
    assert [p.name for p in ticks.SCRATCH.iterdir()] == ["keep.txt"]


def test_build_manifest_round_trips_a_job_with_no_kept_rows(dirs):
    jobs = pd.DataFrame([{"dataset": "bookTicker", "symbol": "BTCUSDT", "day": D, "windows": [(D0 + 500, D0 + 600)]}])
    arc = _archive(jobs)
    man = ticks.build(jobs, dirs / "m.csv", getter=arc)
    assert man["rows_kept"].iloc[0] == 0 and pd.isna(man["first_ts"].iloc[0]) and pd.isna(man["last_ts"].iloc[0])
    arc.calls.clear()
    assert ticks.build(jobs, dirs / "m.csv", getter=arc).equals(man) and arc.calls == []


def test_build_aborts_on_a_holdout_leak(dirs, monkeypatch):
    monkeypatch.setattr(ticks, "_inside", lambda ts, w: np.ones(len(ts), bool))
    day = "2023-12-31"
    jobs = pd.DataFrame([{"dataset": "aggTrades", "symbol": "BTCUSDT", "day": day, "windows": [(ms(day), HOLD)]}])
    arc = FakeArchive()
    arc.add(ticks.daily_key("aggTrades", "BTCUSDT", day), zip_bytes("aggTrades", [HOLD - 5, HOLD + 5]))
    with pytest.raises(PermissionError):  # not collected into the RuntimeError of ordinary failures
        ticks.build(jobs, dirs / "m.csv", getter=arc)
    assert not (dirs / "m.csv").exists() and list(ticks.SCRATCH.iterdir()) == []
    assert not list((dirs / "ticks").rglob("*.parquet"))


def test_a_leg_on_the_first_bookticker_day_is_excluded_not_a_failed_build(dirs):
    # a leg at 2023-05-16 00:01 has a pre-window piece on 2023-05-15, a day with no bookTicker archive
    d16 = ms("2023-05-16")
    legs = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT"], "t_live_ms": [d16 + 60_000, d16 + 3_600_000]})
    arc = FakeArchive()
    for ds, day in [("aggTrades", "2023-05-15"), ("aggTrades", "2023-05-16"), ("bookTicker", "2023-05-16")]:
        arc.add(ticks.daily_key(ds, "BTCUSDT", day), zip_bytes(ds, [ms(day) + 60_000 * i for i in range(120)]))
    with pytest.raises(RuntimeError, match="1 of 4 jobs failed.*bookTicker-2023-05-15"):
        ticks.build(ticks.jobs_from_windows(ticks.leg_windows(legs)), dirs / "m0.csv", getter=arc)

    listing = [(u.removeprefix(bv.BASE_URL), len(b)) for u, b in arc.files.items()]
    avail = ticks.dev_days(["BTCUSDT"], lister=lambda prefix: [kv for kv in listing if kv[0].startswith(prefix)])
    assert avail == {"BTCUSDT": ["2023-05-16"]}
    w = ticks.leg_windows(legs, avail=avail)
    assert [m["leg"] for m in w.attrs["missing"]] == [0] and set(w["leg"]) == {1}
    man = ticks.build(ticks.jobs_from_windows(w), dirs / "m.csv", getter=arc)
    assert len(man) == 2 and (man["rows_kept"] == 70).all()  # minutes 50..119 of the archive


def test_a_second_build_refuses_while_one_runs(dirs):
    # both builds used to share SCRATCH, and the second one's start-up cleanup deleted the first one's zip
    jobs = _jobs()
    arc = _archive(jobs)
    in_flight, release = threading.Event(), threading.Event()

    def slow(url, dest):
        arc(url, dest)
        if url.endswith(".zip"):
            in_flight.set()
            assert release.wait(10)

    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(ticks.build, jobs.iloc[:1], dirs / "a.csv", slow, 1)
        try:
            assert in_flight.wait(10)
            scratch = sorted(p.name for p in ticks.SCRATCH.iterdir())
            assert scratch == [ticks.daily_key(*jobs.iloc[0][["dataset", "symbol", "day"]]).rsplit("/", 1)[1]]
            with pytest.raises(RuntimeError, match="another build"):
                ticks.build(jobs.iloc[4:5], dirs / "b.csv", getter=arc)
            assert sorted(p.name for p in ticks.SCRATCH.iterdir()) == scratch
        finally:
            release.set()
        assert len(first.result(timeout=10)) == 1
    # the lock goes with the build (test_build_collects_failures... covers a failed one)
    assert len(ticks.build(jobs.iloc[4:5], dirs / "b.csv", getter=arc)) == 1
    assert list(ticks.SCRATCH.iterdir()) == []


def test_out_files_inside_the_repo_are_stored_relative():
    p = ROOT / "data" / "ticks" / "aggTrades" / "BTCUSDT" / "x.parquet"
    assert ticks._rel(p) == "data/ticks/aggTrades/BTCUSDT/x.parquet"
    assert ticks._resolve("data/ticks/aggTrades/BTCUSDT/x.parquet") == p.resolve()
    w = [(D0, D0 + 10)]
    f = ticks.out_file_for("aggTrades", "BTCUSDT", D, w)
    assert f.parent == ticks.TICK_DIR / "aggTrades" / "BTCUSDT" and f.name.startswith(f"BTCUSDT-aggTrades-{D}-w")
    assert f != ticks.out_file_for("aggTrades", "BTCUSDT", D, [(D0, D0 + 11)])
    assert f == ticks.out_file_for("aggTrades", "BTCUSDT", D, [(D0, D0 + 5), (D0 + 5, D0 + 10)])  # merged first
