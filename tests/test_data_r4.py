"""Round 4 data plumbing: archive listing, perp-only builds, load-time helpers."""
import io
import zipfile

import httpx
import numpy as np
import pandas as pd
import pytest

from btcq.data import binance_vision as bv
from btcq.data import store
from conftest import T0

NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def page(prefixes=(), keys=(), truncated=False, marker=None):
    body = "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    body += "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    nm = f"<NextMarker>{marker}</NextMarker>" if marker else ""
    return (f'<ListBucketResult xmlns="{NS}"><IsTruncated>{"true" if truncated else "false"}</IsTruncated>'
            f"{nm}{body}</ListBucketResult>")


def test_list_prefixes_pages_through_prefix_only_pages():
    base = "data/futures/um/monthly/klines/"
    seen = []

    def handler(req):
        marker = req.url.params.get("marker", "")
        seen.append(marker)
        if marker == "":
            return httpx.Response(200, text=page([base + "AAAUSDT/", base + "BBBUSDT/"], truncated=True,
                                                 marker=base + "BBBUSDT/"))
        return httpx.Response(200, text=page([base + "CCCUSDT/"]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert bv.list_prefixes(base, client) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert seen == ["", base + "BBBUSDT/"]


def test_listing_retries_on_503_and_429(monkeypatch):
    monkeypatch.setattr(bv.time, "sleep", lambda s: None)
    codes = iter([503, 429, 200])

    def handler(req):
        c = next(codes)
        return httpx.Response(c, text=page(keys=["x/A-1m-2024-01.zip", "x/A-1m-2024-01.zip.CHECKSUM"]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert bv.list_keys("x/", client) == ["x/A-1m-2024-01.zip"]


def test_listing_does_not_retry_a_404():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        bv.list_keys("x/", httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(calls) == 1


def test_trading_span_drops_the_stale_tail():
    m1 = pd.DataFrame({"ts": T0 + 60_000 * np.arange(6), "volume": [1, 2, 0, 3, 0, 0.0]})
    assert list(store.trading_span(m1)["ts"]) == list(T0 + 60_000 * np.arange(4))
    assert len(store.trading_span(m1.assign(volume=0.0))) == 0


def test_fill_funding_gaps_never_doubles_a_settlement():
    h = 3_600_000
    day0 = T0  # Monday 00:00 UTC
    real = pd.DataFrame({"ts": [day0 + 4, day0 + 16 * h + 4, day0 + 24 * h + 3], "rate": [1e-4, 2e-4, 3e-4]})
    f, hours = store.fill_funding_gaps(real, day0, day0 + 24 * h + 10, rate=5e-5)
    # the 08:00 settlement is missing (16h gap); it is filled exactly once
    assert list(f["ts"]) == [day0 + 4, day0 + 8 * h, day0 + 16 * h + 4, day0 + 24 * h + 3]
    assert f.loc[1, "rate"] == 5e-5 and hours == pytest.approx(16.0, abs=0.01)
    # trimmed to the span, and an empty funding file becomes synthetic settlements
    f2, _ = store.fill_funding_gaps(real.iloc[:0], day0 + 1, day0 + 40 * h, rate=1e-4)
    assert list(f2["ts"]) == [day0 + 8 * h, day0 + 16 * h, day0 + 24 * h, day0 + 32 * h]
    f3, hours3 = store.fill_funding_gaps(real, day0 + 10 * h, day0 + 20 * h, rate=1e-4)
    assert list(f3["ts"]) == [day0 + 16 * h + 4] and hours3 == 0.0


def _kline_zip(path, start_ms, n, px=100.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{start_ms + 60_000 * i},{px},{px + 1},{px - 1},{px},10,{start_ms + 60_000 * i + 59_999},"
            f"{px * 10},5,4,{px * 4},0" for i in range(n)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(path.stem + ".csv", "\n".join(rows))
    path.write_bytes(buf.getvalue())
    return path


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(store, "RAW_DIR", raw)
    monkeypatch.setattr(store, "PARQUET_DIR", tmp_path / "parquet")
    sym = "NEWUSDT"
    m1 = _kline_zip(raw / "futures/um/monthly/klines" / sym / "1m" / f"{sym}-1m-2022-01.zip", T0, 600)
    files = {"perp_1m_monthly": [m1]}
    monkeypatch.setattr(bv, "download", lambda name, symbol, keep=None, **kw: files.get(name, []))
    return sym, raw


def test_build_perp_only_tolerates_missing_datasets_and_deletes_raw(fake_store):
    sym, raw = fake_store
    rep = store.build_perp_only(sym)
    assert rep["status"] == "ok" and rep["parquet_verified"] and rep["raw_deleted"]
    assert rep["rows"] == 600 and rep["funding_rows"] == 0
    assert not (raw / "futures/um/monthly/klines" / sym).exists()
    assert len(pd.read_parquet(store.symbol_dir(sym) / "funding.parquet")) == 0
    assert (store.symbol_dir(sym) / "build_report.json").exists()


def test_build_perp_only_keeps_raw_when_read_back_fails(fake_store, monkeypatch):
    sym, raw = fake_store
    real = pd.read_parquet
    monkeypatch.setattr(store.pd, "read_parquet", lambda p, *a, **k: real(p, *a, **k).iloc[:-1])
    rep = store.build_perp_only(sym)
    assert not rep["parquet_verified"] and "raw_deleted" not in rep
    assert (raw / "futures/um/monthly/klines" / sym).exists()


def test_build_perp_only_reports_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PARQUET_DIR", tmp_path)
    monkeypatch.setattr(bv, "download", lambda name, symbol, keep=None, **kw: [])
    assert store.build_perp_only("GHOSTUSDT")["status"] == "no_1m"
