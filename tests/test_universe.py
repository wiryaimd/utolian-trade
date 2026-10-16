"""Round 4 universe: symbol catalogue and the point-in-time monthly ranking."""
import numpy as np
import pandas as pd

from btcq.data import universe as U

DAY = U.DAY_MS
M = pd.Timestamp("2022-06-01", tz="UTC")
MS = int(M.value // 10**6)


def ex(*rows):
    return {"symbols": [dict(zip(["symbol", "contractType", "underlyingType", "underlyingSubType", "status"], r))
                        for r in rows]}


def test_normalize_base():
    assert U.normalize_base("1000SHIBUSDT") == "SHIB"
    assert U.normalize_base("1MBABYDOGEUSDT") == "BABYDOGE"
    assert U.normalize_base("1000000MOGUSDT") == "MOG"
    assert U.normalize_base("1INCHUSDT") == "1INCH"
    assert U.normalize_base("LUNA2USDT") == "LUNA2"
    assert U.normalize_base("BTCUSDT") == "BTC"


def test_catalogue_exclusions_and_flags():
    info = ex(("BTCUSDT", "PERPETUAL", "COIN", ["PoW"], "TRADING"),
              ("DOGEUSDT", "PERPETUAL", "COIN", ["Meme"], "TRADING"),
              ("NVDAUSDT", "TRADIFI_PERPETUAL", "EQUITY", [], "TRADING"),
              ("XAUUSDT", "TRADIFI_PERPETUAL", "COMMODITY", [], "TRADING"),
              ("BTCDOMUSDT", "PERPETUAL", "INDEX", [], "TRADING"),
              ("ABCUSDT", "PERPETUAL", "PREMARKET", [], "TRADING"),
              ("PAXGUSDT", "PERPETUAL", "COIN", ["RWA"], "TRADING"))
    ov = pd.DataFrame({"symbol": ["OLDUSDT", "FTTUSDT"], "exclude": ["1", ""], "is_meme": ["", "0"],
                       "reason": ["test", ""]})
    syms = ["BTCUSDT", "DOGEUSDT", "NVDAUSDT", "XAUUSDT", "BTCDOMUSDT", "ABCUSDT", "PAXGUSDT", "USDCUSDT",
            "BTCUSDT_240329", "ETHBUSD", "LUNAUSDT", "OLDUSDT", "FTTUSDT", "DEFIUSDT"]
    c = U.catalogue(syms, info, ov).set_index("symbol")
    kept = set(c.index[~c["excluded"]])
    assert kept == {"BTCUSDT", "DOGEUSDT", "LUNAUSDT", "FTTUSDT"}
    assert c.loc["NVDAUSDT", "exclude_reason"].startswith("non-crypto")
    assert c.loc["ABCUSDT", "exclude_reason"].startswith("non-crypto")
    assert c.loc["BTCDOMUSDT", "exclude_reason"] == "index" and c.loc["DEFIUSDT", "exclude_reason"] == "index"
    assert c.loc["USDCUSDT", "exclude_reason"] == "stablecoin"
    assert c.loc["PAXGUSDT", "exclude_reason"] == "tokenized gold"
    assert c.loc["BTCUSDT_240329", "exclude_reason"] == "not a USDT perpetual"
    assert c.loc["OLDUSDT", "exclude_reason"] == "override: test"
    assert c.loc["DOGEUSDT", "is_meme"] and not c.loc["BTCUSDT", "is_meme"]
    assert c.loc["BTCUSDT", "is_major"] and not c.loc["DOGEUSDT", "is_major"]
    assert not c.loc["LUNAUSDT", "in_exchange_info"]


def cat_for(symbols, meme=(), majors=()):
    return pd.DataFrame({"symbol": symbols, "base_norm": [U.normalize_base(s) for s in symbols],
                         "excluded": False, "is_meme": [s in meme for s in symbols],
                         "is_major": [s in majors for s in symbols]})


def series(sym, first_day, last_day, vol, ms=MS):
    """Daily rows from ms+first_day*DAY to ms+last_day*DAY inclusive (days relative to M)."""
    days = np.arange(first_day, last_day + 1)
    v = np.broadcast_to(np.asarray(vol, float), days.shape)
    return pd.DataFrame({"symbol": sym, "ts": ms + days * DAY, "quote_volume": v, "count": 1})


def test_age_boundary_is_90_days():
    qv = pd.concat([series("AAAUSDT", -90, -1, 5.0), series("BBBUSDT", -89, -1, 50.0)])
    u = U.build_universe(qv, cat_for(["AAAUSDT", "BBBUSDT"]), [M])
    assert list(u["symbol"]) == ["AAAUSDT"]


def test_must_trade_every_one_of_the_last_7_days():
    a = series("AAAUSDT", -200, -1, 5.0)
    b = series("BBBUSDT", -200, -1, 50.0)
    b.loc[b["ts"] == MS - 3 * DAY, "quote_volume"] = 0.0
    c = series("CCCUSDT", -200, -2, 70.0)      # nothing on the last day
    u = U.build_universe(pd.concat([a, b, c]), cat_for(["AAAUSDT", "BBBUSDT", "CCCUSDT"]), [M])
    assert list(u["symbol"]) == ["AAAUSDT"]


def test_median_counts_missing_days_as_zero():
    sparse = pd.concat([series("AAAUSDT", -200, -200, 1.0), series("AAAUSDT", -40, -1, 100.0)])
    steady = series("BBBUSDT", -200, -1, 10.0)
    u = U.build_universe(pd.concat([sparse, steady]), cat_for(["AAAUSDT", "BBBUSDT"]), [M])
    assert list(u["symbol"]) == ["BBBUSDT", "AAAUSDT"]
    assert u.set_index("symbol").loc["AAAUSDT", "median_qv"] == 0.0


def test_same_base_keeps_the_higher_volume():
    qv = pd.concat([series("SHIBUSDT", -200, -1, 5.0), series("1000SHIBUSDT", -200, -1, 9.0)])
    u = U.build_universe(qv, cat_for(["SHIBUSDT", "1000SHIBUSDT"]), [M])
    assert list(u["symbol"]) == ["1000SHIBUSDT"] and u["n_eligible"].iloc[0] == 1


def test_top_n_tiers_and_flags():
    syms = [f"C{i:02d}USDT" for i in range(25)] + ["BTCUSDT"]
    qv = pd.concat([series(s, -300, -1, 1000.0 - i) for i, s in enumerate(syms)])
    u = U.build_universe(qv, cat_for(syms, meme={"C03USDT"}, majors={"BTCUSDT"}), [M], top_n=20)
    assert len(u) == 20 and list(u["rank"]) == list(range(1, 21))
    assert (u["tier"] == np.where(u["rank"] <= 10, 1, 2)).all()
    assert u.set_index("symbol").loc["C03USDT", "is_meme"]
    assert u["n_eligible"].iloc[0] == 26
    assert "BTCUSDT" not in set(u["symbol"])   # ranked last by volume


def test_point_in_time_invariance():
    syms = [f"C{i:02d}USDT" for i in range(8)]
    qv = pd.concat([series(s, -400, 60, 100.0 + 7 * i) for i, s in enumerate(syms)])
    cat = cat_for(syms + ["NEWUSDT"])
    base = U.build_universe(qv, cat, [M])
    later = qv["ts"] >= MS
    pert = qv.copy()
    pert.loc[later, "quote_volume"] *= np.linspace(0.01, 100, later.sum())
    pert = pd.concat([pert[~(later & (pert["symbol"] == "C00USDT"))], series("NEWUSDT", 0, 60, 1e12)])
    pd.testing.assert_frame_equal(base, U.build_universe(pert, cat, [M]))


def test_month_span():
    m = U.month_starts()
    assert len(m) == 78 and m[0] == pd.Timestamp("2020-04-01", tz="UTC") \
        and m[-1] == pd.Timestamp("2026-09-01", tz="UTC")
