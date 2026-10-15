"""HTML research report: reports/summary.html + reports/tearsheets/<code>.html.

Charts are Plotly specs whose colours are CSS custom properties ("var(--series-1)");
a small script resolves them against the active light/dark theme and re-renders on
theme change, so each mode uses its own validated palette steps.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from btcq.backtest import metrics
from btcq.config import REPORTS_DIR, Costs, Risk
from btcq.research.run import GATES, RESULTS
from btcq.research import registry
from btcq.strategies import ALL, ROUND2

PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"
TEAR = REPORTS_DIR / "tearsheets"

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;
--good:#0ca30c;--good-text:#006300;--critical:#d03b3b;--div-neg:#e34948;--div-mid:#f0efec;--div-pos:#2a78d6;--callout:#eef4fc}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--good:#0ca30c;--good-text:#0ca30c;--critical:#e66767;--div-neg:#e66767;--div-mid:#383835;
--div-pos:#3987e5;--callout:#16263a}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--text-muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;
--good:#0ca30c;--good-text:#0ca30c;--critical:#e66767;--div-neg:#e66767;--div-mid:#383835;--div-pos:#3987e5;--callout:#16263a}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px;font-weight:650}
h2{font-size:19px;margin:40px 0 12px;font-weight:620}
h3{font-size:15px;margin:20px 0 8px;font-weight:620}
p{margin:0 0 12px;max-width:78ch}
.sub{color:var(--text-secondary);margin-bottom:24px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin:12px 0}
.callout{background:var(--callout);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .label{color:var(--text-secondary);font-size:13px}
.tile .value{font-size:28px;font-weight:620;margin-top:2px}
.tile .note{color:var(--text-muted);font-size:12px}
.tw{overflow-x:auto;margin:8px 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--text-secondary);font-weight:560;position:sticky;top:0;background:var(--surface-1)}
td.l,th.l{text-align:left}
tr.hl td{font-weight:620}
.pass{color:var(--good-text)} .fail{color:var(--critical)}
.chart{width:100%;height:320px}
.chart.sm{height:220px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.muted{color:var(--text-muted)} .small{font-size:13px}
a{color:var(--series-1)}
code{font-size:13px}
ul{margin:0 0 12px;padding-left:20px} li{margin:3px 0}
"""

JS = """
const CHARTS = %s;
function tok(v){ if(typeof v==='string' && v.startsWith('var(')){ const n=v.slice(4,-1).trim();
  return getComputedStyle(document.documentElement).getPropertyValue(n).trim()||v;} return v; }
function resolve(o){ if(Array.isArray(o)) return o.map(resolve);
  if(o && typeof o==='object'){ const r={}; for(const k in o) r[k]=resolve(o[k]); return r;} return tok(o); }
function draw(){ for(const [id,spec] of Object.entries(CHARTS)){ const el=document.getElementById(id); if(!el) continue;
  Plotly.react(el, resolve(spec.data), resolve(spec.layout), {displayModeBar:false, responsive:true}); } }
draw();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
"""


def _axis(**kw):
    a = {"gridcolor": "var(--grid)", "linecolor": "var(--axis)", "zeroline": False,
         "tickfont": {"color": "var(--text-muted)", "size": 11}, "showline": True}
    a.update(kw)
    return a


def _layout(**kw):
    lay = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
           "font": {"family": 'system-ui,-apple-system,"Segoe UI",sans-serif', "size": 12,
                    "color": "var(--text-secondary)"},
           "margin": {"l": 58, "r": 18, "t": 10, "b": 40}, "showlegend": False,
           "hoverlabel": {"bgcolor": "var(--surface-1)", "bordercolor": "var(--axis)",
                          "font": {"color": "var(--text-primary)"}},
           "xaxis": _axis(), "yaxis": _axis()}
    for k, v in kw.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            lay[k] = _axis(**v)
        else:
            lay[k] = v
    return lay


def _line(x, y, name, color="var(--series-1)", **kw):
    return {"type": "scatter", "mode": "lines", "x": list(x), "y": [None if not np.isfinite(v) else round(float(v), 6) for v in y],
            "name": name, "line": {"color": color, "width": 2}, **kw}


def _dates(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def _fmt(v, kind="f2"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "pct0":
        return f"{v * 100:.0f}%"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "f3":
        return f"{v:.3f}"
    return f"{v:.2f}"


def _mark(ok: bool) -> str:
    return '<span class="pass">✓ pass</span>' if ok else '<span class="fail">✗ fail</span>'


def _table(headers, rows, left=1, highlight=None) -> str:
    th = "".join(f'<th class="{"l" if i < left else ""}">{html.escape(h)}</th>' for i, h in enumerate(headers))
    body = []
    for ri, r in enumerate(rows):
        cls = ' class="hl"' if highlight is not None and ri == highlight else ""
        tds = "".join(f'<td class="{"l" if i < left else ""}">{c}</td>' for i, c in enumerate(r))
        body.append(f"<tr{cls}>{tds}</tr>")
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _page(title: str, body: str, charts: dict) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style>"
            f'<script src="{PLOTLY}"></script></head><body><main>{body}</main>'
            f"<script>{JS % json.dumps(charts)}</script></body></html>")


def _load(code: str):
    p = RESULTS / f"{code}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if r.get("formed") is False:
        return None
    r["_eq_dev"] = pd.read_parquet(RESULTS / f"{code}_dev_equity.parquet")["equity"]
    r["_tr_dev"] = pd.read_parquet(RESULTS / f"{code}_dev_trades.parquet")
    hp = RESULTS / f"{code}_holdout_equity.parquet"
    r["_eq_hold"] = pd.read_parquet(hp)["equity"] if hp.exists() and "holdout" in r else None
    r["_tr_hold"] = (pd.read_parquet(RESULTS / f"{code}_holdout_trades.parquet")
                     if r["_eq_hold"] is not None else None)
    r["_cfg"] = pd.read_csv(RESULTS / f"{code}_configs.csv")
    r["_dists"] = np.load(RESULTS / f"{code}_dev_dists.npz")
    return r


def _equity_spec(r) -> dict:
    eq0 = Risk().initial_equity
    dev = r["_eq_dev"] / eq0
    data = [_line(_dates(dev.index), dev.to_numpy(), f"Dev ({dev.index[0].year}–2023)",
                  hovertemplate="%{y:.3f}×<extra>Dev</extra>")]
    if r["_eq_hold"] is not None:
        hold = r["_eq_hold"] / eq0 * dev.iloc[-1]
        data.append(_line(_dates(hold.index), hold.to_numpy(), "Holdout (2024–)", "var(--series-2)",
                          hovertemplate="%{y:.3f}×<extra>Holdout</extra>"))
    lay = _layout(yaxis={"type": "log", "title": {"text": "Equity (× start)", "font": {"size": 11}}},
                  hovermode="x unified", showlegend=len(data) > 1,
                  legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": "var(--text-secondary)"}})
    return {"data": data, "layout": lay}


def _dd_spec(r) -> dict:
    parts = [r["_eq_dev"]]
    if r["_eq_hold"] is not None:
        parts.append(r["_eq_hold"] / r["_eq_hold"].iloc[0] * r["_eq_dev"].iloc[-1])
    eq = pd.concat(parts)
    dd = eq / eq.cummax() - 1
    tr = _line(_dates(dd.index), dd.to_numpy(), "Drawdown", hovertemplate="%{y:.1%}<extra></extra>")
    tr["fill"] = "tozeroy"
    tr["fillcolor"] = "rgba(128,128,128,0.10)"
    return {"data": [tr], "layout": _layout(yaxis={"tickformat": ".0%"}, hovermode="x unified")}


def _hist_spec(values, xtitle, marker=None, marker_label="", nbins=50, pct=False) -> dict:
    data = [{"type": "histogram", "x": [round(float(v), 6) for v in values], "nbinsx": nbins,
             "marker": {"color": "var(--series-1)", "line": {"color": "var(--surface-1)", "width": 1}},
             "hovertemplate": "%{x}<br>%{y} obs<extra></extra>"}]
    shapes, ann = [], []
    if marker is not None:
        shapes.append({"type": "line", "x0": marker, "x1": marker, "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"color": "var(--series-2)", "width": 2}})
        ann.append({"x": marker, "yref": "paper", "y": 1, "text": marker_label, "showarrow": False,
                    "xanchor": "left", "yanchor": "top", "font": {"color": "var(--text-primary)", "size": 11},
                    "xshift": 4})
    return {"data": data, "layout": _layout(xaxis={"title": {"text": xtitle, "font": {"size": 11}},
                                                    **({"tickformat": ".1%"} if pct else {})},
                                             yaxis={"title": {"text": "count", "font": {"size": 11}}},
                                             shapes=shapes, annotations=ann, bargap=0.04)}


def _monthly_spec(r) -> dict:
    from btcq.backtest.engine import Result
    eq = pd.concat([r["_eq_dev"]] + ([r["_eq_hold"] / r["_eq_hold"].iloc[0] * r["_eq_dev"].iloc[-1]]
                                     if r["_eq_hold"] is not None else []))
    res = Result(pd.DataFrame(), eq, Risk().initial_equity, eq.index[0])
    m = metrics.monthly_returns(res)
    z = m.to_numpy()
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.01
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data = [{"type": "heatmap", "z": [[None if not np.isfinite(v) else round(float(v), 5) for v in row] for row in z],
             "x": [months[c - 1] for c in m.columns], "y": [str(y) for y in m.index],
             "zmin": -lim, "zmax": lim, "xgap": 2, "ygap": 2,
             "colorscale": [[0, "var(--div-neg)"], [0.5, "var(--div-mid)"], [1, "var(--div-pos)"]],
             "colorbar": {"tickformat": ".0%", "thickness": 10, "outlinewidth": 0,
                          "tickfont": {"color": "var(--text-muted)", "size": 10}},
             "hovertemplate": "%{y} %{x}: %{z:.2%}<extra></extra>"}]
    return {"data": data, "layout": _layout(yaxis={"autorange": "reversed", "showline": False, "type": "category"},
                                             xaxis={"showline": False, "showgrid": False})}


def _dev_row(r) -> list[str]:
    d = r["dev"]
    return [f'<a href="tearsheets/{r["code"]}.html">{r["code"]}</a>', html.escape(r["name"]), r["tf"],
            _fmt(d.get("trades"), "int"), _fmt(d.get("win_rate"), "pct0"), _fmt(d.get("avg_R"), "f3"),
            _fmt(d.get("pf")), _fmt(d.get("t_stat")), _fmt(d.get("sharpe")), _fmt(d.get("cagr"), "pct"),
            _fmt(d.get("max_dd"), "pct"), _fmt(r.get("dsr"), "f3"), _fmt(r.get("pbo")),
            _fmt(r["null"]["p_value"], "f3")]


def _params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


ROUND2_CODES = {s.code for s in ROUND2} | {"C12"}
CHANGE_VS_S1 = {"S1": "baseline", "S1L": "long-only", "S1V": "30% vol-target sizing",
                "S1LV": "long-only + vol-target", "C12": "S1 + S2 sleeves, 1% risk each", "S2": "different signal"}


def _holdout_cell(r) -> str:
    if "holdout_gates" not in r:
        return '<span class="muted">locked</span>'
    if r.get("final_pass") is None:
        return '<span class="muted">info only (contaminated)</span>'
    if r.get("holdout_deviation"):
        ok = all(r["holdout_gates"].values())
        return (_mark(ok).replace("pass", "holdout gates pass").replace("fail", "holdout gates fail")
                + ' <span class="muted">(deviation; dev failed)</span>')
    return _mark(bool(r["final_pass"]))


def _code_link(r) -> str:
    return f'<a href="tearsheets/{r["code"]}.html">{r["code"]}</a>'


def build() -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEAR.mkdir(parents=True, exist_ok=True)
    codes = [s.code for s in ALL] + ["C12"]
    res = [r for r in (_load(c) for c in codes) if r is not None]
    res3 = [r for r in (_load(c) for c in R3_CODES) if r is not None]
    for r in res + res3:
        (TEAR / f"{r['code']}.html").write_text(_tearsheet(r), encoding="utf-8")
    (REPORTS_DIR / "summary.html").write_text(_summary(res, res3), encoding="utf-8")
    return str(REPORTS_DIR / "summary.html")


def _summary(res, res3=()) -> str:
    charts = {}
    c = Costs()
    by = {r["code"]: r for r in res}
    n_trials = len(registry.load().query("period == 'dev'"))
    n_eff = res[0].get("n_eff", float("nan"))
    hist = [json.loads(x) for x in registry.NEFF_LOG.read_text().splitlines()] if registry.NEFF_LOG.exists() else []
    pr_now = hist[-1]["pr"] if hist else n_eff
    finalists = [r for r in res if r.get("dev_pass")]
    f3 = [r for r in res3 if r.get("dev_pass")]
    p3 = [r for r in res3 if r.get("final_pass") is True]
    h3 = [r for r in res3 if "holdout" in r]
    passed = [r for r in res if r.get("final_pass") is True]
    clean = [r for r in res if "holdout" in r and r.get("holdout_status", "clean") == "clean"]
    dirty = [r for r in res if "holdout" in r and r.get("holdout_status") == "contaminated"]

    # ---- verdict
    v = _r3_verdict(res3) if res3 else []
    if res3:
        v.append("<p><b>Rounds 1–2 (BTC only):</b></p>")
    if passed:
        v.append(f"<p><b>{len(passed)} passed every gate:</b> "
                 + ", ".join(f"{r['code']} {html.escape(r['name'])}" for r in passed) + ".</p>")
    else:
        v.append("<p><b>No strategy passed every gate in rounds 1–2.</b></p>")
    if clean:
        items = []
        for r in clean:
            h, d = r["holdout"], r["dev"]
            tag = "passed" if r.get("final_pass") else ("profitable but decayed" if h["avg_R"] > 0 else "failed")
            items.append(f"<li><b>{r['code']}</b> {html.escape(r['name'])}: {tag}. Sharpe went from "
                         f"{d['sharpe']:.2f} in dev to {h['sharpe']:.2f} in the holdout, with {h['avg_R']:+.3f} R "
                         f"per trade over {h['trades']} trades. Longs averaged {h['long_avg_R']:+.2f} R and shorts "
                         f"{h['short_avg_R']:+.2f} R.</li>")
        v.append("<p>Clean out-of-sample tests (2024-01 → 2026-10, never seen before evaluation):</p><ul>"
                 + "".join(items) + "</ul>")
    if dirty:
        v.append("<p>The round-2 variants all passed the dev gates. Their holdout numbers are <b>information only</b>, "
                 "because the ideas were shaped by S1's already-seen holdout: "
                 + "; ".join(f"{r['code']} Sharpe {r['holdout']['sharpe']:.2f}" for r in dirty)
                 + ". Their only clean test left is forward paper trading.</p>")
    v.append("<p class='small muted'>Backtest results are not a guarantee of future returns, and this is research, "
             "not financial advice.</p>")

    gate_names = list(res[0]["gates"].keys()) + ["holdout"]
    gate_rows = [[_code_link(r), "2" if r["code"] in ROUND2_CODES else "1"]
                 + [_mark(x) for x in r["gates"].values()] + [_holdout_cell(r)] for r in res]

    body = [
        "<h1>Crypto perp strategy research</h1>",
        f'<p class="sub">Binance USDⓈ-M perpetuals. Rounds 1–2 traded BTCUSDT only, with dev data from 2017-08 to '
        f'2023-12 (spot prices before 2020). Round 3 trades BTC, ETH, SOL, XRP and BNB as one portfolio, with dev data '
        f'from 2020 to 2023-12. The holdout runs 2024-01 → 2026-10. Costs: taker {c.taker:.2%} + {c.slippage * 1e4:.0f} bp slippage per '
        f"side, plus real funding. Sizing: {Risk().risk_per_trade:.0%} risk per trade unless noted, with leverage "
        f"capped at {Risk().max_leverage:.0f}×.</p>",
        f'<div class="callout">{"".join(v)}</div>',
        '<div class="tiles">'
        f'<div class="tile"><div class="label">Strategies tested</div><div class="value">{len(res) + len(res3)}</div>'
        f'<div class="note">rounds 1–2: {len(res)}, round 3: {len(res3)}</div></div>'
        f'<div class="tile"><div class="label">Configurations backtested</div><div class="value">{n_trials}</div>'
        f'<div class="note">effective N {n_eff:.1f} (ratcheted)</div></div>'
        f'<div class="tile"><div class="label">Passed dev gates</div><div class="value">{len(finalists) + len(f3)}</div>'
        f'<div class="note">{", ".join(r["code"] for r in finalists + f3) or "none"}</div></div>'
        f'<div class="tile"><div class="label">Passed a clean holdout</div><div class="value">{len(passed) + len(p3)}</div>'
        f'<div class="note">{len(clean) + len(h3)} clean holdout tests run</div></div></div>',
        *(_r3_section(res3, charts) if res3 else []),
        "<h2>Rounds 1–2: gate results (BTC only)</h2>",
        "<p class='small muted'>Gates are pre-registered in research/PREREGISTRATION.md, round 2 included. "
        "Each strategy's holdout is evaluated at most once. Contaminated holdouts can't pass or fail.</p>",
        _table(["Strategy", "Round"] + gate_names, gate_rows, left=2),
        "<h2>Dev-period performance (chosen config per strategy)</h2>",
        "<p class='small muted'>The chosen config has the best neighbourhood-average Sharpe among configs with at "
        "least 100 trades. R is the net PnL divided by the risk taken at the stop. S1V and S1LV size to 30% "
        "volatility instead of 1% risk. C12 runs two sleeves at 1% each, so CAGR and drawdown are not comparable "
        "one-to-one, but Sharpe is.</p>",
        _table(["Code", "Strategy", "TF", "Trades", "Win", "Avg R", "PF", "t", "Sharpe", "CAGR", "Max DD",
                "DSR", "PBO", "Null p"], [_dev_row(r) for r in res], left=2),
    ]

    # ---- round 2 comparison
    r2 = [by[k] for k in ("S1", "S1L", "S1V", "S1LV", "S2", "C12") if k in by]
    rows = []
    for r in r2:
        d, h = r["dev"], r.get("holdout", {})
        status = r.get("holdout_status", "clean") if h else "locked"
        rows.append([_code_link(r), CHANGE_VS_S1.get(r["code"], ""), _fmt(d["sharpe"]),
                     f"{_fmt(d['cagr'], 'pct')} / {_fmt(d['max_dd'], 'pct')}",
                     _fmt(h.get("sharpe")), _fmt(h.get("avg_R"), "f3"), _fmt(h.get("trades"), "int"),
                     _fmt(h.get("max_dd"), "pct"), status])
    s1, s1v, c12 = by.get("S1"), by.get("S1V"), by.get("C12")
    notes = []
    if s1 and s1v:
        notes.append(f"<li><b>Vol-target sizing is mostly a leverage change.</b> Dev Sharpe was {s1v['dev']['sharpe']:.2f} "
                     f"vs {s1['dev']['sharpe']:.2f} for S1, with the same signals. CAGR and drawdown scale up together "
                     f"({s1v['dev']['cagr']:.0%} / {s1v['dev']['max_dd']:.0%} vs {s1['dev']['cagr']:.0%} / "
                     f"{s1['dev']['max_dd']:.0%}). S1's 1%-risk-at-ATR-stop sizing already scales inversely with "
                     f"volatility.</li>")
    if "S1L" in by and s1:
        notes.append(f"<li><b>Long-only keeps the dev edge with fewer trades.</b> Dev Sharpe was "
                     f"{by['S1L']['dev']['sharpe']:.2f} over {by['S1L']['dev']['trades']} trades. Its better holdout "
                     f"is expected by construction, since shorts were dropped after seeing them lose there.</li>")
    if c12 and "holdout" in c12:
        notes.append(f"<li><b>The combo diversifies in dev but inherits S2's out-of-sample failure.</b> The sleeves' "
                     f"daily-return correlation was {c12.get('sleeve_corr', float('nan')):.2f}, and dev Sharpe rose to "
                     f"{c12['dev']['sharpe']:.2f} with a t-stat of {c12['dev']['t_stat']:.1f}. In the holdout it fell "
                     f"to {c12['holdout']['sharpe']:.2f}. Peak combined notional was "
                     f"{c12.get('max_notional', float('nan')):.2f}× equity in dev.</li>")
    body += ["<h2>Round 2: S1 variants, S2 holdout, S1+S2 combo</h2>",
             _table(["Code", "Change", "Dev Sharpe", "Dev CAGR / DD", "Holdout Sharpe", "Holdout avg R",
                     "Holdout trades", "Holdout DD", "Holdout status"], rows, left=2),
             f"<ul>{''.join(notes)}</ul>"]

    # ---- holdout equity small multiples
    hold = [r for r in res if "holdout" in r]
    if hold:
        body.append("<h2>Dev and holdout equity</h2><p class='small muted'>Log scale. The holdout segment continues "
                    "from the dev end value.</p><div class='grid2'>")
        for r in sorted(hold, key=lambda r: r.get("holdout_status", "clean") != "clean"):
            cid = f"ho_{r['code']}"
            charts[cid] = _equity_spec(r)
            charts[cid]["layout"]["margin"] = {"l": 48, "r": 12, "t": 30, "b": 30}
            label = r.get("holdout_status", "clean")
            body.append(f'<div class="card"><h3 style="margin-top:0">{r["code"]} · {html.escape(r["name"])} '
                        f'<span class="muted small">({label} holdout)</span></h3>'
                        f'<div id="{cid}" class="chart sm"></div></div>')
        body.append("</div>")

    # ---- dev small multiples
    body.append("<h2>Dev equity curves</h2><p class='small muted'>Each strategy at its chosen config, starting at 1×. "
                "The y-axes are independent and on a log scale.</p><div class='grid2'>")
    for r in res:
        cid = f"sm_{r['code']}"
        eq = r["_eq_dev"] / Risk().initial_equity
        charts[cid] = {"data": [_line(_dates(eq.index), eq.to_numpy(), r["code"],
                                      hovertemplate="%{x}: %{y:.3f}×<extra></extra>")],
                       "layout": _layout(yaxis={"type": "log"}, margin={"l": 48, "r": 12, "t": 6, "b": 30})}
        body.append(f'<div class="card"><h3 style="margin-top:0">{r["code"]} · {html.escape(r["name"])}</h3>'
                    f'<div id="{cid}" class="chart sm"></div></div>')
    body.append("</div>")

    # ---- DSR disclosure
    rows = [[r["code"], _fmt(r.get("psr0"), "f3"), _fmt(r.get("dsr"), "f3"), _fmt(r.get("dsr_n_all"), "f3"),
             _fmt(r.get("dsr_prereg"), "f3")] for r in res]
    body += [
        "<h2>Multiple-testing correction (Deflated Sharpe) and amendments</h2>",
        "<p>The DSR asks whether a Sharpe beats the best Sharpe you would expect from the same number of "
        "<i>skill-less</i> trials by luck alone. Two amendments were made, both after seeing results and both "
        "disclosed in research/PREREGISTRATION.md.</p><ul>"
        "<li><b>Amendment 1 (after round 1).</b> The pre-registered estimator used the cross-trial Sharpe variance. "
        "Fee-destroyed strategies (Sharpe down to −10) inflated it into a luck threshold of about 5.9 annualized "
        "Sharpe, which nothing could beat. It was replaced with the zero-skill sampling variance 1/(T−1) and an "
        "effective trial count from the eigenvalues of the trials' return correlations. This change turned S1 "
        "from fail into pass.</li>"
        f"<li><b>Amendment 2 (after round 2).</b> That eigenvalue count <i>fell</i> from 14.8 to {pr_now:.1f} when "
        "91 near-duplicate S1 variants were added, which would have loosened the correction and let S2 pass at "
        "0.970. The effective N is now ratcheted so it never decreases. This change turned S2 from pass back into "
        "fail.</li></ul>",
        _table(["Code", "PSR vs 0", f"DSR (N_eff {n_eff:.1f})", f"DSR (N={n_trials} independent)",
                "DSR (pre-registered)"], rows),
        "<h2>Method notes and data caveats</h2><ul>"
        "<li>Signals use closed bars only and fill at the next bar's open. Stops and targets are resolved on 1-minute "
        "data, and if both are touched in the same minute the stop is assumed first. "
        f"A look-ahead truncation test passes for all {len(ALL)} strategies.</li>"
        "<li>C12 approximates a shared account by summing the two sleeves' daily returns.</li>"
        "<li>Spot BTCUSDT 1m stands in for the perp before 2020-01, the first month in the perp archive. Funding before "
        "2020 is a synthetic 0.01% per 8h, because this machine's network blocks Binance's REST API. Only the public "
        "archive was reachable.</li>"
        "<li>The 1m archive has a flat, zero-volume stretch on 2023-11-10 16:30–16:50 UTC. Every other perp bar matches "
        "Binance's own 15m/4h klines exactly.</li>"
        "<li>The random-entry null keeps the strategy's own holding times and long/short mix, and charges taker costs "
        "on both sides.</li></ul>",
    ]
    return _page("Crypto perp research", "".join(body), charts)


R3_CODES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]


def _r3_verdict(res3) -> list[str]:
    fin = [r for r in res3 if r.get("dev_pass")]
    passed = [r for r in res3 if r.get("final_pass") is True]
    names = ", ".join(f"{r['code']} {html.escape(r['name'])}" for r in fin)
    v = [f"<p><b>Round 3 (5-market portfolios, target ≥1 trade/week):</b> {len(fin)} of {len(res3)} families "
         f"passed every dev gate{': ' + names if fin else ''}.</p>"]
    near = [(r, [k for k, x in r["gates"].items() if not x]) for r in res3 if not r.get("dev_pass")]
    near = [(r, f) for r, f in near if len(f) == 1]
    if near:
        v.append("<p>Closest: " + "; ".join(
            f"<b>{r['code']}</b> {html.escape(r['name'])} failed only <code>{html.escape(f[0])}</code> "
            f"(DSR {r.get('dsr', 0):.3f}) while trading {r['dev'].get('trades_per_year', 0) / 52:.1f}×/week, "
            f"profitable in {r.get('symbols_positive', 0)}/5 markets, null p {r['null']['p_value']:.3f}"
            for r, f in near) + (". Its holdout was later run as a user-approved deviation (below)."
                                 if all("holdout" in r for r, _ in near)
                                 else ". Its holdout stays locked under the pre-registered rules.") + "</p>")
    if passed:
        v.append("<p><b>Passed the clean holdout too:</b> " + ", ".join(
            f"{r['code']} ({r['holdout'].get('trades_per_year', 0) / 52:.1f} trades/week, holdout Sharpe "
            f"{r['holdout']['sharpe']:.2f}, {r['holdout']['avg_R']:+.3f} R per trade)" for r in passed) + ".</p>")
    dev_ = [r for r in res3 if "holdout" in r and r.get("holdout_deviation")]
    for r in dev_:
        h, ref = r["holdout"], r.get("dev_scope", r["dev"])
        ok = all(r["holdout_gates"].values())
        v.append(f"<p><b>{r['code']} holdout (user-approved deviation, 4 alts, never seen before):</b> "
                 f"{'all holdout gates passed' if ok else 'holdout gates failed'}. {h['trades']} trades "
                 f"({h.get('trades_per_year', 0) / 52:.1f}/week), {h['avg_R']:+.3f} R per trade, Sharpe "
                 f"{h['sharpe']:.2f} vs {ref['sharpe']:.2f} in dev, max drawdown {h['max_dd']:.0%}, t-stat "
                 f"{h['t_stat']:.2f}. Longs {h['long_avg_R']:+.2f} R, shorts {h['short_avg_R']:+.2f} R. It still "
                 "can't be a final pass, because it failed the dev DSR gate.</p>")
    held = [r for r in res3 if "holdout" in r and not r.get("final_pass") and not r.get("holdout_deviation")]
    if held:
        v.append("<p>Holdout failures: " + "; ".join(
            f"{r['code']} Sharpe {r.get('dev_scope', r['dev'])['sharpe']:.2f} → {r['holdout']['sharpe']:.2f}, "
            f"{r['holdout']['avg_R']:+.3f} R per trade" for r in held) + ".</p>")
    return v


def _r3_section(res3, charts) -> list[str]:
    rows = []
    for r in res3:
        d = r["dev"]
        rows.append([_code_link(r), html.escape(r["name"]), r["tf"], _fmt(d.get("trades_per_year", 0) / 52),
                     _fmt(r.get("weeks_with_entry"), "pct0"), _fmt(d.get("avg_R"), "f3"), _fmt(d.get("pf")),
                     _fmt(d.get("t_stat")), _fmt(d.get("sharpe")), _fmt(d.get("max_dd"), "pct"),
                     _fmt(r.get("dsr"), "f3"), _fmt(r["null"]["p_value"], "f3"),
                     f"{r.get('symbols_positive', 0)}/5", _fmt(r.get("confirm_years"))])
    gate_names = list(res3[0]["gates"].keys()) + ["holdout"]
    gate_rows = [[_code_link(r)] + [_mark(x) for x in r["gates"].values()] + [_holdout_cell(r)] for r in res3]
    out = ["<h2>Round 3: 5-market portfolios at the 1h horizon</h2>",
           "<p class='small muted'>Each family runs with identical params on BTC, ETH, SOL, XRP and BNB as "
           "equal-capital sleeves, risking 1% of the account per trade with total notional ≤3×. Fills are delayed "
           "1 minute to model auto-execution. Confirm yrs is how long forward trading would need to reach t = 2 "
           "at the dev edge.</p>",
           _table(["Code", "Family", "TF", "Trades/wk", "Weeks w/ trade", "Avg R", "PF", "t", "Sharpe",
                   "Max DD", "DSR", "Null p", "Symbols +", "Confirm yrs"], rows, left=2),
           "<h3>Round 3 gates</h3>", _table(["Family"] + gate_names, gate_rows)]
    hold = [r for r in res3 if "holdout" in r]
    if hold:
        hrows = []
        for r in hold:
            h, ha = r["holdout"], r.get("holdout_all", {})
            ref = r.get("dev_scope", r["dev"]) if r.get("holdout_scope") == "alts" else r["dev"]
            hrows.append([_code_link(r), html.escape(r.get("holdout_status", "")),
                          _fmt(h.get("trades_per_year", 0) / 52), _fmt(h.get("avg_R"), "f3"), _fmt(h.get("pf")),
                          f"{_fmt(h.get('sharpe'))} <span class='muted'>(dev {_fmt(ref.get('sharpe'))})</span>",
                          _fmt(h.get("max_dd"), "pct"), _fmt(ha.get("sharpe")), _holdout_cell(r)])
        out += ["<h3>Round 3 holdout (2024-01 → 2026-10)</h3>",
                _table(["Family", "Status", "Trades/wk", "Avg R", "PF", "Sharpe", "Max DD",
                        "All 5 incl. BTC Sharpe", "Result"], hrows, left=2)]
    out.append("<div class='grid2'>")
    for r in res3:
        cid = f"r3_{r['code']}"
        charts[cid] = _equity_spec(r)
        charts[cid]["layout"]["margin"] = {"l": 48, "r": 12, "t": 30, "b": 30}
        out.append(f'<div class="card"><h3 style="margin-top:0">{r["code"]} · {html.escape(r["name"])}</h3>'
                   f'<div id="{cid}" class="chart sm"></div></div>')
    out.append("</div>")
    return out


def _tearsheet(r) -> str:
    charts = {}
    d = r["dev"]
    body = [f'<p class="small"><a href="../summary.html">← Summary</a></p>',
            f"<h1>{r['code']} · {html.escape(r['name'])}</h1>",
            f'<p class="sub">{r["tf"]} bars · chosen config: <code>{html.escape(_params(r["params"]))}</code> · '
            f'{r["n_configs"]} configs in grid</p>',
            f'<div class="card"><p><b>Hypothesis.</b> {html.escape(r["hypothesis"])}</p>'
            f'<p style="margin:0"><b>Rules.</b> {html.escape(r["rules"])}</p></div>']
    contaminated = "holdout" in r and r.get("holdout_status") == "contaminated"
    if contaminated:
        body.append('<div class="callout"><p style="margin:0"><b>Holdout contaminated: information only.</b> '
                    "This variant was designed after S1's 2024–26 holdout had been seen, so its holdout numbers "
                    "can't pass or fail it. Forward paper trading is its clean test.</p></div>")
    if r.get("inherited_gates"):
        body.append(f'<p class="small muted">{" and ".join(r["inherited_gates"])} are not defined for one fixed '
                    "portfolio; " + ("they are inherited from S1, where its parameters were chosen"
                                     if r["code"] == "F1" else "the worse component's value is used")
                    + " (pre-registered).</p>")
    status = ("passed all gates" if r.get("final_pass") is True else
              "passed dev gates, holdout info only" if contaminated and r.get("dev_pass") else
              "failed dev gates, holdout info only" if contaminated else
              "passed dev gates, failed holdout" if r.get("dev_pass") and "holdout" in r else
              "failed dev gates; holdout gates passed (user-approved deviation)"
              if r.get("holdout_deviation") and all(r.get("holdout_gates", {}).values()) else
              "failed dev gates, failed holdout" if "holdout" in r and not r.get("dev_pass") else
              "passed dev gates" if r.get("dev_pass") else "failed dev gates")
    tiles = [("Dev trades", _fmt(d.get("trades"), "int"), f"{d.get('trades_per_year', 0):.0f}/yr"),
             ("Avg R / trade", _fmt(d.get("avg_R"), "f3"), f"win rate {_fmt(d.get('win_rate'), 'pct0')}"),
             ("Dev Sharpe", _fmt(d.get("sharpe")), f"{'Equal-weight hold' if r.get('round') == 3 else 'BTC hold'} {_fmt(r['benchmark']['sharpe'])}"),
             ("Verdict", status.split(";")[0].split(",")[0], status)]
    body.append('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{a}</div><div class="value" style="font-size:22px">{b}</div>'
        f'<div class="note">{c}</div></div>' for a, b, c in tiles) + "</div>")

    if r.get("symbol_table"):
        st = r["symbol_table"]
        body += [f"<h2>Per market (dev)</h2><p class='small muted'>{_fmt(d.get('trades_per_year', 0) / 52)} trades "
                 f"per week across the portfolio; a new trade in {_fmt(r.get('weeks_with_entry'), 'pct0')} of weeks; "
                 f"about {_fmt(r.get('confirm_years'))} years of forward trading to reach t = 2 at this edge.</p>",
                 _table(["Market", "Trades", "Win", "Avg R", "Long avg R", "Short avg R", "Sleeve Sharpe"],
                        [[x["symbol"], _fmt(x["trades"], "int"), _fmt(x.get("win_rate"), "pct0"),
                          _fmt(x["avg_R"], "f3"), _fmt(x.get("long_avg_R"), "f3"), _fmt(x.get("short_avg_R"), "f3"),
                          _fmt(x.get("sharpe"))] for x in st])]
    gates = [[html.escape(k), _mark(v)] for k, v in r["gates"].items()]
    for k, v in r.get("holdout_gates", {}).items():
        gates.append([html.escape(k), _mark(v)])
    body += ["<h2>Gates</h2>", _table(["Gate", "Result"], gates)]

    charts["eq"] = _equity_spec(r)
    charts["dd"] = _dd_spec(r)
    body += ["<h2>Equity and drawdown</h2>",
             '<div class="card"><div id="eq" class="chart"></div></div>',
             '<div class="card"><div id="dd" class="chart sm"></div></div>']

    stats_rows = [
        ["Trades", _fmt(d.get("trades"), "int")], ["Win rate", _fmt(d.get("win_rate"), "pct")],
        ["Avg R (median)", f"{_fmt(d.get('avg_R'), 'f3')} ({_fmt(d.get('median_R'), 'f3')})"],
        ["Profit factor (R)", _fmt(d.get("pf"))], ["t-stat of mean R", _fmt(d.get("t_stat"))],
        ["Avg R with 2× costs", _fmt(r["stress_2x"].get("avg_R"), "f3")],
        ["CAGR / total return", f"{_fmt(d.get('cagr'), 'pct')} / {_fmt(d.get('total_return'), 'pct')}"],
        ["Sharpe / Sortino / Calmar", f"{_fmt(d.get('sharpe'))} / {_fmt(d.get('sortino'))} / {_fmt(d.get('calmar'))}"],
        ["Max drawdown (longest, days)", f"{_fmt(d.get('max_dd'), 'pct')} ({d.get('dd_days', 0)})"],
        ["Exposure / avg hold", f"{_fmt(d.get('exposure'), 'pct0')} / {d.get('avg_hold_h', 0):.1f} h"],
        ["Long n / avg R", f"{d.get('long_n', 0)} / {_fmt(d.get('long_avg_R'), 'f3')}"],
        ["Short n / avg R", f"{d.get('short_n', 0)} / {_fmt(d.get('short_avg_R'), 'f3')}"],
        ["Fees / funding per trade (R)", f"{_fmt(d.get('fees_R'), 'f3')} / {_fmt(d.get('funding_R'), 'f3')}"],
        ["DSR (amended / N=224 / pre-reg)",
         f"{_fmt(r.get('dsr'), 'f3')} / {_fmt(r.get('dsr_n_all'), 'f3')} / {_fmt(r.get('dsr_prereg'), 'f3')}"],
        ["PBO (CSCV)", _fmt(r.get("pbo"))], ["Plateau (neighbours profitable)", _fmt(r.get("plateau"), "pct0")],
        ["Random-entry null p", _fmt(r["null"]["p_value"], "f3")],
        ["Bootstrap 5th pct return / 95th pct DD",
         f"{_fmt(r['bootstrap']['ret_p5'], 'pct')} / {_fmt(r['bootstrap']['mdd_p95'], 'pct')}"],
        ["Equal-weight buy & hold (CAGR / DD)" if r.get("round") == 3 else "BTC buy & hold (CAGR / DD)", f"{_fmt(r['benchmark']['cagr'], 'pct')} / {_fmt(r['benchmark']['max_dd'], 'pct')}"],
    ]
    cols = ["Dev"]
    if "holdout" in r:
        h = r["holdout"]
        hv = [_fmt(h["trades"], "int"), _fmt(h["win_rate"], "pct"),
              f"{_fmt(h['avg_R'], 'f3')} ({_fmt(h['median_R'], 'f3')})", _fmt(h["pf"]), _fmt(h["t_stat"]), "–",
              f"{_fmt(h['cagr'], 'pct')} / {_fmt(h['total_return'], 'pct')}",
              f"{_fmt(h['sharpe'])} / {_fmt(h['sortino'])} / {_fmt(h['calmar'])}",
              f"{_fmt(h['max_dd'], 'pct')} ({h['dd_days']})",
              f"{_fmt(h['exposure'], 'pct0')} / {h['avg_hold_h']:.1f} h",
              f"{h['long_n']} / {_fmt(h['long_avg_R'], 'f3')}", f"{h['short_n']} / {_fmt(h['short_avg_R'], 'f3')}",
              f"{_fmt(h['fees_R'], 'f3')} / {_fmt(h['funding_R'], 'f3')}", "–", "–", "–", "–",
              f"– / boot p95 {_fmt(h['boot_mdd_p95'], 'pct')}",
              f"{_fmt(h.get('benchmark', {}).get('cagr'), 'pct')} / "
              f"{_fmt(h.get('benchmark', {}).get('max_dd'), 'pct')}"]
        stats_rows = [row + [v] for row, v in zip(stats_rows, hv)]
        cols.append("Holdout")
    body += ["<h2>Statistics</h2>", _table(["Metric"] + cols, stats_rows)]

    yrs = pd.DataFrame(r["years"])
    if "holdout" in r:
        yrs = pd.concat([yrs.assign(period="dev"), pd.DataFrame(r["holdout"]["years"]).assign(period="holdout")])
    else:
        yrs = yrs.assign(period="dev")
    body += ["<h2>By year</h2>", _table(
        ["Year", "Period", "Trades", "Win", "Avg R", "Sum R", "Return"],
        [[str(int(y.year)), y.period, _fmt(y.trades, "int"), _fmt(y.win_rate, "pct0"), _fmt(y.avg_R, "f3"),
          _fmt(y.sum_R), _fmt(y["return"], "pct")] for _, y in yrs.iterrows()], left=2)]

    charts["mo"] = _monthly_spec(r)
    body += ["<h2>Monthly returns</h2>", '<div class="card"><div id="mo" class="chart" style="height:'
             f'{max(220, 34 * (len(charts["mo"]["data"][0]["y"]) + 2))}px"></div></div>']

    Rv = r["_tr_dev"]["R"].clip(-3, 10).to_numpy()
    charts["rh"] = _hist_spec(Rv, "R multiple per trade (dev, clipped to [-3, 10])", 0.0, "0", nbins=60)
    nd = r["_dists"]["null"]
    charts["nl"] = _hist_spec(nd, "Mean return per trade, random entries", r["null"]["actual_mean_ret"],
                              f"strategy (p={r['null']['p_value']:.3f})", pct=True)
    bm = r["_dists"]["boot_mdd"]
    hm = r.get("holdout", {}).get("trade_mdd")
    charts["bm"] = _hist_spec(bm, "Max drawdown, 10k block bootstraps of dev trades", hm,
                              "holdout" if hm is not None else "", pct=True)
    body += ["<h2>Distributions</h2><div class='grid2'>",
             '<div class="card"><h3 style="margin-top:0">Trade outcomes</h3><div id="rh" class="chart sm"></div></div>',
             '<div class="card"><h3 style="margin-top:0">Strategy vs random-entry null</h3><div id="nl" class="chart sm"></div></div>',
             '<div class="card"><h3 style="margin-top:0">Bootstrap drawdown</h3><div id="bm" class="chart sm"></div></div>',
             "</div>"]

    cfg = r["_cfg"].sort_values("nbhd_sharpe", ascending=False)
    pk = list(r["grid"].keys())
    rows, hl = [], None
    for i, (_, c) in enumerate(cfg.iterrows()):
        if int(c["cfg"]) == r["chosen_cfg"]:
            hl = i
        rows.append([str(c[k]) for k in pk] + [_fmt(c["trades"], "int"), _fmt(c.get("avg_R", np.nan), "f3"),
                                              _fmt(c.get("pf", np.nan)), _fmt(c["sharpe"]),
                                              _fmt(c["nbhd_sharpe"]), _fmt(c.get("max_dd", np.nan), "pct")])
    body += ["<h2>Parameter grid (dev)</h2><p class='small muted'>Sorted by neighbourhood Sharpe; the chosen config "
             "is in bold.</p>",
             _table(pk + ["Trades", "Avg R", "PF", "Sharpe", "Nbhd Sharpe", "Max DD"], rows, left=len(pk), highlight=hl)]
    return _page(f"{r['code']} tearsheet", "".join(body), charts)
