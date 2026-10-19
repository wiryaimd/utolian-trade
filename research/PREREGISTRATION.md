# Pre-registration — BTCUSDT perp strategy research

Written before any parameter sweep was run (2026-10-15). The gates, grids and selection rule below are fixed in code
(`src/btcq/research/run.py`, `src/btcq/strategies/*.py`). Any change to a strategy's logic bumps its `version`,
and the new configs count as additional trials in `research/trials.csv`.

## Data and periods
- Instrument: Binance USDⓈ-M BTCUSDT perpetual. Spot BTCUSDT 1m is used as price history from 2017-08-17 until the
  perp archive begins (2020-01-01).
- **Dev:** 2017-08-17 → 2023-12-31. Funding-based strategies start at 2020-01-01, when real funding history begins.
- **Holdout:** 2024-01-01 → latest. It is locked in code (`store.unlock_holdout`) and evaluated **once**, only for
  strategies that pass every dev gate. Every unlock is logged in `research/holdout_log.jsonl`.

## Execution and costs
- **Signals and fills:** signals are computed on closed bars and filled at the next bar's open.
- **Stops and targets:** checked on the 1m path. If both are touched in the same minute, the stop is assumed first.
  A gap through the stop fills at the open.
- **Fees:** taker 0.05% per side, maker 0.02%. A limit order fills only if price trades through it.
- **Slippage:** 1 bp per side on market and stop fills.
- **Funding:** actual historical funding is charged at every settlement. Before 2020 a synthetic 0.01% per 8h is used.
- **Sizing:** 1% of equity is risked at the stop, notional is capped at 3x equity, and one position is held at a time.

## Selection rule (per strategy)
Among configs with at least 100 dev trades, pick the one with the highest **neighbourhood-average Sharpe**: the mean
of its own annualised Sharpe and that of its grid neighbours (one step away in one parameter). This picks a plateau
rather than a peak.

## Acceptance gates (all must pass)
1. **Trade count:** at least 100 trades in dev, and at least 30 in the holdout.
2. **Dev performance**, net of 1x costs: profit factor in R of at least 1.2, and t-stat of mean R of at least 2.0.
3. **Cost stress:** mean R is still above 0 with all costs doubled.
4. **Deflated Sharpe Ratio of at least 0.95**, with N = all dev trials in the registry and V[SR] = their
   cross-sectional variance.
5. **PBO (CSCV, 16 slices) of at most 0.5** over the strategy's grid.
6. **Plateau:** at least 70% of the chosen config's grid neighbours have positive total return.
7. **Random-entry null:** 1,000 permutations with the same holding times and long/short mix, entered at random
   strategy-bar opens. The strategy's mean per-trade return must beat it with p < 0.05.
8. **Year concentration:** no single calendar year holds more than 50% of total R.
9. **Holdout:** mean R above 0, annualised Sharpe at least 50% of the dev Sharpe, and max drawdown no worse than the
   95th percentile of block-bootstrapped dev drawdowns for the same number of trades.

## Hypotheses and grids
| Code | Hypothesis | Grid size |
|---|---|---|
| S1 | Donchian trend / TSMOM (4h) | 30 |
| S2 | Volatility squeeze breakout (4h) | 36 |
| S3 | 4h trend + 15m pullback | 36 |
| S4 | Asia-range session breakout (15m) | 24 |
| S5 | Intraday time-series momentum (15m) | 32 |
| S6 | Funding crowding contrarian (4h) | 18 |
| S7 | Capitulation / squeeze reversal (15m) | 24 |
| S8 | Range-regime VWAP mean reversion (15m) | 24 |

Total: 224 dev trials before any revision.

Filter overlays (Coinbase premium, DVOL, taker flow, OI) are tested only on strategies that pass, and each one adds
to the trial count.

## Amendment 1 — Deflated Sharpe null (2026-10-15, made AFTER the first dev sweep)

**What happened:** the pre-registered estimate of V[SR] was the cross-sectional variance of all 224 trials' Sharpe
ratios. Several families (S5, S7, S8) lose heavily to fees, with annualized Sharpe as low as −10. That loss comes
from costs, not estimation noise, and it inflates V[SR] to 1.2e-2 per day. The resulting "expected maximum Sharpe
of skill-less trials" was about 5.9 annualized. Under it, every strategy, including any conceivable real one, gets
DSR = 0.000, so the gate is degenerate.

**Amended gate (primary):**
- V[SR] = 1/(T−1). This is the sampling variance of a Sharpe estimate when the true Sharpe is zero, which is the
  False Strategy Theorem's null.
- N = the effective number of independent trials. It is the participation ratio of the eigenvalues of the
  correlation matrix of all trials' daily returns (`stats.effective_trials`), and came out at 14.8 for the 224
  trials.
- The 0.95 threshold is unchanged.

**Disclosure:** this amendment was chosen after seeing results. It turns S1 from fail into pass on this gate
(0.976). The report also shows, for every strategy:
- the DSR with N = 224 treated as independent: S1 = 0.81, which would fail;
- the pre-registered value: 0.000 for all strategies.

No other gate, grid or selection rule was changed.

## Round 2 — registered 2026-10-15, before any round-2 backtest was run

The user asked for: S1 long-only, S1 with volatility-targeted sizing, an S1+S2 combination, and S2 on its untouched
holdout. Every new configuration is recorded in `research/trials.csv`. Afterwards the effective trial count N_eff is
recomputed over all trials from both rounds, and **every** strategy, including the round-1 strategies, is re-gated
with the new N_eff.

### New strategies

| Code | Definition | Grid |
|---|---|---|
| S1L | S1 rules, long entries only. Exits are unchanged, and short breakouts are ignored instead of reversing. | S1 grid (30) |
| S1V | S1 rules on both sides, but sized for 30% annualised volatility (see below). The stop is still active. | S1 grid (30) |
| S1LV | Long-only and vol-targeted. | S1 grid (30) |
| C12 | S1 at its chosen config (n=40, k=3.5, channel) plus S2 at its chosen config (bb_n=20, q=0.3, stop=atr, exit=trail). Each sleeve risks 1% on a shared account, and the combined daily return is the sum of the two sleeves'. | 1 fixed portfolio |

**Vol-target sizing (S1V, S1LV):**
- Notional = equity × 0.30 / σ, capped at 3× equity.
- σ = the standard deviation of 4h log returns over the last 180 bars (30 days), annualised by √(6·365).

**C12 gates:**
- PBO and plateau are not defined for a single fixed portfolio, so C12 inherits the worse of its components: max PBO
  and min plateau.
- Trade-level gates (count, PF, t-stat, cost stress, random-entry null, year share) use the union of both sleeves'
  trades.

### Holdout status (decided now, before results)
- **S2 — clean.** Its 2024–26 data has never been evaluated. S2 missed the dev DSR gate (0.946 < 0.95), so it is
  not a finalist. It is evaluated at the user's explicit request. The result is valid out-of-sample evidence, but S2
  can only reach "final pass" if it also passes the re-gated dev checks.
- **S1L, S1V, S1LV — contaminated.** The long-only idea came from S1's holdout long/short split, and all three share
  S1's signals, whose holdout has already been seen. Their holdout numbers are reported for information only and
  **cannot pass or fail**. Forward paper trading is their real out-of-sample test.
- **C12 — partly contaminated** (through its S1 sleeve). It is reported with that label and can't pass or fail.

## Amendment 2 — monotone effective N (2026-10-15, made AFTER the round-2 dev sweep)

**What happened:** after adding the 91 round-2 trials, the eigenvalue participation ratio *fell* from 14.8 to 8.9.
The trials are near-duplicates of S1, so the S1 cluster's eigenvalue dominates the spectrum. Under Amendment 1 that
would have lowered the luck threshold, and S2's DSR would have risen from 0.946 to 0.970, a "pass" earned purely by
testing more S1 variants. A multiple-testing correction must never become more lenient as trials are added.

**Fix:** the N used in the DSR is the maximum participation ratio ever recorded in
`research/neff_history.jsonl`. It can never decrease. Round 1 recorded 14.8 over 224 series, and round 2 recorded
8.9 over 315 series, so 14.8 is used.

**Disclosure:** this was chosen after seeing round-2 results. It moves S2 back from pass to fail, which is the
conservative direction. The fully conservative DSR, with all 315 trials treated as independent, is still reported
for every strategy.

## Round 3 — registered 2026-10-15, before any round-3 backtest was run

**Goal:** at least 1 trade a week (52/yr), with 2+ a week preferred. Execution is **auto-execute**, modelled as a
1-minute latency on market entries and signal exits; resting stops and targets have no latency.

**Markets:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT and BNBUSDT perpetuals.
- **Slippage per side:** BTC and ETH 1 bp; SOL, XRP and BNB 2 bp. Taker fee 0.05% and maker fee 0.02%.
- **Funding:** each symbol's real funding history.
- **Portfolio:** the same params run on every symbol as equal-capital sleeves, with **no per-symbol tuning**.
- **Sizing:** each sleeve risks 1% × N_symbols of its own capital at the stop, with notional capped at 3× the
  sleeve. That equals 1% of the total account per trade, with total notional ≤ 3×.
- **Portfolio daily return:** the mean of the available sleeves' daily returns.
- **OI availability:** an OI snapshot stamped t is usable from t + 5 minutes.

### Families (1h horizon unless noted)

| Code | Family | Grid | Dev window |
|---|---|---|---|
| F1 | S1 frozen (4h, n=40, k=3.5, channel) on all 5 symbols | 1 | 2020-01 → 2023-12 |
| F2 | 1h Donchian trend, ATR stop, chandelier or channel exit | n{24,48,96} × k{2.5,3.5} × exit{chandelier,channel} = 12 | 2020-01 → 2023-12 |
| F3 | Taker-flow continuation: z of k-hour net taker flow vs 30 days, with a VWAP confirmation. ATR stop at 2×ATR, time exit H | k{4,12,24} × thr{1.5,2.0} × H{6,24} = 12 | 2020-01 → 2023-12 |
| F4 | OI-confirmed N-hour breakout | N{12,24,48} × oi_thr{0, 1σ} × exit{chandelier, time 2N} = 12 | 2021-12 → 2023-12 |
| F5 | OI-flush reversal: 1h OI change z ≤ −z_oi plus a move > 1.5×ATR → fade | z_oi{2.5,3,4} × H{4,12} × exit{time, 50%-retracement target} = 12 | 2021-12 → 2023-12 |
| F6 | Premium dislocation: z of the 1h perp premium vs 30 days beyond ±z → fade | z{2,2.5,3} × H{4,8,24} = 9 | 2020-01 → 2023-12 |
| F7 | Equal-weight ensemble of the F-families that pass every dev gate except frequency (each needing ≥30 trades/yr) | 1 fixed rule | overlap of components |

### Gates
All earlier gates apply: trade count ≥100, PF ≥1.2, t ≥2, 2× cost stress, DSR ≥0.95 on the ratcheted N_eff over
all rounds, PBO ≤0.5, plateau ≥70%, random-entry null p <0.05, and year share ≤50%. Added:
- **Frequency:** at least 52 portfolio trades per year in dev **and** in the holdout. Selection is restricted to
  configs with ≥52 trades/yr.
- **Cross-symbol consistency:** mean R > 0 in at least 3 of the 5 symbols in dev.
- **Reported and used for final ranking, not as a gate:** trades per week, the share of weeks with an entry, and the
  years of forward trading needed to reach t = 2 at the dev edge, (2σ/μ)² ÷ trades per year.

### Holdout (2024-01 → latest), labels fixed now
- **F3, F4, F5, F6: clean** on all 5 symbols. Disclosure: Claude has seen aggregate BTC behaviour in 2024–26
  (longs beat shorts, 2025 was weak).
- **F1, F2:** these are S1-derived. Their holdout gate is evaluated on the **4-alt portfolio**, and the BTC sleeve's
  holdout is information only.
- **F7:** inherits the worst label among its components.
- Each family's holdout is evaluated once, and only if the family passes the dev gates.

## Deviation 1 — F1 holdout, user-approved (2026-10-15, recorded BEFORE the evaluation)

F1 failed the round-3 dev gates on DSR only (0.747 < 0.95), so under the pre-registered rules its holdout stays
locked. The user explicitly authorised a one-time evaluation anyway.

**How it runs:** F1's chosen config (4h, n=40, k=3.5, channel) on the 4-alt portfolio, ETH/SOL/XRP/BNB, for
2024-01 → latest. This is the pre-registered round-3 holdout scope for S1-derived families. The BTC sleeve is
information only, and the holdout gates and costs are unchanged.

**What the result can and can't mean:**
- F1 **cannot** become a "final pass", because it failed a dev gate.
- The result is clean out-of-sample evidence on markets and dates never used for tuning or evaluation, and it will be
  reported as such.
- The run is logged in `research/holdout_log.jsonl`.

## Round 4 — registered 2026-10-16, before any 1d or 1m data of the new markets was downloaded

**Why.** The user wants about one trade a day. Round 3 showed that trading F1 faster destroys its edge. F2 (the 1h
version) is worse at every lookback, and mixing it into F1 lowers Sharpe from 1.20 to 1.10. So the extra trades come
from breadth instead: the same frozen rule on more markets. Most of those markets have never been used in any round,
which makes this a replication test of F1.

**Code and frozen inputs.**
- Code commit `bd901500e2a0ff675b7dc17285049278bca421e6`.
- Files frozen in this commit, all in `research/round4/`:
  - `exchange_info_um_2026-10-16.json`: fetched 2026-10-16 from the public `fapi/v1/exchangeInfo` via the user's
    VPS. SHA256 `918bc3087252d4fee05e9cf8f49039b9c182e3a304a1fef4e63180d18b2fa338`.
  - `symbol_overrides.csv` (SHA256 `eeb364767d13efaeae4c173a70374027267a5b2c06e22ac866fb3d4a9349ac3e`)
  - `symbols.csv` (SHA256 `f7ab96c113b6d2761fa1357b4943d0bd80b93644d7f0dbfd47dc76797bbea0ff`)
- User decisions (2026-10-16):
  - crypto-only universe;
  - meme coins kept, and reported separately as information;
  - 0.30% risk per trade;
  - the decision rule below, fixed in advance.

### Trials
U1 and U1L are version 1 and add 2 dev trials. The trial registry and the N_eff ratchet (Amendment 2) apply, and
every earlier strategy is re-gated afterwards. Stress runs are not trials.

- **U1:** F1 exactly, long and short. That is a 4h Donchian breakout (close crosses the prior 40-bar high or low), an
  initial stop at 3.5×ATR(14), an exit on the opposite 20-bar channel, and decisions executed with a 1-minute latency.
- **U1L:** U1 with short breakouts ignored. Information only, except under the U1L rule below.

### Symbol catalogue
- **Source:** every symbol under `data/futures/um/{monthly,daily}/klines/` in the data.binance.vision archive, listed
  2026-10-16. That is 1,066 symbols, including delisted ones.
- **Kept:** names ending in `USDT` with no `_`.
- **Excluded:**
  - stablecoin bases (USDC, BUSD, TUSD, USDP, FDUSD, DAI, USDE, USD1, RLUSD, PYUSD, USDS, FRAX, LUSD, GUSD, EUR,
    EURI, AEUR, XUSD, USTC, UST);
  - index contracts (BTCDOM, DEFI, FOOTBALL, BLUEBIRD, or `underlyingType` INDEX);
  - non-crypto underlyings (`contractType` TRADIFI_PERPETUAL, or `underlyingType` other than COIN);
  - tokenized gold (PAXG, XAUT);
  - the overrides file. DOTECOUSDT is excluded there as a composite index.
- **Counts:** 678 kept. Excluded: 220 non-crypto, 157 not a USDT perpetual, 5 index, 3 stablecoin, 2 tokenized gold,
  1 override.
- **The snapshot only classifies symbols and gives delisting status.** It never decides eligibility or ranking.
- **Missing from the snapshot:** 29 kept symbols, all delisted coins such as LUNA, MATIC, SRM, EOS, BTT and HNT. They
  are classified by the lists above.
- **Meme flag:** `underlyingSubType` contains "Meme", or the overrides file says so.
- **Normalized base:** strip a leading 1000000, 100000, 10000, 1000 or 1M before a letter.
- **Separate symbols:** renamed tickers (MATIC/POL, BTT/1000BTTC) and LUNA/LUNA2.

### Universe (each UTC month m, from 2020-04 to 2026-09, using only data dated before m)
- **Input:** daily quote volume from the archive's 1d klines. Only volume and dates are stored, never prices.
  Days missing from a truncated monthly file are patched from the daily 1d archive. Days still missing after that
  had no archived trading.
- **Eligible:**
  - the first archived day is at least 90 days before m;
  - quote volume > 0 on each of the 7 days before m;
  - not excluded.
- **Rank:** median daily quote volume over the 90 days before m, with missing days counted as 0. Same normalized
  base: the higher median is kept. Take the top 20.
- **Commit:** `universe.csv` is committed after it is built and before any 1m download.
- **Gated set:** members other than BTC, ETH, SOL, XRP and BNB. Those five are information only, because their F1
  results have been seen.
- **Full universe:** all members, information only. This is what the bot would trade.

### Execution
- **Entries:** only on decisions whose 4h bar closes inside a member month. After a coin leaves the universe, its open
  position runs to the normal exit. Signals are computed on the full price history.
- **Path:** stops are checked on the 1m path. A stop wins when the stop and another event fall in the same minute, and
  a gap through the stop fills at the open.
- **Delisting or settlement:** any stretch of more than 24h with no traded minute (zero volume or no rows)
  counts. That includes a stretch inside the data and one where trading stops more than 24h before the window end.
  - An open position is closed at the last traded 1m close before the stretch, with an extra 1% adverse.
  - No new entries start until 7 days after trading resumes.
  - This covers ordinary delistings and also contracts that were delisted and later relisted under the same
    ticker.
  - The snapshot's status is not used for this rule.
- **Data rules:** drop the trailing zero-volume minutes. Fill funding gaps longer than 9h with the synthetic 0.01%/8h
  rate. No other cleaning.
- **Data report:** written per stage, and committed before any strategy run.
  - Red flags: 1m coverage under 90% in any member month, funding gaps over 5% of member hours, or an archive that
    ends early for a symbol the snapshot lists as TRADING.
  - Any red flag stops the run until the data is fixed.

### Costs
- **Fees:** taker 5 bp per side.
- **Slippage per side, by rank in the month of the fill:**
  - ranks 1–10: 2 bp;
  - ranks 11–20: 3 bp;
  - not a member: 5 bp.
- **Funding:** each symbol's real funding history.
- **2× stress:** fees and slippage doubled; funding and the delisting penalty unchanged.
- **Worst-of-minute stress:** stops fill at the 1m low for longs and the 1m high for shorts, plus slippage.

### Sizing
- **Slots:** 20, one sleeve per symbol worth 1/20 of the account. Each sleeve risks 6% of its own capital at the stop
  with notional capped at 3× the sleeve. That is 0.30% of the account per trade and at most 0.15× per position.
- **Accounting:** account daily return = Σ sleeve returns / 20, with empty sleeves at 0. Trade return = sleeve trade
  return / 20.
- **Cap:** at most 20 open positions, which means total risk at entry ≤ 6% and total notional ≤ 3×.
  - When 20 are open, a new entry is skipped.
  - A signal exit at the open frees its slot for entries in the same minute; a stop exit frees it from the next
    minute.
  - Simultaneous entries are admitted by better rank, then alphabetically.
  - Skipped entries are reported.
- **Two accounts:** gated (all gates) and full universe (information only).

### Windows
- **Stage A (dev):** 2020-04-01 → 2023-12-31.
- **Stage B:** 2024-01-01 → 2026-09-30, the last complete month, because alt funding archives are monthly.
  - It is unlocked once, only if U1 passes every Stage A gate, and the unlock is logged in `holdout_log.jsonl`.
  - Every input is checked before the unlock, using file metadata and timestamps/volume only.
  - If a crash leaves no results, the logged unlock may be resumed exactly once; the resume is logged too.
  - Stage B starts flat.
  - U1L passing alone does not unlock it.

### Stage A gates (gated set)
- **Trades:** at least 100, and at least 52 a year.
- **Edge:** PF ≥ 1.2 in R, and t ≥ 2 on weekly portfolio returns (Monday–Sunday UTC).
- **Stress:** mean R > 0 at 2× costs, and mean R > 0 under worst-of-minute stop fills.
- **Random-entry null:** p < 0.05. Candidate entries come only from member bars. Holding times and directions are the
  strategy's own; costs and seed are per symbol.
- **Concentration:** no calendar year holds more than 50% of total R.
- **Consistency:** at least 60% of gated symbols with 10+ trades have mean R > 0.
- **Inherited from S1:** PBO ≤ 0.5 and plateau ≥ 70% (0.348 and 1.0), as F1 did in round 3.

### Stage B gates (gated set)
- At least 52 trades a year.
- Mean R > 0.
- Sharpe at least 50% of Stage A.
- Trade-sequence max drawdown no worse than the 95th percentile of a block bootstrap of Stage A trades.
- Weekly Sharpe > 0.

### Verdict (gated daily returns 2020-04 → 2026-09, Stage A and Stage B concatenated)
- **REPLICATED:** Stage A passes, Stage B passes, and PSR(0) ≥ 0.95 at N=1. N=1 is justified because the rule, its
  parameters and the universe rule are all fixed before any of these markets' data is seen.
- **STRICT:** REPLICATED, plus DSR ≥ 0.95 at the ratcheted N_eff (14.85 now).
- The DSR on Stage A alone is reported as information.
- **U1L rule:** U1L replaces U1 only if all three hold:
  - U1L is REPLICATED;
  - the PSR of its alpha is at least 0.95, after removing beta to an equal-weight buy-and-hold of the point-in-time
    gated members;
  - its alpha Sharpe beats U1's.

### What each outcome means
- **REPLICATED:** expand the paper bot to the universe at 0.30% risk. That is separate bot work.
- **STRICT, or REPLICATED plus good forward paper results:** candidate for small real money.
- **FAIL:** keep the 5-coin F1 paper bot. No retuning and no variants on this universe. Record that F1's edge may be
  specific to those coins.

### Reported, not gated
- trade-level t;
- results by year and by symbol;
- the meme-coin and major-coin subsets;
- the full universe;
- the share of R from delisted coins;
- concurrency and skipped entries;
- the equal-weight buy-and-hold benchmark.

### Disclosures
- **Already seen:**
  - F1 dev on the 5 majors, 2020–23;
  - the F1 holdout on ETH/SOL/XRP/BNB, 2024–26;
  - aggregate BTC behaviour, 2024–26;
  - general knowledge of crypto 2020–26: the 2021 alt season, the LUNA and FTX collapses, the 2024 meme cycle.
- **Volume ranks include 2024–26:** the universe file holds them, as volume only. The user approved this.
- **Unchanged this round:** the paper bot and F1's parameters.
- **Approximation:** the slot accounting rebalances sleeve capital daily, which is second-order compared with sizing
  on live shared equity.

## Round 5 — registered 2026-10-16, before any ETH/XRP/BNB/SOL spot data is downloaded and before any round-5 backtest

**Why.** Round 4 failed: F1's edge did not carry over to 71 never-used perps, although it held on the 5 majors. The
user now wants about one order a day on the paper bot's 5 markets (BTC, ETH, SOL, XRP and BNB USDT-M perps).

The design panel found no evidenced strategy that makes about one *independent* bet a day after fees on these
markets. The popular 21–23 UTC "evening long" grosses about 9 bp against a 12–14 bp taker round trip, so it was
rejected on arithmetic without spending a trial.

**User decisions (2026-10-16)**
- Daily activity is enough: one order or rebalance a day, even if positions last weeks.
- Families: D1 (a daily long-only Donchian ensemble) and XS (dollar-neutral cross-sectional momentum among the 5).
- Carry and options are out.

**Code:** commit `7aed2c0c55a4ad249e4c54c7727858f849f7c9c3`.

### Protocol change (user decision, 2026-10-16, made after every DSR test in rounds 1–4 had failed)
- **REPLICATED** means a family passes its dev gates and then its one-shot holdout gates.
- **STRICT** means REPLICATED plus DSR ≥ 0.95 on the dev + holdout daily returns, at N = max(ratcheted N_eff after
  round 5, 14.85 + 2), passed as a float.
- It applies to round 5 only. Earlier strategies are not relabelled.
- **Disclosure.** This relaxes round 4's rule, which also required PSR(0) ≥ 0.95, and the decision was taken after
  seeing failures. Each family has one primary config fixed here, and its grid is used only for PBO and plateau,
  never for selection.

### Power table
Monte Carlo with iid Gaussian daily returns, 40,000 simulations, seed 5 (`research/round5/power_table.csv`):
- dev = 1,461 days, passing when weekly t ≥ 2;
- holdout = 1,004 days, passing when mean > 0 and Sharpe ≥ 50% of dev;
- STRICT = DSR at N = 16.85 over 2,465 days.

| True annual Sharpe | P(dev pass) | P(REPLICATED) | P(STRICT) |
|---|---|---|---|
| 0 | 2.4% | 0.41% | 0.02% |
| 0.5 | 15.7% | 6.8% | 1.2% |
| 0.75 | 31.0% | 17.3% | 4.9% |
| 1.0 | 49.9% | 34.2% | 16.0% |
| 1.25 | 68.8% | 54.0% | 34.6% |
| 1.5 | 83.8% | 71.8% | 57.8% |
| 2.0 | 97.7% | 91.5% | 89.3% |

- **Null false-positive rate:** about 0.41% per fixed config (0.02% for STRICT).
- **Upper bounds:** the other gates lower both the power and the false-positive rate.
- **Implied thresholds:** weekly t ≥ 2 needs an annual Sharpe of about 1.0. STRICT needs about 1.34 over 2,465 days.

### Data
- **Execution:** perp 1m only, from 2020-01-01. BTC's stitched pre-2020 spot data is never used for execution.
- **Funding:** real funding per coin.
- **Daily close:** the last 1m close before 00:00 UTC. Decisions are made at 00:00 and filled as takers at 00:01.
- **Warm-up (D1 only)**
  - Spot daily closes before each perp's first day D0, stitched with no basis adjustment.
  - BTC uses the local spot 1m history (2017-08 → 2019-12). ETH, XRP, BNB and SOL use the spot 1d archive up to and
    including the perp's first month + 2 (2020-03, 2020-03, 2020-04, 2020-11), so it never reaches 2024.
  - Missing days of up to 2 in a row are forward-filled.
- **Spot data report:** red flags (any one stops before any strategy run) are:
  - a gap of more than 2 days;
  - a non-positive price;
  - median spot/perp overlap basis above 0.5%;
  - a stitch jump above 1%, or one that can't be measured.
- **The stitch jump**
  - It is the same-day basis at D0.
  - For BTC there is no overlap, so it is measured on adjacent minutes: the spot close at 2019-12-31 23:59 against
    the perp open at 2020-01-01 00:00.
  - BTC's overlap basis can't be checked, and that is reported as such.
- XS uses perp closes only.

### Costs
- **Fees and slippage:** taker 5 bp per side. Slippage per side is 1 bp for BTC and ETH and 2 bp for SOL, XRP and
  BNB.
- **Stress:** the 2× stress doubles fees and slippage. Funding is unchanged.

### D1: daily long-only Donchian ensemble, vol-targeted
After Zarattini, Pagani and Barbon (2025), "Catching Crypto Trends".

**Rule.** For each market and each lookback L ∈ {5, 10, 20, 30, 60, 90, 150, 250, 360} days:
- go long when flat and the close is strictly above the maximum of the prior L closes;
- go flat when the close falls below the trailing level, which is the running maximum since entry of
  MID_L = (max + min of the last L closes, today included) / 2;
- stay flat until L+1 closes exist and σ is ready.

**Sizing.** Each sub-sleeve is sized at entry to min(0.25 / σ, 2.0) × its own capital. σ is the standard deviation
(ddof = 0) of the last 90 daily log returns × √365. Long only, with no other stop.

**Pre-registered engine approximation**
- 45 sub-sleeves (5 coins × 9 lookbacks), each worth 1/45 of the account, sized at entry only (no daily resizing).
- The engine stop is a never-triggering sentinel (0.0).
- R is recomputed as qty × (entry price − trailing level at entry).
- Entries are a level signal, and so are exits (out whenever the state is flat).
- **State-synchronous start:** a position already long at the window start is entered at the first decision
  executed after it.

**Configs:** all 9 lookbacks (primary), {5..60} and {60..360}. All three are subsets of the same 45 runs.

**Dev gates** (on daily and weekly portfolio returns):
- weekly t ≥ 2;
- mean daily return > 0 at 2× costs;
- no calendar year above 50% of the summed daily returns;
- at least 3 of 5 coins with a positive summed contribution;
- PBO ≤ 0.5 over the 3 configs;
- plateau ≥ 70%: both neighbour configs have a positive total return;
- **alpha gate:** the PSR of the alpha after removing beta to an equal-weight buy-and-hold of the 5 perps
  (closes, without funding) is at least 0.95;
- **circular-shift null:** p < 0.05.
  - Each draw uses ONE random offset k ~ U{30 .. n−30} days, applied to every coin and lookback on the shared window
    calendar. Days before a coin's listing are flat.
  - 1,000 draws, seed 55, exact re-simulation through the engine.
  - The statistic is the mean daily net portfolio return.
  - One common offset keeps the coins' joint timing structure. The review showed that independent per-coin offsets
    reject about 11% of the time under no edge when the coins are correlated.

DSR is reported.

**Holdout:** contaminated (the long-only idea came from already-seen holdouts), so it is information only and runs
only inside an XS unlock.

**Outcome:** a dev pass makes D1 a zero-risk shadow sleeve next to F1. It may replace F1 only by user decision after
at least 12 months of shadow running, and it never stacks on F1.

### XS: dollar-neutral cross-sectional momentum among the 5 perps

**Rule**
- **Ranking:** each decision bar ranks the eligible coins by their L-day log return, with ties broken by symbol.
  - Eligible means at least 31 perp closes, a finite score and σ > 0.
- **Sides:** long the top k and short the bottom k, with k = n // 2.
- **Weights:** inverse 30-day volatility within each side.
- **Size:** 0.5× account gross per side at entry, which is an engine size of 2.5w per coin sleeve (5 slots,
  max leverage 2.5).
- **Stop:** each leg has a stop at close × (1 ∓ 4 × MAD30), where MAD30 is the mean absolute daily log return over 30
  days. The stop also defines R.
- **Leg handling**
  - A continuing leg keeps its size.
  - Long → short reverses in the same minute.
  - A stopped-out leg re-enters at the next decision if it is still on its side.

**Configs**
- Weekly decision on the Sunday bar (closing Monday 00:00, filled at 00:01), with L ∈ {7, 14 (primary), 21, 28}.
- Daily decision with L ∈ {14, 21} and a book-level 7-day minimum hold. The whole book is re-formed only when the
  target book changed and at least 7 days have passed since the last re-formation, so it stays k long and k short.

**Dev gates**
- PF ≥ 1.2 (in R);
- weekly t ≥ 2;
- mean R > 0 at 2× costs;
- no entry year above 50% of total R;
- at least 3 of 5 coins with positive mean R;
- PBO ≤ 0.5 over the 6 configs;
- plateau ≥ 70% of the 5 non-primary configs with a positive total return (at least 4 of 5);
- **rank-permutation null p < 0.05:**
  - on every decision date, the eligible coins' ranks are randomly permuted;
  - the permuted ranks go through the same legs, stops and engine;
  - 1,000 draws, seed 5005;
  - the statistic is the mean daily portfolio return with fees and slippage at 0 and funding kept;
- weekly correlation with F1's dev daily returns ≤ 0.3;
- Newey-West HAC alpha t (lag 7, with a T/(T−2) correction) against F1 ≥ 2. Lags 0, 14 and 28 are reported;
- leave-one-coin-out: each 4-coin universe is re-simulated with 5 slots kept, and every run's mean R must be > 0.

The trade count is reported, not gated.

**Holdout gates** (2024-01-01 → 2026-09-30):
- mean R > 0;
- Sharpe ≥ 50% of dev;
- trade-sequence max drawdown ≤ the 95th percentile of a block bootstrap of the dev trades;
- at least 52 trades a year;
- leave-one-coin-out mean R > 0.

**Holdout status:** clean. Disclosure: general knowledge of 2024–26 relative moves (XRP's late-2024 surge, ETH
lagging BTC) points toward success.

### Windows and unlock
- **Dev:** 2020-01-01 → 2023-12-31.
- **Holdout:** 2024-01-01 → 2026-09-30, starting flat.
- **Unlock:** once, only if XS passed every dev gate. Everything is checked before the unlock, from timestamps only.
  - The unlock is logged as `[R5: XS,D1]`.
  - If a crash leaves no XS result, the unlock may be resumed exactly once, logged as `[R5 resume`.
  - XS's verdict is saved before D1's information-only run.

### Decisions fixed in advance
- **XS REPLICATED:** a paper sleeve sharing F1's 3× notional cap, with F1 first.
- **XS STRICT:** candidate for small real money, at the user's decision.
- **D1 dev pass:** shadow sleeve (see above).
- **Any FAIL:** that family is closed on these markets.
- **Nothing passes:** F1 stays alone and the 1-trade/day target is retired for these markets.
- The bot work (a daily/weekly scheduler, netted per-market targets) is separate and happens only after a pass.

### Trials and multiple testing
- 9 dev trials: D1 × 3 and XS × 6, version 1.
- N_eff is recomputed and ratcheted (Amendment 2), and rounds 1–3 are re-gated.
- Round 4's U1/U1L multiple-testing fields are refreshed as information.
- A rerun needs a strategy version bump and adds new trials.

### Reported, not gated
- D1: R statistics, and the share of days with an order.
- D1 exposure tracking: realized sized-at-entry gross exposure against the paper's daily-resized target, as a ratio
  distribution and the share of days outside [0.5, 2].
- XS: worst-of-minute stop fills, ranks/sides/weights, and alpha against the equal-weight hold.
- Results by lookback, coin and year.

### Disclosures
- **Seen in earlier rounds:**
  - S1 and F1 (4h trend) on these markets, dev and holdouts;
  - round 4's result that trend works on these 5 majors but not on 71 other perps;
  - aggregate 2024–26 market behaviour.
- **Approximations:**
  - entry-only sizing;
  - unnetted sub-sleeve costs, which is conservative;
  - an equal-weight benchmark without funding;
  - a stitch with no basis adjustment;
  - SOL's long lookbacks warm up only in 2021.

### Round 5 outcome and post-run notes (2026-10-17, after the dev sweep)
- **D1 failed dev.**
  - Failed: year share 0.610 (one year above 50%).
  - Failed: alpha PSR against the equal-weight benchmark 0.884 (< 0.95).
  - Passed: weekly t 2.49, shift null p 0.009, PBO 0.10, plateau 1.0, 2× costs, 5 of 5 coins positive.
- **XS failed dev.**
  - Failed: PBO 0.578 (> 0.5).
  - Failed: weekly correlation with F1 0.441 (> 0.3).
  - Failed: HAC alpha t against F1 1.41 (< 2).
  - Passed: PF 1.42, weekly t 2.15, rank null p 0.041, leave-one-coin-out positive, plateau 1.0.
- **Consequence (pre-registered):** both families are closed on these markets and no holdout was unlocked. F1 stays
  alone, and the 1-trade/day target is retired for these 5 markets.
- **Independent verification:** three rebuilds from raw dev data, written without project code, reproduced every
  trade and gate value.
- **Wording note:** the text above calls D1's alpha benchmark an "equal-weight buy-and-hold". The registered code
  (7aed2c0) uses the daily-rebalanced equal-weight mean of the listed perps. Literal buy-and-hold variants give an
  alpha PSR of 0.909–0.929, which also fails, and D1 fails the year-share gate independently.

## Round 6 — registered 2026-10-17, before any external price data is downloaded and before any round-6 backtest

**Why.** Rounds 4 and 5 failed. F1's crypto trend edge did not carry over to 71 other perps, and daily-activity
variants on the 5 majors were mostly the same bull-market bet. Crypto coins all move with BTC, so adding coins adds
orders but not independent bets. Trend following has a long record across commodities and equity indices, and those
markets move differently from crypto. Binance now lists them as USDⓈ-M TradFi perps (24/7, 0.02%/0.05% fees).

**User decisions (2026-10-17)**
- The account is eligible for TradFi perps.
- Macro markets only (no single stocks: picking today's big names would be hindsight bias).
- Long and short.
- A new sleeve next to the live crypto F1 paper bot. F1 is unchanged.

**Code:** commit `ac5f312`. New: `src/btcq/data/external.py`, `src/btcq/strategies/r6_macro.py` and
`src/btcq/research/round6.py`; changed: `stats.py` (power_table_td), `registry.py` (optional paths), `run.py` (CLI),
`report.py`, `strategies/__init__.py`, `pyproject.toml` and `.gitattributes`. The 17 files listed in
`round6.CODE_FILES` are hashed at the sweep and checked before the holdout.

### Universe and research series
The universe is fixed: every TradFi perp with at least $5M of 24h volume on Binance on 2026-10-16, single stocks
excluded. The perps only exist since 2025-12, so each is tested on a long external daily proxy (non-rolling where
possible). Slippage tiers come from the same one-day liquidity snapshot.

| Perp | Research series | Source | First date | Slippage per side |
|---|---|---|---|---|
| XAUUSDT | GLD | Yahoo chart API | 2004-11-18 | 2 bp |
| XAGUSDT | SLV | Yahoo | 2006-04-28 | 2 bp |
| XPTUSDT | PPLT | Yahoo | 2010-01-08 | 5 bp |
| XPDUSDT | PALL | Yahoo | 2010-01-08 | 5 bp |
| COPPERUSDT | HG=F, unadjusted front month | Yahoo | 2000-08-30 | 5 bp |
| CLUSDT | WTI spot RWTC | EIA spreadsheet | 1986-01-02 | 2 bp |
| BZUSDT | Brent spot RBRTE | EIA | 1987-05-20 | 2 bp |
| NATGASUSDT | Henry Hub spot RNGWHHD | EIA | 1997-01-07 | 3 bp |
| SPYUSDT | SPY | Yahoo | 1993-01-29 | 3 bp |
| QQQUSDT | QQQ | Yahoo | 1999-03-10 | 2 bp |
| EWYUSDT | EWY | Yahoo | 2000-05-12 | 2 bp |

- **Copper:** HG=F gives about 15 years of dev history; CPER would give about 2. Its roll gaps are disclosed.
- **Prices:** Yahoo's `close`, which is split-adjusted but not dividend-adjusted. A perp holder gets no dividend, and
  the perp's funding has no dividend component. Dividend-adjusted closes are run as information only. EIA prices are
  used as published.

### Data handling (fixed now)
- **Yahoo** (`/v8/finance/chart/{ticker}?period1=-2208988800&period2=<now>&interval=1d&events=div|split`):
  - each row's date is its timestamp in the exchange's own time zone (`meta.exchangeTimezoneName`);
  - a bar stamped 17:00 local or later belongs to the next weekday (an evening session);
  - rows dated on or after the download day are dropped (unfinished), as are null closes; of duplicate dates the
    last is kept.
- **EIA** (`/dnav/{pet|ng}/hist_xls/{series}d.xls`, read with xlrd 2.0.1): sheet "Data 1"; cell (1,1) must equal the
  series id; data from row 3. These are close-only series.
- **Missing days are never filled.**
- **Non-positive closes are dropped.** The only allowed one is CLUSDT 2020-04-20 (WTI spot −36.98, inside the
  holdout). Any other is a red flag. A sensitivity with the CL P&L over the dropped interval (2020-04-17 → 04-21)
  set to zero is reported after the unlock, as information.
- **Storage:** raw downloads (timestamped, never overwritten, kept even when parsing fails) and parsed files stay
  under `data/external/` (not committed). The manifest `research/round6/data_manifest.csv` (download time, URLs, raw
  and parquet SHA256, row counts, splits, drops) is committed. Every load checks the parquet against it and checks
  that every stored date is a UTC midnight in ms, strictly increasing. The download runs once; `--resume` may only
  fill markets that are missing and never overwrites.

### Data report and red flags
`research/round6/data_report.csv` is committed before any strategy run. Holdout rows are used only for dates, counts,
price signs and whether consecutive closes are equal; anything built on returns uses dev rows only. Any one red flag
stops the round; a fix needs a dated amendment before any strategy run.
1. The download or parse fails, or an EIA series key doesn't match.
2. The first date is more than 30 days later than expected.
3. The last date is before 2026-09-29.
4. Any row is dated on a weekend.
5. A non-positive close that isn't allowlisted.
6. A run of more than 3 missing days: more than 3 consecutive reference days missing, or more than 3 weekdays
   between consecutive rows anywhere in the series (this also covers the years before SPY starts in 1993). The
   reference days, from SPY's first to its last date, are the NYSE calendar by rule (weekdays minus the NYSE
   holidays: New Year, MLK from 1998, Presidents, Good Friday, Memorial, Juneteenth from 2022, Independence,
   Labor, Thanksgiving, Christmas, observed on Friday when on a Saturday and on Monday when on a Sunday, except
   that a Saturday New Year's Day is not observed; and the special closures 1994-04-27, 2001-09-11..14,
   2004-06-11, 2007-01-02, 2012-10-29..30, 2018-12-05, 2025-01-09), plus SPY's own dates, plus any date that at
   least 2 of the other US-listed ETFs (GLD, SLV, PPLT, PALL, QQQ, EWY) have. So holes in SPY itself count as
   missing days for SPY in every year. The special closures are excluded everywhere, plus, for BZUSDT only, the
   England & Wales bank holidays (rule-based list with substitute days and the special days of 1995, 1999, 2002,
   2011, 2012, 2020, 2022 and 2023), on which Brent spot is not assessed.
7. More than 3% of reference days missing in a calendar year, judged only on years with at least 120 reference
   days (a partial first or last year can't flag on one or two days).
8. 5 or more equal closes in a row.
9. Null rows above 1%.
10. Dev rows of a Yahoo series with a reverting spike: |r_t| > 15% and |r_t + r_t+1| < 0.2·|r_t| (log returns). For
    EIA series this is reported only, because Henry Hub spot spikes are real.
11. Dev rows: |log return| > ln 1.8 on a listed split date (an unadjusted split), or a listed dev split date with
    no row.
12. Dev rows of a Yahoo series: any |log return| > ln 1.8 (an unadjusted split Yahoo didn't list, or a bad print; no
    real daily move in these ETFs or copper futures comes close).

### Holdout lock
- Dev loads return only rows dated before 2016-01-01. The holdout runs 2016-01-01 → 2026-09-30 (end exclusive
  2026-10-01).
- Round 6 has its own unlock flag and logs `[R6:` in `research/holdout_log.jsonl`. It never unlocks the crypto
  2024+ data.
- The unlock runs once, only if M1 passed every dev gate. Everything is checked before it from timestamps, hashes and
  committed files: M1.json, its dev daily returns, F1's dev daily returns, the round 6 trial registry and run log
  must be committed; the strategy version must equal the swept version; and the SHA256s stored at the sweep must
  still match for the round 6 code paths (data, rule, runner, engine, statistics, registry, CLI), the data manifest
  and F1's daily returns (line endings normalised), so no code or input can change between dev and holdout. The
  holdout bootstrap's size, block and seed are fixed in the round 6 code and stored in M1.json.
- Every Round 6 unlock writes a structured record (round 6, unlock or resume, version) to the holdout log. If a crash
  leaves no holdout result, the unlock may be resumed exactly once, for the same version (`[R6 resume`).
- Once the holdout is touched (any Round 6 unlock record, or a holdout result), Round 6 is closed: the dev sweep can
  never run again, with or without a version bump. A resume record with no earlier unlock is refused and flagged.
- The dev sweep starts in two steps: the first `r6-sweep` only logs a start record (version, code SHA256, data
  manifest SHA256) and stops; the run log is committed; the second `r6-sweep` runs. A sweep is refused if any start
  record for its version, in the working copy or in any committed version of the run log on any branch, has a
  different code or data SHA256. So an interrupted sweep may be retried only with identical code and data, and
  neither a checkout, nor a commit restoring an older log, nor a branch cut before the start can undo it. Any change
  needs a version bump.

### Execution
- One engine row per trade date, stamped 23:59 UTC, with open = high = low = close.
- A decision at a trade date's close fills at the next trading day's close (latency 0). Fridays fill on Monday.
- Sub-sleeves are not netted, which is conservative on costs.
- Holdout positions start in sync with the signal state: a position that is on at 2016-01-01 is entered at the first
  executed decision.

### Rule M1
For each market and each lookback L ∈ {5, 10, 20, 30, 60, 90, 150, 250, 360} trading days (the market's own rows):
- **Long:** when flat, go long if the close is above the maximum of the prior L closes. Exit when the close falls
  below the trailing level, which is the running maximum since entry of MID_L = (max + min of the last L closes,
  today included) / 2.
- **Short:** the mirror rule, computed as the long machine on −close. Long and short can never be open together; a
  flip is an exit plus a reversal at the same fill.
- No position until L+1 closes exist and σ is ready.
- **Sizing:** each sub-sleeve is sized at entry to min(0.25 / σ, 2.0) × its own capital, where σ is the standard
  deviation (ddof 0) of the last 90 daily log returns × √252. There is no daily resizing.
- **Stops:** never-triggering sentinels. R is measured from the trailing level at entry.
- **Accounting:** 99 sub-sleeves (11 markets × 9 lookbacks), each 1/99 of the account. The account's daily return on
  the UTC calendar is the sum of the sub-sleeves' daily returns / 99; days with no trading are 0.
- **Configs:** all9 (primary), fast5 {5..60} and slow5 {60..360}, used only for PBO and plateau, never for selection.
  That is 3 trials.

### Costs
- Taker fee 5 bp per side, plus the slippage tiers above.
- Base funding is exactly 0. That matches 2026's median and Binance's 0% interest component; it is uncertain.
- **Funding stress:** every open position pays 5%/yr of its notional on calendar days, long or short. It is charged
  on each trade date for the calendar days since the market's previous trade date (Mondays carry 3), as |notional /
  equity| at the previous daily mark × 5% × days / 365, and subtracted from the sub-sleeve's daily return (floored
  at −100%). It does not compound into position sizes or trigger extra liquidations (an approximation; the plan's
  two-pass engine version was dropped because the engine stops a sleeve when its realized cash reaches 0, which
  stopped long, profitable positions once the funding they had paid exceeded their cash).
- **2× stress:** fees and slippage doubled.

### Hard errors and liquidation
Positions are sized at entry only, so after a large adverse move a sub-sleeve's leverage keeps growing. A sub-sleeve
is **liquidated** at the first daily mark where its equity ≤ 0 or its open notional ≥ 20× equity (about where an
exchange would close a 2× isolated position): it is closed at max(equity, 0) at that mark and stays flat. A
liquidation is a real outcome, in dev and holdout alike: it is booked this way in every run (base, 2×; the funding
stress uses the base run's positions) and reported. The engine's later trades in that sub-sleeve are dropped, and the
trade open at (or exiting on) that mark is closed there at the booked equity, so its loss is capped at its capital.
- **Integrity errors:** engine positions differ from the state machine on an executed decision (every decision
  executed before a liquidation; without one, every decision but the window's last, which meets the end-of-window
  close); any stop exit.
- **Dev:** an integrity error stops the sweep. A fix needs a strategy version bump. The error is logged (the log is
  then committed), and the same version is refused while that record is in the working copy or in any committed
  version of the log; the committed start record pins the code and data in any case.
- **Holdout:** an integrity error is recorded as a holdout FAIL, with no fix and no rerun.

### Windows
- **Dev:** 1986-01-01 → 2015-12-31. Each market starts from its first ready day, so the early years have few markets.
- **Holdout:** 2016-01-01 → 2026-09-30, one shot.

### Power table
Monte Carlo with iid Gaussian returns on 252 trading days a year and 5-day weeks, 20,000 simulations, seed 6
(`research/round6/power_table.csv`, produced by `r6-power` at commit `ac5f312`):
- dev = 30 years, passing when weekly t ≥ 2;
- holdout = 10.75 years, passing when the mean is > 0 and the Sharpe is ≥ 50% of dev;
- STRICT = DSR ≥ 0.95 at N = 17.85 over dev + holdout.
- **Constant:** the full book for all 30 dev years. **Ramp:** the number of live sub-sleeves follows the
  pre-registered source start dates plus warm-up (source facts, not prices): about 175 market-years, or 15.9
  full-book years of 30. The account's mean scales with the live share and its volatility with its square root.

| True full-book Sharpe | Constant: P(dev pass) | Constant: P(REPLICATED) | Constant: P(STRICT) | Ramp: P(dev pass) | Ramp: P(REPLICATED) | Ramp: P(STRICT) |
|---|---|---|---|---|---|---|
| 0 | 2.0% | 0.55% | 0.02% | 2.0% | 0.59% | 0.01% |
| 0.25 | 27.0% | 13.8% | 2.4% | 16.2% | 8.3% | 1.2% |
| 0.4 | 57.5% | 38.7% | 14.4% | 33.9% | 23.7% | 6.6% |
| 0.5 | 76.7% | 57.9% | 31.7% | 49.4% | 38.7% | 15.6% |
| 0.6 | 89.9% | 73.6% | 54.8% | 65.5% | 56.5% | 31.4% |
| 0.75 | 98.3% | 86.2% | 80.2% | 84.2% | 77.9% | 60.1% |
| 1.0 | 99.95% | 94.1% | 94.0% | 97.6% | 95.3% | 92.4% |

- **Null false-positive rate:** about 0.6% per config for REPLICATED and 0.02% for STRICT.
- **Upper bounds:** the other gates (shift null, PBO, plateau, alpha, year share, markets, F1, MDD) are not
  modelled, so they lower both the power and the false-positive rate.
- **Implied thresholds:** a dev pass needs a realized full-book Sharpe of about 0.37 (constant) or 0.50 (ramp).
- **Funding gate:** a 5%/yr cost on about 0.8× gross exposure at 8–9% account volatility is roughly 0.45 Sharpe
  units (an estimate from typical market volatilities, not from the data). A hurdle below the implied threshold is
  already implied by the weekly-t gate. So the funding gate binds first only in the constant variant (at a true
  Sharpe of 0.5: 60.6% for dev pass and funding gate, against 76.7% for dev alone). In the ramp variant it binds
  only if the true hurdle is above 0.50 (at a true Sharpe of 0.6 with a 0.6 hurdle: 50.2% against 65.5%). This
  corrects the plan's expectation that funding would be the binding gate.

### Dev gates (account returns, all9)
1. Weekly t ≥ 2 (Monday–Sunday UTC weeks of the calendar account).
2. Mean daily return > 0 at 2× fees and slippage.
3. Mean daily return > 0 under the funding stress.
4. No calendar year above 50% of the summed daily returns.
5. At least 7 of 11 markets with a positive summed contribution (sum of their 9 sub-sleeves' daily returns / 99).
6. **Shift null p < 0.05.**
   - Each sub-sleeve's base-run daily return is decomposed exactly into exposure × market return − costs, where
     exposure is its signed notional / equity at each daily mark, forward-filled over non-trading days (no zeros are
     inserted), and costs are the fees and slippage it paid on fill days. On a liquidation day with equity ≤ 0 the
     "cost" also holds the credit of the −100% cap (the loss beyond capital); like all costs it stays in every draw,
     which is conservative.
   - Each draw rolls every sub-sleeve's exposure path circularly within its own active span (from its first ready
     day) by one common offset k ~ U{30 .. S_min − 30} calendar days, where S_min is the shortest active span, against
     the unshifted market returns. Time in market, sizes, leverage paths and costs paid are exactly the base run's;
     only their timing against returns changes. One k keeps markets and lookbacks aligned on unwrapped days only (each
     sub-sleeve wraps at its own span start).
   - 1,000 draws, seed 66, base costs, no funding. Statistic: the mean daily net account return. p = (1 + #{null ≥
     actual}) / 1,001. At run time the decomposition is checked on every calendar day (exposure × return must
     reproduce the engine's daily return on days without a fill or liquidation, and fill days other than a
     liquidation day must carry a cost ≥ 0), and k = 0 must reproduce the engine's statistic.
   - **Calibration (synthetic, before any data):** 4,480 no-edge simulations in 12 scenarios (random walks, drifts,
     GARCH and regime volatility, correlated groups, 3 and 11 markets) gave a false-positive rate of 5.3% ± 0.3% at
     p < 0.05; on the realistic 11-market setups it was 5.9% ± 0.8%. Planted trends of 10 bp/day were detected 94%
     of the time. Known caveats: the null is about 10–15% narrower than the no-edge spread, and draws with k near
     S_min wrap a short-span sub-sleeve's later exposure onto the returns that formed it (conservative); at the 5%
     level the two roughly cancel. Very small p-values (below about 0.02) overstate the evidence when S_min is short.
7. PBO ≤ 0.5 (CSCV over the 3 configs' dev daily returns, 16 slices).
8. Plateau: fast5 and slow5 both have a positive compounded total return.
9. **Alpha:** PSR ≥ 0.95 of the residual after OLS beta to an equal-weight benchmark. The benchmark, literally: each
   market's close-to-close return on its own trade date; each UTC day is the mean over the markets with a return
   that day; days with none are 0. It is a daily-rebalanced equal-weight long of the available markets, with no
   costs and no funding.

### Holdout gates
1. Mean daily return > 0.
2. Sharpe ≥ 50% of the dev Sharpe.
3. Max drawdown of the holdout account ≤ the 95th percentile of a block bootstrap of dev daily returns: block 30,
   10,000 paths of the holdout's length, seed 7, drawn from the first day all 99 sub-sleeves are ready (about
   2011-06) to 2015-12-31. Earlier years have fewer markets and smaller drawdowns, which would make the gate unfairly
   strict; the source window is short.
4. At least 7 of 11 markets with a positive contribution.
5. Weekly correlation with F1's 2020–23 daily returns ≤ 0.3, over the overlap.

### Verdict
- **REPLICATED** = every dev gate and every holdout gate.
- **STRICT** = REPLICATED plus DSR ≥ 0.95 on dev + holdout daily returns at N = max(round 6's own ratcheted N_eff,
  14.85 + 3) = 17.85. 14.85 is the largest crypto N_eff ever recorded; +3 counts one per family tried after round 4
  (D1, XS, M1).

### Decisions fixed in advance
- **REPLICATED:** M1 becomes a paper sleeve next to F1. The bot work (a daily scheduler and netted per-market orders
  on the TradFi perps) is separate and comes after.
- **STRICT:** a candidate for small real money, at the user's decision.
- **FAIL:** M1 is closed on these markets with this data. No reruns or variants.

### Trials and multiple testing
- Round 6 keeps its own registry: `research/round6/trials.csv` and `research/round6/neff_history.jsonl`, with trial
  returns in `reports/results/round6/`. Mixing a 1986–2015 calendar into the crypto set would zero-fill the
  calendars, push correlations to about 0 and inflate the crypto N_eff for good. Round 6 never calls `run.cmd_gates`.
- 3 dev trials (all9, fast5, slow5), version 1. A rerun needs a version bump and adds new trials.
- DSR at max(round 6's own N_eff, 2) and at N = 3 is reported as information.

### Reported, not gated
- R statistics, long vs short, per market, per lookback, per year.
- Netted market orders per trading day from the full-book day (expected well above 1–2 a day, since every sub-sleeve
  entry or exit is an order), and gross and net exposure.
- 2× costs plus funding stress together; dividend-adjusted closes; a literal buy-and-hold benchmark.
- After the unlock: the CL 2020-04 neutralised sensitivity, HAC alpha against F1, and DSR at the other Ns. A check of
  the perps against their proxies over 2026 is run separately (Binance data is reachable only from the VPS).

### Disclosures
- **The holdout isn't blind:** trend-fund performance by year is widely known (2008 and 2014 strong, 2016–19 weak,
  2022 strong).
- **Proxies differ from the perps:** ETF fees of 0.4–0.6%/yr in GLD, SLV, PPLT and PALL; NAV vs spot; HG=F roll
  gaps; Henry Hub spot spikes against gas futures; EIA spot isn't investable; WTI 2020-04-20 is handled by a rule.
- **Dividends are excluded** on purpose, and **funding is uncertain**.
- **Weekend pricing isn't modelled:** Binance prices TradFi perps on weekends from an EWMA of its own order book, so
  Monday gaps and weekend marks may differ from the backtest's Monday fill.
- **Liquidity tiers** come from one snapshot (2026-10-16).
- **One price was already seen:** the data-source probe on 2026-10-17 checked availability, date coverage, gaps,
  null rows and flat bars for many Yahoo and EIA series; the only price value it looked at was the known 2020-04-20
  negative WTI close in CL=F. Nothing was analysed.
- **Seen in earlier rounds:** the crypto trend results (F1, D1, round 4).
- **Changes from the brief and the approved plan, all decided before any data:**
  - The F1 correlation is a holdout gate, not a dev gate, because F1's data (2020–23) lies entirely inside the round
    6 holdout.
  - The shift null forward-fills over non-trading days and rolls each sub-sleeve within its own active span; round
    5's version would insert weekend zeros and roll late-listed markets into their pre-listing years.
  - The plan's null re-simulated shifted states through the engine. Two code reviews on synthetic data found that
    version broken: shifted fixed-size positions have no price-based exit, so they lever up into adverse moves (some
    sub-sleeves went through zero equity, with daily returns in the thousands of percent). Even with a liquidation
    rule, it captured only about 80% of a drift's effect, so under no edge it rejected about 7–9% of the time at a
    nominal 5%. The null now shifts the base run's exposure paths instead (above). On synthetic data a drift moves the
    actual and the null alike (paired effect +0.04 null s.d., standard error 0.07, 60 seeds).
  - The plan made "any sub-sleeve with equity ≤ 0" a hard error. A no-edge simulation showed that one liquidation
    in dev would then stop the round with no verdict and invite a redesign after seeing results. Liquidations are now
    booked as outcomes (above) in both windows, and only integrity errors are hard errors. The liquidation rule and
    the holdout treatment of integrity errors were added after the same reviews.
  - The funding stress is a drag computed from the base run's exposure paths, not a second engine pass (see Costs),
    so its timing-check hard errors are gone.
  - Added to the plan's red flags after the reviews: the NYSE rule calendar in the reference, the England & Wales
    holidays for Brent, the weekday-gap check over every series, the 120-day minimum for the yearly share, a split
    date with no row, and any Yahoo dev move above ln 1.8.
  - Four adversarial code reviews and a sandbox rehearsal of the whole runbook on synthetic data ran before this
    commit; every finding was fixed in the registered code.
- **Approximations:** sizing at entry only, unnetted costs, few markets in the early dev years, and daily closes
  only (no intraday path).

### Amendment 1: data sources (2026-10-17, after the first data report, before any strategy run)
- **What happened.** The first `r6-download` red-flagged 4 of 11 markets (`research/round6/data_report_initial.csv`,
  manifest `research/round6/data_manifest_initial.csv`):
  - EIA WTI (RWTC), Brent (RBRTE) and Henry Hub (RNGWHHD): in 1993–2006 the published spreadsheets miss 6–45%,
    16–50% and 43–77% of trading days a year, with gaps of up to 9, 8 and 28 weekdays. The raw files show it is the
    source, not the parser. After 2007, WTI still misses 16% of 2011 and 3.2% of 2015 and Henry Hub 5.6% of 2011;
    Brent is complete from 2007 (7 missing days in 20 years).
  - Copper (HG=F): null rows were 1.19% (> 1%), but 63 of its 79 nulls are Globex placeholder rows on NYSE holidays;
    on trading days the share is 0.24%.
- **Fix (code commit `6d0d91b`):**
  - CLUSDT: Yahoo CL=F (NYMEX WTI front month, unadjusted) from 2000-08-23, like copper.
  - NATGASUSDT: Yahoo NG=F (NYMEX Henry Hub front month, unadjusted) from 2000-08-30.
  - BZUSDT: EIA RBRTE spot from 2007-01-01; earlier rows are dropped at download.
  - Red flag 9 becomes: null rows on reference days above 1% of the market's reference days. Null dates are
    re-parsed from the verified raw file.
  - Nothing else changes: rule, sizing, costs and slippage tiers, gates, windows, null, verdict, and the CL 2020-04-20
    allowlist (CL=F also printed −37.63 that day).
- **How it was chosen.** The candidate series (CL=F, BZ=F, NG=F, and the EIA series from 2007) were first checked
  with the data report's own checks: dates, null rows, equal closes and dev-only returns, never a holdout price.
  - CL=F and NG=F: 15–16 missing reference days in 26 years, no gap over 2 days.
  - BZ=F: a 12-day gap and 7% missing in one year, so it was rejected.
  - EIA Brent from 2007: clean.
- **Data.** The old EIA files are kept (raw files under `data/external/raw/eia/`, superseded parquets under
  `data/external/superseded/`). The three markets are downloaded again with `r6-download --resume`, and their new data
  must pass the data report before any strategy run.
- **Consequences:**
  - **Proxies:** WTI and gas are now front-month futures instead of spot. They carry roll gaps like copper, and gas
    roll gaps can be large around winter; Henry Hub spot spikes are gone. Brent stays spot.
  - **Dev history shrinks:** energy starts in 2000 and 2007 instead of 1986, 1987 and 1997, and nothing trades before
    SPY (1993-01-29). The dev window stays 1986-01-01 → 2015-12-31, with zero returns before the first ready day.
  - **The shift null's S_min** is still set by PPLT/PALL (2010), so it is unchanged.
  - **Power** (recomputed with `r6-power` at `6d0d91b` into `research/round6/power_table.csv`; the constant variant is
    unchanged): the ramp now has about 137 market-years, or 12.4 full-book years of 30 (was 15.9). A dev pass needs a
    realized full-book Sharpe of about 0.57 (was 0.50), and the funding gate binds under the ramp only if the true
    hurdle is above 0.57.

| True full-book Sharpe | Ramp: P(dev pass) | Ramp: P(REPLICATED) | Ramp: P(STRICT) |
|---|---|---|---|
| 0 | 2.1% | 0.56% | 0.01% |
| 0.25 | 13.3% | 6.8% | 1.0% |
| 0.4 | 27.5% | 19.3% | 4.8% |
| 0.5 | 40.5% | 32.0% | 11.5% |
| 0.6 | 54.7% | 47.6% | 24.3% |
| 0.75 | 74.3% | 69.4% | 49.4% |
| 1.0 | 93.8% | 92.2% | 87.1% |

- **Disclosure:** this amendment was written after seeing the data report's quality statistics (dates, null rows,
  equal closes, dev-only moves). No strategy was run and no holdout price was looked at.

### Round 6 outcome (2026-10-17, after the dev sweep)
- **M1 failed dev: 8 of 9 gates.**
  - Weekly t 0.41 (< 2); account Sharpe 0.07 over 1986–2015.
  - Mean daily return at 2× costs −1.2e-5 and under the funding stress −3.1e-5 (both ≤ 0); the funding stress cost
    1.45%/yr.
  - Year share 187% (2008 alone is 0.186 of a 0.099 total).
  - 6 of 11 markets positive (< 7).
  - Shift null p 0.211 (null mean −6.3e-6/day against the actual +9.1e-6, s.d. 2.0e-5, S_min 1,662).
  - Plateau failed: fast5 −34%, slow5 +76% in total.
  - Alpha PSR against the equal-weight benchmark 0.68 (< 0.95).
  - Passed: PBO 0.0 (slow5 ranks first in and out of sample in every split).
  - DSR 0.075 at N = 17.85. Own N_eff 1.35. No liquidations and no integrity errors.
- **Consequence (pre-registered):** M1 is closed on these markets with this data, with no reruns or variants. The
  2016–2026 holdout was not unlocked and stays unused.
- **Reported, not gated:**
  - Longs earned and shorts lost: −0.33 R per trade for shorts against −0.11 R for longs (R statistics carry the
    0.1%-of-notional floor of the inherited R rule).
  - Slow lookbacks earned and fast ones lost: the summed contributions run from −0.147 at L = 5 to +0.097 at L = 360.
  - Activity from the full-book day (2011-06-14): 4.8 netted market orders per trading day, 20 sub-sleeve entries a
    week, gross exposure 0.76 on average.
  - Dividend-adjusted closes: Sharpe 0.12, weekly t 0.67.
- **Independent verification (three verifiers):**
  - A re-simulation without the project's engine, rule or runner reproduced all 13,087 base trades and 13,087 2×
    trades, every gate value and all 1,000 null draws to float noise.
  - A gate recompute with independent code matched all 9 gates, the PBO logits and the N_eff.
  - A protocol audit found no breach: commit order, start and code hashes, the holdout log, the untouched crypto
    registry, and no price dated 2016 or later in any result.
- **Notes from the verification** (information only; no gate is affected):
  - The profit factor in the summary is on R (0.86); on currency net it is 1.13.
  - The long/short split sums compounded trade returns; the market and lookback splits sum daily returns.
  - The reported 2× + funding mean takes the funding from the 2× run's own exposures.
  - Some fast sub-sleeves lose almost all their capital without being liquidated (SPY L5 ends at 0.24% of its
    capital); under the pre-registered sum/99 accounting they keep their full 1/99 weight.
  - Red flag 9 was redefined in Amendment 1 after copper's null share had been seen (disclosed there). The script
    that checked the candidate series is archived as `research/round6/amendment1_source_check.py`.
  - `power_table.csv` now holds the Amendment 1 version; the original table is in commit `7c6f725`.

## D1 shadow paper book (forward test), registered 2026-10-17 before it starts

**Why.** At the user's request, Round 5's D1 runs on the paper bot next to F1. D1 failed Round 5's dev gates: one year
made 61% of its profit, and it could not be told apart from simply holding the coins (alpha PSR 0.88 against the
equal-weight hold). This run is information only. Its result can never, by itself, make D1 a validated strategy.

**What runs** (code: `src/btcq/live/d1.py`, signals from `src/btcq/strategies/r5_d1.py` as registered in Round 5):
- the Round 5 rule exactly: 5 coins (BTC, ETH, SOL, XRP, BNB USDT-M perps) × 9 lookbacks = 45 long-only sub-sleeves,
  each starting with 1/45 of a $10,000 paper account and keeping its own cash;
- decisions at each UTC daily close on the coin's full perp daily history; taker fills at the open of 00:01 with
  Round 5's costs (5 bp fee, 1–2 bp slippage) and real funding;
- one deviation from the backtest: Round 5 warmed up with spot data before each perp listed; live uses perp history
  only. Every lookback has been flat at least once since then, so the states are the same today.
- D1 is never paused. The Round 5 rule has no kill switch, so the bot's /pause, /resume and 35% drawdown pause act on
  the F1 books only, and D1's drawdown baseline is never reset. A UTC day without prices is filled with the last
  close and not traded, as the research calendar and engine do.
- The coins are pinned to the five above, whatever the bot's SYMBOLS setting is. If D1 is ever switched off while
  the bot runs on, its record ends there. A restart begins a new record (the bot says so), and only the current
  record counts.
- **Parity:** replaying the 2022-03 → 2023-05 archive through the bot reproduced `research.round5.d1_run` trade for
  trade (entry and exit times, prices, returns, R and funding), including a restart in the middle of a day
  (`tests/test_d1.py`).

**How it will be judged:**
- **Engineering checks from month 1** (dashboard): sub-trades per week near the research's 6.2, days with an order
  near 64%, and drawdown within the research's 19%.
- **First look after 12 months, verdict after 24 months.** D1 is called "promising" only if all three hold over the
  forward period:
  1. its alpha against an equal-weight hold of the 5 coins, rebalanced at each UTC daily close (OLS beta on daily
     returns, PSR of the residual, as in Round 5), is ≥ 0.95;
  2. its total return after costs and funding is positive;
  3. its worst drawdown on end-of-day equity (as in the research) is ≤ 30%. The dashboard also shows a stricter
     figure measured on every minute.
- **Otherwise** D1 stays closed.
- **Even if "promising",** real money needs a new pre-registered test and the user's decision. D1 never adds to
  F1's risk.

**Disclosures:**
- 24 months is short. The result will most likely be inconclusive, and it depends on whether the period contains a
  bear market.
- The dashboard shows a running alpha estimate from day 30. Looking at it early changes nothing: the rule above is
  the only verdict.

## Deviation 2: D1 on 2024-2026, owner-approved (2026-10-18, recorded BEFORE the evaluation)

D1 failed Round 5's dev gates. Its holdout was registered as contaminated: information only, and run only inside an
XS unlock. XS failed too, so no Round 5 holdout was opened. On 2026-10-18 the owner asked to see D1 on 2024-2026
anyway, as the most relevant years, accepting that this is outside the registered rules.

**How it runs:**
- The registered rule and primary config (all9), with Round 5's code and costs unchanged. The window is Round 5's
  holdout, 2024-01-01 → 2026-09-30, with the earlier history used for signals as usual. There is one engine run per
  sub-sleeve and no shift null.
- One unlock, logged in `research/holdout_log.jsonl` with the tag `[R5 D1-info:`, made by
  `python -m btcq.research.run r5-d1-info --unlock-holdout`. The command refuses to run twice and never evaluates XS:
  **XS's holdout stays locked and clean.**
- Independent re-checks of the result may load the same window; each such load is logged with the tag
  `[R5 D1-info verify`.
- This is not a new trial: the trial registry and N_eff are unchanged.

**What will be reported (fixed now, before the data is seen):**
- total return, CAGR, Sharpe, worst end-of-day drawdown and weekly t;
- beta and alpha PSR against Round 5's benchmark, the equal-weight daily-rebalanced hold of the listed perps;
- D1 and the hold for each year (2024, 2025, and 2026 to 30 September), the year share, and results per coin and per
  lookback;
- activity (sub-trades per week, days with an order), fees and funding;
- the same account added up the way the live paper book does it: each sub-sleeve keeps its own cash, with no daily
  rebalancing;
- reference yardsticks only, not gates: mean daily return > 0, Sharpe at least half of dev's (≥ 0.68), worst drawdown
  against dev's 19%, and alpha PSR against 0.95.

**What it can and can't mean:**
- It cannot make D1 pass Round 5 or change Round 5's FAIL. It cannot change the registered forward test either: its
  rules, its 12/24-month verdict and its account construction stay as written.
- The data is not clean for this idea. The long-only trend idea came partly from S1's and F1's earlier 2024-2026 runs
  on these markets, and the period's general market history is known. The result is information only.
- Whatever it shows, it is reported in full in the Round 5 records and in `research/D1/README.md`.

### Deviation 2 outcome (2026-10-18, information only)

The run followed the text above: commit `5dd793a` came before the single `[R5 D1-info:` unlock, XS was not evaluated,
and the trial registry and N_eff are unchanged. Two independent checks loaded the window, each logged with the
`[R5 D1-info verify` tag. An independent re-simulation, using none of the project's backtest code, reproduced all 967
trades and every daily return exactly.

D1 on 2024-01-01 → 2026-09-30, after costs and funding:
- +34.6% (CAGR 11.4%), Sharpe 0.92, worst end-of-day drawdown 11.9%, weekly t 1.47;
- beta 0.17 and alpha PSR 0.812 against the equal-weight hold, which made +106.2% (Sharpe 0.75, worst drawdown
  61.7%);
- year share 0.894 (2024): 2024 +31.3%, 2025 +0.4%, 2026 to 30 September +2.1%;
- all 5 coins and all 9 lookbacks positive; XRP gave 46% of the result;
- the live-style account (each sub-sleeve keeps its own cash) made +34.3%, with a 12.5% worst drawdown.

Reference yardsticks: mean daily return > 0, met. Sharpe ≥ 0.68, met. Worst drawdown under dev's 19%, met. Alpha
PSR ≥ 0.95, not met. Both measures D1 failed in dev got worse: year share 0.61 → 0.89 and alpha PSR 0.884 → 0.812.

Consequences, as fixed above: Round 5's FAIL stands, and the forward test continues unchanged. Full tables are in
`research/D1/README.md` section 8.5 and `research/D1/results/holdout_*.csv`.

## Deviation 3: F2 on 2024-2026, owner-approved (2026-10-18, recorded BEFORE the evaluation)

F2 failed Round 3's dev gates on DSR and plateau, so under the registered rules its holdout stays locked. On
2026-10-18 the owner asked for its 2024-2026 backtest anyway.

**How it runs:**
- Round 3's registered holdout procedure, exactly as F1's was run under Deviation 1.
- F2's chosen config: 1h bars, n = 96, k = 3.5, channel exit. Code and costs are unchanged.
- Command: `python -m btcq.research.run r3-validate --strategy F2 --unlock-holdout`, with one unlock logged in
  `research/holdout_log.jsonl`.
- Window: 2024-01 → the latest archive date, as for F1.
- Scope: Round 3's holdout gates on the 4-alt portfolio (ETH/SOL/XRP/BNB); the BTC sleeve is information only.
- No other Round 3 family is evaluated: F3-F6's holdouts stay unused. This is not a new trial, so the registry and
  N_eff are unchanged.
- Independent checks may load the same window; each is logged with the tag `[R3 F2-info verify`.

**What it can and can't mean:**
- F2 cannot become a final pass, because `final_pass` requires a dev pass.
- Its holdout is labelled clean, so this is out-of-sample evidence for F2. Two disclosures limit that:
  - the general 2024-26 behaviour of these markets is known;
  - the 2024-26 result of its close relative F1 has been seen.
- It will be reported in full: the summary, Round 3's holdout gates, the BTC sleeve, and comparisons with dev, with
  F1's 2024-26 result and with holding the coins.

### Deviation 3 outcome (2026-10-18)

**How the run went:**
- The run followed the text above: commit `32b88c3` came before the single `[R3 F2-info:` unlock, and only F2 was
  evaluated.
- One independent check loaded the window, logged as `[R3 F2-info verify]`. It re-simulated F2 with its own code and
  reproduced all 939 trades and every daily return (differences of 1e-14 at most).
- **Disclosure:** before its unlock, that check read only the row counts, the first and last funding timestamps and the
  1m gap list, without logging. No prices or returns were read.

**F2 on 2024-01-01 → 2026-10-13, 4-alt scope, after costs and funding:**
- 751 trades (270 a year), average R +0.134, win rate 28%, profit factor 1.21, trade t 1.21;
- +31.5% (CAGR 10.3%), Sharpe 0.449, worst end-of-day drawdown 56.2%;
- 2024 +33.2%, 2025 -10.3%, 2026 +10.1%;
- longs +0.29 R and shorts -0.05 R; XRP gave 73% of the R;
- against the 4-alt equal-weight hold, which made +88.4% with a 64.1% worst drawdown: alpha PSR 0.78;
- correlation with F1 on the same scope and dates: 0.80 daily. F1 made +80.7%, with Sharpe 0.82 and a 27.6% worst
  drawdown.

**Round 3 holdout gates:**
- Passed: trades ≥ 30, trades a year ≥ 52, average R > 0, and drawdown ≤ the bootstrap p95 (52.9% ≤ 70.4%).
- Failed: Sharpe ≥ 50% of dev (0.449 against 0.471).

`final_pass` is false, as fixed above. Funding data ends on 2026-09-30, so the last 13 days carry no funding, as in
F1's run; the effect is about 0.3%. **Consequence:** F2 stays closed. It adds nothing next to F1.

## Practical rulebook (owner decision, 2026-10-18; applies to research registered after this section)

**Why.** Every strategy so far failed at least one strict gate. The strict gates were built for other people's money.
They also treat the whole project's ~390 trials as one search, so every new idea pays for every old one. The owner
trades his own $10,000 and chose looser, practical rules. They are written here before any test uses them.

**Scope:**
- The rulebook applies to every round or deviation registered after this section.
- Past verdicts stay as recorded. Earlier results may be re-read under these rules as information only, with no
  re-runs or tweaks to rescue them.
- Registered forward tests keep their own written rules, including the D1 shadow book's 12/24-month verdict.

**A strategy "passes in practice" when all of these hold:**
1. **Dev gates (practical):**
   - Enough data: at least 100 trades in dev (or the round's own higher minimum).
   - Mean daily return (or average R) > 0 after doubled fees and slippage.
   - It beats random timing: the round's null test at p < 0.05.
   - DSR ≥ 0.95 within the family:
     - N is the number of configs searched for this family's parameters, including any earlier grid that chose them
       (F1's would be S1's 30), but not the project-wide ratchet;
     - the variance is the null variance 1/(T−1) of the daily Sharpe;
     - formula: PSR against the expected maximum Sharpe of N skill-less trials (`stats.expected_max_sr`).
   - Best calendar year ≤ 65% of the summed daily returns.
   - Where the round has a grid: PBO ≤ 0.5 and its neighbouring configs positive (plateau).
   - Positive in most markets: at least 60%, or the round's own higher bar.
2. **Clean out-of-sample confirmation (required).** It must stay positive on data never used for the idea, the choice
   of markets, or the tuning, using one of:
   - an untouched holdout: mean > 0, Sharpe ≥ 50% of dev, and worst drawdown within the dev bootstrap's 95th
     percentile (or within 1.5× dev's worst drawdown where no bootstrap is defined);
   - a forward paper period whose length is fixed before it starts: at least 6 months and 50 trades, with the same
     three yardsticks.

   A contaminated holdout does not count.
3. **Reported, never gates:**
   - the strict tier (DSR at the ratcheted project N_eff ≥ 0.95);
   - alpha PSR against holding the same markets;
   - correlation with F1;
   - weekly t;
   - year concentration.

**What a pass means:** a candidate for paper trading, or for small real money at the owner's decision, starting at
reduced risk. Real money is never automatic.

**Still not allowed:**
- changing rules or thresholds after seeing results;
- choosing a config, market list or window from holdout results;
- re-running a failed test with changes without registering it as a new trial.

### Practical rulebook, Amendment 1: N is the family's effective number of trials (owner decision, 2026-10-18)

**Disclosure: this amendment was decided AFTER seeing a result it changes.** The F1 + D1 portfolio report
(`research/F1_D1_portfolio/README.md`, commit `dcaf33e`) re-read F1 under the rulebook as first worded:
- With N = 30 (S1's raw grid of configs), F1's within-family DSR is 0.64, a fail.
- S1's 30 configs are so correlated that they behave like 1.71 independent trials. With N_eff = 1.71, the DSR is
  0.99, a pass.

After seeing both numbers, the owner chose the effective count. The reason given: it is the method the project has
always used for N (the strict tier's N_eff), and the raw count treats near-identical configs as independent tries.
The original wording above is kept unchanged so the change stays visible.

**New rule, replacing "N is the number of configs searched" in the DSR gate:**
- N is the family's effective number of independent trials, `stats.effective_trials`: the participation ratio of the
  eigenvalues of the correlation matrix of the family's trial daily returns over the dev window.
- The family is every config searched for its parameters, including any earlier grid that chose them. For F1 that is
  S1's 30 configs, from `S1_trial_returns.parquet`.
- When N_eff < 2 there is no deflation (`expected_max_sr` returns 0), so DSR = PSR(0).
- The variance stays the null variance 1/(T−1).
- Every round reports DSR at both the raw config count and N_eff. The gate uses N_eff.
- Rounds must save their trial daily returns, as earlier rounds did, so N_eff can be computed.

**Effect on the re-read (information only; past verdicts stay as recorded):** F1 now meets every practical rule:
- dev: DSR 0.99, best year 34%, 2× costs, random-timing null p 0.003, 5 of 5 markets positive;
- clean out-of-sample: its 2024-26 holdout passed all 5 gates.

D1 meets the practical dev rules. Its 3 configs behave like 1.13 independent trials, so its DSR is 0.996 at N_eff
(0.97 at the raw N = 3). It still has no clean out-of-sample result, so it needs a forward paper period. Neither re-read changes a recorded verdict, and real money stays the owner's decision.

## D1 practical checkpoint (owner decision, registered 2026-10-18 15:50 UTC, before any closed D1 trade)

**Why.** Under the practical rulebook and its Amendment 1, D1 meets the dev rules but has no clean out-of-sample
result: its 2024-26 holdout is contaminated (Deviation 2). The rulebook accepts a forward paper period fixed in
advance as clean confirmation. D1's shadow paper run on the VPS is that period.

**State when this was registered** (read-only from the VPS database):
- D1 started on 2026-10-17 at 07:33 UTC, and its first orders filled on 2026-10-18 at 00:01 UTC;
- 12 of 45 sub-positions open, paper equity $9,986.54 (−0.13%), 0 closed sub-trades.

**When the check is made:**
- Normally at the first daily close on or after 2027-04-18 00:00 UTC (6 months after the first fills).
- If fewer than 50 D1 sub-trades have closed by then, the check moves to the daily close at which the 50th closes. It
  never moves for any other reason.

**What is measured:** the paper book's own end-of-day equity, from the bot database's `equity` table for book D1,
from its first day to the check. This is the live-style account, in which each sub-sleeve keeps its own cash.

**Yardsticks (all three must hold):**
1. Total return over the period > 0 (mean daily return > 0).
2. Annualised Sharpe of the daily returns (√365) ≥ 0.68, which is half of dev's 1.35.
3. Worst end-of-day drawdown ≤ 28.5%, which is 1.5 × dev's worst drawdown of 18.98%.

**Two choices fixed now:**
- **Drawdown basis.** The rulebook prefers the dev bootstrap's 95th percentile where one is defined. D1's trade-sequence
  bootstrap ignores that its trades overlap, so its drawdowns are far too small: p95 5.2% against a real 19% (D1
  report §6.9). It is therefore not used, and the 1.5 × dev-worst rule applies.
- **Which dev numbers.** The yardsticks use the research-style dev results (Sharpe 1.35, worst drawdown 19%). That is
  stricter than comparing with dev's live-style results (Sharpe 1.12, worst drawdown 28.6%).

**What the result means:**
- **All three met:** D1 "passes in practice". It becomes a candidate for small real money at the owner's decision.
  Running it next to F1 in one account would first need the live executor to net both books' positions.
- **Any one missed:** no practical pass. The check is not repeated or moved, and D1 continues on paper.
- **Either way,** the registered forward test is unchanged and reported separately: first look at 12 months, verdict at
  24 (alpha PSR ≥ 0.95, total return > 0, end-of-day drawdown ≤ 30%).

## Practical rulebook, Amendment 2: PBO is reported when the primary config is fixed in advance (owner decision, 2026-10-19, before any Round 7 data or backtest)

**Disclosure:** decided before Round 7 is run and before any Round 7 data is loaded. It changes no recorded verdict.

**Why:** the rulebook's dev rule "where the round has a grid: PBO ≤ 0.5 and its neighbouring configs positive"
was written for searches that pick the best config. When a round fixes its primary config before the test and
never selects among the grid, PBO does not measure a selection the round makes. With only a few configs it is
also close to a coin flip (with 6 configs, a few CSCV splits decide it).

**New rule, from Round 7 on:** when the registration fixes the primary config in advance and the grid is used only
to judge robustness, PBO is reported, not gated. The plateau rule stays a gate: the registered neighbouring configs
must all be positive. A round that selects its config from the grid keeps the PBO ≤ 0.5 gate.

## Round 7 (RT) — registered 2026-10-19, before any Round 7 data is downloaded and before any Round 7 backtest

**Why.** The owner wants a strategy that trades every day. F1 trades a few times a month, and D1 is only in a
forward test. A daily rotation among the most liquid non-major perps is a different family from everything tried
so far: it is cross-sectional (which coins), not a breakout on each coin alone. XS, a cross-sectional idea on the
5 majors, failed Round 5's dev gates; U1 (F1's breakout on these alts) failed Round 4.

**Owner decisions (2026-10-19):**
- The universe is the non-majors only: BTC, ETH, SOL, XRP and BNB are excluded. Their 2024–26 data has already
  been seen in earlier rounds; none of the non-majors' 2024–26 data has (Round 4's Stage B was never unlocked).
- Judged under the practical rulebook with Amendments 1 and 2 (PBO reported only).
- Long only, at most 1× gross, rebalanced at 00:00 UTC, with realistic turnover costs.

**Code:** commit `d638eaf`. New: `src/btcq/backtest/rotation_sim.py`, `src/btcq/strategies/r7_rotation.py`,
`src/btcq/research/round7.py`, `src/btcq/research/round7_calib.py`, `src/btcq/research/guards.py` and their tests;
changed: `store.py` (a blind quality count and structured unlock records), `stats.py` (`power_table_practical`),
`run.py` (CLI), `report.py`, `strategies/__init__.py`, `.gitattributes`. The 17 files in `round7.CODE_FILES` are
hashed at the sweep start and checked before the holdout. 350 tests pass at that commit.

### Universe `um-rot15-pit-v1`
- The gated (non-major) rows of `research/round4/universe.csv` (Round 4's `um-top20-pit-v1`, point in time: each
  month's top 20 USDT-M perps by median 90-day quote volume before the month, at least 90 days old). SHA256
  `6d79828bb183fce140900b0e1cc52e5d4d1687d7720484abfad0945074dcf36f` (line endings normalised), checked on every load.
- 16 members a month in 2020-06 → 2021-05, 15 a month from 2021-06 → 2026-09. 71 coins appear in dev months, 65 in
  holdout months, 109 in all; 38 appear only in holdout months.
- Membership and the slippage tier of a fill come from the month of the fill day: a decision at the close of the
  last day of a month uses the next month's list.

### Data
- Binance's monthly/daily archive of 1m perp klines and funding, as in Round 4 (`store.build_perp_only`). The 71 dev
  coins are already in the store from Round 4 (full span, never loaded after 2023 by any run). The 38 holdout-only
  coins are built by `r7-download` before the dev sweep.
- **Blind build:** the build's quality report counts extreme bars only on rows before 2024-01-01; the data report
  reads only the timestamps and volume of the 1m files and the timestamps of the funding files. No price or return
  from 2024 on is looked at or summarised before the unlock.
- **Cleaning (Round 4's):** trailing zero-volume minutes after the last traded minute are dropped; funding stretches
  longer than 9h without a settlement are filled at the synthetic 0.01%/8h (00/08/16 UTC settlements inside them),
  and the filled hours are reported per coin.
- **Data report** (`research/round7/data_report.csv`) and **manifest** (`research/round7/manifest.json`: the universe
  SHA256 and the SHA256 of every coin's 1m and funding parquet) are committed before the sweep start record; the
  manifest is re-checked at the sweep and before the holdout.
- **Red flags** (per window in which the coin is a member): the build failed or has no 1m data; no traded minute at
  or after the start of its first member month; the worst member month's 1m coverage is below 90% (counted up to
  the last traded minute); funding gaps above 9h cover more than 5% of the member hours; a coin still listed as
  TRADING whose rows stop more than 1h before the window's end (an incomplete archive). Data that stops because
  trading stopped is a delisting under the rules, reported but not a flag. The flags are recomputed from the report's
  columns on every check, and the manifest pins the report's SHA256. Any red flag stops the round until a dated
  amendment; then `r7-download --resume` rebuilds the failed builds.

### Rule RT
At each decision, the close of UTC day D (the last 1m close of the day), for every coin:
- **Score:** ln(C_D / C_{D−L}), C = the day's last 1m close.
- **Candidate:** a member of the fill day's (D+1) month; at least one traded minute on day D; the decision is not
  inside a no-trade gap or the 7-day cooldown after one (Round 4's delisting rule, below); a 1m row at or after
  00:01 on day D+1; both closes; and no gap inside the lookback: the last finished gap must have ended before the
  minute of C_{D−L} (strictly more than L days before the decision), so C_{D−L} is never a stale pre-gap price.
- **Eligible:** a candidate with score > 0 (absolute momentum).
- **Ranking:** candidates by score, highest first; ties by symbol.
- **Holding:** a held coin is kept while it is eligible and ranks ≤ H. The free slots go to the best-ranked eligible
  coins not held. K = 3 slots. Fewer than 3 eligible coins means the rest is cash (never re-weighted to 1/n).
- **Sizing:** every held coin is resized at each fill to 1/3 of the equity at the decision close: target quantity =
  E_D / 3 / X, where X is the fill price. Cash earns 0%.
- **Execution:** fills at the open of the first 1m row at or after 00:01 UTC on day D+1 (1-minute latency, as F1).
  A held coin with no fill row that day and not in a gap keeps its quantity until the next fill.
- **Configs:** L ∈ {7, 14, 28} days × H ∈ {3, 6}: 6 trials. **The primary is fixed: L = 14, H = 6.** Its plateau
  neighbours are (L = 7, H = 6), (L = 28, H = 6) and (L = 14, H = 3). Nothing is selected from the grid.

### Account and costs
- One account of $10,000, long only, at most 3 positions, gross ≤ 1× at each fill.
- Per fill day, in minute order (as `engine.run`): funding on the old position through the fill minute; a forced
  exit before the fill; the fills; funding on the new position through the day's last minute; a forced exit at or
  after the fill minute; the mark at the day's last close. Both windows start flat and close every position at the
  last close of the window.
- **Fees:** taker 5 bp on the traded notional of every buy, sell and resize.
- **Slippage:** Round 4's tiers by the fill month's rank: 2 bp (rank 1–10), 3 bp (11–20), 5 bp for a coin that is
  not a member that month (an exit after it left the list), on |Δquantity|.
- **Funding:** the real Binance rate at each settlement (any interval), on the quantity held at that minute, at that
  minute's open, as `engine.Market` maps it; synthetic 0.01%/8h before a coin's first real rate.
- **Delisting rule (Round 4's):** more than 24h without a traded minute is a settlement or delisting: a held coin is
  closed at the close of its last traded minute with an extra 1% penalty, and no new entry is allowed until 7 days
  after trading resumes.
- **2× stress:** fees and slippage doubled (funding and the penalty unchanged).
- **Single-coin check:** with daily resizing off, the simulator reproduces `engine.run` exactly for one coin (trades,
  quantities, fees, funding, forced exits and daily marks, to 1e-9); this is a unit test.

### Windows
- **Dev:** 2020-06-01 → 2023-12-31 (1,309 days); scores and volatility use the 70 days before the start. The first
  fill is 2020-06-01 00:01.
- **Holdout:** 2024-01-01 → 2026-09-30 (1,004 days), one shot, started flat on 2024-01-01 from decisions at the
  2023-12-31 close.

### Null
- Random coin choice with the same exposure: on each decision day the candidates' scores are shuffled among the
  candidates of the same volatility tercile. Volatility here is recent volatility: the square root of an EWMA
  (weight 0.06 on the newest day, λ = 0.94) of squared daily log close-to-close returns up to the decision close,
  defined once 30 returns exist. Candidates are sorted by it (ties by symbol) and split into 3 near-equal groups;
  candidates without a volatility form a fourth group. The number of eligible coins each day (so the cash share and
  the market exposure) and the volatility mix of the choices are the actual path's; only which coins are chosen is
  random. The holding rule then runs on the shuffled scores.
- Statistic: the mean daily return with fees and slippage at 0 (funding and the delisting penalty kept), for the
  actual path and each draw, through the same simulator and the same summation. 1,000 draws, seed 7007 (numba's
  generator; the numpy, numba, pandas, scipy and Python versions are pinned at the sweep start),
  p = (1 + #{null ≥ actual}) / 1,001.
- The identity shuffle reproduces the actual statistic bit for bit, and its p-value is 1 (a unit test).

### Null calibration
Run by `r7-calibrate` at commit `d638eaf` (`research/round7/null_calibration.csv`), before any data. Synthetic daily panels of 1,400 days; prices from simple returns. In every **no-edge** scenario
every coin has the same conditional expected simple return each day, so no rule can beat a random choice:
- **iid:** 16 coins, equal volatility, no factor.
- **factor:** volatility 2–8%/day, one market factor (bull, bear and flat regimes in the common drift) whose shocks
  scale with beta ∝ volatility, t(3) shocks.
- **garch:** factor plus volatility clustering (GARCH(1,1) on each coin's own shocks).
- **churn:** factor with 26 coins, 15 members a month, late listings, a delisting with a −30% forced exit, random
  positive funding.
- **cluster:** beta independent of volatility, three sector factors, common jump days (2%) with random loadings.

Information rows: **beta_tilt** (the drift itself scales with beta, so high-beta coins earn more in the bull regime:
a structural tilt, not a no-edge panel) and two planted cross-sectional momentum edges (power). 1,000 panels per
no-edge scenario, 499 draws each, rejection at p < 0.05 (exactly 5% under exchangeability at 499 draws):

| Scenario | Kind | Planted edge | Panels | Rejected at p < 0.05 | s.e. | Mean p |
|---|---|---|---|---|---|---|
| iid | no-edge | – | 1,000 | 5.8% | 0.7% | 0.472 |
| factor | no-edge | – | 1,000 | 4.1% | 0.6% | 0.516 |
| garch | no-edge | – | 1,000 | 4.9% | 0.7% | 0.514 |
| churn | no-edge | – | 1,000 | 6.4% | 0.8% | 0.490 |
| cluster | no-edge | – | 1,000 | 5.7% | 0.7% | 0.486 |
| beta_tilt | tilt (information) | – | 400 | 5.2% | 1.1% | 0.503 |
| factor | planted edge (power) | 0.015 | 200 | 45.5% | 3.5% | 0.139 |
| factor | planted edge (power) | 0.03 | 200 | 89.5% | 2.2% | 0.022 |

The rule fixed before the calibration ran: a rejection rate ≤ 7% in every no-edge scenario; the registration
commands refuse to run unless the committed calibration meets it. **Every no-edge scenario is within the limit** (the largest is churn, 6.4%). Pooled over the 5,000 no-edge
panels the rate is 5.4% ± 0.3%: the null is very slightly liberal, so a p-value just under 0.05 is a little weaker
evidence than it looks. The beta-drift tilt is not picked up as skill (5.2%). A planted momentum edge (each coin's
next return gains edge × its volatility × its standardised 14-day idiosyncratic return) is detected 45.5% of the time
at 0.015 and 89.5% at 0.03. The calibration file stores the SHA256 of the code that decides it
(`4a6e0727ec69e3e94275f560b594323fa15a99826482e51f3d87f27c4dd9907c`: `rotation_sim.py`, `r7_rotation.py`,
`round7_calib.py`), and the round's commands refuse to run unless it still matches.

### Power table
`power_table_practical`, iid Gaussian daily returns, 40,000 simulations, seed 7 (`research/round7/power_table.csv`):
dev = 1,309 days passing when the mean is > 0 and the DSR at N_eff is ≥ 0.95; holdout = 1,004 days passing when its
mean is > 0 and its Sharpe is ≥ 50% of dev's. The null, 2×-cost, year-share, plateau, breadth and drawdown gates are
not modelled (they can only lower these numbers).

| True annual Sharpe | P(dev) N_eff < 2 | P(dev) N_eff 2 | P(dev) N_eff 6 | P(pass) N_eff < 2 | P(pass) N_eff 2 | P(pass) N_eff 6 |
|---|---|---|---|---|---|---|
| 0 | 4.7% | 1.5% | 0.2% | 0.9% | 0.2% | 0.0% |
| 0.5 | 24.2% | 11.1% | 2.3% | 10.6% | 4.1% | 0.6% |
| 0.75 | 41.3% | 22.9% | 6.4% | 24.0% | 11.8% | 2.6% |
| 1.0 | 59.8% | 39.2% | 14.4% | 41.9% | 25.6% | 8.0% |
| 1.25 | 76.5% | 57.6% | 27.7% | 60.8% | 44.0% | 19.3% |
| 1.5 | 88.1% | 74.7% | 45.1% | 75.7% | 63.0% | 36.0% |
| 2.0 | 98.4% | 94.9% | 80.4% | 91.7% | 88.2% | 74.0% |

- **What it means:** 3.6 years of daily returns cannot confirm a modest edge. The 6 configs overlap heavily, so the
  family's N_eff will probably be below 2 (no deflation); then the DSR gate needs a realised dev Sharpe of about 0.87.
  A strategy with a true Sharpe of 1.0 passes dev about 60% of the time and the whole test about 42%; a true 0.5,
  24% and 11%. With no edge at all it passes dev 4.7% and the whole test 0.9% of the time (on these two gates).
- **Implied dev Sharpe needed (approximately):** 0.87 at N_eff < 2, 1.14 at 2, 1.32 at 3, 1.56 at 6.

### Dev gates (primary config, L = 14, H = 6)
1. At least 100 trades. A trade is a holding episode of one coin, from its first buy to its full exit; resizes
   inside it are not counted.
2. Mean daily return > 0 at 2× fees and slippage.
3. Null p < 0.05.
4. DSR ≥ 0.95 at the family's N_eff (Amendment 1): `stats.effective_trials` of the 6 configs' dev daily returns,
   null variance 1/(T−1), no deflation when N_eff < 2.
5. Best calendar year ≤ 65% of the summed dev daily returns (2020 counts from June).
6. Plateau: the 3 neighbours have a positive total return at base costs.
7. Breadth: at least 60% of the coins with at least 5 trades have a positive summed net P&L, with at least 10 such
   coins.

PBO (CSCV over the 6 configs, 16 slices) is reported only (Amendment 2).

### Holdout gates
1. Mean daily return > 0.
2. Sharpe ≥ 50% of the dev Sharpe (both daily × √365).
3. Worst end-of-day drawdown ≤ the 95th percentile of a block bootstrap of the dev daily returns: block 30, 10,000
   paths of 1,004 days, seed 7 (computed at the sweep and stored in `RT.json`).

### Verdict
- **PASSES IN PRACTICE** = every dev gate and every holdout gate. RT then becomes a candidate for a separately
  registered paper book, or small real money at the owner's decision, starting at reduced risk. Nothing is automatic.
- **FAIL:** RT is closed on this universe. No reruns or variants without a new registration and new trials.
- Any error after the holdout unlock (loading, simulation, gates) is a FAIL, recorded as such (no fix, no rerun).

### Holdout lock and run order
1. `r7-calibrate` and `r7-power` (synthetic only), committed with the code before this section.
2. `r7-download`: the blind build of the 38 coins, the data report and the manifest, committed.
3. `r7-sweep` logs a start record and stops. The record pins the version and the SHA256 of the 17 code files, the
   manifest, the universe, `F1_dev_daily.parquet` and this section's committed text, plus the library versions.
   The run log is committed; `r7-sweep` again runs dev. A sweep is refused if any start record for its version, in
   the working copy or any committed version of the log, has different pins. A version runs dev once: a run record
   is logged before it starts, and any error before the dev verdict is logged as a hard error (a fix needs a version
   bump). The verdict (`RT.json` and an `r7-sweep` record with the gates) is saved before anything else is written.
4. `r7-validate --unlock-holdout` runs only if every dev gate passed: the pins must be unchanged, and `RT.json`, the dev
   daily returns, a committed passing `r7-sweep` record, the trial registry, the data report, the manifest, the power
   table, the calibration and `F1_dev_daily.parquet` must be committed. It logs a validate-start record in the run
   log and then one structured `[R7: RT v1]` record in `research/holdout_log.jsonl`. Unlocks are counted over every
   committed version of both logs and their working copies. A crash before the result may be resumed once
   (`[R7 resume`). After a recorded holdout result or failure the version is closed, and once the holdout is touched
   the dev sweep can never run again. Independent checks, only after the holdout verdict, log each load as
   `[R7 verify`.
- **Hard checks of the simulator** (an entry without a price or on a non-candidate, more than 3 holdings, weights
  above 1, equity ≤ 0, the P&L identity) raise errors; in dev they stop the sweep, in the holdout they are a FAIL.

### Trials and multiple testing
- 6 dev trials in the global registry (`research/trials.csv`), version 1, with their daily returns in
  `reports/results/RT_trial_returns.parquet`. The project's ratcheted N_eff is updated; `run.cmd_gates` is not called.
- Reported: DSR at the raw N = 6, at the N_eff of RT's 6 configs together with XS's 6, and the strict tier: DSR on
  dev + holdout at N = max(ratcheted project N_eff, 14.85 + 4 = 18.85) (+4: one per family after Round 4: D1, XS,
  M1, RT).

### Reported, not gated
- PBO; per-year results against the equal-weight hold; per coin; meme coins against the rest; the result without the
  best coin (re-run without it); no daily resize; 2× costs in full.
- Alpha PSR and beta against the membership-masked equal-weight daily-rebalanced hold of the month's members.
- Correlation and HAC alpha against F1 (dev 2020-06 → 2023; holdout against F1's 2024–26 results).
- Weekly t, turnover, fees and funding paid, days with an order, days invested, exits by reason.

### Disclosures
- **Not blind to the period:** the general crypto market of 2024–26 is known (the BTC/ETH/SOL results of earlier
  rounds, and that memes and some new listings rallied). Which non-major coins led in which month has not been
  looked at.
- **Seen before:** the same universe's dev years were used by Round 4 (U1, a breakout), and the 5 majors' results of
  every round. Cross-sectional momentum among the majors (XS) failed. Round 4's data report lists whole-file counts
  of extreme minutes (a move above 30% within a minute or a 50% range) for the 71 dev coins, which include 2024+
  rows; they are counts of bad-looking bars, not returns, and the Round 7 report shows only the blind counts.
- **Memes:** up to 10 of the 15 monthly members in 2024–26 are memes (from the universe file's `is_meme`, which is
  membership, not prices). There is no meme cap; the meme split is reported.
- **Approximations:** taker fills at the 00:01 open with fixed slippage tiers (no market impact model beyond them);
  daily resizes are traded in full, however small; funding is charged at the minute's open; one account, so no
  margin model beyond gross ≤ 1×.
- **Changes from the approved plan, decided before any data:**
  - **The null groups coins by recent (EWMA) volatility, not the plan's 30-day volatility.** A dry run of the
    calibration (synthetic only, before this registration) showed that 30-day volatility lags volatility bursts. On
    a GARCH panel where the generator gave coins in a burst a higher expected return, 30-day groups rejected 7.0%
    (400 panels, 199 draws) and EWMA groups 5.0%; 10-day groups (7.0%) and 5 groups (8.3%) did no better. On
    panels with no edge at all the two were alike (4.8% and 4.3%).
  - **The calibration generator was rebuilt after that dry run.** The first version built log returns with a fixed
    drift correction. Under t(3) shocks and GARCH, that gave high-volatility coins a higher expected simple return,
    which is a real, predictable edge, so those panels were not "no edge"; its "beta" scenario also scaled the drift
    with beta. The dry run rejected 5.3% (iid), 4.1% (beta), 9.9% (garch) and 7.1% (churn) of 1,000 panels each.
    With equal expected simple returns, the garch and churn panels gave 4.8% and 5.8% (400 panels). The beta-drift
    case is kept as an information row (beta_tilt). The cluster scenario was added after the second code review.
  - **The gap-in-lookback boundary is strict** (the last gap must end more than L days before the decision), after
    the first code review found that a gap ending exactly at 00:00 left C_{D−L} as a stale pre-gap price.
  - **PBO is reported, not gated** (Amendment 2 above). The plan already said so; the rulebook needed the amendment.
  - **Any error after the holdout unlock is a FAIL**, not only an integrity error (second review: a crash could
    otherwise leave the holdout opened with no verdict). The verdict is saved before any result file is written.
  - **Pins:** the sweep start record pins the SHA256 of 17 source files (simulator, rule, runner, engine, statistics,
    metrics, store, config, universe, Round 4/5 helpers, registry, CLI, calibration), the data manifest, the universe,
    `F1_dev_daily.parquet`, this section's committed text, and the numpy, numba, pandas, scipy and Python versions.
    The calibration file pins the SHA256 of the code that decides it (simulator, rule and null, calibration module).
  - Two independent code reviews and a re-review of the fixes ran on synthetic data before this commit; every
    finding was fixed in the registered code. One reviewer's independent pure-Python account matched the simulator
    to 7e-16 over fuzzed panels, and engine parity holds for forced exits at 00:00, at the fill minute and with
    missing rows.
- **The universe's holdout membership is known:** which coins were among the 15 most liquid non-majors in each month
  of 2024–26 was computed in Round 4 from daily quote volume (no prices or returns were looked at).

### Round 7 outcome (2026-10-19, after the dev sweep)

**Verdict: RT FAILS dev.** It fails 5 of the 7 registered dev gates. As registered, the holdout was not opened:
`research/holdout_log.jsonl` has no Round 7 record, so the non-majors' 2024–26 data stays clean. RT is closed on this
universe. Results: `reports/results/RT.json`, commit `d5d22ce`; run log `research/round7/runs.jsonl`.

**Dev, primary L = 14, H = 6, 2020-06-01 → 2023-12-31, $10,000 start:**
- 428 trades (holding episodes, 7.2 days on average), orders on 90% of days, invested on 87%.
- Total return −57.4% (CAGR −21.2%), Sharpe 0.22, volatility 95% a year, weekly t 0.42.
- Worst end-of-day drawdown 97.0%: from $66,895 on 2021-04-16 to $2,004 on 2023-10-09. Year-end equity: $11,040,
  $20,404, $4,341, $4,259.
- Costs: fees $2,095 and funding $11,862. The funding was almost all paid in 2020–21 ($1,792 and $10,657; in 2021
  about 39% of the account, summed daily), when long positions in trending alts paid high rates.

| Year | RT | Equal-weight hold of the month's members |
|---|---|---|
| 2020 (from June) | +10.4% | +59.1% |
| 2021 | +84.8% | +253.2% |
| 2022 | −78.7% | −88.0% |
| 2023 | −1.9% | +74.6% |
| 2020-06 → 2023 | −57.4% | +17.9% |

| Dev gate | Result | |
|---|---|---|
| ≥ 100 trades | 428 | pass |
| Mean daily > 0 at 2× fees and slippage | +0.040%/day (total −66.4%) | pass |
| Null p < 0.05 | 0.538 (actual 0.076%/day at zero cost, null mean 0.082%/day) | **fail** |
| DSR ≥ 0.95 at the family N_eff | 0.664 at N_eff 1.31 (no deflation: PSR 0.664) | **fail** |
| Best year ≤ 65% | 2021 = 181% of the summed daily returns | **fail** |
| Plateau: 3 neighbours positive | (7, 6) +14.6%, (28, 6) +81.8%, (14, 3) −33.3% | **fail** |
| Breadth: ≥ 60% of coins with 5+ trades positive, 10+ coins | 14 of 33 (42%) | **fail** |

**Reported, not gated:**
- PBO 0.54.
- All six configs' totals: (7, 3) +34.8%, (7, 6) +14.6%, (14, 3) −33.3%, (14, 6) −57.4%, (28, 3) +155.0%,
  (28, 6) +81.8%. Their Sharpes run from 0.22 to 0.75. These are information only: the primary was fixed in
  advance, and testing a 28-day version now would be a new, post-hoc trial.
- DSR at the raw N = 6: 0.19. With XS's 6 configs (N_eff 3.09): 0.33. Strict tier (dev only, N = 18.85): 0.07.
- Against the equal-weight hold: beta 0.73, alpha −14.8% a year, alpha PSR 0.34.
- F1: weekly correlation 0.16, HAC alpha t −0.10.
- Without the best coin (DOGE, +$12.8k) the total is −70.4%. With no daily resize it is −53.6%. 34 meme trades
  made +$8.4k and the rest −$14.2k.
- No forced (delisting) exit happened in the primary. LUNA was last held 2022-05-02 → 05-03, before its collapse.

**Independent check (before this outcome was written).** A separate agent wrote its own pure-Python simulator from
the registered rules and re-ran the committed dev panel.
- All 6 configs matched to rounding (daily returns within 7e-16), with identical episode counts.
- All 428 primary episodes matched, and so did the 2× run.
- Every gate it recomputed matched `RT.json`.
- The panel matched the raw dev 1m and funding data for LINK, DOGE and LUNA on every day.
- Its own estimate of funding was within 1.1% of the simulator's.
- Its own stratified null gave p = 0.54.
- One wording detail: the null's EWMA uses pandas' bias-adjusted weights, so the newest day weighs about 0.071 when
  only 30 returns exist (0.06 in the limit). With exact recursive weights the groups agree on 99.95% of candidate
  cells and p is 0.545. The verdict does not depend on it.

**What it means, in plain words:**
- Holding the 3 strongest of the liquid alts, re-chosen every day, did no better than picking coins of similar
  volatility at random (p 0.54).
- It made money only in the 2021 bull market, and much less than simply holding all 15. Funding on the long
  positions took a large share of that.
- It then lost almost everything in 2022–23. It traded every day, as wanted, but daily activity did not bring an
  edge here.
- This closes daily cross-sectional momentum on this universe, after XS failed on the majors.

## Execution study X1 — registered 2026-10-19, before any X1 run or download

**Why.** Every backtest so far fills as a taker: 5 bp fee plus 1–2 bp slippage per side. The owner wants more
activity and asked whether post-only limit (maker) orders would make trading cheaper. This study measures, on dev
data only (before 2024-01-01):
- **Q1:** whether maker-first execution raises F1's dev net return, under a conservative fill model, without
  changing which trades F1 takes;
- **Q2:** D1 (settled by arithmetic below);
- **Q3:** what maker-first execution costs per side for three kinds of order (no signal, momentum, reversion), and
  whether any already-tested fast family's dev gross edge lies in the band that execution could change.

It never touches API keys, places no orders and changes nothing in the paper bot.

**Owner decisions (2026-10-19):**
- the plan in `C:\Users\adnya\.claude\plans\partitioned-dazzling-kazoo.md` is approved;
- the work is staged: this registration and Stage A first, then a stop for the owner's OK before any download.

**Code:** commit `8f232eb`. New files:
- `src/btcq/backtest/maker_engine.py` (the maker-first account and single-leg simulator);
- `src/btcq/backtest/queue_sim.py` (the tick-level queue simulator);
- `src/btcq/data/ticks.py` (the stream-and-discard tick downloader);
- `src/btcq/research/x1_execution.py` (the runner);
- `tests/test_engine_maker.py`, `tests/test_queue_sim.py`, `tests/test_ticks.py` and `tests/test_x1.py`.

Changed files: `stats.py` (`block_bootstrap_means`, appended) and `run.py` (the `x1-*` verbs). **`engine.py` is
unchanged.** The start record pins:
- the SHA256 of the 20 files in `x1_execution.CODE_FILES`;
- this section's committed text;
- `F1_dev_daily.parquet` and the exchangeInfo snapshot;
- the numpy, numba, pandas, scipy, pyarrow and Python versions.

### What is already known (from committed dev results, before this registration)
The figures are sums of trade returns ÷ sleeves, in % of the account per year, 2020–2023.

| | F1 | D1 |
|---|---|---|
| Legs a maker order could take | 1,040 of 1,200 (84 exits are same-minute reversals) | 2,581 of 2,608 |
| Ceiling: every eligible leg maker at the taker price, slippage saved, nothing missed | 1.1–2.3 %/yr | 0.19–0.35 %/yr |
| Ceiling with the fee saving only | 0.76–1.50 %/yr | 0.13–0.24 %/yr |

**D1 is NO-GO by arithmetic:** even its full ceiling is below the +0.5%/yr bar of rule (a). No D1 run is made.

### Execution model (pinned)
**Legs.**
- Maker-eligible: entries and decision exits (exit_signal, time, reverse).
- Always taker: stops, targets, forced and end-of-window closes.

**Policy "post, then cross"** (`maker_engine.run`, `MakerPolicy(T, model, fallback)`):
- At m0, the taker reference minute (the bar close plus the registered 1-minute latency), post a post-only limit at
  P = open(m0) − 1 tick for a buy (+ 1 tick for a sell).
- It rests until the first 1m row at or after m0 + T minutes. Whatever is still open there crosses at that row's
  open, paying the taker fee plus the registered slippage.
- **Primary: T = 15, no re-pegging.** Grid T ∈ {5, 15, 60}; T = 5 and T = 60 are the plateau neighbours.
- **The quantity is fixed at post:** risk sizing at the taker reference price o(m0)·(1 + side·slip) and the
  signal's stop, never re-sized.
- **Reversal:** an exit order and an entry order posted at the same decision. The entry is sized on the cash a taker
  exit at m0 would leave. If the old position's stop fires while the exit rests, only the exit order is cancelled.
- **Resting exits:** the position keeps its stop, which is checked before the exit order in each minute (the stop
  wins a tie).
- **Funding:** any funding settlement inside a resting window is a hard error (F1 decides at hh:01 with T ≤ 60, so
  none is expected).
- **T = 0** is the registered taker run. It is bit-identical to `engine.run` (a parity test on synthetic paths and on
  F1 ETHUSDT dev).

**1m models:**
- **strict, the decision basis:** a fill needs low < P (buy) or high > P (sell) in a minute from m0 on, beyond a
  relative tolerance of 1e-9·P. Under price-time priority a print through P means the level at P was cleared, so
  this is a sufficient fill condition apart from sub-minute latency, which check (f) measures. The fill is booked at
  the registered taker price with the maker fee, so a fill saves only the 3 bp fee difference. Unfilled legs pay the
  drift to the fallback.
- **central (reported):** the same fill rule, filled at o(m0), so it credits the registered slippage.
- **optimistic (reported, a model and not a bound):** low ≤ P (high ≥ P), filled at P.

**Tick size per symbol and month** (`research/x1/tick_sizes.csv`):
1. Unique 1m OHLC prices are rounded to 10 significant digits.
2. The candidate tick is the smallest positive gap, snapped to {1, 2, 5}×10^k.
3. It is used if at least 99% of the gaps are whole multiples of it; otherwise the 2026-10-16 exchangeInfo tick is
   used.

**Hard checks** (each is a logged hard error):
- the 5-sleeve taker portfolio reproduces the committed `F1_dev_daily.parquet` to 1e-12;
- every trade's fees equal its legs' maker or taker fees;
- every posted order ends in exactly one outcome;
- mean saving = p_fill × mean saving filled + (1 − p_fill) × mean saving unfilled, and a strict fill saves exactly
  3 bp;
- no trade, daily return or Q3 leg is dated on or after 2024-01-01.

**Tick model** (`queue_sim.simulate_legs`; pinned now, used in Stage B only):
- **Placement:** the order is live at t0 + 60 s, at the best bid (buy) or best ask (sell) of the last bookTicker row
  at or before that instant. The taker benchmark is the opposite best at that instant.
- **Queue ahead** = the bookTicker size at P. Only prints from the other side count:
  - a print through P fills the remainder;
  - at-P volume first consumes the queue ahead, and any excess fills us (partial fills tracked).
- **Book rows:**
  - a bid below P sets the queue ahead to 0, because our order is then the best bid;
  - a bid at P caps the queue ahead at the shown size;
  - **pro-rata variant:** a size drop not explained by trades at P since the previous observation of the level (in
    practice the same-millisecond trades) reduces the queue ahead pro rata.
- At equal timestamps, trades are applied before book rows.
- **Expiry:** whatever is still open at t0 + 60 s + T crosses at the then best opposite price.
- **Savings:** measured against the tick taker benchmark; fees enter only through the maker share.
- **Usable legs:** legs outside the extracted coverage windows, or not complete through the markouts, are excluded
  and counted. A leg during which the archived opposite best reached P is flagged (`book_crossed`), since the model
  fills only on prints.
- **B1 sizing:** generic legs are sized at $1,386 (F1's mean order on $10k). B2 legs use F1's own quantities, and a
  reversal is one order filled exit first.

### Q3: generic cost curves (information only)
**Stage A, on 1m data** (`research/x1/stage_a_q3_costs.csv`). For each of the 5 coins on dev perp rows
(2020-01 → 2023-12):
- **random:** 8,000 legs, rows and then sides drawn from one generator with seed 1102, in BTC, ETH, SOL, XRP, BNB
  order;
- **momentum:** a 1h close crossing above the prior 24-bar high is a buy, below the prior low a sell;
- **reversion:** a 1h log return beyond 2σ of the prior 720 1h returns, traded against.

How legs are run and costed:
- All legs are filled at m0 = the 1h close + 1 min. Only legs whose minutes up to T = 60 plus the 60-minute markout
  are all in the data are used, so every T and model sees the same legs.
- c_eff = (5 bp + slippage) − the mean saving, with a day-clustered 95% CI.

**Stage B1 (after the owner's OK):** the generic legs on B1's days go through the tick model, giving per-class and
per-coin corrections. These form **X1-cost-v1** (`research/x1/x1_cost_v1.csv`, pinned by SHA256).

**Screen** (`research/x1/stage_a_screen.csv`; a re-read of committed dev trade files, no re-runs):
- For S4, S5, S7, S8, F2–F6, the gross edge per trade, with slippage added back on every taker leg, is set against
  the taker round trip and the maker-first round trip.
- The maker-first round trip is computed at T = 15, per trade's coin: the entry class's c_eff + the random class's
  c_eff for decision exits, or the taker cost for stop and other exits.
- "Execution-sensitive" = maker-first < gross < taker. Momentum families are S4, S5, F2, F3 and F4; reversion
  families are S7, S8, F5 and F6.
- It is reported for all dev trades and for entries from 2020 on.
- Any such family needs a new registered round, with new trials and a clean out-of-sample test.

### Decision rules
**F1 GO-SHADOW** requires all of these, for the primary under the strict model against the registered taker run
(dev 2020-01 → 2023-12, 5-coin portfolio daily returns, d = maker − taker):
- (a) mean(d) × 365 ≥ +0.5%/yr;
- (b) the 5th percentile of 10,000 circular block-bootstrap means of d (block 30 days, seed 11) is above 0;
- (c) strict T = 5 and T = 60 also have mean(d) > 0;
- (d) the sum of d is above 0 in at least 3 of the 4 years, and the per-sleeve sum is above 0 in at least 4 of the
  5 coins;
- (e) Sharpe (daily, ddof 1, √365) falls by at most 0.02, and the worst drawdown worsens by at most 1 percentage
  point;
- (f) on the F1 legs in the bookTicker window (B2), the tick model's mean saving (conservative queue) is at least
  the strict 1m model's minus 0.5 bp.

**What the outcomes mean:**
- If strict fails any of (a)–(e) in Stage A, F1 is NO-GO and B2 is never downloaded.
- If (a)–(e) pass, the decision depends on (f), which needs the owner's OK for the B2 download.
- A GO-SHADOW only allows a separately registered forward shadow book. Real money is the owner's decision, after a
  forward pass.

### Data (Stage B; every download needs the owner's OK)
- **B1:** 15 days per coin, seed 1101, from 2023-05-16 → 2023-12-30: bookTicker 5.2 GB + aggTrades 0.5 GB.
- **B2** (only after a strict pass): the F1 legs in 2023-05-16 → 2023-12-31: bookTicker 13.3 GB + aggTrades 1.4 GB.
- **Holdout guard:**
  - a key's last day (a monthly key's month end) must be on or before 2023-12-31;
  - windows reaching 2024-01-01 are dropped;
  - every time column of every parsed row is asserted to be before 2024-01-01.
- **Handling:** files are streamed and discarded (one zip per worker, at most 2 workers). The manifest (key, bytes,
  SHA256, rows) is committed.

### Trials
- F1X, version 1, adds 5 dev trials: strict T = 5, 15 and 60 with fallback, and strict T = 15 and 60 without
  (post-and-cancel).
- Their portfolio daily returns go to `reports/results/F1X_trial_returns.parquet`, and the project N_eff ratchet is
  updated.
- The post-and-cancel runs change F1's trades. They are information only; a limit-only F1 needs a new round.

### Run order
1. The code commit, then this registration.
2. `x1-start` logs a start record with the pins; it is committed.
3. `x1-stage-a` runs once. Any error after its run record is a logged hard error that closes the version; a fix
   needs a version bump. Its outputs are pinned by SHA256 in its log record.
4. Stop for the owner's OK before `x1-download --set B1`, and again before B2.
5. Each Stage B set runs once.

### Changes from the approved plan, decided before any X1 run
- **A new module instead of an engine option.** The maker-first path is `maker_engine.py`, not an option inside
  `engine.py`, because earlier rounds pin `engine.py` by hash.
- **The account level uses the 1m models only.** The tick model checks legs (B1 calibration, check (f)), as the
  approved plan's stages say.
- **Implementation choices the plan left open:**
  - a resting entry still open at the window end, or at a forced exit, crosses at that minute's close and is closed
    with it;
  - a maker entry starts its trailing extreme at its market price (o(m0) under strict);
  - an entry never opens on cash ≤ 0;
  - a maker fill at or beyond its own stop opens nothing, as the engine's limit fills (strict can't hit this rule);
  - the B1 and B2 windows run from 10 min before to 2 h after each leg.

### Disclosures
- **Ceilings computed beforehand.** F1's dev results and the cost ceilings above were computed from committed
  results before this registration.
- **The screen's inputs were already read.** The plan's review read the committed dev trade files and reported
  per-family gross edges per trade in bp: S7 −11.9, S8 −26.8, F5 1.4, F6 4.5, F2 100.8, F4 47.9, S4 17.7, S5 14.7,
  F3 24.8.
- **Partial Stage A results were seen before this registration.** While testing and reviewing the code, agents ran
  the maker-first engine on real dev data (dev loaders only; nothing from 2024 or later). What they saw:
  - the taker run reproduces F1 on all 5 coins exactly;
  - with strict T = 15, entries filled as maker 577 of 600 times (BTC 114/118, ETH 135/136, SOL 107/109, XRP
    106/115, BNB 115/122) and decision exits 422 of 440, with no stop pre-emption and no funding inside a window;
  - at strict T = 5: ETH 133/136, BTC 110/118;
  - two sleeve equities while checking a float-tolerance fix: XRP strict T = 15 ended at $16,370.89 without the fix
    and $16,243.48 with it; ETH strict T = 5 at $13,647.20 and $13,627.85;
  - Q3-style random legs: XRP strict T = 5 had a fill rate of 0.81 and a mean saving of −1.07 bp without the fix,
    0.78 and −1.57 bp with it (T = 15: 0.89 and −1.26 bp, then 0.88 and −1.75 bp); BTC strict T = 5 was 1.87 bp,
    then 1.77 bp.

  The decision rules were fixed in the approved plan before any of this, and none was changed afterwards. The only
  code change made after these numbers were seen was the float-tolerance fix to the touch test (exact one-tick
  touches had been counted as trade-throughs), and it moves every figure in the conservative direction. The
  reviewer's point is recorded: with ticks this small, strict fills nearly every leg, so its result sits close to the
  fee-only ceiling, and check (f) is the operative test of whether the 1m model is too optimistic.
- **bookTicker exists only from 2023-05-16 in dev,** so the calibration covers one fairly calm regime.
## Execution study X1 — Stage A outcome (2026-10-19, after the Stage A run)

**Verdict: F1 NO-GO.** Under the strict model the primary policy (T = 15) adds +0.34%/yr, below the +0.5%/yr bar of
rule (a). It also fails rule (d): 2 of 4 years and 3 of 5 coins are positive. As registered, B2 is never downloaded,
and the paper bot stays taker. **D1 is NO-GO by arithmetic** (registered). B1 is not downloaded: it needs the
owner's OK.

**How the run went:**
- Commits: code `8f232eb`, registration `e89b4c9`, start record `3bc6eaf`, result `87ab1b5`.
- `x1-stage-a` ran once on dev data, with no hard error, and nothing dated 2024 or later was loaded.
- The taker run reproduced `F1_dev_daily.parquet` exactly: max |difference| 0. That is CAGR 44.9%, Sharpe 1.20,
  worst drawdown 28.1% and 600 trades.
- Every hard check passed.
- Results: `reports/results/X1_stage_a.json`; tables in `research/x1/`.

**Strict model, F1 dev 2020-01 → 2023-12 (5-coin portfolio, against the taker run):**

| Run | Gain vs taker | Sharpe | Worst DD | Maker fills / orders |
|---|---|---|---|---|
| taker (registered F1) | – | 1.202 | 28.06% | – |
| strict T = 5 | +0.294 %/yr | 1.210 | 28.1% | 93.9% of 1,040 |
| **strict T = 15 (primary)** | **+0.337 %/yr** | **1.211** | **28.14%** | **96.5% of 1,040** |
| strict T = 60 | +0.147 %/yr | 1.206 | 28.0% | 98.4% of 1,040 |

| Rule | Result | |
|---|---|---|
| (a) gain ≥ +0.5%/yr | +0.337%/yr | **fail** |
| (b) bootstrap 5% bound > 0 | 1.44e-6 a day | pass |
| (c) T = 5 and T = 60 > 0 | +0.294 and +0.147 %/yr | pass |
| (d) ≥ 3 of 4 years and ≥ 4 of 5 coins | years 2020 −0.05%, 2021 +0.67%, 2022 −0.10%, 2023 +0.83% (2 of 4); coins BTC +3.2%, ETH +3.6%, SOL −0.3%, XRP −2.9%, BNB +3.2% (3 of 5), summed daily differences | **fail** |
| (e) Sharpe −0.02 at most, drawdown +1 pp at most | 1.211 vs 1.202; 28.14% vs 28.06% | pass |

**Where the strict result comes from:**
- **The fills.** 1,004 of 1,040 orders filled as maker: 581 of 600 entries and 423 of 440 exits, half a minute
  after posting on average. Each fill saved the 3 bp fee difference, and fees fell from $9,766 to $5,035.
- **The orders that ran away.** The other 36 orders were the ones where price ran away. After 15 minutes they
  crossed at prices 58.7 bp worse on average.
- **Net effect.** +0.86 bp per order, which is about $34 a year on a $10,000 account.
- **XRP and SOL lose.** XRP's tick is 0.7–5.9 bp, so filling on a print through the post price takes a larger move.
  SOL's loss is mostly one unfilled exit on 2022-10-25 (−211 bp).

**Reported, not gated:**
- **Central model** (fills credited with the registered slippage): +0.85%/yr at T = 15, which would pass (a)–(e).
- **Optimistic model** (a touch fills, at the post price): +1.35%/yr, which would also pass.
- These are the plan's upper models, not the decision basis; check (f) would have been the test of whether the 1m
  models are too generous.
- **Post-and-cancel (information only, it changes F1's trades):**
  - at T = 15, 19 of 600 entries were missed; those trades averaged −0.05 R against +0.34 R for the trades taken;
  - at T = 60, 8 were missed, averaging +0.43 R.
- **Tick sizes:** all 231 coin-months were inferred from the 1m price grid, with no exchangeInfo fallback. The tick
  is 0.002–0.06 bp on BTC, up to 0.7–0.8 bp on ETH and BNB, up to 3.2 bp on SOL, and 0.7–5.9 bp on XRP.

**Q3: what a maker-first order costs per side** (all 5 coins, T = 15; the taker cost is 6.6 bp: the 5 bp fee plus
the 1–2 bp slippage):

| Kind of order | Legs | strict c_eff | central | optimistic | strict markout 5 min after a fill |
|---|---|---|---|---|---|
| random | 39,999 | 5.68 bp (saves 0.92, CI 0.79–1.04) | 4.18 | 2.52 | −3.7 bp |
| momentum (1h breakout) | 8,832 | 5.95 bp (saves 0.64, CI 0.18–1.11) | 4.41 | 2.66 | −7.5 bp |
| reversion (1h 2σ move, faded) | 8,654 | 6.75 bp (saves −0.18, CI −0.76–0.40) | 5.25 | 3.42 | −1.7 bp |

Fills are adversely selected: after a fill, the price keeps moving through the order. The conservative model
saves under 1 bp a side, and nothing at all for orders that fade a move.

**Screen** (`research/x1/stage_a_screen.csv`; the 1m costs above; information only):
- **Strict:** no earlier fast family is in the band where execution could change it.
- **Central and optimistic:** only S4 qualifies, and only for its entries from 2020 on (gross 7.3 bp per trade
  against a maker-first round trip of 6.4–6.8 bp and a taker round trip of 12.0 bp).
- **The other families sit outside the band:**
  - F2, F3, F4, S5 and all-years S4 earn more than the taker round trip, so they failed for other reasons;
  - S7, S8, F5 and F6 earn less than even the optimistic maker round trip.

**Independent check (before this outcome was written).** Four verifiers, three of them using none of the X1 code:
1. **Rebuild.** One wrote its own F1 and maker-first simulator from the registered rules. It reproduced the taker
   run bit for bit (all 600 trades and every daily return) and strict T = 5, 15 and 60 to about 1e-16, including all
   1,040 legs, every fill minute and every gate value.
2. **Gates.** One recomputed every gate, including the bootstrap with its own code.
3. **Q3 and screen.** One re-simulated all 57,489 Q3 legs (0 mismatches), rebuilt the momentum and reversion events
   from its own 1h bars, re-derived the tick table (231 of 231 months) and recomputed the screen and the cost rows.
4. **Protocol audit.** It found:
   - linear commits, with the code and the registration unchanged after they were committed;
   - every pin matching at HEAD, and exactly one run record and one result;
   - `research/holdout_log.jsonl` unchanged, no date of 2024 or later in any output, and no tick download.

Not rebuilt independently: the central and optimistic F1 runs, which are reported only.

**Robustness (from the checks).**
- **Rule (a) fails under every setting tried:** without the float-tolerance fix, +0.41%/yr; with the 2026 exchangeInfo
  ticks instead of the inferred table, −0.05%/yr.
- **Rule (d) is fragile:** without the tolerance fix it would pass (4 of 4 years, 4 of 5 coins). The verdict rests
  on rule (a).

**Erratum to the registration's Disclosures** (the registered section stays as committed, because it is pinned by
SHA256):
- **The counts came from mixed settings.** The fill counts quoted there as seen before registration (577 of 600
  entries, 422 of 440 exits, and the per-coin and T = 5 counts) came from review runs that did not all use the
  registered tick table. Some used the 2026 exchangeInfo ticks: BTC 0.1, BNB 0.01.
- **The committed run differs.** It has 581 and 423, and the per-coin entry counts differ (BTC 116, SOL 106, XRP
  104, BNB 120).
- **One sentence was wrong.** The sentence "the only code change … moves every figure in the conservative direction"
  is wrong: the tick source also differed, and for BTC and BNB that raised fills, the GO-favouring direction.
- **The Q3 figures quoted there** (XRP and BTC random legs) do not match the committed legs either. The BTC figure
  matches the exchangeInfo tick, and the XRP sample could not be identified.
- **No effect on the result.** The code was frozen before the registration, the tick rule is in the registered
  text, and the result is NO-GO.

**What it means, in plain words:**
- **For F1:** on these coins a resting limit order nearly always fills, because the tick is tiny. That saves the
  3 bp fee difference. But the few orders that don't fill are exactly the breakouts that run away, and waiting
  15 minutes on those costs about 59 bp. Net, maker-first execution would add about 0.3% a year to F1, below the bar
  and negative on XRP and SOL. **The paper bot keeps taker orders.**
- **For faster strategies:** the conservative model cuts the cost per side from 6.6 bp to about 5.7 bp. Only the
  optimistic model, which fills on a touch at a better price, gets near 2.5 bp. Only tick data (B1, 5.7 GB) can say
  which is closer to reality. Even under the optimistic model, no earlier fast family except S4's 2020+ entries
  sits where cheaper execution would flip it.

## Execution study X1 — Stage B1 outcome (2026-10-19, after the owner-approved B1 download)

**What ran:**
- **The download.** The owner approved the B1 download on 2026-10-19. 150 archives were downloaded, checksum-verified,
  streamed and discarded (commit `e031e66`): bookTicker and aggTrades for 15 days per coin (seed 1101), 75 coin-days
  between 2023-05-22 and 2023-12-30, 5.48 GB.
  - **Kept:** 403M book rows and 35.6M trades, as parquet under `data/ticks/` (8.9 GB, not committed; more than the
    plan's estimate because B1 keeps whole days). No row is on or after 2024-01-01.
- **The stage.** `x1-stage-b --set B1` ran once (commit `b8cc816`). It simulated the 587 complete generic Q3 legs
  that fall on those days through the tick model, at every T and in both queues: 3,522 runs.
  - **Excluded:** 30, 33 and 48 legs at T = 5, 15 and 60. One had no book: it went live at exactly 00:00 after a
    day that was not downloaded. The others were legs whose 60-minute markout ran past the day loaded.

**X1-cost-v1** (`research/x1/x1_cost_v1.csv`; all 5 coins; conservative queue; tick and 1m savings on the same
legs; the taker pays 6.6 bp per side):

| Kind of order | T | Legs | Tick model saving | 1m strict | 1m central | Correction (tick − strict) | c_eff per side (v1) |
|---|---|---|---|---|---|---|---|
| random | 5 | 410 | 2.16 bp | 1.37 | 2.78 | +0.79 | 4.97 bp |
| random | 15 | 407 | 2.48 bp | 1.69 | 3.22 | +0.79 | **4.89 bp** |
| random | 60 | 400 | 2.26 bp | 1.55 | 3.12 | +0.70 | 4.87 bp |
| momentum (1h breakout) | 5 | 75 | 1.23 bp | 2.03 | 3.51 | −0.81 | 6.60 bp |
| momentum | 15 | 75 | 2.51 bp | 1.86 | 3.34 | +0.64 | **5.31 bp** |
| momentum | 60 | 71 | 2.02 bp | 1.70 | 3.20 | +0.31 | 5.31 bp |
| reversion (1h 2σ, faded) | 5 | 72 | 1.89 bp | 0.99 | 2.52 | +0.90 | 6.19 bp |
| reversion | 15 | 72 | 3.16 bp | 2.59 | 4.22 | +0.57 | **6.18 bp** |
| reversion | 60 | 68 | 3.18 bp | 2.60 | 4.22 | +0.58 | 5.64 bp |

c_eff (v1) = the Stage A full-dev strict c_eff minus the correction. The pro-rata queue gives the same or slightly
higher savings (random T = 15: 2.63 bp).

**How to read it:**
- **The tick model lands between the two 1m models.** It saves 0.6–0.8 bp per side more than the strict 1m model,
  and 0.6–1.0 bp less than the central one.
- **The maker-first cost per side at T = 15** comes to about:
  - 4.9 bp for an order with no signal (26% below the taker);
  - 5.3 bp after a breakout;
  - 6.2 bp when fading a move.

  A maker-first round trip therefore costs about 10–12 bp against the taker's 13.2 bp.
- **Precision.** The random correction (+0.79 bp, day-clustered SE about 0.15–0.17) and the reversion correction
  (+0.57 bp, SE about 0.10) are resolved. The momentum correction is not:
  - its T = 15 95% CI is about −0.5 to +1.8 bp;
  - a single BTC leg (2023-06-05 16:01) joined a book with a momentary 20-tick spread and never filled, and it alone
    turns the T = 5 correction from +0.97 into −0.81 bp.

  Per-coin momentum and reversion cells (11–18 legs each) are noise; use the ALL rows.
- **The strict 1m model is far too harsh on XRP.** Its random correction is +2.45 bp. With a 1–6 bp tick, a print
  through one tick inside the open is rare, while fills at the touch from the queue are common.

**F1, information only (this changes no verdict).**
- **Why it matters.** Stage A's registered verdict (NO-GO, on the strict model) stands, and B2, which would have
  measured F1's own legs, stays undownloaded as registered. But B1 shows that the strict model understates real
  savings by 0.6–0.8 bp per side on generic orders.
- **The arithmetic.** F1's shortfall against the bar is 0.16%/yr, about 0.4 bp per leg. That is inside the range of
  the correction for breakouts (+0.64 bp, CI −0.5 to +1.8). So F1's real maker-first gain is probably between about
  +0.3 and +1.0%/yr: $30–100 a year on a $10,000 account, close to the bar rather than clearly below it.
- **What it would take to settle it.** Only a new registered version measuring F1's own legs on tick data could do
  that.

**Independent check (before this outcome was written).** Three verifiers:
1. **Hand re-simulation.** One wrote its own tick event loop from the registered text. It reproduced every committed
   tick leg exactly (586 legs × 3 T × 2 queues, including the 3 partial fills and a full T = 60 leg).
2. **Cost model recompute.** One recomputed all 54 cost-model rows from the legs file (largest difference 1.8e-15).
   It also confirmed:
   - the 1m savings are bit-identical to Stage A's;
   - the B1 legs are exactly the complete Q3 legs on the downloaded days;
   - the tick data is sound on 200 sampled legs: bid < ask everywhere, the 1m open inside the spread on 198, a
     1-tick spread on 196, and the joined book at most 0.93 s old.
3. **Protocol audit.** It found everything clean:
   - the commit chain `e0ed891` → `e031e66` → `b8cc816`, with `src/` unchanged since `8f232eb`;
   - the registered section's pin intact, one download record and one Stage B1 record, every pin equal to the start
     record, and no error;
   - the 75 days re-derived offline from seed 1101, nothing dated 2024 or later in any row, and the holdout log
     unchanged.

**Notes from the check** (none moves a cost-model value by more than 0.005 bp):
- **Pro-rata wording.** The registered text says "since the previous observation of the level". The code does not
  cut the queue pro rata on the first book row after the level was unseen (it only caps it there). The other reading
  moves the reported pro-rata mean by at most 0.002 bp. The corrections use the conservative queue only.
- **Midnight exclusions.** Four random legs (2, 3 and 4 at T = 5, 15 and 60) were excluded at midnight even though
  the next day was downloaded (BNB 09-17/18, ETH 09-20/21), because each coin-day is simulated on its own. All four
  filled within 5 minutes; including them would move at most 9 cells by up to 0.0044 bp.
- **Exclusion counts.** These were not logged by the run; they are recorded above.
- **`book_crossed`.** The flag covers only the time an order rests (237 of 1,115 filled rows, and about 25% of all
  legs): the archived opposite best reached the post price, which the model, filling only on prints, does not count
  as a fill.
- **Scope of X1-cost-v1.** It applies corrections measured in 2023-05 → 12 to 2020–2023 1m costs, whose taker
  baseline includes the registered 1–2 bp slippage. It is a planning number for a future registered round, not a
  validated cost.

**In plain words:**
- **For a faster strategy:** on these coins, posting a limit order and crossing after 15 minutes would cost about
  1–1.7 bp less per side than a market order for ordinary or breakout orders, and about 0.4 bp less when fading a
  move. A round trip costs roughly 10–12 bp instead of 13.2 bp. That helps, but it does not change the arithmetic
  for a strategy whose edge is a few bp per trade, and the screen found no earlier fast family in that band.
- **For F1:** the effect is worth roughly $30–100 a year on $10,000. The paper bot stays taker as registered.

## Round 8 (IBS) — registered 2026-10-20, before any Round 8 data is loaded and before any Round 8 backtest

**Why.** The owner wants more frequent trading. The current state:
- F1 is the only strategy with a clean passing holdout.
- D1 is in a forward paper test.
- Rounds 4–7 and X1 failed or closed.
- Short-term mean reversion in crypto is closed: S7, S8, F5 and F6 earned at or below costs.

The equity-index markets were never tried for mean reversion. Round 6 tested only trend (M1) on the TradFi proxies, M1
failed dev, and their 2016-01-01 → 2026-09-30 holdout was never opened.

Round 8 asks whether a published daily mean-reversion rule, fixed before any data, survives:
- Binance perp costs;
- a random-timing null;
- an alpha test against buy-and-hold.

**Owner decisions.**
- **2026-10-19, in planning, before any data** (plan `C:\Users\adnya\.claude\plans\floofy-crafting-pony.md`):
  - Long-only and long-short are both tested, as two co-primaries, each judged on its own gates.
  - The universe is SPY, QQQ and EWY only, so the commodity markets' 2016+ holdout stays clean.
  - The holding period is 1 day, the source paper's rule.
  - Approving the plan also approved the round's own gates, including **alpha as a gate**, which the practical
    rulebook lists as reported only.
- **2026-10-20, after the first null calibration failed and before any data** (owner's words): "Option A: keep the null
  and scenarios, require p < 0.01 per primary, with every no-edge row <= 2.5% at p < 0.01 on a fresh calibration (new
  seed) before registering." See Null calibration below.

**Code:** commit `1aff072`.
- **New:**
  - `src/btcq/data/external_ohlc.py`;
  - `src/btcq/strategies/r8_ibs.py`, which holds the rule and the null;
  - `src/btcq/backtest/daily_sim.py`;
  - `src/btcq/research/round8.py`;
  - `src/btcq/research/round8_calib.py`;
  - `tests/test_external_ohlc.py`, `test_daily_sim.py`, `test_r8_ibs.py` and `test_round8.py`.
- **Changed:** `run.py` (the `r8-*` verbs) and `.gitattributes`.
- **Unchanged:** `engine.py`, `external.py` and `stats.py`.
- **Pins:** the 18 files in `round8.CODE_FILES` are hashed at the sweep start and checked before the holdout.
- **Tests:** 673 tests pass at that commit.

### Rule
Source: Pagonidis, "The IBS Effect: Mean Reversion in Equity ETFs", NAAIM 2014.

At each trade date t of a market (its own rows), IBS_t = (C_t − L_t) / (H_t − L_t). IBS is undefined, so there is no
signal, when H ≤ L or when H, L or C is missing.
- **IBS-L:** go long at the close of t if IBS_t ≤ 0.20 and hold to the next trade date's close. The position is renewed
  while the signal repeats; otherwise it is exited at that close.
- **IBS-LS:** IBS-L plus a short at the close of t if IBS_t ≥ 0.80, held and renewed the same way. A switch between long
  and short is a flip at the same close.
- **Sizing (Round 6's):** min(0.25 / σ, 2.0) × the sleeve's equity at entry or flip.
  - σ = the standard deviation (ddof 0) of the last 90 daily log returns of the market's own closes × √252.
  - There is no position until σ exists.
  - A renewed position is never resized.
- **Grid (12 configs, used only for N_eff, the plateau and PBO):**
  - direction {L, LS} × θ {0.15, 0.20, 0.25} × hold h {1, 2};
  - LS shorts at IBS ≥ 1 − θ;
  - with h = 2 the position follows the latest non-zero signal of the last 2 closes, and is flat if there is none.
- **Primaries (fixed): IBS-L = (L, 0.20, 1) and IBS-LS = (LS, 0.20, 1).**
  - The plateau neighbours of each are θ 0.15 and θ 0.25 at h = 1, and θ 0.20 at h = 2, in the same direction.
  - Nothing is selected from the grid.

### Data (no download)
| Perp | Proxy | First date | Slippage per side |
|---|---|---|---|
| SPYUSDT | SPY | 1993-01-29 | 3 bp |
| QQQUSDT | QQQ | 1999-03-10 | 2 bp |
| EWYUSDT | EWY | 2000-05-12 | 2 bp |

- **Source:** Round 6's raw Yahoo chart files (`data/external/raw/yahoo/{SPY,QQQ,EWY}_20261016T*.json`). Their SHA256s
  are pinned in `research/round6/data_manifest.csv`. The slippage tiers are Round 6's 2026-10-16 snapshot.
- **Parser** `external_ohlc.parse_yahoo_ohlc`:
  - it applies `external.parse_yahoo`'s date rules: exchange-local trade dates; a bar stamped 17:00 local or later
    belongs to the next weekday; the unfinished bar of the download day and null closes are dropped; of duplicate dates
    the last is kept;
  - it keeps open, high and low (NaN when Yahoo gives null);
  - each of these is a hard error:
    - its (date, close) rows differ from `parse_yahoo`'s;
    - a raw file's SHA256 differs from Round 6's manifest;
    - after Round 6's cleaning, any (date, close) differs from Round 6's verified parquet, row for row.
- **Outputs:**
  - derived parquets under `data/external/ohlc/`, not committed;
  - `research/round8/data_manifest.csv` (the raw, Round 6 parquet and derived parquet SHA256s and the row counts),
    committed and checked on every load together with Round 6's manifest pins.
- **Lock:**
  - Dev loads return only rows dated before 2016-01-01.
  - The holdout (2016-01-01 → 2026-09-30) needs Round 8's own unlock (`external_ohlc.unlock_r8`). It writes a
    structured `round: 8` record tagged `[R8:` in `research/holdout_log.jsonl`.
  - Round 8's loader can load only these 3 markets. It never unlocks the crypto 2024+ data or Round 6's other markets.
- **Blind data report** (`research/round8/data_report.csv`), committed before the sweep start record:
  - holdout rows give only counts: rows, null open/high/low, OHLC consistency violations, and days with H = L;
  - dev rows also get range checks;
  - rows after 2026-09-30 are only counted.
- **Red flags.** Any one stops the round until a dated amendment:
  1. a raw-file SHA256 or a close differs from Round 6 (raised while building);
  2. null H or L on more than 0.5% of the rows with a close;
  3. L ≤ min(O, C) ≤ max(O, C) ≤ H fails, with a relative tolerance of 1e-6, on more than 0.1% of rows;
  4. H = L on more than 1% of dev rows;
  5. any dev row with (H − L) / C > 30%. Dev rows between 15% and 30% are listed.
- IBS does not change with split adjustments, because Yahoo adjusts O, H, L and C alike. Closes are not
  dividend-adjusted, as in Round 6 and as for a perp holder.

### Execution, account and costs
- **Fills:** a decision at trade date t's close fills at C_t, the same close (`daily_sim.simulate`, lag 0).
  - Reported only, never trials: a fill at the next open O_{t+1} and at the next close C_{t+1} (lag 1).
  - A decision at the window's last close is not opened, because it would be held past the window. An open position is
    closed there.
- **Simulator** (`src/btcq/backtest/daily_sim.py`):
  - signed positions, sized at entry or flip only;
  - fees and slippage on every fill;
  - daily marks at each close;
  - Round 6's liquidation rule: equity ≤ 0 or notional ≥ 20× equity at a mark, or an exit that leaves the cash ≤ 0,
    closes the sleeve at max(equity, 0), and it stays flat. It is never expected.
  - **With lag 1 it reproduces `engine.run` on close-only rows to 1e-12:** trades, fees and daily marks, tested on
    fuzzed paths.
- **Account (one per primary):**
  - 3 slots of $10,000 / 3, each compounding its own sleeve;
  - the account's daily return = the sum of the slots' daily returns / 3, on the UTC calendar;
  - days without trading, and markets that aren't ready yet, count as 0;
  - Sharpe is annualised with √365.
- **Costs:**
  - **base:** 5 bp taker plus the slippage tier, per side, on every fill;
  - **2×:** fees and slippage doubled;
  - funding is 0 in the base case.
  - **Funding stress:** 5%/yr on |notional / equity| at the previous trade date's mark, over the calendar days since it
    (Mondays carry 3), long or short. It is subtracted from the sleeve's daily return (floored at −100%), as Round 6's
    drag.
  - Reported only: 2× costs plus funding together.
  - Reported only: **maker-first**, from `research/x1/x1_cost_v1.csv` (T = 15, ALL).
    - Entries use the reversion class, 6.18 bp per side against the 6.6 bp taker (−0.42 bp).
    - Exits use the random class, 4.89 bp (−1.71 bp).
    - Labelled *calibrated on Binance crypto majors in 2023, not TradFi books; a resting order also moves the fill off
      the close*.

### Windows
- **Dev:** 1993-07-01 → 2015-12-31, 8,219 calendar days.
  - Each market trades once σ exists: SPY from the start, QQQ from about 1999-07-19, EWY from about 2000-09-20 (the
    source first dates + 90 NYSE trading days).
  - That is 18.1 full-book years of 22.5.
- **Holdout:** 2016-01-01 → 2026-09-30, 3,926 days, one shot.
  - Positions start in sync with the signal: the state machine runs on all loaded rows, and the first executed decision
    is the first holdout trade date's close.

### Null: date-level relocation of held days
- **The unit:** a decision date d of the union trade-date grid, with its position vector over the 3 markets. Each entry
  is the market's side, or 0.
- **Each draw** moves every held date, whole vector at once, to a date d′ drawn uniformly with replacement from the
  eligible dates of d's stratum, and re-sizes there with σ_{m,d′}.
- **Strata (date level):**
  - **era:** the set of markets whose σ exists;
  - **volatility:** the EWMA volatility tercile (λ = 0.94) of the slot-scaled equal-weight index, with terciles within
    each era. The index's daily return is the sum of the markets' close-to-close returns / 3;
  - **trend:** the index above or below its 200-day SMA;
  - **gap:** whether the next trade date is at least 2 calendar days away (a weekend or holiday follows);
  - dates before the EWMA (30 index returns) and the SMA (200 grid rows) exist form their own volatility/trend cell,
    still split by era and gap.
- **Eligible destinations:** window dates, excluding the window's last date, on which every ready market has a row, σ
  and a next close inside the window.
- **Statistic:** the mean over held market-days of side × min(0.25 / σ_{m,d}, 2) × the next close-to-close simple return,
  at zero cost. That is the account's gross P&L sized on each day; costs are the same in every draw.
- **Settings:**
  - 1,000 draws, seed 8008 (numpy's PCG64);
  - p = (1 + #{null ≥ actual}) / 1,001;
  - **each primary needs p < 0.01** (owner decision 2026-10-20; see Null calibration).
- **Unit tests:** the identity draw reproduces the statistic exactly and gives p = 1, and a draw keeps each date's
  position vector whole and inside its stratum.
- **What it controls for:** the drift (beta), volatility regimes, trend regimes and the weekend, by stratum. The alpha
  gate checks beta separately.
- **What it does not:** real predictability that comes from noise in the closing print (bid-ask bounce, auction
  imbalance, EWY's premium to a stale NAV). See Disclosures.
- **Known small bias** (found by the code review; disclosed, not changed):
  - A pool date e carries the next return r_{e+1}, which also decides whether e+1 is a held low-IBS date, and e and e+1
    usually share a stratum.
  - So the pools of strata with many held dates have slightly depressed means. The null is centred about 0.07–0.14 null
    standard deviations too low, which makes the test slightly liberal.
  - The calibration measures the null with this bias included.
  - IBS-L's null is also somewhat conservative: the z spread is about 0.85–0.9, because held dates are a
    without-replacement quarter of their strata while the null draws with replacement.

### Null calibration
The rule, fixed in the approved plan before any calibration: in every no-edge scenario, for both rules, the rejection
rate must be ≤ 7% at p < 0.05 and ≤ 3.5% at p < 0.025 (the gate's threshold then).

**The first calibration failed** (seed 88, 499 draws, 1,000 panels per no-edge scenario; code commit `ea9c569`; the file
is in that commit):
- 5 of the 6 no-edge scenarios passed for both rules.
- **varprem** failed:
  - IBS-L 7.5% at p < 0.05 (3.4% at p < 0.025);
  - IBS-LS 8.7% at p < 0.05 and 4.5% at p < 0.025.
- The other rows at p < 0.05 / p < 0.025:

  | Scenario | IBS-L | IBS-LS |
  |---|---|---|
  | iid | 1.9% / 0.6% | 4.7% / 2.5% |
  | garch | 2.9% / 1.2% | 4.5% / 2.5% |
  | regime | 2.4% / 0.8% | 4.1% / 2.2% |
  | factor | 2.3% / 0.8% | 4.8% / 2.4% |
  | weekend | 3.6% / 1.3% | 4.6% / 2.3% |

- The planted κ = 0.10 edge was detected 99.5% / 100% of the time.
- **Why varprem fails:** the expected return rises with each market's own conditional variance, and in this scenario
  each market has its own GJR volatility. A low-IBS day raises that market's next-day variance, and so its expected
  return. The date-level strata cannot see one market's own variance.
- **Diagnostics on the same panels** (information, run after the failure):
  - splitting the strata by grid-row parity or by weekday would remove the small bias above, but IBS-LS still failed
    (7.0–7.2% / 3.6–3.7%);
  - at p < 0.01 every row was at or below 2.5% (varprem 0.7% / 2.1%, iid 0.2% / 0.8%).

**Owner decision (2026-10-20):** keep the null and the scenarios. Each primary's null gate becomes p < 0.01. The new
rule: every no-edge row, for both rules, must reject at most 2.5% at p < 0.01 (the Bonferroni share of a 5% family error
for the two primaries) on a fresh calibration with a new seed. The rates at p < 0.05 and p < 0.025 are reported only,
because the test no longer uses those levels.

**The registered calibration** (`r8-calibrate` at commit `1aff072`, `research/round8/null_calibration.csv`):
- seed 1088, 999 draws (at 999 draws, p < 0.01 means at most 8 draws at or above the actual, nominal 0.9%; the dev test
  at 1,000 draws is nominal 0.999%);
- 1,000 panels per no-edge scenario, 400 for the bull row, 200 per planted κ;
- the registration commands refuse to run unless the committed calibration meets the rule, was run by the current
  calibration code (`707d15e6b847544e2d114c8bb318d95fbc5c09703f69fd6723b7678436a40450`: `r8_ibs.py`, `r6_macro.py`, `ta.py`, `round8_calib.py`) under the same numpy, pandas,
  numba and Python versions, and detects the planted κ = 0.10 edge at p < 0.01 in at least 80% of panels for both rules.

**Generator:**
- Each market's daily simple return is R = μ_t + σ_t z, so the expected simple return is μ_t on every day by
  construction (Round 7's lesson).
- The day is one Brownian bridge in log price from the previous close to log(1 + R). The first step, the overnight gap,
  carries 25% of the variance. Then 78 equal intraday steps run from the open to the close, so H and L are the path's
  extremes.
- Daily volatilities are 1.2%, 1.7% and 2.1%.
- Panels follow the dev ramp: market 0 alone, then markets 1 and 2 ready about 6.0 and 7.2 years into about 5,595 window
  days, on a Monday–Friday calendar. Returns are clipped at −50%.

| Scenario | What it is |
|---|---|
| iid | constant volatility, drift 3 bp/day, Gaussian shocks, independent markets |
| garch | GJR-GARCH (ω 0.02, α 0.03, γ 0.10, β 0.90) with unit-variance t(4) shocks, independent markets |
| varprem | garch, with an expected return of 3 bp + 3 σ_t²: low-IBS days predict higher volatility and so a higher return |
| regime | a market-wide bull/bear drift of ±8 bp/day, switching with probability 1/252 a day |
| factor | one factor (loadings 0.95 / 0.895 / 0.737: ρ 0.85, 0.70, 0.66) with a common GJR variance and t(4) shocks; shared intraday paths, so IBS signals coincide across markets |
| weekend | after a weekend: 3× the daily variance (the extra variance overnight) and a −5 bp drift |
| bull (information) | drift 8 bp/day |
| planted (power) | factor + regime, plus +κσ_t after IBS ≤ 0.2 and −κσ_t after IBS ≥ 0.8, κ ∈ {0.05, 0.10} |

| Scenario | Kind | Panels | IBS-L p < 0.01 (s.e.) | IBS-L p < 0.025 / 0.05 | IBS-LS p < 0.01 (s.e.) | IBS-LS p < 0.025 / 0.05 |
|---|---|---|---|---|---|---|
| iid | no-edge | 1,000 | 0.5% (0.2%) | 1.1% / 3.5% | 0.7% (0.3%) | 2.6% / 4.9% |
| garch | no-edge | 1,000 | 0.3% (0.2%) | 1.4% / 3.7% | 1.7% (0.4%) | 3.1% / 5.8% |
| varprem | no-edge | 1,000 | 1.1% (0.3%) | 2.2% / 4.4% | 1.5% (0.4%) | 3.6% / 7.1% |
| regime | no-edge | 1,000 | 0.2% (0.1%) | 0.9% / 2.7% | 0.8% (0.3%) | 2.3% / 4.7% |
| factor | no-edge | 1,000 | 0.1% (0.1%) | 0.9% / 2.3% | 0.8% (0.3%) | 2.4% / 4.9% |
| weekend | no-edge | 1,000 | 0.3% (0.2%) | 1.3% / 2.8% | 0.9% (0.3%) | 2.5% / 4.8% |
| bull | information | 400 | 0.8% (0.4%) | 1.2% / 3.8% | 1.5% (0.6%) | 2.8% / 6.5% |
| planted κ = 0.05 | power | 200 | 35.0% (3.4%) | 53.5% / 69.0% | 74.0% (3.1%) | 84.5% / 93.5% |
| planted κ = 0.10 | power | 200 | 98.0% (1.0%) | 100.0% / 100.0% | 100.0% (0.0%) | 100.0% / 100.0% |

**Every no-edge row is within the limit** for both rules (the highest is garch IBS-LS,
1.7%). Pooled over the 6,000 no-edge panels the rate at p < 0.01 is 0.42% for
IBS-L and 1.07% for IBS-LS, against a nominal 1.0%. At p < 0.05, which the test no longer uses,
varprem's IBS-LS row is 7.1%: the
null is liberal there, as in the first calibration. The planted κ = 0.10 edge is detected at p < 0.01 in
98% (IBS-L) and
100% (IBS-LS) of panels.

### Power table
`round8.power_table` (`research/round8/power_table.csv`, `r8-power` at commit `1aff072`):
- iid Gaussian calendar-day returns, 200,000 simulations per row from exact sufficient statistics, seed 8;
- **dev** = 8,219 days. A dev pass needs mean > 0 and DSR ≥ 0.95 at N_eff (null variance 1/(T − 1));
- **holdout** = 3,926 days. A holdout pass needs mean > 0 and Sharpe ≥ 50% of dev's;
- ρ = the share of the true dev Sharpe that survives into the holdout.

Three variants:
- **constant:** the full book every dev day.
- **ramp:** the dev mean scales with the live share of the 3 slots, and the variance with it, from the ready dates above
  (18.1 full-book years).
- **ramp+alpha:** the row's Sharpe is the true *alpha* Sharpe.
  - The strategy is β·b + a, where b is the account-scaled hold at an annual Sharpe of 0.45, and β·sd(b) = 0.5 sd(a), so
    the hold explains 20% of the variance.
  - Dev also needs alpha PSR ≥ 0.95. The holdout also needs alpha mean > 0 and alpha Sharpe ≥ 50% of dev's.
  - β is fitted by OLS in each window.

**constant** (22.5 full-book dev years; P(pass) at N_eff < 2 / 3):

| True dev Sharpe | P(dev) N_eff < 2 / 3 | P(pass) ρ = 1 | P(pass) ρ = 0.5 | P(pass) ρ = 0 |
|---|---|---|---|---|
| 0 | 5.0% / 0.6% | 1.2% / 0.1% | 1.2% / 0.1% | 1.2% / 0.1% |
| 0.25 | 32.2% / 9.6% | 16.4% / 4.0% | 11.4% / 2.6% | 7.1% / 1.6% |
| 0.5 | 76.7% / 44.9% | 57.3% / 31.2% | 34.3% / 17.1% | 13.5% / 6.0% |
| 0.75 | 97.3% / 85.7% | 85.1% / 74.0% | 47.8% / 39.9% | 11.3% / 8.6% |
| 1 | 99.9% / 98.8% | 93.9% / 92.8% | 49.8% / 48.9% | 6.0% / 5.8% |

**ramp** (18.1 full-book dev years; P(pass) at N_eff < 2 / 3):

| True dev Sharpe | P(dev) N_eff < 2 / 3 | P(pass) ρ = 1 | P(pass) ρ = 0.5 | P(pass) ρ = 0 |
|---|---|---|---|---|
| 0 | 5.0% / 0.6% | 1.2% / 0.1% | 1.2% / 0.1% | 1.2% / 0.1% |
| 0.25 | 28.1% / 7.6% | 14.5% / 3.2% | 10.1% / 2.1% | 6.0% / 1.2% |
| 0.5 | 68.4% / 35.6% | 52.1% / 25.1% | 31.6% / 13.9% | 12.9% / 5.0% |
| 0.75 | 93.9% / 75.6% | 84.0% / 66.5% | 49.9% / 37.3% | 12.8% / 8.6% |
| 1 | 99.5% / 96.0% | 95.1% / 91.6% | 55.8% / 53.0% | 8.1% / 7.3% |

**ramp+alpha** (18.1 full-book dev years; P(pass) at N_eff < 2 / 3):

| True dev Sharpe (alpha) | P(dev) N_eff < 2 / 3 | P(pass) ρ = 1 | P(pass) ρ = 0.5 | P(pass) ρ = 0 |
|---|---|---|---|---|
| 0 | 4.9% / 3.1% | 1.1% / 0.6% | 1.1% / 0.7% | 1.1% / 0.6% |
| 0.25 | 27.6% / 20.1% | 13.5% / 9.3% | 9.2% / 6.3% | 5.7% / 3.9% |
| 0.5 | 67.9% / 56.7% | 50.0% / 40.9% | 29.9% / 23.8% | 11.7% / 9.1% |
| 0.75 | 93.7% / 87.8% | 82.4% / 76.8% | 47.5% / 43.6% | 11.9% / 10.6% |
| 1 | 99.5% / 98.4% | 94.3% / 93.2% | 53.6% / 52.7% | 7.3% / 7.0% |

- **The dev Sharpe needed** (constant): about 0.35 at N_eff < 2, 0.46 at 2, 0.53 at 3 and 0.57 at 4.
- **Decay decides the holdout.** At McLean–Pontiff-like decay (ρ ≈ 0.4–0.5), even a strong rule passes about half the
  time. A fully decayed edge (ρ = 0) still passes the holdout about 6–18% of the time after a dev
  pass, at true Sharpes of 0.5–1.0.
- **Not modelled** (each can only lower these numbers):
  - the null gate at p < 0.01, which at the calibration's planted κ = 0.05 edge (about 6 bp) detects
    35% / 74%, against 48% / 82% at p < 0.025 in the first calibration;
  - 2× costs, funding, year share, plateau, breadth and the holdout drawdown gate;
  - weekend zeros in the calendar series;
  - the correlation between the markets (the ramp's variance scales with the live share, as in Round 6).

### Dev gates (each primary on its own config)
1. At least 100 trades (holding episodes). Expected: about 2,000 for IBS-L and about 5,000 for IBS-LS.
2. Mean daily return > 0 at 2× fees and slippage.
3. Mean daily return > 0 under the funding stress at base costs.
4. **Null p < 0.01.**
5. DSR ≥ 0.95 at the family N_eff (Amendment 1):
   - N_eff = `stats.effective_trials` of all 12 configs' dev daily returns;
   - null variance 1/(T − 1);
   - no deflation when N_eff < 2.
6. **Alpha (round-specific):** PSR ≥ 0.95 of the residual after an OLS beta (`stats.beta_alpha`) to the account-scaled
   buy-and-hold.
   - The hold, literally: each market's close-to-close return on its own trade date, summed over the markets with a
     return that day and divided by 3. Days with none are 0, and there are no costs.
   - Why not Round 6's mean of the available markets: it would triple the benchmark's weight after 2000 against the
     SPY-only years and bias one OLS beta.
7. Best calendar year ≤ 65% of the summed dev daily returns (1993 counts from July).
8. Plateau: all 3 neighbours have a positive total return at base costs.
9. Breadth: at least 2 of 3 markets have a positive summed contribution (the sleeve's daily returns / 3) at base costs.

PBO (CSCV over the 12 configs, 16 slices) is reported only (Amendment 2).

### Holdout gates
These apply to the candidate only. One unlock serves both primaries; only primaries that passed dev are evaluated.
1. Mean daily return > 0.
2. Sharpe ≥ 50% of the dev Sharpe.
3. Worst end-of-day drawdown ≤ the 95th percentile of a dev block bootstrap:
   - block 30, 10,000 paths of 3,926 days, seed 7;
   - drawn from the first day all 3 markets are ready (about 2000-09-20) to 2015-12-31;
   - computed and stored at the sweep for each primary.
4. Alpha mean > 0, with the beta from an OLS fit inside the holdout against the holdout's account-scaled hold.
5. Alpha Sharpe ≥ 50% of the dev alpha Sharpe.

### Verdict and decision rule
- **PASSES IN PRACTICE** = the candidate passes every dev gate and every holdout gate.
- **Candidate:** the primary that passed every dev gate. If both did, the one with the higher dev alpha PSR; if those tie
  (the PSR saturates at 1), the higher dev alpha Sharpe; then IBS-L.
  - The candidate is recorded in the dev verdict (`reports/results/round8/IBS.json` and the `r8-sweep` record) before
    the unlock.
  - **Only the candidate can pass.** The other primary's holdout result is information only and can never be promoted,
    so the holdout never chooses.
- **FAIL:** the rule is closed on these markets, with no reruns or variants. **Any error after the unlock is a FAIL.**
- **Checkpoint:** after a dev pass, stop for the owner's OK before the unlock, because the unlock uses up these markets'
  only clean holdout.
- **What a pass means:** a separately registered paper book only, with no real orders and no API keys.
  - Its first step is the never-run check of the SPY, QQQ and EWY perps against the proxies, from the VPS.
  - Real money stays the owner's decision after a forward pass.
  - F1 and D1 are unchanged.

### Holdout lock and run order
1. **Code and tests,** on synthetic data, with two independent code reviews and a re-review of the fixes. Then
   `r8-calibrate` and `r8-power` (synthetic), committed with the code, and then this section.
2. **`r8-data`:** the OHLC re-parse of Round 6's raw files, the blind data report and the manifest, committed.
   - Any red flag stops the round.
   - It refuses to run again once the report or the manifest exists.
3. **`r8-sweep`** logs a start record and stops.
   - **The record pins:**
     - the version;
     - the SHA256 of the 18 code files, both data manifests, the data report, `F1_dev_daily.parquet` and this section's
       committed text;
     - the numpy, numba, pandas, scipy, pyarrow and Python versions.
   - **The run:** the run log (`research/round8/runs.jsonl`) is committed, and the second `r8-sweep` runs dev once for
     both primaries.
   - **Refusals:** a sweep is refused if any start record for its version, in the working copy or in any committed
     version of the log, has different pins.
   - **Hard errors:** a run record is logged before the run starts, and any error before the dev verdict is logged as a
     hard error. A version bump may run only after every earlier started version has a committed hard error.
   - **Once any version has a dev verdict (PASS or FAIL), the dev sweep never runs again,** with or without a version
     bump.
   - **The verdict:** `reports/results/round8/IBS.json` is saved first. The `r8-sweep` record logs the candidate, every
     pin and the yardsticks: each primary's dev pass, gates, Sharpe, alpha PSR, alpha Sharpe and drawdown limit.
4. **An independent rebuild of dev** from this text, before the outcome is written.
   - **Outcomes are written as separate top-level sections (`## Round 8 ... outcome`), never inside this section, whose
     committed text is pinned.**
5. **The holdout,** only for a dev pass and only with the owner's OK: `r8-validate --unlock-holdout`, once
   (`[R8: IBS v1]`).
   - **It checks:**
     - the pins and the committed verdict;
     - a committed `r8-sweep` record whose candidate, yardsticks and pins equal `IBS.json`'s, and exactly one logged
       verdict;
     - the trial registry, the data report, the manifests, the calibration, the power table and
       `F1_dev_daily.parquet`;
     - that the holdout log has no Round 6 record.
   - **What it logs:** a validate-start record in the run log, then the structured unlock record.
   - **There is one unlock and no resume:**
     - any Round 8 holdout-log record or validate-start record, in any committed version of the logs or their working
       copies, counts as a touch;
     - a touch without a recorded result is recorded as a FAIL;
     - any error after the unlock is a FAIL.
   - Once the holdout is touched, the dev sweep can never run again.
   - Independent checks after the holdout verdict log each load as `[R8 verify`; such a load is refused before the
     unlock.

**Code freeze:** the 18 pinned files must not change between the sweep start record and the holdout. They include shared
modules such as `run.py`, `stats.py` and `config.py`. By design, any edit blocks the Round 8 holdout for good.

### Trials and multiple testing
- **Own registry,** as in Round 6, because the 1993–2015 calendar would distort the crypto N_eff:
  - `research/round8/trials.csv` and `research/round8/neff_history.jsonl`;
  - the 12 trial return series in `reports/results/round8/`;
  - version 1. A version bump, allowed after a hard error only, adds trials.
- **Family N_eff** over all 12 configs, used for both primaries' DSR.
- **Reported:**
  - DSR at the raw N = 12;
  - N_eff together with M1's 3 configs, and each primary's DSR at it;
  - the strict tier: DSR on dev (and on dev + holdout after the unlock) at N = max(the project's ratcheted N_eff, Round
    8's own, 14.85 + 5 = 19.85).

### Reported, not gated
- PBO and weekly t.
- Holdout alpha PSR, a stricter alternative the owner could ask for.
- Weekly correlation with F1 and its HAC alpha:
  - over 2020–23 against `F1_dev_daily.parquet`;
  - over 2024–26 against F1's committed holdout returns;
  - these overlaps lie inside the holdout, so this is never a gate.
- Results by market and by year; Mondays against other days; the null statistic by the decision date's volatility
  tercile and gap.
- The next-open and next-close fill sensitivities; Roll's implied spread by market and era (1993–2000, 2001–08,
  2009–15); 2× costs plus funding; maker-first.
- Turnover, days with orders, days in market, fees paid, and the funding stress cost per year.

### Disclosures
- **Clean status of 2016-01-01 → 2026-09-30 for SPY, QQQ and EWY:**
  - never loaded by any strategy run;
  - no `[R6` or Round 6 record in `holdout_log.jsonl` (now enforced);
  - Round 6's data report used holdout rows only for dates, counts, signs and runs of equal closes;
  - the Amendment 1 source check used only dev returns;
  - the 2026 perps-against-proxies check was never run.
- **Not blind:**
  - General market history is known: the 2016–26 US bull market, with the drawdowns of 2018 Q4, 2020 and 2022, and
    Korea's.
  - The planner knows, in general, practitioner commentary that daily mean-reversion rules on US index ETFs kept working
    for much of the time after 2016, with no specific figures.
  - Seen during planning, in search snippets only and not read further:
    - a 2023 study (Pandey & Joshi, arXiv 2306.12434) reported that IBS "held up" on more recent country-ETF data;
    - a blog headline claimed an IBS strategy gained on SPY.
- **Order of events (accepted by the owner with the plan):**
  1. the snippet about the 2023 study was seen while sources were being searched;
  2. after that, the planner recommended IBS over RSI(2);
  3. the owner chose IBS.

  So holdout-period information about other ETFs may have softly shaped the choice of idea. No Round 8 data was used,
  the rule and its thresholds come from the 2014 paper, and no parameter was chosen with data after 2013.
- **Dev mostly replicates the paper's own sample.** 1993–2013 lies inside it; only 2014–15 and the holdout come after.
  - Amendment 1's "any earlier grid that chose them" cannot be applied in full: the author's own search over thresholds
    and indicators cannot be counted, so N_eff over the 12 configs under-deflates.
  - **The holdout is the real evidence.** The dev DSR and null show a replication, not a discovery.
- **The main execution risk:** the signal and the fill share one closing price.
  - Transient noise in the ETF's closing print is booked as edge: bid-ask bounce, auction imbalance, and EWY's premium or
    discount to a stale NAV. A perp order at the close cannot capture it, and IBS-L leans on EWY.
  - Reported in dev: Roll's implied spread, next to the next-open and next-close fills.
  - Costs are charged conservatively: a full round trip on every episode, with renewals free.
  - The paper's data source (CSI) may have used dividend-adjusted closes, which would overstate its SPY gross figures by
    about 1 bp a day.
- **Seen before:**
  - Round 6's M1 ran on these dev closes, and Round 6's data report checked dev returns.
  - An unregistered exploratory long-only D1 trend run (2026-10-17) also used SPY, QQQ and EWY dev closes up to 2015. It
    was trend only, with no mean-reversion statistic. This was added to the plan after approval, as a disclosure only.
- **Proxies differ from the perps:**
  - the ETF's official close against the perp's price at the close;
  - Binance's EWMA pricing at weekends (Friday positions carry the weekend);
  - the perps are listed only since 2025-12, and their funding is unknown;
  - the perp's own US-session high and low may differ from the ETF's;
  - the slippage tiers come from one snapshot;
  - dividends are excluded.
- **Multiplicity:** 2 co-primaries, so the null threshold is split and N_eff is shared.
- **The rulebook is extended:** the alpha gates and the p < 0.01 null gate are stricter than the practical rulebook,
  by owner decision.
- **The null's threshold was set after a calibration failed.** No data was involved, and the scenarios and the null are
  unchanged. Every synthetic run of the null seen before this registration:
  - **The development pilot** (300 panels × 199 draws, seed 88's streams) at p < 0.05 / p < 0.025:

    | Scenario | IBS-L | IBS-LS |
    |---|---|---|
    | iid | 1.0% / 0.7% | 5.7% / 2.3% |
    | garch | 3.7% / 1.0% | 3.0% / 1.7% |
    | varprem | 8.0% / 1.7% | 8.3% / 4.0% |
    | regime | 1.0% / 0.7% | 5.0% / 1.3% |
    | factor | 2.7% / 1.3% | 4.3% / 2.0% |
    | weekend | 2.7% / 1.3% | 3.0% / 1.0% |

    Planted κ = 0.10 was detected 98–100% of the time.
  - **Diagnostics** (600 fresh panels, 199 draws) at p < 0.05 / p < 0.025:

    | Variant | IBS-L | IBS-LS |
    |---|---|---|
    | varprem as registered | 5.3% / 1.8% | 6.0% / 3.0% |
    | without the leverage effect | 2.5% / 0.7% | 4.8% / 3.0% |
    | on the factor structure | 3.7% / 1.2% | 4.8% / 2.0% |

  - The first calibration and its diagnostics, above.
  - Two reviewers' own synthetic runs (iid, garch, varprem and factor, 400–1,200 panels), with mean z +0.06 to +0.27.

### Changes from the approved plan, decided before any data
- **The null's gate is p < 0.01 per primary, and the calibration rule is ≤ 2.5% at p < 0.01** (owner decision
  2026-10-20, above). The plan had p < 0.025, with limits of 7% and 3.5%.
- **The null lives with the rule** (`r8_ibs.py`, as in Round 7), not in `round8.py`. That way the calibration's SHA256
  pin covers exactly the rule, the null, the sizing they reuse and the calibration code.
- **No resume and no rerun.**
  - Round 7's one-time resume after a crash is dropped: the plan says any error after the unlock is a FAIL.
  - The dev sweep never runs again once any version has a verdict: the plan says a FAIL closes the rule with no reruns
    or variants, and a rerun after a pass could re-roll the candidate.
- **Implementation choices the plan left open:**
  - the last-close convention of same-close fills (above);
  - the EWMA (the recursion with weight 0.06 on the newest squared index return, started at the first return and
    defined from the 30th);
  - the terciles: a rank split of the window's dates in each era, with ties by date. The strata are a conditioning
    variable of the null, never a signal;
  - warm-up dates are split by era and gap;
  - a held market-day without a next close inside the window earns nothing and leaves its vector;
  - a held date whose stratum has no eligible date stays in place (counted; not expected);
  - the null's random stream is numpy's PCG64;
  - the next-open sensitivity fills a missing open at the close;
  - the strict tier's N also includes Round 8's own ratchet;
  - the candidate's tie-break by dev alpha Sharpe;
  - the account-scaled hold and the null's index count a market from its second row, not from when its σ exists. That
    is about 90 trading days early for QQQ and EWY, reading the plan's "available" as "has a return that day", as in
    Round 6.
- **Calibration:**
  - 400 bull panels and 200 panels for each planted κ: 6,800 panels in all (the plan said about 7,200);
  - the generator's free parameters are those in the table above. The plan fixed only the scenario list, λ = 3, the
    drift sizes, the t(4) shocks and the bridge;
  - the 80% detection threshold for the planted κ = 0.10 edge was set after the development pilot.
- **Power** is computed from exact sufficient statistics of Gaussian days. That is equivalent to simulating the paths,
  and was checked against brute-force simulation by a unit test and by a reviewer in 12 cells. The alpha variant's
  β·sd(b) = 0.5 sd(a) and the hold's 0.45 Sharpe were chosen before any data.
- **Data:**
  - red flags 2 and 3 are shares of the dev + holdout rows (counts); flag 4 uses dev rows only;
  - the Round 8 manifest also pins Round 6's raw and parquet SHA256s, and every load re-checks both.
- **From the code reviews** (every finding was fixed or is listed here):
  - a verify load is refused before the unlock and counts as a touch;
  - the unlock refuses a second call and any reason carrying a round tag;
  - the preflight compares every pin and yardstick of `IBS.json` with the committed `r8-sweep` record;
  - an exit that leaves a sleeve's cash ≤ 0 is booked as a liquidation at 0;
  - the data report excludes rows after 2026-09-30 from its shares.
  - Independent checks by the reviewers:
    - lag-1 parity with `engine.run` on 400 fuzzed seeds (18,854 trades);
    - the same-close P&L identity on 300 seeds;
    - the power function against brute force;
    - `cmd_sweep` and `cmd_validate` end to end on synthetic data.
- **Known limits, accepted:** the validate step is single-phase, so its start and unlock records stay uncommitted until
  they are committed after the run.
