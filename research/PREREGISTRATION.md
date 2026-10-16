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
