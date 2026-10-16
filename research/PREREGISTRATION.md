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
