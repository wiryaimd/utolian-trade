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
