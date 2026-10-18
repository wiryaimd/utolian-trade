# F1 + D1 as one portfolio (2020-01-01 → 2026-09-30)

Information only, from the saved and verified daily results of both strategies (no new data was loaded):

- **F1:** 5-coin portfolio. Dev uses `reports/results/F1_dev_daily.parquet`. 2024-26 uses the mean of the 5 sleeves in
  `F1_holdout_sym_daily.parquet`; that rebuild matches F1.json exactly (+70.2%, Sharpe 0.76 to 2026-10-13).
- **D1:** `D1_dev_daily.parquet` and `D1_holdout_daily.parquet` (Deviation 2).
- **Costs:** fees, slippage and real funding are already inside both daily series.
- **Script:** `f1d1_portfolio.py`. Output: `splits.csv`, `order_days.csv`, `summary.json`.

**Caveats:**
- Both strategies' 2024-26 results had already been seen. F1's was a clean holdout; D1's was contaminated.
- Both research runs restart on 2024-01-01: open positions are closed on 2023-12-31 and re-entered, a small extra cost.
- Running both in one real Binance account would need the live executor to net F1's and D1's positions on the same
  coins. Today only F1 can trade live; D1 is paper only.

## Capital split, rebalanced to target at the start of each month

| F1 / D1 | 2020-2026 total | per year | Sharpe | worst drop | worst year | 2024-26 total (Sharpe, drop) |
|---|---|---|---|---|---|---|
| 100 / 0 | +656% | 34.9% | 1.04 | 28.1% (2022) | +0.5% (2025) | +71.3% (0.77, 26.7%) |
| 75 / 25 | +540% | 31.6% | 1.12 | 22.4% | +1.4% (2025) | +64.5% (0.84, 22.4%) |
| 60 / 40 | +470% | 29.4% | 1.18 | 19.9% | +1.6% (2025) | +59.5% (0.89, 19.9%) |
| 50 / 50 | +425% | 27.8% | 1.23 | 18.3% | +1.7% (2025) | +55.9% (0.93, 18.3%) |
| 40 / 60 | +381% | 26.2% | 1.27 | 16.6% | +1.6% (2025) | +52.0% (0.97, 16.6%) |
| **32 / 68 (equal risk, chosen from 2020-23 only)** | +347% | 24.8% | 1.29 | 15.5% | +0.3% (2022) | +48.8% (1.00, 15.3%) |
| 25 / 75 | +318% | 23.6% | 1.31 | 16.0% | -0.9% (2022) | +45.8% (1.02, 14.2%) |
| 0 / 100 | +221% | 18.9% | 1.19 | 19.0% | -5.2% (2022) | +34.6% (0.92, 11.9%) |

- **Equal-risk split:** the weight that gives F1 and D1 the same 2020-23 volatility (F1 36%, D1 17%). The 2024-26 data
  was not used to choose it, so its 2024-26 row is a fair test of the rule.
- **Two pots never rebalanced** (like the paper bot's two books) gives similar results. For example, 50/50 makes
  +439% with a 21.2% worst drop.
- **Correlation of F1 and D1:** 0.44 daily, 0.58 weekly.

## Days with orders (fills of either strategy, UTC days)

| | days with at least one order | orders a week (one per coin per day) | longest stretch with no order |
|---|---|---|---|
| F1 alone | 41% | 5.1 | 21 days |
| D1 alone | 65% | 9.2 | 15 days |
| **F1 + D1** | **77%** (70-82% each year) | 12.4 | 13 days |

## Practical rulebook re-read (information only; past verdicts unchanged)

| | F1 | D1 |
|---|---|---|
| Trades | 600 ✓ | 1,304 ✓ |
| 2× costs | ✓ | ✓ |
| Beats random timing | ✓ (p 0.003) | ✓ (p 0.009) |
| Best year ≤ 65% | 34% ✓ | 61% ✓ |
| Markets positive | 5/5 ✓ | 5/5 ✓ |
| PBO / plateau | (single config) | 0.10 / 1.0 ✓ |
| **DSR within the family** | **0.99 at N_eff = 1.71** ✓ (S1's 30 configs; 0.64 at the raw N = 30) | **0.996 at N_eff = 1.13** ✓ (0.97 at the raw N = 3) |
| Clean out-of-sample | ✓ (2024-26 holdout passed all 5 gates) | ✗ none yet (holdout contaminated); needs a forward paper period |

Under the rulebook as first worded (raw config count), F1's DSR was 0.64. After seeing that, the owner amended the
rulebook to use the family's effective number of independent trials. That is the project's usual N_eff method, and
S1's 30 near-identical configs count as 1.71 trials. The amendment is recorded in PREREGISTRATION.md as "Practical
rulebook, Amendment 1", with this disclosure.

Under the amended rulebook, F1 meets every practical rule, including clean out-of-sample confirmation. D1 meets the
dev rules but still needs a forward paper period. Both re-reads are information only: no recorded verdict changes,
and real money stays the owner's decision.
