# Execution study X1: maker-first ("post, then cross") execution

This study asks whether post-only limit orders would make the owner's trading cheaper than market (taker) orders.
The registration and its outcome are in `research/PREREGISTRATION.md`, under "## Execution study X1" and
"## Execution study X1 — Stage A outcome".

## Stage A (2026-10-19): F1 NO-GO
- **Result:** maker-first execution would add +0.34%/yr to F1 under the conservative (strict) model. That is below
  the +0.5%/yr bar, and only 2 of 4 years and 3 of 5 coins are positive.
  - **Why it is so small:** 96.5% of the orders fill and save 3 bp each, but the 3.5% where price runs away cost
    about 59 bp each.
- **Paper bot:** stays taker. D1 was NO-GO by arithmetic.
- **Downloads:** the B2 tick data is never downloaded. The B1 tick data (5.7 GB, for the faster-strategy cost
  model) needs the owner's OK.
- **Cost per side for generic orders**, conservative model, T = 15: 5.7 bp (random), 6.0 bp (breakout) and 6.8 bp
  (fading a move). The taker pays 6.6 bp. The optimistic model gives 2.5–3.4 bp; B1 would tell which is closer to
  reality.

## Stage B1 (2026-10-19, owner-approved download): cost model X1-cost-v1
- **Data:** 75 coin-days of bookTicker and aggTrades (2023-05-22 → 2023-12-30, 5.48 GB downloaded). The extracted
  parquet takes 8.9 GB under `data/ticks/` and is not committed.
- **Result:** the tick queue model saves 0.6–0.8 bp per side more than the strict 1m model on the same legs. Real
  fills therefore sit between the strict and central models.
- **Maker-first cost per side at T = 15** (the taker pays 6.6 bp):

  | Order class | Cost per side | Precision |
  |---|---|---|
  | random | 4.9 bp | resolved |
  | breakout | 5.3 bp | unresolved, CI about ±1.2 bp |
  | fading a move | 6.2 bp | resolved |

  Use the ALL-coin rows only; the per-coin breakout and fade cells are noise.
- **F1, information only:** F1's real gain is probably +0.3 to +1.0%/yr, close to the +0.5%/yr bar. The registered
  NO-GO stands; settling it would need a new registered version on F1's own tick legs (B2).

## Files
| File | What it holds |
|---|---|
| `runs.jsonl` | Start record (pins), Stage A run record and result record |
| `tick_sizes.csv` | Tick per coin and month, inferred from the 1m price grid (231 months, no fallback) |
| `stage_a_f1_legs.csv` | F1's 1,040 maker-eligible legs under the primary policy (strict, T = 15) |
| `stage_a_q3_costs.csv` | Generic cost curves: class × coin × T × model, with day-clustered 95% CIs |
| `stage_a_screen.csv` | Earlier fast families' gross edge against taker and maker-first round trips |
| `../../reports/results/X1_stage_a.json` | Every gate value, run summary, order count and pin |
| `../../reports/results/F1X_trial_returns.parquet` | Portfolio daily returns of the 5 F1X trials |
| `../../reports/results/X1_q3_legs.parquet` | Per-leg 1m savings of the 57,489 generic legs (Stage B input) |
| `tick_manifest_B1.csv` | The 150 B1 archives: key, bytes, SHA256, rows kept, out file |
| `x1_cost_v1.csv` | X1-cost-v1: tick vs 1m saving per class × coin × T, corrections and corrected c_eff |
| `../../reports/results/X1_stage_b.json`, `X1_stage_b1_legs.parquet` | Stage B1 summary and per-leg tick results |

## Commands
```
python -m btcq.research.run x1-start                 # start record (done, committed)
python -m btcq.research.run x1-stage-a               # Stage A (done once; refuses to rerun)
python -m btcq.research.run x1-download --set B1     # done once (owner-approved), manifest committed
python -m btcq.research.run x1-stage-b --set B1      # done once; refuses to rerun
# B2 is never downloaded: Stage A's strict model failed (a) and (d)
python -m btcq.research.run x1-report
```
