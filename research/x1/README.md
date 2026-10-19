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

## Commands
```
python -m btcq.research.run x1-start                 # start record (done, committed)
python -m btcq.research.run x1-stage-a               # Stage A (done once; refuses to rerun)
python -m btcq.research.run x1-download --set B1     # NOT RUN: needs the owner's OK (5.7 GB, stream-and-discard)
python -m btcq.research.run x1-stage-b --set B1      # NOT RUN
python -m btcq.research.run x1-report
```
