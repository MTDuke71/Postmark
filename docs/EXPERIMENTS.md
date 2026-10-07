# Experiment record

Every change to search, evaluation or time management is gated by an SPRT
(spec TST-4): 8+0.08, 1 thread, 16 MB hash, UHO_2024_8mvs_+090_+099,
alpha = beta = 0.05, bounds [0, 5] unless stated. Each row is one run.
A rejected change is kept on a `rejected/<feature>` branch so it can be
retried after other improvements land; several of Huginn's features lost
until an unrelated fix made them gain (see LESSONS_LEARNED.md §2).

## M3: search features

| # | Feature | Result | Elo | Games | W / L / D | LLR | Bench | Where |
|---|---------|--------|-----|-------|-----------|-----|-------|-------|
| 1 | Transposition table cutoffs | pass | +53.5 ± 17.8 | 884 | 386 / 251 / 247 | 2.96 | 3954657 | 56a2526 |
| 2 | Principal variation search | pass | +38.5 ± 15.2 | 1240 | 505 / 368 / 367 | 2.96 | 2945642 | f1674fc |
| 3 | Null-move pruning | pass | +116.9 ± 26.2 | 558 | 296 / 115 / 147 | 2.95 | 2048445 | f9cffd4 |
| 4 | Killer-move ordering | pass | +141.8 ± 27.3 | 460 | 248 / 70 / 142 | 2.95 | 1856386 | cc3baaa |
| 5 | History heuristic | pass | +41.9 ± 15.9 | 1108 | 435 / 302 / 371 | 2.95 | 1736272 | 0350156 |
| 6 | Late move reductions | pass | +101.9 ± 23.5 | 484 | 215 / 77 / 192 | 2.95 | 388300 | ff35fb0 |
| 7 | Reverse futility pruning | pass | +80.4 ± 20.4 | 510 | 186 / 70 / 254 | 2.95 | 265027 | c669351 |
| 8 | Futility pruning | pass | +9.9 ± 6.9 | 4504 | 1194 / 1066 / 2244 | 2.94 | 248136 | c88d0bf |
| 9 | SEE pruning in quiescence | pass | +22.0 ± 11.3 | 1692 | 496 / 389 / 807 | 2.97 | 100340 | 94a8b24 |
| 10 | SEE capture ordering (losing captures last) | pass | +38.6 ± 15.2 | 914 | 284 / 183 / 447 | 2.95 | 97142 | 8ccaf41 |
| 11 | Check extension (unrestricted, +1 ply on every checking move) | **fail** | −33.0 ± 15.1 | 1110 | 247 / 352 / 511 | −2.97 | 113718 | rejected/check-extension (ac7f26b) |
| 12 | Mate-distance pruning | stopped (undecided) | +2.7 ± 4.0 | 13500 | 3635 / 3531 / 6334 | 0.22 | 97142 | rejected/mate-distance-pruning (343acc8) |
| 13 | Continuation history (1- and 2-ply, i32, bounded) | **fail** (stopped at LLR −2.46) | −6.5 ± 8.4 | 3200 | 808 / 868 / 1524 | −2.46 | 97416 | rejected/continuation-history (72388e0) |
| 14 | Aspiration windows (depth >= 6, ±50, failing side ×2) | pass | +17.6 ± 9.9 | 2256 | 652 / 538 / 1066 | 2.96 | 96965 | 41fc9a6 |
| 15 | Internal iterative reductions (depth >= 4, no hash move) | pass | +13.2 ± 8.3 | 3244 | 950 / 827 / 1467 | 2.96 | 91901 | df08cf4 |
| 16 | Singular extensions (depth >= 8, TT depth >= depth−3, margin 2·depth) | **fail** | −11.3 ± 9.5 | 2190 | 511 / 582 / 1097 | −2.96 | 91901 | rejected/singular-extensions (4da447e) |
| 17 | Late move pruning (depth <= 6, 3 + depth² quiets searched) | pass | +57.9 ± 18.2 | 642 | 226 / 120 / 296 | 2.97 | 57419 | 71d5cff |
| 18 | Continuation history retry (1-ply, i16) | **fail** (stopped at LLR −1.56) | −2.1 ± 7.5 | 4000 | 1030 / 1054 / 1916 | −1.56 | 57041 | rejected/continuation-history-1ply (13d3fe9) |

Elo is relative to the previous row's build, not to the M2 baseline, so the
column does not sum. Bench is the fixed-depth node count (TST-2).

### Rejected: notes for retries

- **Check extension (11).** Bench rose 17%; at 8+0.08 the extra plies cost
  more than they found. Candidates for a retry: extend only checks with
  SEE >= 0, cap total extension along a path, or revisit after singular
  extensions, when the extension budget is shared.
- **Mate-distance pruning (12).** Stopped after 13,500 games and 11.5 hours
  with LLR 0.22: a gain of about +2.7 Elo (LOS 91%) that [0, 5] cannot
  decide. Exact and harmless; retry bundled with another change, or as a
  [-5, 0] non-regression test.
- **Continuation history (13).** Stopped at LLR -2.46 (84% of the way to
  rejection) after 3,200 games with a steady -6 to -7 Elo. At depth 15 it
  searched 3.7% fewer nodes at 9% lower NPS, so the ordering helps but the
  table reads cost more than they save. Retry with i16 entries, with only
  the one-ply entry (or a down-weighted two-ply one), and after the later
  pruning features, when ordering matters more.
- **Singular extensions (16).** -11 Elo with Huginn's recipe. At 8+0.08
  most iterations end at depth 8-12, so the exclusion search runs only
  near the root, where it costs the most and extends the least. Retry at
  a longer time control or with a higher minimum depth; the TT is already
  depth-preferred, so Huginn's starvation problem does not apply.
- **Continuation history retry (18).** The cheap form (one-ply, i16)
  removed the slowdown but gained nothing: -2 +/- 7.5 after 4,000 games,
  stopped. Two attempts now; park until a longer time control or an
  evaluation change makes quiet-move ordering worth more.
