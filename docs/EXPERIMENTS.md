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
