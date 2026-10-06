# Lessons Learned from Huginn

Compiled 2026-10-05 from the experiment record of Huginn, the C++ engine that
preceded Postmark.

**How to read this.** Every entry records what happened *in Huginn*. An idea
that failed there may work in Postmark, and one that gained there may not;
the two engines differ in move generation, evaluation, test conditions and the
order features arrived. Use this document to decide what to try, in what
order, and where to be careful. Never use it as a reason to skip a test:
whatever is tried must still pass Postmark's own SPRT (spec PRN-2).

Hand-crafted evaluation and Texel tuning are left out, since Postmark goes
straight to NNUE.

**Sources**, all under `Huginn/docs/`: `BACKLOG.md` (BL), `BACKLOG-archive-2.0.md`
(A20), `-2.1.md` (A21), `-2.2.md` (A22), `BASELINE_LADDER.md` (LADDER),
`SEARCH_AND_EVAL.md` (S&E), `SPRT_QUEUE_TEST_PLAN.md` (QUEUE),
`SMP_MEASUREMENT_RUNBOOK.md` (RUNBOOK). Item numbers are Huginn backlog items.

**Huginn's test conditions**, unless stated: 10+0.1, one thread, 64 MB hash,
`noob_3moves.epd`, 4 games at once, SPRT bounds [0, 10], capped at 1,000
games, run on two machines (Ryzen 7800X3D and Intel 13700KF). Elo figures are
self-play against the previous build.

---

## 1. Search features that gained

Largest first. "Pooled" means the two machines' results combined.

| Feature | Result | Notes |
|---|---|---|
| Futility pruning per move, not per node (A21 #45) | +345 ± 61 and +355 ± 72 | The old version dropped the whole node, captures and checks included. The fix skips only quiet, non-promoting, non-checking moves. Margin `100 + 50·depth`, depth ≤ 3. |
| Iteration-start time gate and TT clear on new game (A21 #46/#47) | +127 ± 25 | See §5. |
| Mate-distance adjustment in the TT made to work (A20 #13) | +104 ± 62 (100 games) | The ply counter was never incremented, so the adjustment did nothing. |
| Repetition detection fix (A21 #44) | about +76 alone | Used the size of a grow-only buffer as the history length. |
| SEE split of good and bad captures in ordering (A21) | +50 ± 21, +29 ± 15 | Captures with SEE < 0 sort below every quiet move. Two weaker variants were flat or negative (§2). |
| Check-aware quiescence (A22 #52) | +40 ± 18, +45 ± 19 | In check: no stand-pat, all evasions, mate if none. |
| Contempt of 25 cp (A20 #16) | +40 ± 41 (200 games) | Never swept. Hurt results against a much stronger opponent. |
| SEE pruning in quiescence (A20) | about +24 | Skips captures with SEE < 0. Start-position depth-11 nodes fell tenfold. |
| Razoring (S&E) | +35 (100 games) | Depth 2-4, margin 400, reduces depth by one instead of returning. |
| Legal-move count driving PVS and LMR (A22 #57) | +30 ± 16 | A pseudo-legal index had been counting illegal moves as "late". More nodes at fixed depth, yet stronger. |
| TT bound classification fix (A20 #23) | about +24 pooled | Bounds were classified against the already-raised alpha. |
| Singular extensions (A22 #62) | +14.9 ± 10.6 pooled | Depth ≥ 8, TT depth ≥ depth − 3, exclusion search at `(depth−1)/2` with margin `2·depth`. |
| Aspiration windows, fifth attempt (A22 #17) | +14.5 ± 10.6 pooled | From depth 6, ±50, widen only the failing side ×2. Four earlier attempts lost (§2, §3). |
| LMR by table `ln(depth)·ln(move)/2` (A20) | about +14 (100 games) | Min depth 3, from the fourth move; captures, promotions, checks and in-check nodes exempt. |
| History-modulated LMR (A22 #63) | +13.6 ± 10.7 pooled | ±1 ply at history ±4096. Later found half-dormant (§2). |
| TT ageing (A22 #42) | flat at blitz, +16 ± 17 at 60+0.6 | 6-bit search date. |
| Mate-distance pruning (A21 #43) | +15 ± 18, +10 ± 17 | |
| Soft and hard time limits (BL #65) | +10.6 ± 10.6 (2,000 games) | See §5. |

Smaller gains: avoiding a repetition at the root when winning, en-passant key
normalisation (+8, and 23% fewer nodes), pin-aware SEE (+7), counter-move at a
low ordering score (+7).

Present in Huginn from the start and never measured in isolation: check
extension, killers, reverse futility (depth ≤ 6, margin `80·depth`), null move
(flat R = 4, depth ≥ 5).

## 2. Search features that were flat or lost

| Feature | Result | Stated reason |
|---|---|---|
| Late move pruning (A20 #7, BL) | −254, −56, −18, about 0 across four attempts | Never shipped. First version counted list position instead of quiets searched. Converged to zero as ordering improved. |
| Continuation history, 1-ply (A20 #3) | about 0, and −9 at higher weight | Added to an unbounded butterfly history about 100-1000× larger, so it had no effect; when forced, it overrode the counter-move. Suggested redo: bound both tables first. |
| Aspiration windows, attempts 1-4 | −75, −49, −24 to −42, −34 | Gained only after soundness bugs were fixed (§3). |
| Null-move verification search (A21 #43) | about −15 | Cost 53% more nodes. |
| Depth- and eval-scaled null-move R (A21 #43) | about +2 | `R = 2 + depth/4` against flat R = 4. Postmark's `3 + depth/4` was not tested there. |
| Exempting PV nodes from reverse futility (A21) | −13, −22 | Lost despite 20-43% fewer nodes. |
| Exempting PV nodes from futility (A21) | −13, −27 | |
| Weaker SEE ordering variants (A20 #6) | −16, +2 | Bad captures below history quiets only, or below killers only. |
| Counter-move at a high ordering score (A20 #13) | −114 | Placed above queen promotions. |
| Any single repetition scored as a draw (A20 #28) | −40, then about 0 | First version polluted the TT. |
| Four-entry TT buckets (A22 #42b) | −9, then about 0 | |
| Larger hash (A22 #31, BL #40) | flat | 64, 256 and 1,024 MB at 60+0.6. |
| Butterfly history reworks (BL #68-#70) | all about 0 | Gravity bound ±16384 was −1 ± 17. Entries below −4096 outnumbered those above +4096 about 40 to 1, so "reduce less" almost never fired. |
| PEXT slider attacks (A21 #32) | 3-5% slower | On Zen 4. |

## 3. Order and interactions

- **Pruning and window tricks failed before the soundness bugs were fixed and
  gained afterwards.** Aspiration windows went from −34 to +14.5 with
  essentially the same code, once repetition, Zobrist, quiescence and PVS bugs
  were gone. The same pattern showed for LMR's check exemption and LMP.
- **Gains do not add.** Two features worth about +100 each gave +31 together.
  About +122 of summed self-play gain became +33 to +54 against an outside
  engine, while bug-fix gains carried over at about 0.86.
- **Bundles hide losers.** A +62 bundle contained a feature worth about −15,
  found only by removing one piece at a time.
- **History-driven LMR depends on how history is updated.** Changing when
  penalties are applied collapsed move ordering.
- **Always-replace TT schemes** starved singular extensions of the mid-depth
  entries they need.

## 4. Testing method

- **Bounds of [0, 10] could not settle gains of 7-10 Elo**: one test was still
  undecided after 2,000 games.
- **Small samples misled repeatedly**: +80 after 40 games became +46; +33 after
  200 became +2.
- **Node count at fixed depth is not a strength measure.** 13% more nodes was
  stronger; 40% more was flat; 20-43% fewer lost. It is a check that two
  builds are identical, nothing more.
- **Check the baseline is the baseline.** A stale build setting once produced
  baseline-against-baseline on both machines. Before a match, the old build
  must reproduce its known bench and the new build must differ.
- **Some changes cannot show in a bench from fresh positions**, such as TT
  ageing, which acts only from the second search.
- **Run nothing else on a machine hosting a timed match.** One gate read −17,
  and +3 when rerun on an idle machine.
- **Test against the immediate predecessor**, not through a shared older
  baseline.
- **Bugs both builds share cancel out in every comparison.** Huginn's worst
  bugs were found by watching real games and by code audits, not by SPRT.
- **Tactical test suites and Elo disagree**: 16 more solutions coincided with
  −3 Elo.
- **Screening by depth reached must use the real time per move.** A 1,000 ms
  screen gave the wrong sign for NNUE when game moves averaged about 260 ms.
- **Holding standard input open**: piping `go` then `quit` into the engine cut
  searches short in Huginn's harness.

## 5. Time management

- **The largest time-related gain was using the clock at all.** Refusing to
  start an iteration after a quarter of the budget left about 75% of the
  clock unused; moving the gate to half was most of a +127 result.
- **Final scheme:** allotment `time/20 + inc/2` (or `time/movestogo + inc/2`).
  Soft limit equals the allotment and is checked between iterations; hard
  limit is `min(3·soft, time/2, …)` and aborts mid-search. Clock use rose
  from 83% to 97%.
- **The `time/2` cap matters**: without it, three times the soft limit can
  reach 60% of the clock when the increment is large relative to the time
  left.
- **A floor on the budget caused forfeits** with 1-10 ms left; the floor became
  1 ms.
- Not tried: extending the time when the best move keeps changing, and other
  divisors than 20.

## 6. Lazy SMP

- Built in gated steps: lockless TT, shared TT, helper threads, then the
  `Threads` option, each shown not to change single-thread play.
- **Result at 8 threads against 1:** +199 ± 26 and +173 ± 26.
- **Helpers must actually search differently.** A first version gave only two
  distinct schedules and reached depth 1.49× faster with 8 threads; proper
  iteration skipping and per-thread window offsets gave 3.61×. The broken
  version passed every functional test.
- Multi-threaded matches must run one game at a time.
- Helpers must not read input or print; a shared TT date must be advanced once
  per search, not once per thread.

## 7. NNUE

**Network.** 768 inputs to 256, for each of two perspectives, clipped ReLU,
one output. No king buckets. A king-bucketed variant in a sibling engine was
flat even at 42 million positions.

**Inference.** The accumulator was updated inside the position's add, remove
and move primitives, so there was no separate stack. Verified by comparing
incremental and from-scratch values at every node of a 4.9-million-node walk.
Speed against the hand-crafted evaluation: 0.36× scalar, 0.58× with AVX2
accumulator updates, parity with an AVX2 forward pass as well.

**Data generation.**
- Self-play at fixed depth 7, on 15 threads, about 750-780 positions per second.
- 8% random moves before ply 20, for variety.
- Positions kept only if not in check, the move played is not a capture or
  promotion, ply ≥ 10 and |score| < 1500.
- Label: the search score. Whether the game result was mixed in is not
  recorded.

**Training.** A custom Python trainer on the RX 9070 XT, about 23 seconds per
epoch, 32 epochs.

**Results.**

| Network | Data | Result |
|---|---|---|
| v1 | 14.2 million positions, 5 hours of generation | +162 ± 35 against the hand-crafted evaluation |
| v2 | 11.2 million new positions labelled by v1, mixed with the first set (25.4 million) | +70 ± 25 against v1 |

A sibling engine's curve with the same architecture: 1 million positions was
−460, 11 million was parity, 22.6 million was +177. Data volume was the main
lever.

**What went wrong.**
- The network was loaded from a file named after the executable; a renamed
  binary silently fell back to the old evaluation. The loader checked only the
  file size.
- The first network scored king and bishop against king as +408. The
  insufficient-material check must stay in front of the network.
- Existing match games were useless as training data: near-duplicate
  positions and labels from varying depths.
- A load message printed at startup broke the UCI protocol's silence.

## 8. Bugs that cost the most time

| Bug | How it was found |
|---|---|
| Futility dropping whole nodes | Comparing against another engine's source |
| Repetition length taken from a grow-only buffer | Printing the count per iteration; only appeared with a warm TT |
| Zobrist table one row too short | A small wobble in an otherwise fixed node count |
| History index folding two pieces together | Comparing against a sibling engine |
| Ply counter never incremented | Four-way bisection |
| TT bounds classified against the raised alpha | Outside code review |
| History table not zeroed between searches | King-and-queen endgames, where quiet ordering decides everything |
| Quiescence blind to check | Independent audit |
| A test target that ran zero tests and passed | Noticed late; it had never run |
| Tablebase integration | Estimated at an hour, took nine commits; scores had leaked into the TT |

## 9. What this suggests for Postmark

These are prompts for tests, not conclusions.

1. **PEXT against magic bitboards on the Ryzen.** Huginn measured PEXT as
   slower on this CPU family. Postmark's AVX2 build uses PEXT; a speed-only
   comparison (bench nodes per second, spec TST-6) would settle it.
2. **Games at once.** Huginn ran 4; Postmark runs 10 on 8 physical cores, and
   one Postmark SPRT had a cluster of time losses. Consider 8.
3. **Futility pruning** should skip individual quiet moves and never the node.
4. **Late move pruning** has the worst record in Huginn; test it late and
   count quiets searched.
5. **Continuation history** needs bounded tables. Postmark's history already
   has a bound, which was the stated missing piece.
6. **History penalties.** Postmark penalises the quiet moves tried before a
   cutoff. A similar change in Huginn enlarged the tree; the pending SPRT of
   Postmark's history heuristic is the test of that.
7. **Aspiration windows** after the main pruning features, not before.
8. **The NNUE network file** should be embedded in the binary (spec EVL-4),
   which removes the fallback failure altogether.
9. **Data volume first.** Plan the first network around 10-20 million
   positions; below that the sibling engine's network lost to hand-crafted
   evaluation.
10. **Watch games.** Several of Huginn's largest gains came from bugs that no
    self-play test could see.
