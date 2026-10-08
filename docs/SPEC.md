# Postmark — Engine Specification

Status: **Agreed 1.4** (2026-10-07)

Postmark is a UCI chess engine written in Rust. Its single yardstick is playing
strength: a change that affects play is accepted only if it is shown to gain
Elo (or, for refactors, not to lose it).

Requirement IDs (`BRD-1`, `SRC-3`, …) are stable and may be cited in commits and
tests. "Must" is binding; "should" is the default unless a measurement says
otherwise.

---

## 1. Principles

| ID | Requirement |
|----|-------------|
| PRN-1 | Playing strength is the primary goal. Where clarity and speed conflict in a hot path, speed wins; everywhere else, write idiomatic Rust. |
| PRN-2 | All code is written fresh for Postmark; none is copied or ported from Huginn or any other engine. Huginn may be used as a test opponent, and its experiment record and design notes may be consulted to decide what to try and in what order. Whatever is tried must still pass Postmark's own SPRT gate. |
| PRN-9 | Postmark is licensed under the GNU GPL, version 3 or later (changed from the Unlicense in 1.4). The copyleft licence is chosen so that networks, training data and tooling from GPL-licensed engines, in particular Huginn and FableR, and GPL-obligated public datasets can be used. PRN-2 is unchanged: licence compatibility permits copying code, the fresh-code rule still forbids it. Design notes, measurements and trained networks may be used freely. |
| PRN-3 | The engine crate has **zero runtime dependencies**. Exceptions must be listed in this document (currently: Syzygy probing, §9.2). Dev-, build- and tooling-only dependencies are permitted. |
| PRN-4 | `unsafe` is permitted in hot paths (unchecked indexing, SIMD intrinsics, transposition-table access) only when it is benchmarked as a win and carries a `// SAFETY:` comment stating the invariant relied upon. |
| PRN-5 | Debug builds must check, via `debug_assert!`, every invariant that `unsafe` code relies on. |
| PRN-6 | The engine must be deterministic: single-threaded search to a fixed depth from a fixed position always visits the same number of nodes. |
| PRN-7 | Every function, type, constant and module carries a `///` (or `//!`) doc comment, private items included. A function's comment states what it does, its parameters and return value, and any preconditions the caller must uphold; `unsafe` functions add a `# Safety` section and functions that can panic a `# Panics` section. |
| PRN-8 | Non-obvious techniques (magic/PEXT indexing, pruning and reduction conditions, tapered-eval interpolation, TT packing) are explained in the doc comment at the point of use, including why the technique is sound, not just what the code does. |

## 2. Toolchain and build targets

| ID | Requirement |
|----|-------------|
| BLD-1 | Stable Rust, edition 2024. No nightly features. |
| BLD-2 | Two release builds, selected at compile time: **`cargo v3`** (x86-64-v3: AVX2, POPCNT, BMI2) is the primary build; **`cargo generic`** (baseline x86-64) is the portable fallback. The tier is set by the compiler's target CPU, and code selects its implementation with `cfg(target_feature = ...)`, so a binary can never contain instructions its tier does not enable. |
| BLD-3 | Slider attacks use PEXT in the `v3` build and magic bitboards in the `generic` build, behind one common interface. |
| BLD-4 | Release profile: `opt-level = 3`, fat LTO, `codegen-units = 1`, `panic = "abort"`. |
| BLD-6 | The repository is a cargo workspace. The engine crate (a library plus the UCI binary) is the only crate bound by PRN-3. Tooling (training-data generation driver, test scripts) lives in a separate tools crate, created when the first tool is needed, and may use dependencies. |
| BLD-5 | Windows is the development platform; the code must also build on Linux. ARM64 is out of scope. |

## 3. Board representation

| ID | Requirement |
|----|-------------|
| BRD-1 | Position state is held as bitboards (per piece type and per colour) plus a 64-entry mailbox for piece-on-square lookup. |
| BRD-2 | Little-endian rank-file square mapping (a1 = 0, h8 = 63). |
| BRD-3 | Moves are encoded in 16 bits (from, to, flags). |
| BRD-4 | A Zobrist key is maintained incrementally for the full position; separate pawn and material keys are maintained once an evaluation or history table needs them. |
| BRD-5 | Make/unmake must restore the position bit-for-bit, including the Zobrist key. Irreversible state is kept on a stack. |
| BRD-6 | FEN parsing and output round-trip exactly. Malformed FEN is rejected with an error and never panics. |
| BRD-7 | Standard chess only. Chess960 is out of scope; castling is stored as conventional rights. |

## 4. Move generation

| ID | Requirement |
|----|-------------|
| MGN-1 | Move generation is **fully legal**, using checker and pin masks; no make-then-test legality check. |
| MGN-2 | Staged generation is supported: captures/promotions and quiets can be generated separately. |
| MGN-3 | Moves are written to a fixed-capacity stack-allocated list; no heap allocation during search. |
| MGN-4 | Perft must match published node counts for the standard suite (start position, Kiwipete, and the usual en-passant, promotion and castling stress positions), including start position depth 6 = 119,060,324 and Kiwipete depth 5 = 193,690,690. |
| MGN-5 | A `perft` command (with `divide` output) is available from the engine binary. |

## 5. Search

### 5.1 v0.1 baseline

| ID | Requirement |
|----|-------------|
| SRC-1 | Iterative deepening, fail-soft negamax alpha-beta with a principal variation. |
| SRC-2 | Quiescence search over captures and promotions, with check evasions when in check. |
| SRC-3 | Move ordering: PV/hash move first, then captures by MVV-LVA. |
| SRC-4 | Draw detection: threefold repetition (twofold inside the search tree), fifty-move rule, insufficient material. |
| SRC-5 | Mate scores are distance-to-mate adjusted and reported as `score mate N`. |
| SRC-6 | The search polls for `stop` and time expiry and returns the best move from the last fully completed iteration (or the partial iteration if it has already improved on it). |

### 5.2 Structure (binding from v0.1)

| ID | Requirement |
|----|-------------|
| SRC-7 | All per-search mutable state (position, search stack, history tables, node counters, PV) lives in a per-thread struct. Nothing is global except the transposition table and the stop flag. |
| SRC-8 | The transposition table is shared and lockless: entries are packed, read and written without locks, and any move read from it is validated against the current position before use. |
| SRC-9 | Search parameters (margins, reductions) are named constants in one place, structured so they can later be exposed for SPSA tuning. |

### 5.3 Later milestones

Each item is added individually and must pass the SPRT gate (§8) on its own:
transposition table cutoffs, aspiration windows, principal variation search,
null-move pruning, late move reductions, killer/history/continuation-history
ordering, static exchange evaluation, reverse futility and futility pruning,
late move pruning, check and singular extensions, internal iterative reductions.

## 6. Evaluation

| ID | Requirement |
|----|-------------|
| EVL-1 | Evaluation sits behind a single interface so the hand-crafted evaluator can be replaced by NNUE without changing the search. The interface supports incremental update on make/unmake. |
| EVL-2 | v0.1: tapered (middlegame/endgame) material plus piece-square tables, updated incrementally. |
| EVL-3 | No hand-crafted terms are added beyond EVL-2. The hand-crafted evaluator exists to validate the search and to bootstrap NNUE training data. |
| EVL-4 | NNUE replaces the hand-crafted evaluator once the modern search features are in (M4): incrementally updated accumulator, AVX2 inference in the `v3` build with a scalar fallback, network embedded in the binary. The training data must be generated by Postmark's own search. The evaluator that labels the first round may be an external network loaded through EVL-6 (a bootstrap teacher, so that the first net does not have to climb out of the small-corpus regime from a material-only teacher); later rounds use Postmark's own networks as teacher. Every corpus records the engine build, the labelling evaluator and the search depth that produced it. |
| EVL-6 | The network file format is the layout shared by Huginn and FableR: 768 input features indexed `(colour * 6 + piece) * 64 + square`, `square ^ 56` flip for the black perspective, two perspective accumulators of a hidden size read from the file, clipped ReLU, 16-bit weights, output in centipawns / 16. Their networks therefore load into Postmark and Postmark's into them, which allows evaluation to be held constant while searches are compared. Postmark's writer adds a magic header and a version field; the reader also accepts the headerless form those engines write. |
| EVL-5 | Scores are in centipawns from the side to move's perspective. |

## 7. UCI and time management

| ID | Requirement |
|----|-------------|
| UCI-1 | Supported commands: `uci`, `isready`, `ucinewgame`, `position` (`startpos` / `fen`, with `moves`), `go`, `stop`, `quit`, `setoption`. |
| UCI-2 | `go` supports `wtime`/`btime`/`winc`/`binc`/`movestogo`, `movetime`, `depth`, `nodes`, and `infinite`. |
| UCI-3 | Options in v0.1: `Hash` (MB), `Threads` (accepted, fixed at 1 until Lazy SMP ships). |
| UCI-4 | `info` lines report `depth`, `seldepth`, `score`, `nodes`, `nps`, `time`, `hashfull` and `pv` at least once per completed iteration. |
| UCI-5 | Input is read on a dedicated thread so `stop` and `isready` are answered while searching. |
| UCI-6 | Unknown or malformed input is ignored; the engine never panics on input. |
| UCI-7 | Non-UCI commands: `bench` (§8), `perft`, and `d` (print the board). |
| TIM-1 | Time management uses a soft limit (don't start a new iteration) and a hard limit (abort the search), with a configurable move-overhead margin. |
| TIM-2 | The engine must not lose on time at the standard test time control (§8) over a 10,000-game run. |

## 8. Testing and acceptance

| ID | Requirement |
|----|-------------|
| TST-1 | CI runs `cargo test`, `cargo clippy` with warnings denied, `cargo fmt --check`, and the perft suite on every push. |
| TST-8 | Documentation (PRN-7) is enforced in CI: the `missing_docs`, `clippy::missing_docs_in_private_items`, `clippy::missing_safety_doc` and `clippy::missing_panics_doc` lints are denied, and `cargo doc` must build without warnings. |
| TST-2 | `bench` searches a fixed set of positions to a fixed depth and prints total nodes and NPS. The node count is recorded in every commit message that touches the engine (`Bench: 1234567`). |
| TST-3 | A commit declared non-functional must leave the bench node count unchanged. |
| TST-4 | **Merge gate.** Every change to search, evaluation or time management must pass a fastchess SPRT against the current `main`: time control 8+0.08, 1 thread, 16 MB hash, α = β = 0.05, the `UHO_2024_8mvs_+090_+099` unbalanced opening book, games played in pairs with colours reversed. |
| TST-5 | Elo bounds: `[0, 5]` for strength changes during early development, tightening to `[0, 3]` once typical gains fall below 5 Elo. Refactors and simplifications use non-regression bounds `[-5, 0]`. |
| TST-6 | Speed-only changes are verified by bench NPS on the same machine and an unchanged node count; they need no SPRT. |
| TST-7 | Each tagged release is played in a gauntlet that includes Huginn, and the result is recorded in the release notes. |

## 9. Optional features

### 9.1 Lazy SMP
`Threads` becomes effective: helper threads run the same search with their own
state (SRC-7) over the shared table (SRC-8). Accepted by SPRT at 4 threads
against the single-threaded build at equal time.

### 9.2 Syzygy tablebases
`SyzygyPath` and `SyzygyProbeLimit` options; WDL probing in search and DTZ
probing at the root. This is the one permitted exception to PRN-3 — the choice
between binding Fathom and using a Rust implementation is deferred to that
milestone.

### 9.3 Pondering and MultiPV
`Ponder` option with `go ponder` / `ponderhit`; `MultiPV` option reporting the
N best root lines.

### 9.4 Opening book
Internal Polyglot (`.bin`) book support via `OwnBook` and `BookFile` options,
off by default. The book is never used during SPRT testing.

## 10. Roadmap

| Milestone | Content | Exit criterion |
|-----------|---------|----------------|
| M0 | Cargo project, build profiles and features, CI | CI green on an empty engine |
| M1 | Board, FEN, Zobrist, make/unmake, legal move generation | Perft suite passes (MGN-4) |
| **M2 — v0.1** | UCI, baseline search (§5.1), material + PST eval, time management, `bench` | Completes a 1,000-game fastchess match with no crashes, illegal moves or time losses |
| M3 | Modern search features (§5.3), one SPRT each | Each feature passed or rejected on record |
| M4 | NNUE (EVL-4, EVL-6), including data generation and training pipeline | SPRT pass against the material + PST evaluator |
| M5 | Lazy SMP (§9.1) | SPRT pass at 4 threads |
| M6 | Pondering and MultiPV (§9.3) | Functional tests pass; no single-PV regression |
| M7 | Syzygy (§9.2) | Correct WDL/DTZ on a test suite; non-regression SPRT |
| M8 | Polyglot book (§9.4) | Functional tests pass |

## 11. Out of scope

Chess960 and other variants, ARM64 builds, a built-in GUI, protocols other than
UCI (no XBoard/CECP), and cloud or distributed search.

## 12. Open questions

None. Changes to this document from here on are made by amending the relevant
requirement and bumping the version in the status line.
