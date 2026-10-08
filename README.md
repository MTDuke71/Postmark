# Postmark
Postmark Chess Engine

A UCI chess engine written in Rust. The requirements and roadmap are in
[docs/SPEC.md](docs/SPEC.md).

## Building

Requires stable Rust (the toolchain is pinned by `rust-toolchain.toml`).

| Command | Output | CPU required |
|---------|--------|--------------|
| `cargo v3` | `target/v3/release/postmark` | AVX2, BMI2, POPCNT (x86-64-v3) |
| `cargo generic` | `target/generic/release/postmark` | any x86-64 |

## Running

Run with no arguments, `postmark` speaks UCI on standard input and output,
for use with any UCI chess GUI or match runner. Options: `Hash` (MB),
`Threads` (accepted; one thread for now), `Move Overhead` (ms) and
`EvalFile` (path of a network in the format shared with Huginn and FableR,
to evaluate with instead of the built-in evaluator; `<empty>` restores the
default).

Besides the standard UCI commands it understands:

| Command | Effect |
|---------|--------|
| `d` | Print the current position. |
| `eval` | Print the static evaluation of the current position. |
| `perft <depth>` | Count the positions reachable in `depth` moves, listed per move. |
| `bench [depth]` | Search a fixed set of positions and print the node count and speed. |

A command given on the command line is run once and the engine exits, e.g.
`postmark bench`.

## Checks

These are the checks CI runs on every push:

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps --document-private-items
```

CI also runs the full perft suite, which is skipped by default because it is
slow unoptimised:

```
cargo test --workspace --release -- --include-ignored
```
