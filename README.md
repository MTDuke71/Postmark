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

## Checks

These are the checks CI runs on every push:

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps --document-private-items
```
