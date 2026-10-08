//! The `bench` command: a fixed search whose node count identifies the
//! engine's behaviour (TST-2).
//!
//! Searching the same positions to the same depth always visits the same
//! number of nodes (PRN-6), so the total is a fingerprint of the search and
//! evaluation. Any change that alters play changes it; a change that claims
//! not to alter play must leave it untouched (TST-3). The reported speed is
//! used to measure speed-only changes (TST-6).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use crate::nnue::Network;
use crate::params::DEFAULT_HASH_MB;
use crate::position::Position;
use crate::search::Searcher;
use crate::timeman::Limits;
use crate::tt::TranspositionTable;

/// Depth each bench position is searched to unless another is requested.
pub const DEFAULT_DEPTH: i32 = 6;

/// The bench positions: a spread of openings, tactical middlegames and
/// endgames, including the six perft positions.
pub const POSITIONS: [&str; 12] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9",
    "2r3k1/5ppp/p3p3/1p1n4/3P4/1B3P2/PP3KPP/2R5 b - - 0 1",
    "8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - - 0 1",
    "8/8/4k3/8/2p5/8/1P2K3/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 b - - 0 1",
];

/// Searches every bench position to `depth` and returns the total node
/// count and the time taken, printing one line per position. The search
/// evaluates with `network` if one is given, otherwise with the
/// hand-crafted tables.
///
/// The transposition table is emptied before each position so that the
/// result does not depend on what was searched before.
///
/// # Panics
///
/// Panics if the table cannot be allocated, or if a bench position failed
/// to parse; the latter would be a bug in this crate, ruled out by its
/// tests.
pub fn run(depth: i32, network: Option<&Arc<Network>>) -> (u64, Duration) {
    let table = TranspositionTable::new(DEFAULT_HASH_MB)
        .expect("not enough memory for the bench transposition table");
    let stop = AtomicBool::new(false);
    let limits = Limits {
        depth: Some(depth),
        ..Limits::default()
    };

    let start = Instant::now();
    let mut total = 0;
    for (index, fen) in POSITIONS.iter().enumerate() {
        let position = Position::from_fen(fen).expect("bench positions are valid");
        table.clear();
        let mut searcher =
            Searcher::new(position, &table, &stop, &limits, Instant::now(), 0, network);
        let result = searcher.run(&mut |_| {});
        let best = result
            .best_move
            .map_or_else(|| "0000".to_string(), |mv| mv.to_string());
        println!(
            "{:>2}/{} nodes {:>9} best {:<5} {fen}",
            index + 1,
            POSITIONS.len(),
            result.nodes,
            best
        );
        total += result.nodes;
    }
    (total, start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_is_reproducible() {
        let (first, _) = run(2, None);
        let (second, _) = run(2, None);
        assert_eq!(first, second);
        assert!(first > 0);
    }
}
