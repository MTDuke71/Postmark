//! Perft: counting the leaf positions of the legal move tree.
//!
//! The counts for well-known positions are published, so comparing against
//! them verifies move generation and make/unmake together (MGN-4).

use crate::movegen::{GenKind, generate};
use crate::moves::{Move, MoveList};
use crate::position::Position;

/// Returns the number of positions reachable from `position` in exactly
/// `depth` legal moves. `position` is unchanged on return.
///
/// At depth 1 the answer is the number of legal moves, so the moves are
/// counted without being played ("bulk counting"). That is valid only
/// because move generation is fully legal.
pub fn perft(position: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate(position, GenKind::All, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for &mv in list.iter() {
        position.make_move(mv);
        nodes += perft(position, depth - 1);
        position.unmake_move(mv);
    }
    nodes
}

/// Returns each legal move in `position` with the perft count of the subtree
/// below it at `depth`; the counts sum to `perft(position, depth)`.
///
/// Comparing this breakdown with another engine's is the standard way to
/// locate a move generation bug. For `depth` 0 the list is empty.
pub fn divide(position: &mut Position, depth: u32) -> Vec<(Move, u64)> {
    let mut list = MoveList::new();
    if depth > 0 {
        generate(position, GenKind::All, &mut list);
    }
    list.iter()
        .map(|&mv| {
            position.make_move(mv);
            let nodes = perft(position, depth - 1);
            position.unmake_move(mv);
            (mv, nodes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::START_FEN;

    /// The standard perft suite: each position with its published node
    /// counts for depths 1, 2, 3, …
    const SUITE: [(&str, &[u64]); 6] = [
        (
            START_FEN,
            &[20, 400, 8_902, 197_281, 4_865_609, 119_060_324],
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2_039, 97_862, 4_085_603, 193_690_690],
        ),
        (
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[14, 191, 2_812, 43_238, 674_624, 11_030_083],
        ),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9_467, 422_333, 15_833_292],
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[44, 1_486, 62_379, 2_103_487, 89_941_194],
        ),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2_079, 89_890, 3_894_594, 164_075_551],
        ),
    ];

    /// Checks every suite position at each depth whose count is at most
    /// `node_limit`.
    fn check_suite(node_limit: u64) {
        for (fen, counts) in SUITE {
            let mut position = Position::from_fen(fen).unwrap();
            for (index, &expected) in counts.iter().enumerate() {
                if expected > node_limit {
                    break;
                }
                let depth = index as u32 + 1;
                assert_eq!(perft(&mut position, depth), expected, "{fen} depth {depth}");
            }
        }
    }

    #[test]
    fn perft_suite_shallow() {
        check_suite(700_000);
    }

    #[test]
    #[ignore = "hundreds of millions of nodes; run in release (CI does)"]
    fn perft_suite_full() {
        check_suite(u64::MAX);
    }

    #[test]
    fn divide_sums_to_perft() {
        let mut position = Position::startpos();
        let parts = divide(&mut position, 3);
        assert_eq!(parts.len(), 20);
        assert_eq!(parts.iter().map(|&(_, nodes)| nodes).sum::<u64>(), 8_902);
        assert!(divide(&mut position, 0).is_empty());
    }

    /// Walks the move tree checking, at every node, that the incremental key
    /// matches a from-scratch computation, that the staged generators agree
    /// with the full one, and that unmaking a move restores the position
    /// exactly.
    fn verify_tree(position: &mut Position, depth: u32) {
        assert_eq!(
            position.key(),
            position.compute_key(),
            "{}",
            position.to_fen()
        );

        let mut all = MoveList::new();
        generate(position, GenKind::All, &mut all);
        let mut staged = MoveList::new();
        generate(position, GenKind::Noisy, &mut staged);
        let noisy_count = staged.len();
        generate(position, GenKind::Quiet, &mut staged);
        for (index, mv) in staged.iter().enumerate() {
            let is_noisy = mv.is_capture() || mv.promotion().is_some();
            assert_eq!(is_noisy, index < noisy_count, "{} {mv}", position.to_fen());
        }
        let sorted = |list: &MoveList| {
            let mut moves: Vec<String> = list.iter().map(Move::to_string).collect();
            moves.sort();
            moves
        };
        assert_eq!(sorted(&all), sorted(&staged), "{}", position.to_fen());

        if depth == 0 {
            return;
        }
        let before = position.clone();
        for &mv in all.iter() {
            position.make_move(mv);
            verify_tree(position, depth - 1);
            position.unmake_move(mv);
            assert_eq!(*position, before, "after {mv}");
        }
    }

    #[test]
    fn make_unmake_keys_and_staging_are_consistent() {
        for (fen, _) in SUITE {
            let mut position = Position::from_fen(fen).unwrap();
            verify_tree(&mut position, 2);
        }
    }
}
