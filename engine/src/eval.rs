//! Static evaluation (EVL-1, EVL-2, EVL-5).
//!
//! The search talks to evaluation only through the [`Evaluator`] trait, so
//! the hand-crafted evaluator here can later be replaced by a neural network
//! without touching the search. The trait is shaped around what such a
//! network needs: it is told about every move made and unmade, so it can
//! keep its own state up to date incrementally instead of recomputing it at
//! each node.

use std::sync::Arc;

use crate::moves::Move;
use crate::nnue::{Network, NnueEvaluator};
use crate::position::Position;
use crate::types::{Color, Piece, PieceKind, Square};

/// A static evaluator that follows the search down and up the move tree.
pub trait Evaluator {
    /// Discards all state and starts again from `position`. Must be called
    /// before the first [`Evaluator::evaluate`], and whenever the position
    /// changes other than through the two methods below.
    fn refresh(&mut self, position: &Position);

    /// Records that `mv` is about to be played. `position` is the position
    /// *before* the move, which is what identifies the pieces involved.
    fn make_move(&mut self, position: &Position, mv: Move);

    /// Records that the most recently made move has been taken back.
    fn unmake_move(&mut self);

    /// Returns the score of `position` in centipawns from the point of view
    /// of the side to move (EVL-5). `position` must be the one this
    /// evaluator has been following.
    fn evaluate(&self, position: &Position) -> i32;
}

/// The evaluator a search uses: the network when one is loaded, otherwise
/// the hand-crafted tables.
///
/// An enum rather than a trait object so that the choice costs a predictable
/// branch at each call instead of an indirect one, and so that the search
/// owns its evaluator by value (SRC-7).
#[derive(Clone, Debug)]
pub enum Eval {
    /// Material and piece-square tables (EVL-2).
    Pst(PstEvaluator),
    /// The neural network (EVL-4).
    Nnue(NnueEvaluator),
}

impl Eval {
    /// Creates an evaluator following `position`: the network evaluator if
    /// `network` is given, otherwise the hand-crafted one.
    pub fn new(network: Option<&Arc<Network>>, position: &Position) -> Eval {
        match network {
            Some(network) => Eval::Nnue(NnueEvaluator::new(Arc::clone(network), position)),
            None => Eval::Pst(PstEvaluator::new(position)),
        }
    }
}

impl Evaluator for Eval {
    #[inline]
    fn refresh(&mut self, position: &Position) {
        match self {
            Eval::Pst(evaluator) => evaluator.refresh(position),
            Eval::Nnue(evaluator) => evaluator.refresh(position),
        }
    }

    #[inline]
    fn make_move(&mut self, position: &Position, mv: Move) {
        match self {
            Eval::Pst(evaluator) => evaluator.make_move(position, mv),
            Eval::Nnue(evaluator) => evaluator.make_move(position, mv),
        }
    }

    #[inline]
    fn unmake_move(&mut self) {
        match self {
            Eval::Pst(evaluator) => evaluator.unmake_move(),
            Eval::Nnue(evaluator) => evaluator.unmake_move(),
        }
    }

    #[inline]
    fn evaluate(&self, position: &Position) -> i32 {
        match self {
            Eval::Pst(evaluator) => evaluator.evaluate(position),
            Eval::Nnue(evaluator) => evaluator.evaluate(position),
        }
    }
}

/// A pair of scores for the same feature: its worth in the middlegame and
/// in the endgame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tapered {
    /// Middlegame value in centipawns.
    mg: i32,
    /// Endgame value in centipawns.
    eg: i32,
}

/// Material values by piece kind (pawn … king).
const MATERIAL: [Tapered; 6] = [
    Tapered { mg: 100, eg: 125 },
    Tapered { mg: 320, eg: 310 },
    Tapered { mg: 335, eg: 325 },
    Tapered { mg: 500, eg: 550 },
    Tapered { mg: 975, eg: 1000 },
    Tapered { mg: 0, eg: 0 },
];

/// How much each piece kind contributes to the game phase. The start
/// position totals [`PHASE_MAX`]; pawns and kings do not count.
const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];

/// The phase of a position with all minor and major pieces on the board.
const PHASE_MAX: i32 = 24;

/// Middlegame bonus for a pawn by rank: advancing gains space.
const PAWN_ADVANCE_MG: [i32; 8] = [0, 0, 2, 6, 12, 25, 50, 0];

/// Endgame bonus for a pawn by rank: nearer to promotion is worth more.
const PAWN_ADVANCE_EG: [i32; 8] = [0, 0, 5, 12, 25, 50, 90, 0];

/// Middlegame king score by rank: stay on the back rank.
const KING_RANK_MG: [i32; 8] = [20, 0, -30, -50, -60, -60, -60, -60];

/// Middlegame king score by file: prefer the castled squares to the centre.
const KING_FILE_MG: [i32; 8] = [15, 25, 5, -10, -10, 5, 25, 15];

/// Returns the positional score of a White piece of kind `kind` on the
/// square with the given `file` and `rank` (both 0-7), without material.
///
/// The tables are produced from a few simple rules rather than listed
/// square by square: pieces are better towards the centre, pawns are better
/// advanced, rooks belong on the seventh rank, and the king hides in the
/// middlegame and centralises in the endgame. They are deliberately crude;
/// their job is to make the search testable and to produce the first
/// training data for the network that replaces them (EVL-3).
const fn positional(kind: PieceKind, file: usize, rank: usize) -> Tapered {
    // Distance from the nearer edge, 0-3, along each axis.
    let file_centre = if file < 4 { file } else { 7 - file } as i32;
    let rank_centre = if rank < 4 { rank } else { 7 - rank } as i32;
    // 0 on the rim … 3 on the four central squares.
    let centre = if file_centre < rank_centre {
        file_centre
    } else {
        rank_centre
    };
    let rank_i = rank as i32;

    match kind {
        PieceKind::Pawn => {
            // Central pawns on ranks 3-5 control the centre.
            let central = if rank >= 2 && rank <= 4 {
                file_centre * 4
            } else {
                0
            };
            Tapered {
                mg: PAWN_ADVANCE_MG[rank] + central,
                eg: PAWN_ADVANCE_EG[rank],
            }
        }
        PieceKind::Knight => Tapered {
            mg: centre * 12 - 18 + rank_i * 2,
            eg: centre * 8 - 12,
        },
        PieceKind::Bishop => Tapered {
            mg: centre * 6 - 6 - if rank == 0 { 8 } else { 0 },
            eg: centre * 5 - 7,
        },
        PieceKind::Rook => {
            let seventh = rank == 6;
            Tapered {
                mg: file_centre * 4 - 4 + if seventh { 15 } else { 0 },
                eg: if seventh { 10 } else { 0 },
            }
        }
        PieceKind::Queen => Tapered {
            mg: centre * 3 - 4,
            eg: centre * 8 - 10,
        },
        PieceKind::King => Tapered {
            mg: KING_RANK_MG[rank] + KING_FILE_MG[file],
            eg: centre * 14 - 20,
        },
    }
}

/// Builds the combined material and positional table: for every piece on
/// every square, its value to its own side.
///
/// Black's entries are White's mirrored top to bottom, so that both sides
/// are scored by the same rules.
const fn build_piece_square() -> [[Tapered; 64]; 12] {
    let mut table = [[Tapered { mg: 0, eg: 0 }; 64]; 12];
    let mut kind = 0;
    while kind < 6 {
        let mut square = 0;
        while square < 64 {
            let (file, rank) = (square % 8, square / 8);
            let bonus = positional(PieceKind::ALL[kind], file, rank);
            let value = Tapered {
                mg: MATERIAL[kind].mg + bonus.mg,
                eg: MATERIAL[kind].eg + bonus.eg,
            };
            table[kind][square] = value;
            // XOR with 56 flips the rank and keeps the file.
            table[kind + 6][square ^ 56] = value;
            square += 1;
        }
        kind += 1;
    }
    table
}

/// Value of each piece (by [`Piece::index`]) on each square, to its owner.
static PIECE_SQUARE: [[Tapered; 64]; 12] = build_piece_square();

/// The running totals from which the evaluation is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Totals {
    /// Middlegame score, White minus Black.
    mg: i32,
    /// Endgame score, White minus Black.
    eg: i32,
    /// Sum of [`PHASE_WEIGHT`] over all pieces on the board.
    phase: i32,
}

impl Totals {
    /// Totals of an empty board.
    const ZERO: Totals = Totals {
        mg: 0,
        eg: 0,
        phase: 0,
    };

    /// Adds (`sign` = 1) or removes (`sign` = -1) `piece` on `square`.
    #[inline]
    fn update(&mut self, piece: Piece, square: Square, sign: i32) {
        let value = PIECE_SQUARE[piece.index()][square.index()];
        // Scores are White minus Black, so Black's pieces count negatively.
        let side = match piece.color() {
            Color::White => sign,
            Color::Black => -sign,
        };
        self.mg += side * value.mg;
        self.eg += side * value.eg;
        self.phase += sign * PHASE_WEIGHT[piece.kind().index()];
    }

    /// Returns the totals of `position`, computed from scratch.
    fn of(position: &Position) -> Totals {
        let mut totals = Totals::ZERO;
        for square in position.occupied() {
            if let Some(piece) = position.piece_on(square) {
                totals.update(piece, square, 1);
            }
        }
        totals
    }
}

/// The hand-crafted evaluator: tapered material and piece-square tables,
/// updated incrementally (EVL-2).
///
/// It keeps one set of running totals per move on the current search path, so unmaking
/// a move is just dropping the newest entry.
#[derive(Clone, Debug)]
pub struct PstEvaluator {
    /// Totals for each position on the path; the last entry is current.
    stack: Vec<Totals>,
}

impl PstEvaluator {
    /// Creates an evaluator following `position`.
    pub fn new(position: &Position) -> PstEvaluator {
        let mut stack = Vec::with_capacity(256);
        stack.push(Totals::of(position));
        PstEvaluator { stack }
    }

    /// Returns the totals of the current position.
    #[inline]
    fn current(&self) -> Totals {
        // The stack is never empty: `new` and `refresh` push an entry and
        // `unmake_move` only removes entries pushed by `make_move`.
        self.stack.last().copied().unwrap_or(Totals::ZERO)
    }
}

impl Evaluator for PstEvaluator {
    fn refresh(&mut self, position: &Position) {
        self.stack.clear();
        self.stack.push(Totals::of(position));
    }

    fn make_move(&mut self, position: &Position, mv: Move) {
        let mut totals = self.current();
        let us = position.side_to_move();
        let (from, to) = (mv.from(), mv.to());

        if let Some(moved) = position.piece_on(from) {
            totals.update(moved, from, -1);
            let placed = match mv.promotion() {
                Some(kind) => Piece::new(us, kind),
                None => moved,
            };
            totals.update(placed, to, 1);
        }

        if mv.is_en_passant() {
            let victim = Position::en_passant_victim(to);
            totals.update(Piece::new(!us, PieceKind::Pawn), victim, -1);
        } else if mv.is_capture()
            && let Some(captured) = position.piece_on(to)
        {
            totals.update(captured, to, -1);
        }

        if mv.is_castle() {
            let (rook_from, rook_to) = Position::castling_rook_squares(mv, to);
            let rook = Piece::new(us, PieceKind::Rook);
            totals.update(rook, rook_from, -1);
            totals.update(rook, rook_to, 1);
        }

        self.stack.push(totals);
    }

    fn unmake_move(&mut self) {
        debug_assert!(self.stack.len() > 1);
        self.stack.pop();
    }

    /// Blends the middlegame and endgame scores by the game phase: with all
    /// pieces on the board the middlegame score is used, with none the
    /// endgame score, and in between a weighted average. This "tapering"
    /// avoids a sudden jump in the evaluation when a piece is exchanged.
    fn evaluate(&self, position: &Position) -> i32 {
        let totals = self.current();
        debug_assert_eq!(totals, Totals::of(position));
        // Promotions can push the phase past its starting value.
        let phase = totals.phase.min(PHASE_MAX);
        let white = (totals.mg * phase + totals.eg * (PHASE_MAX - phase)) / PHASE_MAX;
        match position.side_to_move() {
            Color::White => white,
            Color::Black => -white,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::{GenKind, generate};
    use crate::moves::MoveList;

    fn evaluate(fen: &str) -> i32 {
        let position = Position::from_fen(fen).unwrap();
        PstEvaluator::new(&position).evaluate(&position)
    }

    #[test]
    fn start_position_is_balanced() {
        assert_eq!(evaluate(crate::position::START_FEN), 0);
    }

    #[test]
    fn score_is_from_the_side_to_move() {
        let white = evaluate("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
        let black = evaluate("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
        assert!(white > 800);
        assert_eq!(white, -black);
    }

    #[test]
    fn mirrored_positions_score_the_same() {
        // The second position is the first with colours and ranks swapped.
        let a = evaluate("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let b = evaluate("r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1");
        assert_eq!(a, b);
    }

    /// Checks that incremental totals match a from-scratch count throughout
    /// the move tree (the `debug_assert` in `evaluate` does the comparison).
    fn walk(position: &mut Position, evaluator: &mut PstEvaluator, depth: u32) {
        evaluator.evaluate(position);
        if depth == 0 {
            return;
        }
        let mut list = MoveList::new();
        generate(position, GenKind::All, &mut list);
        for &mv in list.iter() {
            evaluator.make_move(position, mv);
            position.make_move(mv);
            walk(position, evaluator, depth - 1);
            position.unmake_move(mv);
            evaluator.unmake_move();
        }
    }

    #[test]
    fn incremental_update_matches_refresh() {
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ];
        for fen in fens {
            let mut position = Position::from_fen(fen).unwrap();
            let mut evaluator = PstEvaluator::new(&position);
            walk(&mut position, &mut evaluator, 3);
        }
    }
}
