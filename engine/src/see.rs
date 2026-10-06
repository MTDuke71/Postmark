//! Static exchange evaluation: the material outcome of a sequence of
//! captures on one square, worked out without searching.
//!
//! When a piece captures on a square, the opponent may recapture, the first
//! side may recapture again, and so on. Each side will keep capturing only
//! while it pays. Assuming both sides always bring their least valuable
//! attacker next (which keeps the most at stake for the opponent and the
//! least at risk for oneself), the sequence is determined, and its net
//! material result can be computed by simply playing it out on the
//! bitboards. That is the static exchange evaluation (SEE) of the move.
//!
//! Only [`see_ge`] is offered, which answers whether the exchange is worth
//! at least a threshold rather than computing the exact value. The question
//! the search asks is always of that form ("does this capture lose
//! material?"), and the threshold form can stop as soon as the answer is
//! settled, which is usually after one or two captures.

use crate::attacks::{bishop_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::params::SEE_VALUE;
use crate::position::Position;
use crate::types::{Color, PieceKind, Square};

/// Returns the value of a piece kind for the purpose of exchanges.
#[inline]
fn value(kind: PieceKind) -> i32 {
    SEE_VALUE[kind.index()]
}

/// Returns the least valuable kind of piece among `pieces`, with the square
/// of one such piece, or `None` if the set is empty.
#[inline]
fn least_valuable(position: &Position, pieces: Bitboard) -> Option<(PieceKind, Square)> {
    for kind in PieceKind::ALL {
        let candidates = pieces & position.kind(kind);
        if candidates.any() {
            return Some((kind, candidates.lsb()));
        }
    }
    None
}

/// Returns `true` if the exchange started by `mv` in `position` gains at
/// least `threshold` centipawns for the side to move, assuming both sides
/// capture with their least valuable piece first and stop when continuing
/// would not pay.
///
/// Castling moves nothing onto an attacked square and captures nothing, so
/// it is worth exactly zero. A move that gives check or has some other
/// positional point is not credited for it; this is a material count only.
///
/// The sequence is played out on a copy of the occupancy. Removing an
/// attacker from the occupancy can uncover a slider behind it on the same
/// line (a battery such as a rook behind a rook), so after each capture the
/// bishop and rook attacks through the vacated square are recomputed and
/// any newly revealed attackers are added.
///
/// The running `balance` is the gain so far minus the threshold, from the
/// point of view of the side that has just captured, and it is negated as
/// the move passes to the other side. Two early exits settle most calls:
/// if capturing the victim for free would not reach the threshold, nothing
/// will; and if the capture reaches the threshold even when the capturing
/// piece is then lost, nothing the opponent does can prevent it.
pub fn see_ge(position: &Position, mv: Move, threshold: i32) -> bool {
    if mv.is_castle() {
        return threshold <= 0;
    }
    let from = mv.from();
    let to = mv.to();
    let mut occupied = position.occupied() ^ from.bb();

    // What the first capture wins, and what piece then stands on `to`.
    let mut gain = if mv.is_en_passant() {
        occupied ^= Position::en_passant_victim(to).bb();
        value(PieceKind::Pawn)
    } else {
        position.piece_on(to).map_or(0, |piece| value(piece.kind()))
    };
    let mut next_kind = position
        .piece_on(from)
        .map_or(PieceKind::Pawn, |piece| piece.kind());
    if let Some(promotion) = mv.promotion() {
        // A promoting pawn becomes the new piece before it can be taken.
        gain += value(promotion) - value(PieceKind::Pawn);
        next_kind = promotion;
    }

    let mut balance = gain - threshold;
    if balance < 0 {
        return false;
    }
    balance -= value(next_kind);
    if balance >= 0 {
        return true;
    }

    // From here on the question is whether the opponent can recapture and
    // make it pay. The victim square is cleared so that `attackers_to` can
    // look through it along lines; the mover's piece, now standing there,
    // is the next thing to be captured.
    occupied ^= to.bb();
    let bishops_and_queens = position.kind(PieceKind::Bishop) | position.kind(PieceKind::Queen);
    let rooks_and_queens = position.kind(PieceKind::Rook) | position.kind(PieceKind::Queen);
    let mut attackers = position.attackers_to(to, occupied) & occupied;
    let mut side: Color = !position.side_to_move();
    // Whether the side to move in the original position comes out ahead,
    // given the captures so far. It flips with every further capture.
    let mut result = true;

    loop {
        attackers &= occupied;
        let Some((kind, square)) = least_valuable(position, attackers & position.color(side))
        else {
            break;
        };
        // `side` recaptures, so the outcome flips: what was a gain for the
        // other side is now at risk.
        result = !result;
        if kind == PieceKind::King {
            // A king may capture only if nothing can take it back. If the
            // other side still has an attacker, this recapture is illegal
            // and the exchange ends with the previous result.
            if (attackers & position.color(!side)).any() {
                result = !result;
            }
            break;
        }
        // Seen from `side`, the balance is negated (its gain is the other
        // side's loss), shifted by one because the other side needed only
        // to reach the threshold while `side` must now beat it, and reduced
        // by the piece it is putting at risk.
        balance = -balance - 1 - value(kind);
        if balance >= 0 {
            // Even if this piece is lost, `side` is ahead: the exchange
            // settles in its favour, so the other side would not recapture.
            break;
        }
        occupied ^= square.bb();
        match kind {
            PieceKind::Pawn | PieceKind::Bishop => {
                attackers |= bishop_attacks(to, occupied) & bishops_and_queens;
            }
            PieceKind::Rook => {
                attackers |= rook_attacks(to, occupied) & rooks_and_queens;
            }
            PieceKind::Queen => {
                attackers |= (bishop_attacks(to, occupied) & bishops_and_queens)
                    | (rook_attacks(to, occupied) & rooks_and_queens);
            }
            PieceKind::Knight | PieceKind::King => {}
        }
        side = !side;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::{GenKind, generate};
    use crate::moves::MoveList;

    /// Returns the legal move written `uci` in `position`.
    fn mv(position: &Position, uci: &str) -> Move {
        let mut list = MoveList::new();
        generate(position, GenKind::All, &mut list);
        (0..list.len())
            .map(|i| list[i])
            .find(|m| m.to_string() == uci)
            .unwrap_or_else(|| panic!("no move {uci}"))
    }

    fn position(fen: &str) -> Position {
        Position::from_fen(fen).unwrap()
    }

    #[test]
    fn free_pawn_is_a_gain() {
        let p = position("1k6/8/8/3p4/4P3/8/8/1K6 w - - 0 1");
        let m = mv(&p, "e4d5");
        assert!(see_ge(&p, m, 0));
        assert!(see_ge(&p, m, 100));
        assert!(!see_ge(&p, m, 101));
    }

    #[test]
    fn pawn_takes_pawn_defended_by_pawn_is_even() {
        let p = position("1k6/8/4p3/3p4/4P3/8/8/1K6 w - - 0 1");
        let m = mv(&p, "e4d5");
        assert!(see_ge(&p, m, 0));
        assert!(!see_ge(&p, m, 1));
    }

    #[test]
    fn rook_takes_defended_pawn_loses() {
        let p = position("k7/8/4p3/3p4/8/8/8/K2R4 w - - 0 1");
        let m = mv(&p, "d1d5");
        assert!(!see_ge(&p, m, 0));
        assert!(see_ge(&p, m, -400));
        assert!(!see_ge(&p, m, -399));
    }

    #[test]
    fn battery_behind_the_first_capturer_is_seen() {
        // Rook takes pawn, pawn takes rook, the rook behind (an x-ray
        // through d2) recaptures: +100 - 500 + 100 = -300.
        let p = position("k7/8/4p3/3p4/8/8/3R4/K2R4 w - - 0 1");
        let m = mv(&p, "d2d5");
        assert!(see_ge(&p, m, -300));
        assert!(!see_ge(&p, m, -299));
    }

    #[test]
    fn king_cannot_recapture_a_defended_piece() {
        // Rxd7 and the king may not take back, because the queen on d1,
        // uncovered when the rook leaves d2, would then capture it: the
        // pawn is won for free.
        let p = position("3k4/3p4/8/8/8/8/3R4/K2Q4 w - - 0 1");
        let m = mv(&p, "d2d7");
        assert!(see_ge(&p, m, 100));
        assert!(!see_ge(&p, m, 101));
    }

    #[test]
    fn king_may_recapture_an_undefended_piece() {
        let p = position("3k4/3p4/8/8/8/8/8/K2R4 w - - 0 1");
        let m = mv(&p, "d1d7");
        assert!(see_ge(&p, m, -400));
        assert!(!see_ge(&p, m, -399));
    }

    #[test]
    fn en_passant_is_a_pawn_capture() {
        let p = position("k7/8/8/3pP3/8/8/8/K7 w - d6 0 2");
        let m = mv(&p, "e5d6");
        assert!(see_ge(&p, m, 100));
        assert!(!see_ge(&p, m, 101));
    }

    #[test]
    fn castling_is_worth_nothing() {
        let p = position("k7/8/8/8/8/8/8/4K2R w K - 0 1");
        let m = mv(&p, "e1g1");
        assert!(see_ge(&p, m, 0));
        assert!(!see_ge(&p, m, 1));
    }
}
