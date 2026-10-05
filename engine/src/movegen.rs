//! Legal move generation (MGN-1, MGN-2).
//!
//! Moves are generated legal from the start, rather than generated loosely
//! and tested by playing them. Three facts about the position make that
//! possible:
//!
//! * **Checkers** — the enemy pieces attacking our king. With two checkers
//!   only the king can move. With one, every other piece must capture the
//!   checker or block the line to it; those squares form the *check mask*.
//! * **Pinned pieces** — our pieces that stand alone between our king and an
//!   enemy slider. A pinned piece may move only along the line through the
//!   king and itself, since any other move exposes the king.
//! * **King danger** — the king may not step onto an attacked square, where
//!   attacks are computed with the king itself lifted off the board so that
//!   it cannot hide behind its own former position.
//!
//! En passant is the one move these rules do not cover, because it removes a
//! piece from a square other than the destination; it is verified directly.

use crate::attacks::{
    between, bishop_attacks, king_attacks, knight_attacks, line, pawn_attacks, queen_attacks,
    rook_attacks,
};
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveList};
use crate::position::{CastlingRights, Position};
use crate::types::{Color, PieceKind, Square};

/// Which moves to generate (MGN-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenKind {
    /// Every legal move.
    All,
    /// Captures (including en passant) and all promotions.
    Noisy,
    /// Every legal move that is not [`GenKind::Noisy`], including castling.
    Quiet,
}

/// A slider attack lookup: attacks from a square given the occupied squares.
type SliderAttacks = fn(Square, Bitboard) -> Bitboard;

/// The promotion pieces, strongest first.
const PROMOTION_KINDS: [PieceKind; 4] = [
    PieceKind::Queen,
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Knight,
];

/// Appends the legal moves of the requested `kind` for the side to move in
/// `position` to `list`.
///
/// `Noisy` and `Quiet` are disjoint and together produce exactly the moves of
/// `All`. An empty result for `All` means checkmate or stalemate.
pub fn generate(position: &Position, kind: GenKind, list: &mut MoveList) {
    let us = position.side_to_move();
    let them = !us;
    let ours = position.color(us);
    let theirs = position.color(them);
    let occupied = position.occupied();
    let king = position.king_square(us);
    let noisy = kind != GenKind::Quiet;
    let quiet = kind != GenKind::Noisy;

    // King moves. Attacks are probed with the king removed from the
    // occupancy, so that squares behind it on a checking line count as
    // attacked.
    let without_king = occupied ^ king.bb();
    let king_targets = king_attacks(king)
        & !ours
        & match kind {
            GenKind::All => Bitboard::ALL,
            GenKind::Noisy => theirs,
            GenKind::Quiet => !theirs,
        };
    for to in king_targets {
        if (position.attackers_to(to, without_king) & theirs).is_empty() {
            list.push(Move::new(king, to, capture_flag(theirs, to)));
        }
    }

    let checkers = position.attackers_to(king, occupied) & theirs;
    if checkers.more_than_one() {
        // Double check: no capture or block can answer both checkers.
        return;
    }

    // Squares a non-king piece may move to without leaving the king in
    // check: anywhere when not in check, otherwise the checker's square or a
    // square between it and the king (empty for a knight or pawn check).
    let check_mask = if checkers.any() {
        between(king, checkers.lsb()) | checkers
    } else {
        Bitboard::ALL
    };
    let mut targets = Bitboard::EMPTY;
    if noisy {
        targets |= theirs;
    }
    if quiet {
        targets |= !occupied;
    }
    targets &= check_mask;

    let pinned = pinned_pieces(position, us, king);

    // A pinned knight can never stay on the pin line, so it has no moves.
    for from in position.pieces(us, PieceKind::Knight) & !pinned {
        push_piece_moves(list, from, knight_attacks(from) & targets, theirs);
    }

    let sliders: [(PieceKind, SliderAttacks); 3] = [
        (PieceKind::Bishop, bishop_attacks),
        (PieceKind::Rook, rook_attacks),
        (PieceKind::Queen, queen_attacks),
    ];
    for (piece_kind, attacks_of) in sliders {
        for from in position.pieces(us, piece_kind) {
            let mut attacks = attacks_of(from, occupied) & targets;
            if pinned.contains(from) {
                attacks &= line(king, from);
            }
            push_piece_moves(list, from, attacks, theirs);
        }
    }

    generate_pawn_moves(position, kind, check_mask, pinned, list);

    if quiet && checkers.is_empty() {
        generate_castling(position, list);
    }
}

/// Returns the flag for a non-pawn move to `to`: a capture if an enemy piece
/// stands there, otherwise quiet.
#[inline]
fn capture_flag(theirs: Bitboard, to: Square) -> u16 {
    if theirs.contains(to) {
        Move::CAPTURE
    } else {
        Move::QUIET
    }
}

/// Appends one move from `from` to each square in `targets`, flagged as a
/// capture where the destination holds an enemy piece.
#[inline]
fn push_piece_moves(list: &mut MoveList, from: Square, targets: Bitboard, theirs: Bitboard) {
    for to in targets {
        list.push(Move::new(from, to, capture_flag(theirs, to)));
    }
}

/// Returns the pieces of `us` that are pinned to their king on `king`.
///
/// An enemy slider pins a piece when it would attack the king on an empty
/// board and exactly one piece stands between the two. If that single piece
/// is ours it is pinned (if it is theirs, it is a potential discovered check
/// against us, which does not restrict our moves).
fn pinned_pieces(position: &Position, us: Color, king: Square) -> Bitboard {
    let them = !us;
    let occupied = position.occupied();
    let queens = position.kind(PieceKind::Queen);
    let diagonal = position.kind(PieceKind::Bishop) | queens;
    let straight = position.kind(PieceKind::Rook) | queens;
    let snipers = position.color(them)
        & ((bishop_attacks(king, Bitboard::EMPTY) & diagonal)
            | (rook_attacks(king, Bitboard::EMPTY) & straight));

    let mut pinned = Bitboard::EMPTY;
    for sniper in snipers {
        let blockers = between(king, sniper) & occupied;
        if blockers.any() && !blockers.more_than_one() {
            pinned |= blockers & position.color(us);
        }
    }
    pinned
}

/// Appends the legal pawn moves of the requested `kind`.
///
/// `check_mask` and `pinned` are as computed in [`generate`]. Each pawn's
/// destinations are restricted to the check mask and, if the pawn is pinned,
/// to the line through the king and the pawn.
fn generate_pawn_moves(
    position: &Position,
    kind: GenKind,
    check_mask: Bitboard,
    pinned: Bitboard,
    list: &mut MoveList,
) {
    let us = position.side_to_move();
    let them = !us;
    let theirs = position.color(them);
    let occupied = position.occupied();
    let king = position.king_square(us);
    let noisy = kind != GenKind::Quiet;
    let quiet = kind != GenKind::Noisy;
    let push = us.pawn_push();
    let start_rank = match us {
        Color::White => 1,
        Color::Black => 6,
    };
    let promotion_rank = them.back_rank();

    for from in position.pieces(us, PieceKind::Pawn) {
        let allowed = if pinned.contains(from) {
            check_mask & line(king, from)
        } else {
            check_mask
        };

        // Pushes. A pawn is never on its last rank, so `to` is on the board.
        let to = from.offset(push);
        if !occupied.contains(to) {
            if allowed.contains(to) {
                if to.rank() == promotion_rank {
                    if noisy {
                        push_promotions(list, from, to, false);
                    }
                } else if quiet {
                    list.push(Move::new(from, to, Move::QUIET));
                }
            }
            // The double push needs the first square empty but not allowed:
            // it may block a check on the far square while passing through
            // one that would not.
            if quiet && from.rank() == start_rank {
                let to = to.offset(push);
                if !occupied.contains(to) && allowed.contains(to) {
                    list.push(Move::new(from, to, Move::DOUBLE_PUSH));
                }
            }
        }

        if !noisy {
            continue;
        }

        for to in pawn_attacks(us, from) & theirs & allowed {
            if to.rank() == promotion_rank {
                push_promotions(list, from, to, true);
            } else {
                list.push(Move::new(from, to, Move::CAPTURE));
            }
        }

        // En passant removes the captured pawn from a square other than the
        // destination, so neither the check mask nor the pin test applies
        // (for instance, both pawns can vanish from the king's rank at once
        // and expose it to a rook). Instead, build the occupancy after the
        // capture and ask directly whether the king would be attacked. The
        // captured pawn is still in the position's piece sets, so it is
        // masked out of the attackers.
        if let Some(ep_square) = position.ep_square()
            && pawn_attacks(us, from).contains(ep_square)
        {
            let victim = Position::en_passant_victim(ep_square);
            let after = (occupied ^ from.bb() ^ victim.bb()) | ep_square.bb();
            if (position.attackers_to(king, after) & theirs & !victim.bb()).is_empty() {
                list.push(Move::new(from, ep_square, Move::EN_PASSANT));
            }
        }
    }
}

/// Appends the four promotions of a pawn moving from `from` to `to`.
#[inline]
fn push_promotions(list: &mut MoveList, from: Square, to: Square, capture: bool) {
    for kind in PROMOTION_KINDS {
        list.push(Move::new(from, to, Move::promotion_flag(kind, capture)));
    }
}

/// Appends the legal castling moves. The caller must have established that
/// the side to move is not in check.
///
/// Castling requires the right to be intact, every square between king and
/// rook to be empty, and the two squares the king crosses and lands on not
/// to be attacked. (On the queen's side the b-file square must be empty but
/// may be attacked, as only the rook passes over it.) A castling right
/// guarantees the king and rook are on their home squares.
fn generate_castling(position: &Position, list: &mut MoveList) {
    let us = position.side_to_move();
    let theirs = position.color(!us);
    let occupied = position.occupied();
    let rank = us.back_rank();
    let square = |file| Square::from_file_rank(file, rank);
    let safe = |square: Square| (position.attackers_to(square, occupied) & theirs).is_empty();
    let king = square(4);

    if position.castling().contains(CastlingRights::kingside(us))
        && !occupied.contains(square(5))
        && !occupied.contains(square(6))
        && safe(square(5))
        && safe(square(6))
    {
        list.push(Move::new(king, square(6), Move::KING_CASTLE));
    }

    if position.castling().contains(CastlingRights::queenside(us))
        && !occupied.contains(square(1))
        && !occupied.contains(square(2))
        && !occupied.contains(square(3))
        && safe(square(2))
        && safe(square(3))
    {
        list.push(Move::new(king, square(2), Move::QUEEN_CASTLE));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the legal moves of `kind` in `fen` as sorted UCI strings.
    fn moves(fen: &str, kind: GenKind) -> Vec<String> {
        let position = Position::from_fen(fen).unwrap();
        let mut list = MoveList::new();
        generate(&position, kind, &mut list);
        let mut moves: Vec<String> = list.iter().map(Move::to_string).collect();
        moves.sort();
        moves
    }

    #[test]
    fn start_position_has_twenty_moves() {
        assert_eq!(moves(crate::position::START_FEN, GenKind::All).len(), 20);
    }

    #[test]
    fn double_check_allows_only_king_moves() {
        // Knight on f6 and rook on e1 both check the king on e8.
        let fen = "4k3/8/5N2/8/8/8/8/4RK2 b - - 0 1";
        assert_eq!(moves(fen, GenKind::All), ["e8d8", "e8f7", "e8f8"]);
    }

    #[test]
    fn single_check_must_be_answered() {
        // Rook e1 checks; the bishop can block on e6 or e4, or the king steps aside.
        let fen = "4k3/8/8/3b4/8/8/8/4RK2 b - - 0 1";
        assert_eq!(
            moves(fen, GenKind::All),
            ["d5e4", "d5e6", "e8d7", "e8d8", "e8f7", "e8f8"]
        );
    }

    #[test]
    fn pinned_piece_moves_only_along_the_pin() {
        // The e-file rook is pinned by the rook on e8; the knight on d2 is
        // pinned by the bishop on a5 and cannot move at all.
        let fen = "4r1k1/8/8/b7/8/8/3NR3/4K3 w - - 0 1";
        let all = moves(fen, GenKind::All);
        assert!(all.iter().all(|mv| !mv.starts_with("d2")));
        let rook: Vec<&String> = all.iter().filter(|mv| mv.starts_with("e2")).collect();
        assert_eq!(rook, ["e2e3", "e2e4", "e2e5", "e2e6", "e2e7", "e2e8"]);
    }

    #[test]
    fn en_passant_exposing_the_king_along_the_rank_is_illegal() {
        // Capturing e5xd6 would remove both pawns from the fifth rank and
        // leave the king on a5 attacked by the rook on h5.
        let fen = "4k3/8/8/K2pP2r/8/8/8/8 w - d6 0 1";
        assert!(!moves(fen, GenKind::All).contains(&"e5d6".to_string()));
        // Without the rook the capture is legal.
        let fen = "4k3/8/8/K2pP3/8/8/8/8 w - d6 0 1";
        assert!(moves(fen, GenKind::All).contains(&"e5d6".to_string()));
    }

    #[test]
    fn en_passant_can_capture_a_checking_pawn() {
        // The pawn that just played d7-d5 checks the king on e4.
        let fen = "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1";
        assert!(moves(fen, GenKind::All).contains(&"e5d6".to_string()));
    }

    #[test]
    fn castling_rules() {
        let both = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let all = moves(both, GenKind::All);
        assert!(all.contains(&"e1g1".to_string()) && all.contains(&"e1c1".to_string()));

        // A rook on f8 attacks f1: no king-side castling, queen-side is fine.
        let all = moves("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1", GenKind::All);
        assert!(!all.contains(&"e1g1".to_string()) && all.contains(&"e1c1".to_string()));

        // A rook on b8 attacks only b1, which the king does not cross.
        let all = moves("1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", GenKind::All);
        assert!(all.contains(&"e1c1".to_string()));

        // No castling out of check.
        let all = moves("4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1", GenKind::All);
        assert!(!all.contains(&"e1g1".to_string()) && !all.contains(&"e1c1".to_string()));

        // A piece between king and rook blocks castling.
        let all = moves("4k3/8/8/8/8/8/8/RN2K1NR w KQ - 0 1", GenKind::All);
        assert!(!all.contains(&"e1g1".to_string()) && !all.contains(&"e1c1".to_string()));
    }

    #[test]
    fn promotions_are_noisy() {
        let fen = "3n3k/4P3/8/8/8/8/8/4K3 w - - 0 1";
        let noisy = moves(fen, GenKind::Noisy);
        assert_eq!(
            noisy,
            [
                "e7d8b", "e7d8n", "e7d8q", "e7d8r", "e7e8b", "e7e8n", "e7e8q", "e7e8r"
            ]
        );
        assert!(
            moves(fen, GenKind::Quiet)
                .iter()
                .all(|mv| mv.starts_with("e1"))
        );
    }

    #[test]
    fn checkmate_and_stalemate_have_no_moves() {
        assert!(moves("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", GenKind::All).is_empty());
        assert!(moves("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", GenKind::All).is_empty());
    }
}
