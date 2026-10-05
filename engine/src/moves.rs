//! Move encoding and the fixed-capacity move list.

use std::fmt;
use std::ops::Deref;

use crate::types::{PieceKind, Square};

/// A move packed into 16 bits (BRD-3).
///
/// | Bits  | Meaning            |
/// |-------|--------------------|
/// | 0-5   | origin square      |
/// | 6-11  | destination square |
/// | 12-15 | move flag          |
///
/// The flag values are the associated constants of this type. They are laid
/// out so that bit 2 of the flag (value 4) is set for every capture and bit 3
/// (value 8) for every promotion, which makes those tests a single mask.
/// Castling is encoded as the king's own two-square move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move(u16);

impl Move {
    /// A move that changes nothing special: not a capture, promotion,
    /// double push or castle.
    pub const QUIET: u16 = 0;
    /// A pawn advancing two squares from its starting rank.
    pub const DOUBLE_PUSH: u16 = 1;
    /// Castling on the king's side.
    pub const KING_CASTLE: u16 = 2;
    /// Castling on the queen's side.
    pub const QUEEN_CASTLE: u16 = 3;
    /// An ordinary capture: the captured piece stands on the destination.
    pub const CAPTURE: u16 = 4;
    /// An en-passant capture: the captured pawn stands beside the origin.
    pub const EN_PASSANT: u16 = 5;
    /// Flag bit set on every promotion. The two low bits select the promoted
    /// piece (knight, bishop, rook, queen) and [`Move::CAPTURE`] may be added.
    pub const PROMOTION: u16 = 8;

    /// The all-zero value (a1 to a1), which is never a legal move; used to
    /// fill unused move-list slots and, later, to mean "no move".
    pub const NULL: Move = Move(0);

    /// Packs a move from its origin, destination and flag (one of the flag
    /// constants, or the result of [`Move::promotion_flag`]).
    #[inline]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        debug_assert!(flag < 16);
        Move(from.index() as u16 | (to.index() as u16) << 6 | flag << 12)
    }

    /// Returns the flag for a promotion to `kind`, capturing if `capture`.
    ///
    /// `kind` must be a knight, bishop, rook or queen; this is checked in
    /// debug builds only.
    #[inline]
    pub const fn promotion_flag(kind: PieceKind, capture: bool) -> u16 {
        debug_assert!(matches!(
            kind,
            PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen
        ));
        // Knight..Queen have indices 1..4, giving the selector 0..3.
        let flag = Move::PROMOTION | (kind.index() as u16 - 1);
        if capture { flag | Move::CAPTURE } else { flag }
    }

    /// Returns the origin square.
    #[inline]
    pub const fn from(self) -> Square {
        Square::new((self.0 & 63) as u8)
    }

    /// Returns the destination square.
    #[inline]
    pub const fn to(self) -> Square {
        Square::new((self.0 >> 6 & 63) as u8)
    }

    /// Returns the four-bit move flag.
    #[inline]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    /// Returns `true` if the move captures a piece, including en passant and
    /// capturing promotions.
    #[inline]
    pub const fn is_capture(self) -> bool {
        self.flag() & Move::CAPTURE != 0
    }

    /// Returns `true` if the move is an en-passant capture.
    #[inline]
    pub const fn is_en_passant(self) -> bool {
        self.flag() == Move::EN_PASSANT
    }

    /// Returns `true` if the move is a two-square pawn push.
    #[inline]
    pub const fn is_double_push(self) -> bool {
        self.flag() == Move::DOUBLE_PUSH
    }

    /// Returns `true` if the move castles on either side.
    #[inline]
    pub const fn is_castle(self) -> bool {
        self.flag() == Move::KING_CASTLE || self.flag() == Move::QUEEN_CASTLE
    }

    /// Returns the piece kind a pawn promotes to, or `None` if the move is
    /// not a promotion.
    #[inline]
    pub const fn promotion(self) -> Option<PieceKind> {
        if self.flag() & Move::PROMOTION == 0 {
            return None;
        }
        Some(match self.flag() & 3 {
            0 => PieceKind::Knight,
            1 => PieceKind::Bishop,
            2 => PieceKind::Rook,
            _ => PieceKind::Queen,
        })
    }
}

/// Formats the move in UCI long algebraic notation: origin, destination and,
/// for promotions, the lower-case piece letter (`e2e4`, `e7e8q`, `e1g1`).
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(kind) = self.promotion() {
            write!(f, "{}", kind.to_char())?;
        }
        Ok(())
    }
}

/// A list of moves stored inline, with no heap allocation (MGN-3).
///
/// The list dereferences to a slice of the moves pushed so far.
pub struct MoveList {
    /// Storage; only the first `len` entries are meaningful.
    moves: [Move; MoveList::CAPACITY],
    /// Number of moves pushed.
    len: usize,
}

impl MoveList {
    /// Maximum number of moves the list can hold. No position reachable in a
    /// legal game has more than 218 legal moves.
    pub const CAPACITY: usize = 256;

    /// Creates an empty list.
    #[inline]
    pub const fn new() -> MoveList {
        MoveList {
            moves: [Move::NULL; MoveList::CAPACITY],
            len: 0,
        }
    }

    /// Appends a move.
    ///
    /// If the list is already full the move is dropped. That can only happen
    /// in artificial positions that could never arise in a game; dropping
    /// keeps the engine from crashing on them (and is flagged in debug
    /// builds).
    #[inline]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MoveList::CAPACITY);
        if self.len < MoveList::CAPACITY {
            self.moves[self.len] = mv;
            self.len += 1;
        }
    }
}

impl Default for MoveList {
    fn default() -> MoveList {
        MoveList::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline]
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(name: &str) -> Square {
        Square::parse(name).unwrap()
    }

    #[test]
    fn fields_round_trip() {
        let mv = Move::new(
            sq("g7"),
            sq("h8"),
            Move::promotion_flag(PieceKind::Rook, true),
        );
        assert_eq!(mv.from(), sq("g7"));
        assert_eq!(mv.to(), sq("h8"));
        assert_eq!(mv.promotion(), Some(PieceKind::Rook));
        assert!(mv.is_capture());
        assert!(!mv.is_en_passant());
        assert_eq!(mv.to_string(), "g7h8r");
    }

    #[test]
    fn flag_predicates() {
        let quiet = Move::new(sq("g1"), sq("f3"), Move::QUIET);
        assert!(!quiet.is_capture() && quiet.promotion().is_none() && !quiet.is_castle());
        assert!(Move::new(sq("e5"), sq("d6"), Move::EN_PASSANT).is_capture());
        assert!(Move::new(sq("e1"), sq("g1"), Move::KING_CASTLE).is_castle());
        assert!(Move::new(sq("e1"), sq("c1"), Move::QUEEN_CASTLE).is_castle());
        assert!(Move::new(sq("e2"), sq("e4"), Move::DOUBLE_PUSH).is_double_push());
        let promo = Move::new(
            sq("a7"),
            sq("a8"),
            Move::promotion_flag(PieceKind::Queen, false),
        );
        assert!(!promo.is_capture());
        assert_eq!(promo.to_string(), "a7a8q");
    }

    #[test]
    fn list_holds_pushed_moves_in_order() {
        let mut list = MoveList::new();
        assert!(list.is_empty());
        let a = Move::new(sq("e2"), sq("e4"), Move::DOUBLE_PUSH);
        let b = Move::new(sq("g1"), sq("f3"), Move::QUIET);
        list.push(a);
        list.push(b);
        assert_eq!(&list[..], [a, b]);
    }
}
