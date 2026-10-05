//! Fundamental chess types: colours, piece kinds, pieces and squares.

use std::fmt;
use std::ops::Not;

use crate::bitboard::Bitboard;

/// One of the two sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    /// The side that moves first.
    White = 0,
    /// The side that moves second.
    Black = 1,
}

impl Color {
    /// Returns this colour as an array index (White = 0, Black = 1).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the square-index change of a single pawn push for this colour:
    /// `+8` for White (up the board) and `-8` for Black.
    #[inline]
    pub const fn pawn_push(self) -> i8 {
        match self {
            Color::White => 8,
            Color::Black => -8,
        }
    }

    /// Returns the rank index (0-7) on which this colour's pieces start:
    /// 0 (rank 1) for White, 7 (rank 8) for Black.
    #[inline]
    pub const fn back_rank(self) -> u8 {
        match self {
            Color::White => 0,
            Color::Black => 7,
        }
    }
}

impl Not for Color {
    type Output = Color;

    #[inline]
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// A kind of piece, without regard to colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceKind {
    /// Pawn.
    Pawn = 0,
    /// Knight.
    Knight = 1,
    /// Bishop.
    Bishop = 2,
    /// Rook.
    Rook = 3,
    /// Queen.
    Queen = 4,
    /// King.
    King = 5,
}

impl PieceKind {
    /// Every piece kind, in index order.
    pub const ALL: [PieceKind; 6] = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];

    /// Returns this kind as an array index (Pawn = 0 … King = 5).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the lower-case letter used for this kind in FEN and UCI
    /// (`p`, `n`, `b`, `r`, `q`, `k`).
    pub const fn to_char(self) -> char {
        match self {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        }
    }
}

/// A piece of a particular colour.
///
/// The discriminant is `colour * 6 + kind`, so the twelve pieces index a table
/// directly and `Option<Piece>` occupies a single byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    /// White pawn.
    WhitePawn = 0,
    /// White knight.
    WhiteKnight = 1,
    /// White bishop.
    WhiteBishop = 2,
    /// White rook.
    WhiteRook = 3,
    /// White queen.
    WhiteQueen = 4,
    /// White king.
    WhiteKing = 5,
    /// Black pawn.
    BlackPawn = 6,
    /// Black knight.
    BlackKnight = 7,
    /// Black bishop.
    BlackBishop = 8,
    /// Black rook.
    BlackRook = 9,
    /// Black queen.
    BlackQueen = 10,
    /// Black king.
    BlackKing = 11,
}

impl Piece {
    /// Every piece, in index order.
    pub const ALL: [Piece; 12] = [
        Piece::WhitePawn,
        Piece::WhiteKnight,
        Piece::WhiteBishop,
        Piece::WhiteRook,
        Piece::WhiteQueen,
        Piece::WhiteKing,
        Piece::BlackPawn,
        Piece::BlackKnight,
        Piece::BlackBishop,
        Piece::BlackRook,
        Piece::BlackQueen,
        Piece::BlackKing,
    ];

    /// Returns the piece of the given colour and kind.
    #[inline]
    pub const fn new(color: Color, kind: PieceKind) -> Piece {
        Piece::ALL[color.index() * 6 + kind.index()]
    }

    /// Returns this piece as an array index (0-11).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the colour of this piece.
    #[inline]
    pub const fn color(self) -> Color {
        if (self as u8) < 6 {
            Color::White
        } else {
            Color::Black
        }
    }

    /// Returns the kind of this piece.
    #[inline]
    pub const fn kind(self) -> PieceKind {
        PieceKind::ALL[self as usize % 6]
    }

    /// Returns the FEN letter for this piece: upper case for White, lower
    /// case for Black.
    pub const fn to_char(self) -> char {
        let lower = self.kind().to_char();
        match self.color() {
            Color::White => lower.to_ascii_uppercase(),
            Color::Black => lower,
        }
    }

    /// Parses a FEN piece letter, returning `None` if `c` is not one of
    /// `PNBRQKpnbrqk`.
    pub fn from_char(c: char) -> Option<Piece> {
        Piece::ALL.into_iter().find(|piece| piece.to_char() == c)
    }
}

/// A square of the board.
///
/// Squares are numbered in little-endian rank-file order (BRD-2): a1 = 0,
/// b1 = 1, … h1 = 7, a2 = 8, … h8 = 63. The wrapped index is always below 64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Square(u8);

impl Square {
    /// Creates a square from its index.
    ///
    /// `index` must be below 64; this is checked in debug builds only.
    #[inline]
    pub const fn new(index: u8) -> Square {
        debug_assert!(index < 64);
        Square(index)
    }

    /// Creates a square from a file (0 = a … 7 = h) and a rank (0 = rank 1 …
    /// 7 = rank 8). Both must be below 8; this is checked in debug builds only.
    #[inline]
    pub const fn from_file_rank(file: u8, rank: u8) -> Square {
        debug_assert!(file < 8 && rank < 8);
        Square(rank * 8 + file)
    }

    /// Returns the square as an array index (0-63).
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the file of the square (0 = a … 7 = h).
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// Returns the rank of the square (0 = rank 1 … 7 = rank 8).
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// Returns the bitboard containing only this square.
    #[inline]
    pub const fn bb(self) -> Bitboard {
        Bitboard(1 << self.0)
    }

    /// Returns the square `delta` indices away (e.g. `+8` is one rank up).
    ///
    /// The caller must ensure the result is on the board and that the step
    /// does not wrap around a board edge; the range is checked in debug
    /// builds only.
    #[inline]
    pub const fn offset(self, delta: i8) -> Square {
        Square::new((self.0 as i8 + delta) as u8)
    }

    /// Parses a square in algebraic notation (`"a1"` … `"h8"`), returning
    /// `None` for anything else.
    pub fn parse(text: &str) -> Option<Square> {
        match *text.as_bytes() {
            [file @ b'a'..=b'h', rank @ b'1'..=b'8'] => {
                Some(Square::from_file_rank(file - b'a', rank - b'1'))
            }
            _ => None,
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_round_trips_through_colour_and_kind() {
        for piece in Piece::ALL {
            assert_eq!(Piece::new(piece.color(), piece.kind()), piece);
            assert_eq!(Piece::from_char(piece.to_char()), Some(piece));
        }
        assert_eq!(Piece::from_char('x'), None);
    }

    #[test]
    fn optional_piece_fits_in_one_byte() {
        assert_eq!(std::mem::size_of::<Option<Piece>>(), 1);
    }

    #[test]
    fn square_mapping_is_little_endian_rank_file() {
        assert_eq!(Square::parse("a1"), Some(Square::new(0)));
        assert_eq!(Square::parse("h1"), Some(Square::new(7)));
        assert_eq!(Square::parse("a8"), Some(Square::new(56)));
        assert_eq!(Square::parse("h8"), Some(Square::new(63)));
        for index in 0..64 {
            let square = Square::new(index);
            assert_eq!(Square::parse(&square.to_string()), Some(square));
            assert_eq!(Square::from_file_rank(square.file(), square.rank()), square);
        }
    }

    #[test]
    fn malformed_squares_are_rejected() {
        for text in ["", "a", "i1", "a9", "a0", "e44", "E4"] {
            assert_eq!(Square::parse(text), None, "{text:?}");
        }
    }
}
