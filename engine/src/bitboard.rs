//! Bitboards: sets of squares stored as one bit per square in a `u64`.

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::types::Square;

/// A set of squares. Bit `n` is set when the square with index `n` (see
/// [`Square`]) is a member.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// The empty set.
    pub const EMPTY: Bitboard = Bitboard(0);

    /// The set of all 64 squares.
    pub const ALL: Bitboard = Bitboard(u64::MAX);

    /// Returns the set of the eight squares on `rank` (0 = rank 1 … 7 = rank 8).
    #[inline]
    pub const fn rank(rank: u8) -> Bitboard {
        Bitboard(0xFF << (rank * 8))
    }

    /// Returns `true` if the set has no members.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if the set has at least one member.
    #[inline]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Returns `true` if the set has two or more members.
    ///
    /// `x & (x - 1)` clears the lowest set bit, so the result is non-zero
    /// exactly when a second bit exists.
    #[inline]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    /// Returns the number of squares in the set.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns `true` if `square` is a member of the set.
    #[inline]
    pub const fn contains(self, square: Square) -> bool {
        self.0 & square.bb().0 != 0
    }

    /// Returns the lowest-indexed square in the set.
    ///
    /// The set must not be empty; this is checked in debug builds only.
    #[inline]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square::new(self.0.trailing_zeros() as u8)
    }
}

/// Iterating a bitboard yields its squares from lowest index to highest,
/// removing each one as it is returned.
impl Iterator for Bitboard {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let square = self.lsb();
        self.0 &= self.0 - 1;
        Some(square)
    }
}

/// Implements a binary set operator and its assigning form for [`Bitboard`].
macro_rules! impl_set_operator {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl $trait for Bitboard {
            type Output = Bitboard;

            #[inline]
            fn $method(self, rhs: Bitboard) -> Bitboard {
                Bitboard(self.0 $op rhs.0)
            }
        }

        impl $assign_trait for Bitboard {
            #[inline]
            fn $assign_method(&mut self, rhs: Bitboard) {
                self.0 = self.0 $op rhs.0;
            }
        }
    };
}

impl_set_operator!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
impl_set_operator!(BitOr, bitor, BitOrAssign, bitor_assign, |);
impl_set_operator!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl Not for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration_yields_squares_in_ascending_order() {
        let set = Square::new(3).bb() | Square::new(40).bb() | Square::new(63).bb();
        let squares: Vec<usize> = set.map(Square::index).collect();
        assert_eq!(squares, [3, 40, 63]);
    }

    #[test]
    fn membership_and_counting() {
        let set = Square::new(0).bb() | Square::new(9).bb();
        assert!(set.contains(Square::new(9)));
        assert!(!set.contains(Square::new(1)));
        assert_eq!(set.count(), 2);
        assert!(set.more_than_one());
        assert!(!Square::new(5).bb().more_than_one());
        assert!(!Bitboard::EMPTY.more_than_one());
        assert!(Bitboard::EMPTY.is_empty());
        assert_eq!(set.lsb(), Square::new(0));
    }

    #[test]
    fn rank_masks() {
        assert_eq!(Bitboard::rank(0).0, 0xFF);
        assert_eq!(Bitboard::rank(7).0, 0xFF00_0000_0000_0000);
    }
}
