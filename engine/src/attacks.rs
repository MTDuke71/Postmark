//! Attack sets for every piece, and the geometry tables used for pins and
//! checks.
//!
//! Knight, king and pawn attacks depend only on the square and are computed
//! at compile time. Bishop and rook attacks also depend on which squares are
//! occupied; they are looked up in a table built once, on first use, and
//! indexed either with PEXT or with magic multiplication (BLD-3).

use std::sync::LazyLock;

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

/// A step on the board as a (file change, rank change) pair.
type Step = (i8, i8);

/// The eight knight jumps.
const KNIGHT_STEPS: [Step; 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

/// The eight king steps.
const KING_STEPS: [Step; 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// The four directions a rook slides in.
const ROOK_DIRECTIONS: [Step; 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// The four directions a bishop slides in.
const BISHOP_DIRECTIONS: [Step; 4] = [(1, 1), (-1, 1), (-1, -1), (1, -1)];

/// Builds the attack table of a piece that jumps by fixed `steps`: for each
/// square, the set of squares reached by one step that stays on the board.
const fn leaper_table(steps: &[Step]) -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let mut square = 0;
    while square < 64 {
        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;
        let mut bits = 0u64;
        let mut i = 0;
        while i < steps.len() {
            let to_file = file + steps[i].0;
            let to_rank = rank + steps[i].1;
            if to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8 {
                bits |= 1 << (to_rank * 8 + to_file);
            }
            i += 1;
        }
        table[square] = Bitboard(bits);
        square += 1;
    }
    table
}

/// Knight attacks from each square.
static KNIGHT_ATTACKS: [Bitboard; 64] = leaper_table(&KNIGHT_STEPS);

/// King attacks from each square.
static KING_ATTACKS: [Bitboard; 64] = leaper_table(&KING_STEPS);

/// Pawn capture targets, indexed by the pawn's colour and then its square.
static PAWN_ATTACKS: [[Bitboard; 64]; 2] = [
    leaper_table(&[(-1, 1), (1, 1)]),
    leaper_table(&[(-1, -1), (1, -1)]),
];

/// Returns the squares attacked by a pawn of `color` standing on `square`.
///
/// Read the other way round, `pawn_attacks(color, square)` is also the set of
/// squares from which a pawn of the *opposite* colour attacks `square`.
#[inline]
pub fn pawn_attacks(color: Color, square: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][square.index()]
}

/// Returns the squares attacked by a knight on `square`.
#[inline]
pub fn knight_attacks(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index()]
}

/// Returns the squares attacked by a king on `square`.
#[inline]
pub fn king_attacks(square: Square) -> Bitboard {
    KING_ATTACKS[square.index()]
}

/// Returns the squares attacked by a bishop on `square` when the squares in
/// `occupied` hold pieces. The first occupied square in each direction is
/// included (it may be capturable); squares beyond it are not.
#[inline]
pub fn bishop_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    let tables = &*TABLES;
    tables.attacks[tables.bishop[square.index()].index(occupied)]
}

/// Returns the squares attacked by a rook on `square` when the squares in
/// `occupied` hold pieces. The first occupied square in each direction is
/// included (it may be capturable); squares beyond it are not.
#[inline]
pub fn rook_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    let tables = &*TABLES;
    tables.attacks[tables.rook[square.index()].index(occupied)]
}

/// Returns the squares attacked by a queen on `square` given `occupied`: the
/// union of the bishop and rook attacks.
#[inline]
pub fn queen_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(square, occupied) | rook_attacks(square, occupied)
}

/// Returns the squares strictly between `a` and `b` when they share a rank,
/// file or diagonal, and the empty set otherwise (including when the two are
/// adjacent or equal).
#[inline]
pub fn between(a: Square, b: Square) -> Bitboard {
    TABLES.between[a.index()][b.index()]
}

/// Returns the whole rank, file or diagonal through `a` and `b`, edge to edge
/// and including both squares, or the empty set if they are not aligned (or
/// are the same square).
#[inline]
pub fn line(a: Square, b: Square) -> Bitboard {
    TABLES.line[a.index()][b.index()]
}

/// Returns the square one `step` from `square`, or `None` if that leaves the
/// board.
fn step_from(square: Square, step: Step) -> Option<Square> {
    let file = square.file() as i8 + step.0;
    let rank = square.rank() as i8 + step.1;
    ((0..8).contains(&file) && (0..8).contains(&rank))
        .then(|| Square::from_file_rank(file as u8, rank as u8))
}

/// Computes slider attacks the slow way, by walking outwards from `square` in
/// each of `directions` until the board edge or an occupied square (which is
/// included). Used only to build and to test the lookup tables.
fn ray_attacks(square: Square, occupied: Bitboard, directions: &[Step]) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    for &direction in directions {
        let mut current = square;
        while let Some(next) = step_from(current, direction) {
            attacks |= next.bb();
            if occupied.contains(next) {
                break;
            }
            current = next;
        }
    }
    attacks
}

/// Returns the *relevant occupancy* mask of a slider on `square`: the squares
/// whose occupancy can change its attack set.
///
/// These are the squares along each ray, excluding the last one before the
/// board edge. A piece on that last square blocks nothing behind it, so the
/// attack set is the same whether it is occupied or not; leaving it out
/// halves the table size for each ray.
fn relevant_mask(square: Square, directions: &[Step]) -> Bitboard {
    let mut mask = Bitboard::EMPTY;
    for &direction in directions {
        let mut current = square;
        while let Some(next) = step_from(current, direction) {
            if step_from(next, direction).is_none() {
                break;
            }
            mask |= next.bb();
            current = next;
        }
    }
    mask
}

/// How to find one square's slice of the shared slider attack table.
///
/// The slice has one slot per subset of the relevant occupancy mask. The slot
/// for an occupancy is found by compressing the masked occupancy bits into a
/// dense index:
///
/// * With BMI2, the `PEXT` instruction does exactly that: it gathers the bits
///   of the occupancy selected by the mask into the low bits of the result.
/// * Without it, a *magic* multiplier does the same job. Multiplying the
///   masked occupancy by the magic shifts copies of its bits towards the top
///   of the word, and the top `n` bits (for an `n`-bit mask) are taken as the
///   index. The magic is found by trial so that two occupancies share a slot
///   only when they have the same attack set, which makes the lookup exact.
#[derive(Clone, Copy)]
struct SliderEntry {
    /// Relevant occupancy mask for the square.
    mask: u64,
    /// Magic multiplier for the square.
    #[cfg(not(target_feature = "bmi2"))]
    magic: u64,
    /// Right shift that keeps the top `mask.count_ones()` bits of the product.
    #[cfg(not(target_feature = "bmi2"))]
    shift: u32,
    /// Index of the square's first slot in the shared table.
    offset: usize,
}

impl SliderEntry {
    /// Returns the index in the shared attack table for the given occupancy.
    #[inline(always)]
    fn index(&self, occupied: Bitboard) -> usize {
        #[cfg(target_feature = "bmi2")]
        {
            // SAFETY: this block is compiled only when the `bmi2` target
            // feature is enabled for the whole build, so every CPU the binary
            // is allowed to run on implements PEXT.
            let compressed = unsafe { core::arch::x86_64::_pext_u64(occupied.0, self.mask) };
            self.offset + compressed as usize
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            let product = (occupied.0 & self.mask).wrapping_mul(self.magic);
            self.offset + (product >> self.shift) as usize
        }
    }

    /// Builds the entry for one square and fills its slots.
    ///
    /// `subsets` lists every subset of `mask` with its attack set, and `slots`
    /// is the square's slice of the shared table (starting at `offset`). With
    /// PEXT each subset has its own slot, so the table is simply filled.
    #[cfg(target_feature = "bmi2")]
    fn fit(
        mask: Bitboard,
        offset: usize,
        subsets: &[(Bitboard, Bitboard)],
        slots: &mut [Bitboard],
        _seed: &mut u64,
    ) -> SliderEntry {
        let entry = SliderEntry {
            mask: mask.0,
            offset,
        };
        for &(occupied, attacks) in subsets {
            slots[entry.index(occupied) - offset] = attacks;
        }
        entry
    }

    /// Builds the entry for one square and fills its slots.
    ///
    /// `subsets` lists every subset of `mask` with its attack set, and `slots`
    /// is the square's slice of the shared table (starting at `offset`).
    /// Candidate magics are drawn from the sequence seeded by `seed` until
    /// one maps every subset to a slot without a harmful collision, i.e. no
    /// two subsets with different attack sets land in the same slot.
    ///
    /// Candidates are the AND of three random numbers, which leaves about one
    /// bit in eight set; sparse multipliers are far more likely to work. A
    /// candidate is rejected early if it moves fewer than six mask bits into
    /// the top byte, since it could not spread the indices well enough.
    #[cfg(not(target_feature = "bmi2"))]
    fn fit(
        mask: Bitboard,
        offset: usize,
        subsets: &[(Bitboard, Bitboard)],
        slots: &mut [Bitboard],
        seed: &mut u64,
    ) -> SliderEntry {
        use crate::rng::splitmix64;

        // `stamp[slot]` records the attempt that last wrote the slot, so the
        // table does not have to be cleared between attempts.
        let mut stamp = vec![0u32; slots.len()];
        let mut attempt = 0;
        loop {
            let magic = splitmix64(seed) & splitmix64(seed) & splitmix64(seed);
            if (mask.0.wrapping_mul(magic) >> 56).count_ones() < 6 {
                continue;
            }
            attempt += 1;
            let entry = SliderEntry {
                mask: mask.0,
                magic,
                shift: 64 - mask.count(),
                offset,
            };
            let fits = subsets.iter().all(|&(occupied, attacks)| {
                let slot = entry.index(occupied) - offset;
                if stamp[slot] == attempt {
                    slots[slot] == attacks
                } else {
                    stamp[slot] = attempt;
                    slots[slot] = attacks;
                    true
                }
            });
            if fits {
                return entry;
            }
        }
    }
}

/// Lookup tables that are too costly to compute at compile time.
struct Tables {
    /// Per-square lookup data for rooks.
    rook: [SliderEntry; 64],
    /// Per-square lookup data for bishops.
    bishop: [SliderEntry; 64],
    /// Attack sets for both sliders, shared; each square owns one slice.
    attacks: Box<[Bitboard]>,
    /// See [`between()`].
    between: Box<[[Bitboard; 64]]>,
    /// See [`line()`].
    line: Box<[[Bitboard; 64]]>,
}

/// The tables, built on first use.
static TABLES: LazyLock<Tables> = LazyLock::new(Tables::build);

impl Tables {
    /// Seed of the magic-candidate sequence; fixed so that every build finds
    /// the same magics.
    const MAGIC_SEED: u64 = 0x4D61_6769_6353_6565;

    /// Number of slots in the shared slider table: 102,400 for rooks plus
    /// 5,248 for bishops (the sum over all squares of 2^mask-bits).
    const SLIDER_SLOTS: usize = 102_400 + 5_248;

    /// Builds every table.
    fn build() -> Tables {
        let mut seed = Tables::MAGIC_SEED;
        let mut attacks = Vec::with_capacity(Tables::SLIDER_SLOTS);
        let rook = Tables::build_slider(&ROOK_DIRECTIONS, &mut attacks, &mut seed);
        let bishop = Tables::build_slider(&BISHOP_DIRECTIONS, &mut attacks, &mut seed);

        let mut between = vec![[Bitboard::EMPTY; 64]; 64].into_boxed_slice();
        let mut line = vec![[Bitboard::EMPTY; 64]; 64].into_boxed_slice();
        let rows = between.iter_mut().zip(line.iter_mut());
        for (index, (between_row, line_row)) in rows.enumerate() {
            let from = Square::new(index as u8);
            for direction in ROOK_DIRECTIONS.into_iter().chain(BISHOP_DIRECTIONS) {
                let opposite = (-direction.0, -direction.1);
                let whole_line =
                    ray_attacks(from, Bitboard::EMPTY, &[direction, opposite]) | from.bb();
                // Walk away from `from`; `passed` is everything stepped over
                // so far, which is exactly the squares between the two ends.
                let mut passed = Bitboard::EMPTY;
                let mut current = from;
                while let Some(next) = step_from(current, direction) {
                    between_row[next.index()] = passed;
                    line_row[next.index()] = whole_line;
                    passed |= next.bb();
                    current = next;
                }
            }
        }

        Tables {
            rook,
            bishop,
            attacks: attacks.into_boxed_slice(),
            between,
            line,
        }
    }

    /// Builds the per-square entries for one slider type, appending each
    /// square's slots to `attacks`.
    fn build_slider(
        directions: &[Step],
        attacks: &mut Vec<Bitboard>,
        seed: &mut u64,
    ) -> [SliderEntry; 64] {
        std::array::from_fn(|index| {
            let square = Square::new(index as u8);
            let mask = relevant_mask(square, directions);
            let size = 1usize << mask.count();
            let offset = attacks.len();
            attacks.resize(offset + size, Bitboard::EMPTY);

            // Enumerate every subset of the mask with the carry-rippler
            // trick: `(subset - mask) & mask` steps to the next subset, and
            // returns to zero after the last one.
            let mut subsets = Vec::with_capacity(size);
            let mut subset = 0u64;
            loop {
                let occupied = Bitboard(subset);
                subsets.push((occupied, ray_attacks(square, occupied, directions)));
                subset = subset.wrapping_sub(mask.0) & mask.0;
                if subset == 0 {
                    break;
                }
            }

            SliderEntry::fit(mask, offset, &subsets, &mut attacks[offset..], seed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::splitmix64;

    fn sq(name: &str) -> Square {
        Square::parse(name).unwrap()
    }

    #[test]
    fn leaper_attack_counts() {
        assert_eq!(knight_attacks(sq("a1")).count(), 2);
        assert_eq!(knight_attacks(sq("e4")).count(), 8);
        assert_eq!(king_attacks(sq("h8")).count(), 3);
        assert_eq!(king_attacks(sq("d5")).count(), 8);
    }

    #[test]
    fn pawn_attacks_respect_colour_and_edges() {
        assert_eq!(
            pawn_attacks(Color::White, sq("e4")),
            sq("d5").bb() | sq("f5").bb()
        );
        assert_eq!(
            pawn_attacks(Color::Black, sq("e4")),
            sq("d3").bb() | sq("f3").bb()
        );
        assert_eq!(pawn_attacks(Color::White, sq("a2")), sq("b3").bb());
        assert_eq!(pawn_attacks(Color::Black, sq("h7")), sq("g6").bb());
    }

    #[test]
    fn slider_lookups_match_ray_walking() {
        let mut state = 7;
        for index in 0..64 {
            let square = Square::new(index);
            for _ in 0..200 {
                // AND-ing two draws gives a realistic, sparser occupancy.
                let occupied = Bitboard(splitmix64(&mut state) & splitmix64(&mut state));
                assert_eq!(
                    rook_attacks(square, occupied),
                    ray_attacks(square, occupied, &ROOK_DIRECTIONS)
                );
                assert_eq!(
                    bishop_attacks(square, occupied),
                    ray_attacks(square, occupied, &BISHOP_DIRECTIONS)
                );
            }
        }
    }

    #[test]
    fn slider_table_has_expected_size() {
        assert_eq!(TABLES.attacks.len(), Tables::SLIDER_SLOTS);
    }

    #[test]
    fn between_and_line() {
        assert_eq!(between(sq("a1"), sq("d4")), sq("b2").bb() | sq("c3").bb());
        assert_eq!(between(sq("e1"), sq("e2")), Bitboard::EMPTY);
        assert_eq!(between(sq("a1"), sq("b3")), Bitboard::EMPTY);
        assert_eq!(between(sq("h4"), sq("e4")), sq("f4").bb() | sq("g4").bb());
        assert_eq!(line(sq("c4"), sq("f4")), Bitboard::rank(3));
        assert_eq!(line(sq("b2"), sq("g7")).count(), 8);
        assert_eq!(line(sq("a1"), sq("b3")), Bitboard::EMPTY);
    }
}
