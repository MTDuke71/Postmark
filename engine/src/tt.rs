//! The transposition table: a fixed-size hash table of search results,
//! shared by all search threads without locks (SRC-8).

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::moves::Move;
use crate::search::{MATE_BOUND, MAX_PLY};

/// What a stored score says about the true value of the position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    /// The slot is empty.
    None = 0,
    /// The true score is at most the stored score (no move beat alpha).
    Upper = 1,
    /// The true score is at least the stored score (a move caused a cutoff).
    Lower = 2,
    /// The stored score is exact.
    Exact = 3,
}

/// A search result read from the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TtEntry {
    /// Best move found, or [`Move::NULL`] if none was recorded. It may not
    /// be legal in the probing position (see [`TranspositionTable::probe`]).
    pub mv: Move,
    /// Score, already converted to the probing node's distance from the root.
    pub score: i32,
    /// Depth the position was searched to.
    pub depth: i32,
    /// Meaning of `score`.
    pub bound: Bound,
}

/// One table slot: two 64-bit words.
///
/// `data` packs the entry (see [`pack`]); `check` holds the position's key
/// XOR-ed with `data`. Threads write the two words without any lock, so a
/// reader can see the first word from one write and the second from
/// another. XOR-ing the key with the data makes that detectable: a torn
/// slot fails the `check ^ data == key` test (except with negligible
/// probability) and is treated as a miss, so no corrupt entry is ever used.
/// This is why plain atomic loads and stores, with no ordering guarantees
/// between them, are sufficient.
#[derive(Default)]
struct Slot {
    /// Position key XOR `data`.
    check: AtomicU64,
    /// Packed entry.
    data: AtomicU64,
}

/// Number of bits of the search generation kept in each entry.
const GENERATION_BITS: u32 = 6;

/// Packs an entry into 64 bits:
///
/// | Bits  | Field                    |
/// |-------|--------------------------|
/// | 0-15  | move                     |
/// | 16-31 | score (signed 16-bit)    |
/// | 32-39 | depth (0-255)            |
/// | 40-41 | bound                    |
/// | 42-47 | generation               |
fn pack(mv: Move, score: i32, depth: i32, bound: Bound, generation: u8) -> u64 {
    mv.raw() as u64
        | (score as i16 as u16 as u64) << 16
        | (depth.clamp(0, 255) as u64) << 32
        | (bound as u64) << 40
        | (generation as u64) << 42
}

/// Extracts the bound from packed data.
fn bound_of(data: u64) -> Bound {
    match data >> 40 & 3 {
        1 => Bound::Upper,
        2 => Bound::Lower,
        3 => Bound::Exact,
        _ => Bound::None,
    }
}

/// Extracts the depth from packed data.
fn depth_of(data: u64) -> i32 {
    (data >> 32 & 0xFF) as i32
}

/// Extracts the generation from packed data.
fn generation_of(data: u64) -> u8 {
    (data >> 42) as u8 & ((1 << GENERATION_BITS) - 1)
}

/// Converts a score from "relative to the root" to "relative to this node"
/// for storage.
///
/// Mate scores encode the distance to mate from the root. The same position
/// can be reached at different distances from the root, so a mate score is
/// stored as the distance from the *stored node* instead, by adding the
/// node's ply back on, and converted again on retrieval.
fn score_to_table(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

/// Converts a stored score back to "relative to the root" for a node at
/// `ply`; the inverse of [`score_to_table`].
fn score_from_table(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

/// The transposition table.
pub struct TranspositionTable {
    /// The slots; a position's slot is chosen from its key.
    slots: Vec<Slot>,
    /// Counter identifying the current search, used to prefer replacing
    /// entries left over from earlier searches.
    generation: AtomicU8,
}

impl TranspositionTable {
    /// Creates a table of about `megabytes` MB, with every slot empty.
    ///
    /// Returns `None` if that much memory cannot be allocated.
    pub fn new(megabytes: usize) -> Option<TranspositionTable> {
        let bytes = megabytes.max(1).checked_mul(1024 * 1024)?;
        let count = bytes / std::mem::size_of::<Slot>();
        let mut slots = Vec::new();
        slots.try_reserve_exact(count).ok()?;
        slots.resize_with(count, Slot::default);
        Some(TranspositionTable {
            slots,
            generation: AtomicU8::new(0),
        })
    }

    /// Returns the slot for `key`.
    ///
    /// The key is mapped to an index by taking the high half of the 128-bit
    /// product `key * len`. This spreads keys evenly over any table length,
    /// not just powers of two, and is cheaper than a remainder.
    #[inline]
    fn slot(&self, key: u64) -> &Slot {
        let index = ((key as u128 * self.slots.len() as u128) >> 64) as usize;
        &self.slots[index]
    }

    /// Empties the table.
    pub fn clear(&self) {
        for slot in &self.slots {
            slot.check.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Marks the start of a new search, ageing all existing entries.
    pub fn new_search(&self) {
        let next = (self.generation.load(Ordering::Relaxed) + 1) & ((1 << GENERATION_BITS) - 1);
        self.generation.store(next, Ordering::Relaxed);
    }

    /// Looks up the position with the given `key`, for a node `ply` moves
    /// from the root.
    ///
    /// A returned entry belongs to this position unless two positions share
    /// both a slot and a full 64-bit key, which is rare but possible. The
    /// caller must therefore check that the entry's move is legal in its
    /// position before playing it (SRC-8).
    #[inline]
    pub fn probe(&self, key: u64, ply: usize) -> Option<TtEntry> {
        let slot = self.slot(key);
        let data = slot.data.load(Ordering::Relaxed);
        let check = slot.check.load(Ordering::Relaxed);
        let bound = bound_of(data);
        if check ^ data != key || bound == Bound::None {
            return None;
        }
        Some(TtEntry {
            mv: Move::from_raw(data as u16),
            score: score_from_table((data >> 16) as u16 as i16 as i32, ply),
            depth: depth_of(data),
            bound,
        })
    }

    /// Records a search result for the position with the given `key`,
    /// reached `ply` moves from the root.
    ///
    /// Each position has exactly one slot, so storing may evict another
    /// position. The existing entry is kept only if it is for a *different*
    /// position, was written during the current search, and was searched
    /// deeper than the new result: such an entry saves more work than the
    /// one that would replace it. When the slot already holds this position
    /// and the new result has no best move, the old move is preserved.
    #[inline]
    pub fn store(&self, key: u64, mv: Move, score: i32, depth: i32, bound: Bound, ply: usize) {
        debug_assert!(ply <= MAX_PLY);
        let slot = self.slot(key);
        let generation = self.generation.load(Ordering::Relaxed);
        let old_data = slot.data.load(Ordering::Relaxed);
        let old_check = slot.check.load(Ordering::Relaxed);
        let same_position = old_check ^ old_data == key;

        if !same_position
            && bound_of(old_data) != Bound::None
            && generation_of(old_data) == generation
            && depth_of(old_data) > depth
        {
            return;
        }

        let mv = if mv == Move::NULL && same_position {
            Move::from_raw(old_data as u16)
        } else {
            mv
        };
        let data = pack(mv, score_to_table(score, ply), depth, bound, generation);
        slot.data.store(data, Ordering::Relaxed);
        slot.check.store(key ^ data, Ordering::Relaxed);
    }

    /// Returns how full the table is, in parts per thousand, counting only
    /// entries written during the current search. The figure is estimated
    /// from the first thousand slots, as the UCI protocol suggests.
    pub fn hashfull(&self) -> usize {
        let generation = self.generation.load(Ordering::Relaxed);
        let sample = &self.slots[..self.slots.len().min(1000)];
        let used = sample
            .iter()
            .map(|slot| slot.data.load(Ordering::Relaxed))
            .filter(|&data| bound_of(data) != Bound::None && generation_of(data) == generation)
            .count();
        used * 1000 / sample.len().max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::MATE;
    use crate::types::Square;

    fn mv() -> Move {
        Move::new(Square::new(12), Square::new(28), Move::DOUBLE_PUSH)
    }

    #[test]
    fn stored_entries_are_found() {
        let table = TranspositionTable::new(1).unwrap();
        assert_eq!(table.probe(42, 0), None);
        table.store(42, mv(), -123, 7, Bound::Lower, 0);
        let entry = table.probe(42, 0).unwrap();
        assert_eq!(
            entry,
            TtEntry {
                mv: mv(),
                score: -123,
                depth: 7,
                bound: Bound::Lower
            }
        );
        assert_eq!(table.probe(43, 0), None);
        table.clear();
        assert_eq!(table.probe(42, 0), None);
    }

    #[test]
    fn mate_scores_are_relative_to_the_probing_node() {
        let table = TranspositionTable::new(1).unwrap();
        // Mate found 3 plies below a node that is itself 5 plies from the root.
        table.store(7, mv(), MATE - 8, 3, Bound::Exact, 5);
        // Probed from a node 2 plies from the root, the mate is 3 plies away.
        assert_eq!(table.probe(7, 2).unwrap().score, MATE - 5);
        table.store(9, mv(), -(MATE - 8), 3, Bound::Exact, 5);
        assert_eq!(table.probe(9, 2).unwrap().score, -(MATE - 5));
    }

    #[test]
    fn a_missing_move_does_not_erase_a_known_one() {
        let table = TranspositionTable::new(1).unwrap();
        table.store(5, mv(), 10, 4, Bound::Exact, 0);
        table.store(5, Move::NULL, -20, 6, Bound::Upper, 0);
        let entry = table.probe(5, 0).unwrap();
        assert_eq!((entry.mv, entry.score, entry.depth), (mv(), -20, 6));
    }

    #[test]
    fn hashfull_counts_only_the_current_search() {
        let table = TranspositionTable::new(1).unwrap();
        assert_eq!(table.hashfull(), 0);
        // A key near zero maps to the first slot, which is in the sample.
        table.store(1, mv(), 0, 1, Bound::Exact, 0);
        assert_eq!(table.hashfull(), 1);
        table.new_search();
        assert_eq!(table.hashfull(), 0);
        assert!(table.probe(1, 0).is_some());
    }
}
