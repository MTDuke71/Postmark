//! Deterministic pseudo-random numbers for table generation.
//!
//! The engine never needs unpredictable randomness: Zobrist keys and magic
//! multipliers only need well-mixed bits, and they must be identical on every
//! run so that searches are reproducible (PRN-6).

/// Advances `state` and returns the next value of the SplitMix64 sequence.
///
/// SplitMix64 adds a fixed odd constant to the state and then scrambles the
/// result with two multiply-xorshift rounds. Because the increment is odd the
/// state visits all 2^64 values before repeating, so every output in a run is
/// distinct.
pub const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_is_reproducible_and_varied() {
        let (mut a, mut b) = (1, 1);
        let first: Vec<u64> = (0..8).map(|_| splitmix64(&mut a)).collect();
        let second: Vec<u64> = (0..8).map(|_| splitmix64(&mut b)).collect();
        assert_eq!(first, second);
        for (i, x) in first.iter().enumerate() {
            assert!(!first[..i].contains(x));
        }
    }
}
