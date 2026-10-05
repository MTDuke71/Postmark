//! Zobrist hashing keys (BRD-4).
//!
//! A position's key is the XOR of one random number per feature present:
//! each piece on each square, the castling-rights set, the en-passant file
//! (when a capture is possible) and the side to move. Because XOR is its own
//! inverse, making a move updates the key by XOR-ing out the features that
//! disappear and XOR-ing in those that appear.

use crate::rng::splitmix64;

/// The table of random numbers from which position keys are built.
pub struct Zobrist {
    /// One key per piece (indexed by [`Piece::index`](crate::types::Piece::index))
    /// per square.
    pub piece: [[u64; 64]; 12],
    /// One key per castling-rights set, indexed by its four-bit value.
    pub castling: [u64; 16],
    /// One key per en-passant file.
    pub ep_file: [u64; 8],
    /// Key XOR-ed in when Black is to move.
    pub side: u64,
}

impl Zobrist {
    /// Seed of the key sequence. Any value works; it is fixed so that keys
    /// are the same in every build.
    const SEED: u64 = 0x506F_7374_6D61_726B;

    /// Fills the table from the SplitMix64 sequence, at compile time.
    const fn new() -> Zobrist {
        let mut state = Zobrist::SEED;
        let mut table = Zobrist {
            piece: [[0; 64]; 12],
            castling: [0; 16],
            ep_file: [0; 8],
            side: 0,
        };

        let mut piece = 0;
        while piece < 12 {
            let mut square = 0;
            while square < 64 {
                table.piece[piece][square] = splitmix64(&mut state);
                square += 1;
            }
            piece += 1;
        }

        // The empty rights set keeps key 0 so that it contributes nothing.
        let mut rights = 1;
        while rights < 16 {
            table.castling[rights] = splitmix64(&mut state);
            rights += 1;
        }

        let mut file = 0;
        while file < 8 {
            table.ep_file[file] = splitmix64(&mut state);
            file += 1;
        }

        table.side = splitmix64(&mut state);
        table
    }
}

/// The engine's Zobrist keys.
pub static ZOBRIST: Zobrist = Zobrist::new();
