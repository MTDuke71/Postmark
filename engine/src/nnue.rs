//! The neural-network evaluator (EVL-4, EVL-6): an efficiently updatable
//! neural network, "NNUE".
//!
//! The network is the smallest of the classic designs. Its input is 768
//! yes/no features, one per (colour, piece kind, square): a position turns
//! on at most 32 of them. A first layer of weights maps those features to a
//! hidden vector of `H` sums, the *accumulator*; a clipped ReLU squashes the
//! sums; and one output neuron combines them into a score. The board is fed
//! to the first layer twice, once from each side's point of view (the
//! black view flips the board and swaps the colours), and the output layer
//! sees the side to move's view first. That ordering is how the network
//! knows whose turn it is.
//!
//! What makes it fast is that the accumulator is a plain sum over the
//! active features, so a move changes it by only the few features it turns
//! off and on. The evaluator keeps one accumulator per ply on the search
//! path and derives each from its parent with those few additions and
//! subtractions; unmaking a move is dropping the newest entry.
//!
//! The file format is the layout shared with Huginn and FableR, so that
//! their networks run here and Postmark's run there (EVL-6). Postmark's
//! writer puts a small header in front; the reader also accepts the bare
//! form those engines write, and infers the hidden size from the length.
//!
//! All arithmetic is in integers. The first layer's weights and biases are
//! the trained values times 256, so the accumulator counts in 256ths; the
//! clipped ReLU is therefore `clamp(0, 256)`. The output weights are also
//! times 256 and the output bias times 256 × 256, so the final sum is in
//! 65536ths of the network's output unit, which is a sixteenth of a pawn.
//! Dividing by 4096 (= 65536 / 16) gives centipawns.

use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::eval::Evaluator;
use crate::moves::Move;
use crate::position::Position;
use crate::types::{Color, Piece, PieceKind, Square};

/// Number of input features: 2 colours × 6 piece kinds × 64 squares.
pub const FEATURES: usize = 768;

/// Upper limit of the clipped ReLU, in accumulator units (256ths).
const CLIP: i32 = 256;

/// Divisor that turns the output sum into centipawns (see the module
/// documentation).
const OUTPUT_DIVISOR: i32 = 4096;

/// The first bytes of a network file written by Postmark.
const MAGIC: [u8; 4] = *b"PMNN";

/// The only header version this build reads and writes.
const VERSION: u32 = 1;

/// Length of Postmark's header: magic, version and hidden size.
const HEADER_LEN: usize = 12;

/// Hidden sizes must be multiples of this, so that the vectorised paths
/// never need a remainder loop.
const HIDDEN_MULTIPLE: usize = 16;

/// Largest hidden size accepted; above this the accumulator stack would be
/// unreasonably large.
const MAX_HIDDEN: usize = 4096;

/// Why a network could not be loaded.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be read.
    Io(io::Error),
    /// The header names a version this build does not understand.
    Version(u32),
    /// The hidden size in the header, or implied by the length of a bare
    /// file, is not a multiple of 16 between 16 and 4096.
    Hidden(usize),
    /// The data is not the length the layout requires.
    Length(usize),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(error) => write!(f, "{error}"),
            LoadError::Version(version) => write!(f, "unsupported network version {version}"),
            LoadError::Hidden(hidden) => write!(f, "unsupported hidden size {hidden}"),
            LoadError::Length(len) => write!(f, "{len} bytes is not a valid network"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(error: io::Error) -> LoadError {
        LoadError::Io(error)
    }
}

/// A trained network, ready to evaluate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Network {
    /// Hidden size `H`: the length of each perspective's accumulator.
    hidden: usize,
    /// First-layer weights, `FEATURES` rows of `H`: row `f` is what feature
    /// `f` adds to the accumulator.
    w1: Box<[i16]>,
    /// First-layer bias: the accumulator of an empty board.
    b1: Box<[i16]>,
    /// Output weights, `2 × H`: the side to move's view first.
    w2: Box<[i16]>,
    /// Output bias.
    b2: i32,
}

impl Network {
    /// Builds a network from its parts. `w1` must hold `FEATURES × hidden`
    /// values, `b1` `hidden` and `w2` `2 × hidden`, and `hidden` must be
    /// acceptable to [`Network::from_bytes`].
    ///
    /// # Panics
    ///
    /// Panics if the lengths do not match `hidden`.
    pub fn new(hidden: usize, w1: Vec<i16>, b1: Vec<i16>, w2: Vec<i16>, b2: i32) -> Network {
        assert!(check_hidden(hidden).is_ok(), "hidden size {hidden}");
        assert_eq!(w1.len(), FEATURES * hidden);
        assert_eq!(b1.len(), hidden);
        assert_eq!(w2.len(), 2 * hidden);
        Network {
            hidden,
            w1: w1.into_boxed_slice(),
            b1: b1.into_boxed_slice(),
            w2: w2.into_boxed_slice(),
            b2,
        }
    }

    /// Returns the hidden size.
    #[inline]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Number of bytes the weights take in a file, excluding any header.
    const fn payload_len(hidden: usize) -> usize {
        2 * (FEATURES * hidden + hidden + 2 * hidden) + 4
    }

    /// Reads a network from the file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a network in
    /// either accepted form (see [`Network::from_bytes`]).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Network, LoadError> {
        Network::from_bytes(&std::fs::read(path)?)
    }

    /// Parses a network from the contents of a file.
    ///
    /// Two forms are accepted. Postmark's own starts with a 12-byte header:
    /// the magic `PMNN`, a little-endian `u32` version (1) and a
    /// little-endian `u32` hidden size. The bare form written by Huginn
    /// and FableR has no header; its hidden size is whatever makes the
    /// length come out right. Both are followed by the weights in order:
    /// `w1` (`768 × H` little-endian `i16`, feature-major), `b1` (`H`
    /// `i16`), `w2` (`2 × H` `i16`) and `b2` (one `i32`).
    ///
    /// # Errors
    ///
    /// Returns an error if the version is unknown, the hidden size is not a
    /// multiple of 16 between 16 and 4096, or the length does not match.
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, LoadError> {
        let (hidden, payload) = if bytes.starts_with(&MAGIC) {
            if bytes.len() < HEADER_LEN {
                return Err(LoadError::Length(bytes.len()));
            }
            let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if version != VERSION {
                return Err(LoadError::Version(version));
            }
            let hidden = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (hidden, &bytes[HEADER_LEN..])
        } else {
            // Solve payload_len(hidden) == len for hidden.
            let hidden = bytes.len().saturating_sub(4) / (2 * (FEATURES + 1 + 2));
            (hidden, bytes)
        };
        check_hidden(hidden)?;
        if payload.len() != Network::payload_len(hidden) {
            return Err(LoadError::Length(bytes.len()));
        }

        let mut words = payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i16::from_le_bytes(pair));
        let mut take = |n: usize| -> Vec<i16> { words.by_ref().take(n).collect() };
        let w1 = take(FEATURES * hidden);
        let b1 = take(hidden);
        let w2 = take(2 * hidden);
        let tail = &payload[payload.len() - 4..];
        let b2 = i32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        Ok(Network::new(hidden, w1, b1, w2, b2))
    }

    /// Serialises the network in Postmark's form, header included.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LEN + Network::payload_len(self.hidden));
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        for &word in self.w1.iter().chain(self.b1.iter()).chain(self.w2.iter()) {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&self.b2.to_le_bytes());
        bytes
    }

    /// Returns the first-layer weights of feature `feature`.
    #[inline]
    fn row(&self, feature: usize) -> &[i16] {
        &self.w1[feature * self.hidden..(feature + 1) * self.hidden]
    }
}

/// Checks that `hidden` is a size the evaluator supports.
fn check_hidden(hidden: usize) -> Result<(), LoadError> {
    if hidden == 0 || hidden > MAX_HIDDEN || !hidden.is_multiple_of(HIDDEN_MULTIPLE) {
        Err(LoadError::Hidden(hidden))
    } else {
        Ok(())
    }
}

/// Returns the index of the feature "`piece` stands on `square`" as seen
/// from `perspective`.
///
/// From White's point of view the board is taken as it is. From Black's
/// the ranks are mirrored (`square ^ 56`) and the colours swapped, so that
/// Black looking at its own position sees exactly the pattern White would
/// see looking at the mirror image. One set of first-layer weights then
/// serves both sides.
#[inline]
fn feature(perspective: Color, piece: Piece, square: Square) -> usize {
    let (color, square) = match perspective {
        Color::White => (piece.color().index(), square.index()),
        Color::Black => ((!piece.color()).index(), square.index() ^ 56),
    };
    (color * 6 + piece.kind().index()) * 64 + square
}

/// The network evaluator: accumulators for every position on the current
/// search path, updated incrementally (EVL-4).
#[derive(Clone, Debug)]
pub struct NnueEvaluator {
    /// The network.
    network: Arc<Network>,
    /// Accumulators for each ply, one pair per entry: White's view then
    /// Black's, each `H` long. Only the first `(top + 1) × stride` values
    /// are in use.
    stack: Vec<i16>,
    /// Index of the entry for the current position.
    top: usize,
}

impl NnueEvaluator {
    /// Plies of accumulators allocated up front; the stack grows if a search
    /// goes deeper.
    const INITIAL_PLIES: usize = 256;

    /// Creates an evaluator following `position`.
    pub fn new(network: Arc<Network>, position: &Position) -> NnueEvaluator {
        let stride = 2 * network.hidden();
        let mut evaluator = NnueEvaluator {
            network,
            stack: vec![0; NnueEvaluator::INITIAL_PLIES * stride],
            top: 0,
        };
        evaluator.refresh(position);
        evaluator
    }

    /// Returns the network.
    pub fn network(&self) -> &Arc<Network> {
        &self.network
    }

    /// Length of one stack entry: both perspectives' accumulators.
    #[inline]
    fn stride(&self) -> usize {
        2 * self.network.hidden()
    }

    /// Returns `true` if the current accumulators equal those computed from
    /// scratch for `position`: the check behind the debug assertion in
    /// [`Evaluator::evaluate`] and the incremental-update test, which runs
    /// in release builds too.
    fn matches(&self, position: &Position) -> bool {
        let mut fresh = vec![0i16; self.stride()];
        NnueEvaluator::compute(&self.network, position, &mut fresh);
        self.current() == &fresh[..]
    }

    /// Returns the accumulators of the current position.
    #[inline]
    fn current(&self) -> &[i16] {
        let stride = self.stride();
        &self.stack[self.top * stride..(self.top + 1) * stride]
    }

    /// Fills `accumulators` (both perspectives) with the first-layer sums of
    /// `position`, computed from scratch.
    fn compute(network: &Network, position: &Position, accumulators: &mut [i16]) {
        let hidden = network.hidden();
        let (white, black) = accumulators.split_at_mut(hidden);
        white.copy_from_slice(&network.b1);
        black.copy_from_slice(&network.b1);
        for square in position.occupied() {
            if let Some(piece) = position.piece_on(square) {
                add(white, network.row(feature(Color::White, piece, square)));
                add(black, network.row(feature(Color::Black, piece, square)));
            }
        }
    }
}

/// The pieces a move puts down and picks up. At most two of each: castling
/// places the king and the rook; a capture or a promotion removes two
/// pieces (the mover or the pawn, and the victim).
struct Placements {
    /// Pieces arriving on squares.
    on: [(Piece, Square); 2],
    /// How many entries of `on` are used.
    on_len: usize,
    /// Pieces leaving squares.
    off: [(Piece, Square); 2],
    /// How many entries of `off` are used.
    off_len: usize,
}

impl Placements {
    /// No placements; the unused entries hold a placeholder.
    const NONE: Placements = Placements {
        on: [(Piece::WhitePawn, Square::new(0)); 2],
        on_len: 0,
        off: [(Piece::WhitePawn, Square::new(0)); 2],
        off_len: 0,
    };

    /// Records that `piece` arrives on `square`.
    #[inline]
    fn place(&mut self, piece: Piece, square: Square) {
        self.on[self.on_len] = (piece, square);
        self.on_len += 1;
    }

    /// Records that `piece` leaves `square`.
    #[inline]
    fn lift(&mut self, piece: Piece, square: Square) {
        self.off[self.off_len] = (piece, square);
        self.off_len += 1;
    }

    /// Returns the placements of `mv` in `position`.
    fn of(position: &Position, mv: Move) -> Placements {
        let mut placements = Placements::NONE;
        let us = position.side_to_move();
        let (from, to) = (mv.from(), mv.to());

        if let Some(moved) = position.piece_on(from) {
            placements.lift(moved, from);
            let placed = match mv.promotion() {
                Some(kind) => Piece::new(us, kind),
                None => moved,
            };
            placements.place(placed, to);
        }

        if mv.is_en_passant() {
            let victim = Position::en_passant_victim(to);
            placements.lift(Piece::new(!us, PieceKind::Pawn), victim);
        } else if mv.is_capture()
            && let Some(captured) = position.piece_on(to)
        {
            placements.lift(captured, to);
        }

        if mv.is_castle() {
            let (rook_from, rook_to) = Position::castling_rook_squares(mv, to);
            let rook = Piece::new(us, PieceKind::Rook);
            placements.lift(rook, rook_from);
            placements.place(rook, rook_to);
        }
        placements
    }
}

/// Adds `row` to `accumulator` element-wise.
#[inline]
fn add(accumulator: &mut [i16], row: &[i16]) {
    for (sum, &weight) in accumulator.iter_mut().zip(row) {
        *sum = sum.wrapping_add(weight);
    }
}

/// Subtracts `row` from `accumulator` element-wise.
#[inline]
fn sub(accumulator: &mut [i16], row: &[i16]) {
    for (sum, &weight) in accumulator.iter_mut().zip(row) {
        *sum = sum.wrapping_sub(weight);
    }
}

/// Writes `parent` plus the `on` rows minus the `off` rows into `child`.
///
/// The three shapes a chess move can take (a quiet move, a capture or
/// promotion, and castling) each get a single fused loop, so that the
/// child is read and written once rather than copied and then adjusted
/// row by row; the compiler vectorises these loops. Anything else falls
/// back to the row-by-row form.
#[inline]
fn update(parent: &[i16], child: &mut [i16], on: &[&[i16]], off: &[&[i16]]) {
    match (on, off) {
        ([a], [s]) => {
            for (((sum, &p), &a), &s) in child.iter_mut().zip(parent).zip(*a).zip(*s) {
                *sum = p.wrapping_add(a).wrapping_sub(s);
            }
        }
        ([a], [s, t]) => {
            for ((((sum, &p), &a), &s), &t) in child.iter_mut().zip(parent).zip(*a).zip(*s).zip(*t)
            {
                *sum = p.wrapping_add(a).wrapping_sub(s).wrapping_sub(t);
            }
        }
        ([a, b], [s, t]) => {
            for (((((sum, &p), &a), &b), &s), &t) in
                child.iter_mut().zip(parent).zip(*a).zip(*b).zip(*s).zip(*t)
            {
                *sum = p
                    .wrapping_add(a)
                    .wrapping_add(b)
                    .wrapping_sub(s)
                    .wrapping_sub(t);
            }
        }
        _ => {
            child.copy_from_slice(parent);
            for row in on {
                add(child, row);
            }
            for row in off {
                sub(child, row);
            }
        }
    }
}

/// Returns the output of the network for the accumulators `us` (side to
/// move) and `them`, in centipawns for the side to move.
///
/// Each accumulator value is clipped to `0..=256` and multiplied by its
/// output weight; the sum, plus the output bias, is scaled to centipawns.
/// The sum is taken in 32 bits, which the quantised weights cannot
/// overflow: 512 products of at most 256 × 32767 fit with room to spare.
///
/// This is the portable version, used by the `generic` build. The `v3`
/// build uses [`forward`] below, which computes the same sum with AVX2;
/// integer addition is exact, so the two agree bit for bit.
#[cfg(not(target_feature = "avx2"))]
#[inline]
fn forward(network: &Network, us: &[i16], them: &[i16]) -> i32 {
    let hidden = network.hidden();
    let (w_us, w_them) = network.w2.split_at(hidden);
    let mut sum = 0i32;
    for (&value, &weight) in us.iter().zip(w_us) {
        sum += (value as i32).clamp(0, CLIP) * weight as i32;
    }
    for (&value, &weight) in them.iter().zip(w_them) {
        sum += (value as i32).clamp(0, CLIP) * weight as i32;
    }
    (sum + network.b2) / OUTPUT_DIVISOR
}

/// The AVX2 version of the output layer: see the portable [`forward`] for
/// what it computes.
///
/// Sixteen accumulator values are clipped at a time with a vector min and
/// max, then `madd` multiplies each by its weight and adds the products in
/// pairs into eight 32-bit lanes, which are summed at the end. The pair
/// sums cannot overflow: each product is at most 256 × 32767.
#[cfg(target_feature = "avx2")]
#[inline]
fn forward(network: &Network, us: &[i16], them: &[i16]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm256_add_epi32,
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_max_epi16, _mm256_min_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
    };

    /// Values per 256-bit vector.
    const LANES: usize = 16;

    let hidden = network.hidden();
    let (w_us, w_them) = network.w2.split_at(hidden);
    debug_assert!(hidden.is_multiple_of(LANES));
    debug_assert!(us.len() == hidden && them.len() == hidden);

    // SAFETY: this function is compiled only when the `avx2` target
    // feature is enabled, so the intrinsics are always available. The
    // loads are unaligned loads of 16 values at offsets below `hidden`,
    // which is a multiple of 16 (checked by `check_hidden` when the
    // network was built), from slices that are each `hidden` long (checked
    // by `Network::new` for the weights and asserted above for the
    // accumulators).
    unsafe {
        let zero = _mm256_setzero_si256();
        let clip = _mm256_set1_epi16(CLIP as i16);
        let mut sum = _mm256_setzero_si256();
        for (values, weights) in [(us, w_us), (them, w_them)] {
            for offset in (0..hidden).step_by(LANES) {
                let value = _mm256_loadu_si256(values.as_ptr().add(offset) as *const __m256i);
                let weight = _mm256_loadu_si256(weights.as_ptr().add(offset) as *const __m256i);
                let clipped = _mm256_min_epi16(_mm256_max_epi16(value, zero), clip);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(clipped, weight));
            }
        }
        // Fold the eight lanes: upper half onto lower, then pairs, then
        // neighbours.
        let halves = _mm_add_epi32(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256(sum, 1),
        );
        let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32(halves, 0b01_00_11_10));
        let total = _mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0b10_11_00_01));
        (_mm_cvtsi128_si32(total) + network.b2) / OUTPUT_DIVISOR
    }
}

impl Evaluator for NnueEvaluator {
    fn refresh(&mut self, position: &Position) {
        self.top = 0;
        let stride = self.stride();
        NnueEvaluator::compute(&self.network, position, &mut self.stack[..stride]);
    }

    fn make_move(&mut self, position: &Position, mv: Move) {
        let placements = Placements::of(position, mv);
        let network = &self.network;
        let hidden = network.hidden();
        let stride = 2 * hidden;
        if (self.top + 2) * stride > self.stack.len() {
            self.stack.resize((self.top + 2) * stride, 0);
        }
        let (parent, child) =
            self.stack[self.top * stride..(self.top + 2) * stride].split_at_mut(stride);
        self.top += 1;

        // Each perspective's accumulator is the parent's plus the rows of
        // the arriving pieces minus the rows of the leaving ones.
        for (perspective, offset) in [(Color::White, 0), (Color::Black, hidden)] {
            let row = |(piece, square)| network.row(feature(perspective, piece, square));
            let on = placements.on.map(row);
            let off = placements.off.map(row);
            update(
                &parent[offset..offset + hidden],
                &mut child[offset..offset + hidden],
                &on[..placements.on_len],
                &off[..placements.off_len],
            );
        }
    }

    fn unmake_move(&mut self) {
        debug_assert!(self.top > 0);
        self.top -= 1;
    }

    fn evaluate(&self, position: &Position) -> i32 {
        let hidden = self.network.hidden();
        let entry = self.current();
        debug_assert!(
            self.matches(position),
            "accumulator drifted from the position"
        );
        let (white, black) = entry.split_at(hidden);
        match position.side_to_move() {
            Color::White => forward(&self.network, white, black),
            Color::Black => forward(&self.network, black, white),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::{GenKind, generate};
    use crate::moves::MoveList;
    use crate::rng::splitmix64;

    /// Builds a network of hidden size `hidden` with small pseudo-random
    /// weights, so that the tests exercise real arithmetic.
    fn random_network(hidden: usize, seed: u64) -> Arc<Network> {
        let mut state = seed;
        let mut next =
            |range: i64| (splitmix64(&mut state) % (2 * range as u64 + 1)) as i64 - range;
        let w1 = (0..FEATURES * hidden).map(|_| next(40) as i16).collect();
        let b1 = (0..hidden).map(|_| next(100) as i16).collect();
        let w2 = (0..2 * hidden).map(|_| next(300) as i16).collect();
        let b2 = next(5000) as i32;
        Arc::new(Network::new(hidden, w1, b1, w2, b2))
    }

    fn evaluate(network: &Arc<Network>, fen: &str) -> i32 {
        let position = Position::from_fen(fen).unwrap();
        NnueEvaluator::new(Arc::clone(network), &position).evaluate(&position)
    }

    #[test]
    fn feature_index_matches_the_shared_layout() {
        // A white knight on f3 (square 21) from White's view.
        let knight = Piece::WhiteKnight;
        let f3 = Square::parse("f3").unwrap();
        assert_eq!(feature(Color::White, knight, f3), (0 * 6 + 1) * 64 + 21);
        // From Black's view it is an enemy knight on f6 (square 45).
        assert_eq!(feature(Color::Black, knight, f3), (1 * 6 + 1) * 64 + 45);
        // A black king on e8 is White's enemy king on 60, Black's own king on e1.
        let e8 = Square::parse("e8").unwrap();
        assert_eq!(
            feature(Color::White, Piece::BlackKing, e8),
            (1 * 6 + 5) * 64 + 60
        );
        assert_eq!(
            feature(Color::Black, Piece::BlackKing, e8),
            (0 * 6 + 5) * 64 + 4
        );
    }

    #[test]
    fn file_round_trips_in_both_forms() {
        let network = random_network(32, 7);
        let with_header = network.to_bytes();
        assert_eq!(with_header.len(), HEADER_LEN + Network::payload_len(32));
        assert_eq!(Network::from_bytes(&with_header).unwrap(), *network);
        assert_eq!(
            Network::from_bytes(&with_header[HEADER_LEN..]).unwrap(),
            *network
        );
        let on_disk = Network::from_bytes(&with_header[HEADER_LEN..]).unwrap();
        assert_eq!(on_disk.hidden(), 32);
    }

    #[test]
    fn malformed_files_are_rejected() {
        let network = random_network(16, 3);
        let mut bytes = network.to_bytes();
        bytes[4] = 9;
        assert!(matches!(
            Network::from_bytes(&bytes),
            Err(LoadError::Version(9))
        ));
        let mut bytes = network.to_bytes();
        bytes[8] = 17;
        assert!(matches!(
            Network::from_bytes(&bytes),
            Err(LoadError::Hidden(17))
        ));
        let mut bytes = network.to_bytes();
        bytes.pop();
        assert!(matches!(
            Network::from_bytes(&bytes),
            Err(LoadError::Length(_))
        ));
        assert!(matches!(
            Network::from_bytes(b"PMNN"),
            Err(LoadError::Length(4))
        ));
        assert!(matches!(
            Network::from_bytes(&[0; 100]),
            Err(LoadError::Hidden(0))
        ));
        let mut bare = network.to_bytes()[HEADER_LEN..].to_vec();
        bare.push(0);
        assert!(matches!(
            Network::from_bytes(&bare),
            Err(LoadError::Length(_))
        ));
        assert!(Network::from_file("no/such/file.nnue").is_err());
    }

    #[test]
    fn mirrored_positions_score_the_same() {
        // The second position is the first with colours and ranks swapped,
        // so the network must see identical features from the mover's view.
        let network = random_network(16, 11);
        let a = evaluate(
            &network,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        );
        let b = evaluate(
            &network,
            "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
        );
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    #[test]
    fn empty_board_scores_the_bias_only() {
        // Two bare kings: only two features are on. Check the arithmetic by hand.
        let network = random_network(16, 5);
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let evaluator = NnueEvaluator::new(Arc::clone(&network), &position);
        let mut white = network.b1.to_vec();
        let mut black = network.b1.to_vec();
        for (piece, square) in [(Piece::WhiteKing, "e1"), (Piece::BlackKing, "e8")] {
            let square = Square::parse(square).unwrap();
            add(
                &mut white,
                network.row(feature(Color::White, piece, square)),
            );
            add(
                &mut black,
                network.row(feature(Color::Black, piece, square)),
            );
        }
        let expected = forward(&network, &white, &black);
        assert_eq!(evaluator.evaluate(&position), expected);
    }

    /// Walks the move tree, comparing the incremental accumulators with a
    /// from-scratch computation at every node.
    fn walk(position: &mut Position, evaluator: &mut NnueEvaluator, depth: u32) {
        assert!(evaluator.matches(position), "{}", position.to_fen());
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
        let network = random_network(16, 23);
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        ];
        for fen in fens {
            let mut position = Position::from_fen(fen).unwrap();
            let mut evaluator = NnueEvaluator::new(Arc::clone(&network), &position);
            walk(&mut position, &mut evaluator, 3);
            // The walk must leave the evaluator where it started.
            let fresh = NnueEvaluator::new(Arc::clone(&network), &position);
            assert_eq!(evaluator.evaluate(&position), fresh.evaluate(&position));
        }
    }

    #[test]
    fn stack_grows_past_its_initial_size() {
        let network = random_network(16, 1);
        let mut position = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mut evaluator = NnueEvaluator::new(Arc::clone(&network), &position);
        let mut moves = Vec::new();
        for _ in 0..NnueEvaluator::INITIAL_PLIES + 20 {
            let mut list = MoveList::new();
            generate(&position, GenKind::All, &mut list);
            let mv = *list.iter().find(|mv| !mv.is_capture()).unwrap();
            evaluator.make_move(&position, mv);
            position.make_move(mv);
            moves.push(mv);
        }
        evaluator.evaluate(&position);
        for mv in moves.into_iter().rev() {
            position.unmake_move(mv);
            evaluator.unmake_move();
        }
        let fresh = NnueEvaluator::new(network, &position);
        assert_eq!(evaluator.evaluate(&position), fresh.evaluate(&position));
    }
}
