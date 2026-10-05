//! The chess position: piece placement, game state, FEN conversion and
//! making and unmaking moves.

use std::fmt;

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceKind, Square};
use crate::zobrist::ZOBRIST;

/// FEN of the standard starting position.
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// The set of castling moves that are still permitted, as four bits.
///
/// A right records only that the king and the relevant rook have not moved;
/// whether castling is possible right now is decided by move generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastlingRights(u8);

impl CastlingRights {
    /// No castling rights.
    pub const NONE: CastlingRights = CastlingRights(0);
    /// White may castle king-side.
    pub const WHITE_KINGSIDE: CastlingRights = CastlingRights(1);
    /// White may castle queen-side.
    pub const WHITE_QUEENSIDE: CastlingRights = CastlingRights(2);
    /// Black may castle king-side.
    pub const BLACK_KINGSIDE: CastlingRights = CastlingRights(4);
    /// Black may castle queen-side.
    pub const BLACK_QUEENSIDE: CastlingRights = CastlingRights(8);

    /// For each square, the rights that survive a move from or to it.
    ///
    /// A right is lost when its king or rook moves, or when that rook is
    /// captured on its home square. All of those involve a move touching e1,
    /// a1, h1, e8, a8 or h8, so AND-ing the rights with the entries for a
    /// move's origin and destination handles every case without branching.
    const KEEP: [u8; 64] = {
        let mut keep = [0b1111; 64];
        keep[0] = 0b1101; // a1: White queen-side
        keep[4] = 0b1100; // e1: both White rights
        keep[7] = 0b1110; // h1: White king-side
        keep[56] = 0b0111; // a8: Black queen-side
        keep[60] = 0b0011; // e8: both Black rights
        keep[63] = 0b1011; // h8: Black king-side
        keep
    };

    /// Returns the king-side right of `color`.
    #[inline]
    pub const fn kingside(color: Color) -> CastlingRights {
        match color {
            Color::White => CastlingRights::WHITE_KINGSIDE,
            Color::Black => CastlingRights::BLACK_KINGSIDE,
        }
    }

    /// Returns the queen-side right of `color`.
    #[inline]
    pub const fn queenside(color: Color) -> CastlingRights {
        match color {
            Color::White => CastlingRights::WHITE_QUEENSIDE,
            Color::Black => CastlingRights::BLACK_QUEENSIDE,
        }
    }

    /// Returns `true` if every right in `other` is present in `self`.
    #[inline]
    pub const fn contains(self, other: CastlingRights) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the rights as a four-bit table index (0-15).
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The reason a FEN string was rejected (BRD-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenError {
    /// A required field is absent; the payload names it.
    MissingField(&'static str),
    /// The piece-placement field is not eight ranks of eight squares, or
    /// contains an unknown character.
    BadPlacement,
    /// A side does not have exactly one king.
    KingCount,
    /// A pawn stands on the first or eighth rank.
    PawnOnBackRank,
    /// The side-to-move field is neither `w` nor `b`.
    BadSideToMove,
    /// The castling field is malformed, repeats a letter, or claims a right
    /// whose king or rook is not on its home square.
    BadCastling,
    /// The en-passant field is not a square, or no pawn could have just made
    /// the double push it implies.
    BadEnPassant,
    /// A move counter is not a number in the range 0-65535.
    BadCounter,
    /// There is text after the sixth field.
    TrailingData,
    /// The side that is not to move is in check, so its king could be
    /// captured.
    OpponentInCheck,
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::MissingField(name) => write!(f, "missing {name} field"),
            FenError::BadPlacement => f.write_str("malformed piece placement"),
            FenError::KingCount => f.write_str("each side must have exactly one king"),
            FenError::PawnOnBackRank => f.write_str("pawn on the first or eighth rank"),
            FenError::BadSideToMove => f.write_str("side to move must be 'w' or 'b'"),
            FenError::BadCastling => f.write_str("invalid castling rights"),
            FenError::BadEnPassant => f.write_str("invalid en passant square"),
            FenError::BadCounter => f.write_str("invalid move counter"),
            FenError::TrailingData => f.write_str("unexpected text after the last field"),
            FenError::OpponentInCheck => f.write_str("the side not to move is in check"),
        }
    }
}

impl std::error::Error for FenError {}

/// Formats the position as a text diagram, White at the bottom, followed by
/// its FEN and Zobrist key.
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const RULE: &str = "  +---+---+---+---+---+---+---+---+";
        writeln!(f, "{RULE}")?;
        for rank in (0..8).rev() {
            write!(f, "{} |", rank + 1)?;
            for file in 0..8 {
                let piece = self.piece_on(Square::from_file_rank(file, rank));
                write!(f, " {} |", piece.map_or(' ', Piece::to_char))?;
            }
            writeln!(
                f,
                "
{RULE}"
            )?;
        }
        writeln!(f, "    a   b   c   d   e   f   g   h")?;
        writeln!(f)?;
        writeln!(f, "Fen: {}", self.to_fen())?;
        write!(f, "Key: {:016X}", self.key)
    }
}

/// State that a move destroys and that cannot be recomputed when the move is
/// taken back, saved by [`Position::make_move`] (BRD-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Undo {
    /// The piece the move captured, if any.
    captured: Option<Piece>,
    /// Castling rights before the move.
    castling: CastlingRights,
    /// En-passant square before the move.
    ep_square: Option<Square>,
    /// Halfmove clock before the move.
    halfmove_clock: u16,
    /// Zobrist key before the move.
    key: u64,
}

/// A chess position together with the history needed to take moves back.
///
/// Piece placement is stored twice (BRD-1): as bitboards, for set-wise
/// questions such as "which squares do the black rooks occupy", and as a
/// mailbox, for "what stands on this square". The two are always consistent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Position {
    /// Squares occupied by each piece kind, of either colour.
    kinds: [Bitboard; 6],
    /// Squares occupied by each colour.
    colors: [Bitboard; 2],
    /// The piece on each square.
    mailbox: [Option<Piece>; 64],
    /// The side to move.
    side: Color,
    /// Remaining castling rights.
    castling: CastlingRights,
    /// The square a pawn skipped with a double push on the previous move.
    ep_square: Option<Square>,
    /// Halfmoves since the last capture or pawn move (fifty-move rule).
    halfmove_clock: u16,
    /// Number of the current full move, starting at 1.
    fullmove_number: u16,
    /// Zobrist key of the position, maintained incrementally (BRD-4).
    key: u64,
    /// One entry per move made and not yet unmade.
    history: Vec<Undo>,
}

impl Position {
    /// Returns the standard starting position.
    ///
    /// # Panics
    ///
    /// Panics only if [`START_FEN`] failed to parse, which would be a bug in
    /// this crate and is ruled out by its tests.
    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).expect("the start position FEN is valid")
    }

    /// Returns a board with no pieces, White to move.
    fn empty() -> Position {
        Position {
            kinds: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            mailbox: [None; 64],
            side: Color::White,
            castling: CastlingRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            key: 0,
            history: Vec::with_capacity(256),
        }
    }

    /// Returns the side to move.
    #[inline]
    pub fn side_to_move(&self) -> Color {
        self.side
    }

    /// Returns the squares occupied by pieces of `kind`, of either colour.
    #[inline]
    pub fn kind(&self, kind: PieceKind) -> Bitboard {
        self.kinds[kind.index()]
    }

    /// Returns the squares occupied by `color`.
    #[inline]
    pub fn color(&self, color: Color) -> Bitboard {
        self.colors[color.index()]
    }

    /// Returns the squares occupied by pieces of the given colour and kind.
    #[inline]
    pub fn pieces(&self, color: Color, kind: PieceKind) -> Bitboard {
        self.colors[color.index()] & self.kinds[kind.index()]
    }

    /// Returns every occupied square.
    #[inline]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    /// Returns the piece on `square`, if any.
    #[inline]
    pub fn piece_on(&self, square: Square) -> Option<Piece> {
        self.mailbox[square.index()]
    }

    /// Returns the square of `color`'s king.
    #[inline]
    pub fn king_square(&self, color: Color) -> Square {
        self.pieces(color, PieceKind::King).lsb()
    }

    /// Returns the remaining castling rights.
    #[inline]
    pub fn castling(&self) -> CastlingRights {
        self.castling
    }

    /// Returns the en-passant square: the square skipped by a pawn that made
    /// a double push on the previous move. It is set after every double push,
    /// whether or not an en-passant capture is actually possible.
    #[inline]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep_square
    }

    /// Returns the number of halfmoves since the last capture or pawn move.
    #[inline]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    /// Returns the Zobrist key of the position.
    #[inline]
    pub fn key(&self) -> u64 {
        self.key
    }

    /// Returns the square of the pawn removed by an en-passant capture onto
    /// `ep_square`.
    ///
    /// That pawn stands on the same file, one rank nearer its own side: rank
    /// 5 for an en-passant square on rank 6, and rank 4 for one on rank 3.
    /// In both cases the two square indices differ only in bit 3, so flipping
    /// that bit converts one into the other.
    #[inline]
    pub fn en_passant_victim(ep_square: Square) -> Square {
        Square::new(ep_square.index() as u8 ^ 8)
    }

    /// Returns every piece, of either colour, that attacks `square` when the
    /// occupied squares are `occupied`.
    ///
    /// Passing an occupancy other than the real one answers "what would
    /// attack this square if these pieces were moved", which is how move
    /// generation tests king moves and en-passant captures. Pieces are taken
    /// from the real position, so a piece hypothetically removed must be
    /// masked out of the result by the caller.
    ///
    /// The lookup works backwards from the target: a knight attacks `square`
    /// exactly when a knight placed on `square` would attack it, and the same
    /// symmetry holds for kings and sliders. Pawns are the exception, since
    /// they capture in one direction only, so the pawn attack table of the
    /// opposite colour is used.
    #[inline]
    pub fn attackers_to(&self, square: Square, occupied: Bitboard) -> Bitboard {
        let pawns = self.kind(PieceKind::Pawn);
        let queens = self.kind(PieceKind::Queen);
        (pawn_attacks(Color::Black, square) & pawns & self.color(Color::White))
            | (pawn_attacks(Color::White, square) & pawns & self.color(Color::Black))
            | (knight_attacks(square) & self.kind(PieceKind::Knight))
            | (king_attacks(square) & self.kind(PieceKind::King))
            | (bishop_attacks(square, occupied) & (self.kind(PieceKind::Bishop) | queens))
            | (rook_attacks(square, occupied) & (self.kind(PieceKind::Rook) | queens))
    }

    /// Returns the enemy pieces giving check to the side to move.
    #[inline]
    pub fn checkers(&self) -> Bitboard {
        self.attackers_to(self.king_square(self.side), self.occupied()) & self.color(!self.side)
    }

    /// Returns `true` if the side to move is in check.
    #[inline]
    pub fn in_check(&self) -> bool {
        self.checkers().any()
    }

    /// Returns `true` if neither side has enough material to deliver mate:
    /// no pawns, rooks or queens, and at most one knight or bishop in total
    /// (king against king, or king and minor piece against king).
    pub fn has_insufficient_material(&self) -> bool {
        let heavy =
            self.kind(PieceKind::Pawn) | self.kind(PieceKind::Rook) | self.kind(PieceKind::Queen);
        let minors = self.kind(PieceKind::Knight) | self.kind(PieceKind::Bishop);
        heavy.is_empty() && !minors.more_than_one()
    }

    /// Returns `true` if the current position counts as a draw by
    /// repetition for a search that is `plies_from_root` moves below its
    /// root (SRC-4).
    ///
    /// A position repeated *inside* the search tree is scored as a draw on
    /// its first repetition: if a line can return to a position once it can
    /// do so again, so searching further cannot change the verdict. A
    /// position that only matches one from the game before the root needs
    /// two earlier occurrences, the genuine threefold rule, because the
    /// engine must not claim a draw the rules do not yet grant.
    ///
    /// Only positions since the last capture or pawn move can repeat, and
    /// only those with the same side to move, so the scan steps back two
    /// plies at a time no further than the halfmove clock. The nearest
    /// possible repetition is four plies back.
    pub fn is_repetition(&self, plies_from_root: usize) -> bool {
        let played = self.history.len();
        let reach = played.min(self.halfmove_clock as usize);
        let mut earlier = 0;
        for back in (4..=reach).step_by(2) {
            // `history[n].key` is the key of the position before move `n`,
            // i.e. the position `played - n` plies ago.
            if self.history[played - back].key == self.key {
                if back <= plies_from_root {
                    return true;
                }
                earlier += 1;
                if earlier == 2 {
                    return true;
                }
            }
        }
        false
    }

    /// Returns the en-passant contribution to the Zobrist key.
    ///
    /// The en-passant file is hashed only when a pawn of the side to move
    /// stands where it could capture onto the en-passant square. Otherwise
    /// the square has no effect on the available moves, and hashing it would
    /// make two positions with identical play look different, which would
    /// hide repetitions.
    #[inline]
    fn ep_key(&self) -> u64 {
        match self.ep_square {
            Some(square)
                if (pawn_attacks(!self.side, square) & self.pieces(self.side, PieceKind::Pawn))
                    .any() =>
            {
                ZOBRIST.ep_file[square.file() as usize]
            }
            _ => 0,
        }
    }

    /// Computes the Zobrist key from scratch.
    ///
    /// The result always equals [`Position::key`]; this function exists to
    /// set the key of a newly parsed position and to verify the incremental
    /// updates in tests.
    pub fn compute_key(&self) -> u64 {
        let mut key = ZOBRIST.castling[self.castling.index()] ^ self.ep_key();
        if self.side == Color::Black {
            key ^= ZOBRIST.side;
        }
        for square in self.occupied() {
            if let Some(piece) = self.piece_on(square) {
                key ^= ZOBRIST.piece[piece.index()][square.index()];
            }
        }
        key
    }

    /// Places `piece` on the empty `square`, updating the key.
    #[inline]
    fn add_piece(&mut self, piece: Piece, square: Square) {
        debug_assert!(self.mailbox[square.index()].is_none());
        self.kinds[piece.kind().index()] |= square.bb();
        self.colors[piece.color().index()] |= square.bb();
        self.mailbox[square.index()] = Some(piece);
        self.key ^= ZOBRIST.piece[piece.index()][square.index()];
    }

    /// Removes and returns the piece on `square`, updating the key.
    ///
    /// Panics if the square is empty, which means the move being made or
    /// unmade does not belong to this position.
    #[inline]
    fn remove_piece(&mut self, square: Square) -> Piece {
        let piece = self.mailbox[square.index()]
            .take()
            .expect("no piece on the square a move refers to");
        self.kinds[piece.kind().index()] ^= square.bb();
        self.colors[piece.color().index()] ^= square.bb();
        self.key ^= ZOBRIST.piece[piece.index()][square.index()];
        piece
    }

    /// Returns the origin and destination of the rook in a castling move
    /// whose king lands on `king_to` (g1/g8 king-side, c1/c8 queen-side).
    #[inline]
    pub(crate) fn castling_rook_squares(mv: Move, king_to: Square) -> (Square, Square) {
        if mv.flag() == Move::KING_CASTLE {
            (king_to.offset(1), king_to.offset(-1))
        } else {
            (king_to.offset(-2), king_to.offset(1))
        }
    }

    /// Plays `mv`, which must be a legal move in this position as produced
    /// by [`generate`](crate::movegen::generate).
    ///
    /// # Panics
    ///
    /// Panics if `mv` refers to a piece that is not on the board. Any other
    /// illegal move is not detected and leaves the position inconsistent.
    pub fn make_move(&mut self, mv: Move) {
        let us = self.side;
        let (from, to) = (mv.from(), mv.to());
        let key_before = self.key;

        // The old en-passant key must be removed before any pawn moves,
        // because whether it was hashed depends on the pawn placement.
        self.key ^= self.ep_key();

        let captured = if mv.is_en_passant() {
            Some(self.remove_piece(Position::en_passant_victim(to)))
        } else if mv.is_capture() {
            Some(self.remove_piece(to))
        } else {
            None
        };

        self.history.push(Undo {
            captured,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            key: key_before,
        });

        let moved = self.remove_piece(from);
        let placed = match mv.promotion() {
            Some(kind) => Piece::new(us, kind),
            None => moved,
        };
        self.add_piece(placed, to);

        if mv.is_castle() {
            let (rook_from, rook_to) = Position::castling_rook_squares(mv, to);
            let rook = self.remove_piece(rook_from);
            self.add_piece(rook, rook_to);
        }

        let old_rights = self.castling;
        self.castling = CastlingRights(
            old_rights.0 & CastlingRights::KEEP[from.index()] & CastlingRights::KEEP[to.index()],
        );
        self.key ^= ZOBRIST.castling[old_rights.index()] ^ ZOBRIST.castling[self.castling.index()];

        // A double push leaves the skipped square, midway between origin and
        // destination, open to en passant.
        self.ep_square = mv
            .is_double_push()
            .then(|| Square::new((from.index() + to.index()) as u8 / 2));

        self.halfmove_clock = if moved.kind() == PieceKind::Pawn || captured.is_some() {
            0
        } else {
            self.halfmove_clock.saturating_add(1)
        };
        if us == Color::Black {
            self.fullmove_number = self.fullmove_number.wrapping_add(1);
        }

        self.side = !us;
        self.key ^= ZOBRIST.side;
        // Evaluated last: it depends on the new side to move and pawns.
        self.key ^= self.ep_key();
    }

    /// Takes back `mv`, which must be the move most recently made and not yet
    /// unmade. The position is restored exactly, including its key (BRD-5).
    ///
    /// # Panics
    ///
    /// Panics if no move has been made, or if `mv` is not the last move made.
    pub fn unmake_move(&mut self, mv: Move) {
        let undo = self
            .history
            .pop()
            .expect("unmake_move without a move to take back");
        self.side = !self.side;
        let us = self.side;
        let (from, to) = (mv.from(), mv.to());

        let placed = self.remove_piece(to);
        let moved = if mv.promotion().is_some() {
            Piece::new(us, PieceKind::Pawn)
        } else {
            placed
        };
        self.add_piece(moved, from);

        if mv.is_castle() {
            let (rook_from, rook_to) = Position::castling_rook_squares(mv, to);
            let rook = self.remove_piece(rook_to);
            self.add_piece(rook, rook_from);
        }

        if let Some(piece) = undo.captured {
            let square = if mv.is_en_passant() {
                Position::en_passant_victim(to)
            } else {
                to
            };
            self.add_piece(piece, square);
        }

        if us == Color::Black {
            self.fullmove_number = self.fullmove_number.wrapping_sub(1);
        }
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        // The piece helpers above also touched the key; the saved value
        // replaces it rather than relying on those updates cancelling out.
        self.key = undo.key;
    }

    /// Parses a position in Forsyth-Edwards Notation.
    ///
    /// The six standard fields are expected. The two move counters may be
    /// omitted, in which case they default to `0` and `1`.
    ///
    /// # Errors
    ///
    /// Returns a [`FenError`] if the text is malformed or describes a
    /// position the engine cannot safely play from: a side without exactly
    /// one king, a pawn on the first or eighth rank, castling rights without
    /// the king and rook at home, an impossible en-passant square, or the
    /// side not to move being in check. This function never panics (BRD-6).
    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        let mut fields = fen.split_whitespace();
        let mut field = |name| fields.next().ok_or(FenError::MissingField(name));
        let placement = field("piece placement")?;
        let side = field("side to move")?;
        let castling = field("castling rights")?;
        let en_passant = field("en passant square")?;
        let halfmove_clock = fields.next();
        let fullmove_number = fields.next();
        if fields.next().is_some() {
            return Err(FenError::TrailingData);
        }

        let mut position = Position::empty();
        position.parse_placement(placement)?;

        position.side = match side {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError::BadSideToMove),
        };

        position.parse_castling(castling)?;
        position.parse_en_passant(en_passant)?;

        let counter = |text: Option<&str>, default| match text {
            Some(text) => text.parse::<u16>().map_err(|_| FenError::BadCounter),
            None => Ok(default),
        };
        position.halfmove_clock = counter(halfmove_clock, 0)?;
        position.fullmove_number = counter(fullmove_number, 1)?;

        // With the side not to move in check, the side to move could capture
        // the king, which move generation assumes can never happen.
        let them = !position.side;
        let attackers = position.attackers_to(position.king_square(them), position.occupied());
        if (attackers & position.color(position.side)).any() {
            return Err(FenError::OpponentInCheck);
        }

        position.key = position.compute_key();
        Ok(position)
    }

    /// Parses the piece-placement field of a FEN into this (empty) position
    /// and checks the king and pawn constraints.
    fn parse_placement(&mut self, placement: &str) -> Result<(), FenError> {
        let mut rows = 0;
        for (row, text) in placement.split('/').enumerate() {
            if row >= 8 {
                return Err(FenError::BadPlacement);
            }
            // FEN lists rank 8 first.
            let rank = 7 - row as u8;
            let mut file = 0;
            for c in text.chars() {
                if file >= 8 {
                    return Err(FenError::BadPlacement);
                }
                match c {
                    '1'..='8' => file += c as u8 - b'0',
                    _ => {
                        let piece = Piece::from_char(c).ok_or(FenError::BadPlacement)?;
                        self.add_piece(piece, Square::from_file_rank(file, rank));
                        file += 1;
                    }
                }
            }
            if file != 8 {
                return Err(FenError::BadPlacement);
            }
            rows += 1;
        }
        if rows != 8 {
            return Err(FenError::BadPlacement);
        }

        for color in [Color::White, Color::Black] {
            if self.pieces(color, PieceKind::King).count() != 1 {
                return Err(FenError::KingCount);
            }
        }
        if (self.kind(PieceKind::Pawn) & (Bitboard::rank(0) | Bitboard::rank(7))).any() {
            return Err(FenError::PawnOnBackRank);
        }
        Ok(())
    }

    /// Parses the castling field of a FEN. Each right is accepted only if the
    /// king and rook it needs are on their home squares, because castling
    /// moves are generated on that assumption.
    fn parse_castling(&mut self, castling: &str) -> Result<(), FenError> {
        if castling == "-" {
            return Ok(());
        }
        for c in castling.chars() {
            let (color, rook_file) = match c {
                'K' => (Color::White, 7),
                'Q' => (Color::White, 0),
                'k' => (Color::Black, 7),
                'q' => (Color::Black, 0),
                _ => return Err(FenError::BadCastling),
            };
            let right = if rook_file == 7 {
                CastlingRights::kingside(color)
            } else {
                CastlingRights::queenside(color)
            };
            let king_home = Square::from_file_rank(4, color.back_rank());
            let rook_home = Square::from_file_rank(rook_file, color.back_rank());
            if self.castling.contains(right)
                || self.piece_on(king_home) != Some(Piece::new(color, PieceKind::King))
                || self.piece_on(rook_home) != Some(Piece::new(color, PieceKind::Rook))
            {
                return Err(FenError::BadCastling);
            }
            self.castling = CastlingRights(self.castling.0 | right.0);
        }
        Ok(())
    }

    /// Parses the en-passant field of a FEN, once the side to move is known.
    ///
    /// The square is accepted only if the opponent could have just played the
    /// double push it implies: it is on the right rank, the pushed pawn is in
    /// front of it, and both the square and the pawn's origin are empty.
    fn parse_en_passant(&mut self, en_passant: &str) -> Result<(), FenError> {
        if en_passant == "-" {
            return Ok(());
        }
        let square = Square::parse(en_passant).ok_or(FenError::BadEnPassant)?;
        let expected_rank = match self.side {
            Color::White => 5,
            Color::Black => 2,
        };
        if square.rank() != expected_rank {
            return Err(FenError::BadEnPassant);
        }
        // Seen from the en-passant square, the pushed pawn's origin lies in
        // the direction the side to move advances.
        let origin = square.offset(self.side.pawn_push());
        let pushed_pawn = Position::en_passant_victim(square);
        if self.piece_on(pushed_pawn) != Some(Piece::new(!self.side, PieceKind::Pawn))
            || self.piece_on(square).is_some()
            || self.piece_on(origin).is_some()
        {
            return Err(FenError::BadEnPassant);
        }
        self.ep_square = Some(square);
        Ok(())
    }

    /// Returns the position in Forsyth-Edwards Notation. Parsing a canonical
    /// FEN and converting it back yields the same string (BRD-6).
    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(90);
        for rank in (0..8).rev() {
            let mut empty = 0u8;
            for file in 0..8 {
                match self.piece_on(Square::from_file_rank(file, rank)) {
                    Some(piece) => {
                        if empty > 0 {
                            fen.push((b'0' + empty) as char);
                            empty = 0;
                        }
                        fen.push(piece.to_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push_str(match self.side {
            Color::White => " w ",
            Color::Black => " b ",
        });

        if self.castling == CastlingRights::NONE {
            fen.push('-');
        } else {
            let letters = [
                (CastlingRights::WHITE_KINGSIDE, 'K'),
                (CastlingRights::WHITE_QUEENSIDE, 'Q'),
                (CastlingRights::BLACK_KINGSIDE, 'k'),
                (CastlingRights::BLACK_QUEENSIDE, 'q'),
            ];
            for (right, letter) in letters {
                if self.castling.contains(right) {
                    fen.push(letter);
                }
            }
        }

        match self.ep_square {
            Some(square) => fen.push_str(&format!(" {square}")),
            None => fen.push_str(" -"),
        }
        fen.push_str(&format!(
            " {} {}",
            self.halfmove_clock, self.fullmove_number
        ));
        fen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_fens_round_trip() {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w Kq d6 12 34",
            "4k3/8/8/8/8/8/8/4K2R b K - 99 150",
        ];
        for fen in fens {
            assert_eq!(Position::from_fen(fen).unwrap().to_fen(), fen);
        }
    }

    #[test]
    fn counters_are_optional() {
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - -").unwrap();
        assert_eq!(position.to_fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    }

    #[test]
    fn start_position_fields() {
        let position = Position::startpos();
        assert_eq!(position.side_to_move(), Color::White);
        assert_eq!(position.occupied().count(), 32);
        assert_eq!(
            position.king_square(Color::Black),
            Square::parse("e8").unwrap()
        );
        assert_eq!(
            position.piece_on(Square::parse("d1").unwrap()),
            Some(Piece::WhiteQueen)
        );
        assert!(!position.in_check());
        assert_eq!(position.key(), position.compute_key());
    }

    #[test]
    fn malformed_fens_are_rejected() {
        use FenError::*;
        let cases = [
            ("", MissingField("piece placement")),
            ("4k3/8/8/8/8/8/8/4K3", MissingField("side to move")),
            ("4k3/8/8/8/8/8/8/4K3 w", MissingField("castling rights")),
            ("4k3/8/8/8/8/8/8/4K3 w -", MissingField("en passant square")),
            ("4k3/8/8/8/8/8/8 w - - 0 1", BadPlacement),
            ("4k3/8/8/8/8/8/8/4K3/8 w - - 0 1", BadPlacement),
            ("4k3/8/8/8/8/8/8/4K4 w - - 0 1", BadPlacement),
            ("4k3/8/8/8/8/8/8/4K2 w - - 0 1", BadPlacement),
            ("4k3/8/8/8/8/8/8/4Kx2 w - - 0 1", BadPlacement),
            ("4k3/9/8/8/8/8/8/4K3 w - - 0 1", BadPlacement),
            ("4k3/8/8/8/8/8/8/RNBQKBNRR w - - 0 1", BadPlacement),
            ("8/8/8/8/8/8/8/4K3 w - - 0 1", KingCount),
            ("4k3/8/8/8/8/8/8/3KK3 w - - 0 1", KingCount),
            ("P3k3/8/8/8/8/8/8/4K3 w - - 0 1", PawnOnBackRank),
            ("4k3/8/8/8/8/8/8/4K3 x - - 0 1", BadSideToMove),
            ("4k3/8/8/8/8/8/8/4K3 w K - 0 1", BadCastling),
            ("4k3/8/8/8/8/8/8/4K2R w KK - 0 1", BadCastling),
            ("4k3/8/8/8/8/8/8/4K2R w X - 0 1", BadCastling),
            ("4k3/8/8/8/8/8/8/4K3 w - e9 0 1", BadEnPassant),
            ("4k3/8/8/8/8/8/8/4K3 w - e6 0 1", BadEnPassant),
            ("4k3/8/8/4p3/8/8/8/4K3 w - e3 0 1", BadEnPassant),
            ("4k3/4n3/8/4p3/8/8/8/4K3 w - e6 0 1", BadEnPassant),
            ("4k3/8/8/8/8/8/8/4K3 w - - x 1", BadCounter),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 70000", BadCounter),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra", TrailingData),
            ("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1", OpponentInCheck),
            ("8/8/8/8/8/8/8/3kK3 w - - 0 1", OpponentInCheck),
        ];
        for (fen, error) in cases {
            assert_eq!(Position::from_fen(fen), Err(error), "{fen:?}");
        }
    }

    /// Plays the UCI moves in `moves` from the start position.
    fn play(moves: &str) -> Position {
        use crate::movegen::{GenKind, generate};
        use crate::moves::MoveList;
        let mut position = Position::startpos();
        for text in moves.split_whitespace() {
            let mut list = MoveList::new();
            generate(&position, GenKind::All, &mut list);
            let mv = *list.iter().find(|mv| mv.to_string() == text).unwrap();
            position.make_move(mv);
        }
        position
    }

    #[test]
    fn repetition_inside_and_before_the_search_tree() {
        // One knight shuffle: the start position has now occurred twice.
        let once = play("g1f3 g8f6 f3g1 f6g8");
        assert!(once.is_repetition(4), "repeats a position inside the tree");
        assert!(
            !once.is_repetition(0),
            "only one earlier occurrence in the game"
        );
        // Two shuffles: third occurrence, a draw wherever the root is.
        let twice = play("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
        assert!(twice.is_repetition(0));
        // A pawn move in between makes the earlier positions unreachable.
        let reset = play("g1f3 g8f6 f3g1 f6g8 e2e4");
        assert!(!reset.is_repetition(5));
    }

    #[test]
    fn insufficient_material() {
        let draw = [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
            "4kb2/8/8/8/8/8/8/4K3 w - - 0 1",
        ];
        for fen in draw {
            assert!(
                Position::from_fen(fen).unwrap().has_insufficient_material(),
                "{fen}"
            );
        }
        let play_on = [
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4KNN1 w - - 0 1",
            "4kb2/8/8/8/8/8/8/4KB2 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
        ];
        for fen in play_on {
            assert!(
                !Position::from_fen(fen).unwrap().has_insufficient_material(),
                "{fen}"
            );
        }
    }

    #[test]
    fn en_passant_is_hashed_only_when_capturable() {
        // After 1. e4 no black pawn can capture on e3, so the key matches the
        // same position without the en-passant square.
        let with_ep =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        let without =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        assert_eq!(with_ep.unwrap().key(), without.unwrap().key());

        // Here the d4 pawn can capture on e3, so the keys must differ.
        let with_ep =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        let without =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        assert_ne!(with_ep.unwrap().key(), without.unwrap().key());
    }
}
