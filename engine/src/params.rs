//! Tunable engine parameters, gathered in one place (SRC-9).
//!
//! Every number here is a candidate for automated tuning. Keeping them
//! together, as named constants, means a tuner can later replace them with
//! adjustable values without hunting through the search code.

/// The search checks the clock and the stop flag when the node count is a
/// multiple of this value plus one. It must be one less than a power of two,
/// since it is applied as a bit mask. Smaller values react faster to a
/// deadline but spend more time reading the clock.
pub const NODE_POLL_MASK: u64 = 1023;

/// Ordering score of the hash move (the best move remembered from an earlier
/// search of the position), above everything else.
pub const ORDER_HASH_MOVE: i32 = 1_000_000;

/// Ordering score added to a promotion to a queen, above any capture.
pub const ORDER_QUEEN_PROMOTION: i32 = 200_000;

/// Ordering score added to every capture; the victim and attacker adjust it
/// (see [`ORDER_VICTIM_WEIGHT`]).
pub const ORDER_CAPTURE: i32 = 100_000;

/// Ordering score of the first killer move at a ply: below every capture,
/// above the other quiet moves.
pub const ORDER_KILLER_FIRST: i32 = 90_000;

/// Ordering score of the second killer move at a ply.
pub const ORDER_KILLER_SECOND: i32 = 80_000;

/// Largest magnitude a history score can reach. It must stay below
/// [`ORDER_KILLER_SECOND`] so that history only ranks the remaining quiet
/// moves among themselves.
pub const HISTORY_MAX: i32 = 16_384;

/// A history update is this value times the square of the remaining depth:
/// cutoffs found by deeper searches are better evidence.
pub const HISTORY_BONUS_SCALE: i32 = 16;

/// Upper limit of a single history update.
pub const HISTORY_BONUS_MAX: i32 = 2_000;

/// Multiplier of the captured piece's kind index in a capture's ordering
/// score. It exceeds the largest attacker index, so the victim always
/// dominates: the order is most valuable victim first, and among equal
/// victims the least valuable attacker first (MVV-LVA).
pub const ORDER_VICTIM_WEIGHT: i32 = 8;

/// Least remaining depth at which null-move pruning is tried.
pub const NULL_MOVE_MIN_DEPTH: i32 = 3;

/// Plies removed from the search that follows a null move, before the
/// depth-dependent part (see [`NULL_MOVE_DEPTH_DIVISOR`]).
pub const NULL_MOVE_REDUCTION: i32 = 3;

/// The null-move reduction grows by one ply for every this many plies of
/// remaining depth.
pub const NULL_MOVE_DEPTH_DIVISOR: i32 = 4;

/// Least remaining depth at which late move reductions apply.
pub const LMR_MIN_DEPTH: i32 = 3;

/// Number of moves searched at full depth before reductions begin.
pub const LMR_FULL_DEPTH_MOVES: usize = 3;

/// Constant term of the late-move-reduction formula, in hundredths of a ply.
pub const LMR_BASE_PERCENT: i32 = 75;

/// Divisor of the logarithmic term of the late-move-reduction formula, in
/// hundredths: the reduction is `base + ln(depth) * ln(move number) / divisor`.
pub const LMR_DIVISOR_PERCENT: i32 = 225;

/// Number of moves the remaining time is spread over when the time control
/// does not say how many moves are left.
pub const TIME_MOVES_TO_GO: u64 = 20;

/// Percentage of the increment added to each move's time allotment.
pub const TIME_INCREMENT_PERCENT: u64 = 75;

/// Percentage of a move's allotment after which no new iteration is started
/// (the soft limit). The next iteration usually costs more than all earlier
/// ones together, so starting it late would mostly waste the time.
pub const TIME_SOFT_PERCENT: u64 = 60;

/// Multiple of a move's allotment at which the search is aborted outright
/// (the hard limit).
pub const TIME_HARD_MULTIPLIER: u64 = 3;

/// Percentage of the remaining clock time that the hard limit may never
/// exceed, whatever the allotment.
pub const TIME_HARD_MAX_PERCENT: u64 = 80;

/// Default time, in milliseconds, held back on every move to cover the delay
/// between the engine choosing a move and the opponent's clock starting.
pub const DEFAULT_MOVE_OVERHEAD_MS: u64 = 10;

/// Default transposition table size in megabytes.
pub const DEFAULT_HASH_MB: usize = 16;
