//! The search: iterative deepening alpha-beta with quiescence (SRC-1 to
//! SRC-8).
//!
//! Scores are in centipawns from the point of view of the side to move at
//! the node being searched ("negamax"): a child's score is negated to give
//! its worth to the parent. A window `(alpha, beta)` is carried down the
//! tree. `alpha` is the score the side to move is already sure of and `beta`
//! the most the opponent will allow; once a move scores at least `beta` the
//! opponent would never enter this node, so the remaining moves are skipped
//! (a *cutoff*). The search is *fail-soft*: it returns the best score it
//! actually found even when that lies outside the window.

use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::{Evaluator, PstEvaluator};
use crate::movegen::{GenKind, generate};
use crate::moves::{Move, MoveList};
use crate::params::{
    ASPIRATION_MIN_DEPTH, ASPIRATION_WINDOW, FUTILITY_MARGIN_BASE, FUTILITY_MARGIN_PER_PLY,
    FUTILITY_MAX_DEPTH, HISTORY_BONUS_MAX, HISTORY_BONUS_SCALE, HISTORY_MAX, IIR_MIN_DEPTH,
    LMP_BASE, LMP_DEPTH_SCALE, LMP_MAX_DEPTH, LMR_BASE_PERCENT, LMR_DIVISOR_PERCENT,
    LMR_FULL_DEPTH_MOVES, LMR_MIN_DEPTH, NODE_POLL_MASK, NULL_MOVE_DEPTH_DIVISOR,
    NULL_MOVE_MIN_DEPTH, NULL_MOVE_REDUCTION, ORDER_BAD_CAPTURE, ORDER_CAPTURE, ORDER_HASH_MOVE,
    ORDER_KILLER_FIRST, ORDER_KILLER_SECOND, ORDER_QUEEN_PROMOTION, ORDER_VICTIM_WEIGHT,
    RFP_MARGIN, RFP_MAX_DEPTH,
};
use crate::position::Position;
use crate::see::see_ge;
use crate::timeman::{Limits, TimeBudget};
use crate::tt::{Bound, TranspositionTable};
use crate::types::PieceKind;

/// Greatest distance from the root, in plies, that the search will reach.
pub const MAX_PLY: usize = 128;

/// A score larger than any the search can return; the initial window is
/// `(-INFINITE, INFINITE)`.
pub const INFINITE: i32 = 32_000;

/// Score of delivering checkmate at the root. A mate found `n` plies from
/// the root scores `MATE - n`, so that shorter mates are preferred, and
/// being mated scores the negative (SRC-5).
pub const MATE: i32 = 31_000;

/// Scores at or beyond this magnitude are mate scores; ordinary evaluations
/// always lie strictly inside.
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

/// Score of a drawn position.
pub const DRAW: i32 = 0;

/// Late move reductions in plies, indexed by remaining depth and by the
/// move's position in the ordering (both capped at 63).
///
/// The reduction is `base + ln(depth) * ln(move number) / divisor`, rounded
/// down. It grows with depth, because a deep search can afford to lose more
/// plies, and with the move number, because the later a move is ordered the
/// less likely it is to be best. The logarithms make it grow quickly at
/// first and slowly afterwards.
static LMR_TABLE: LazyLock<[[u8; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0; 64]; 64];
    for (depth, row) in table.iter_mut().enumerate().skip(1) {
        for (number, reduction) in row.iter_mut().enumerate().skip(1) {
            let log = (depth as f64).ln() * (number as f64).ln();
            let plies = LMR_BASE_PERCENT as f64 / 100.0 + log * 100.0 / LMR_DIVISOR_PERCENT as f64;
            *reduction = plies as u8;
        }
    }
    table
});

/// A progress report, produced once per completed iteration (UCI-4).
#[derive(Clone, Debug)]
pub struct SearchInfo<'a> {
    /// Depth of the completed iteration.
    pub depth: i32,
    /// Greatest ply reached in the iteration, including quiescence.
    pub seldepth: usize,
    /// Score of the position for the side to move.
    pub score: i32,
    /// Nodes searched since the search began.
    pub nodes: u64,
    /// Time since the search began.
    pub time: Duration,
    /// Transposition table fill in parts per thousand.
    pub hashfull: usize,
    /// The principal variation: the line of best play found.
    pub pv: &'a [Move],
}

/// Formats the report as a UCI `info` line.
///
/// A mate score is reported as `score mate N`, with `N` in full moves and
/// negative when the engine is the one being mated; the mate distance in
/// plies is `MATE - |score|`, which is rounded up to moves.
impl fmt::Display for SearchInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "info depth {} seldepth {} score ",
            self.depth, self.seldepth
        )?;
        if self.score.abs() >= MATE_BOUND {
            let moves = (MATE - self.score.abs() + 1) / 2;
            write!(f, "mate {}", if self.score > 0 { moves } else { -moves })?;
        } else {
            write!(f, "cp {}", self.score)?;
        }
        let millis = self.time.as_millis();
        let nps = self.nodes as u128 * 1000 / millis.max(1);
        write!(
            f,
            " nodes {} nps {} time {} hashfull {} pv",
            self.nodes, nps, millis, self.hashfull
        )?;
        for mv in self.pv {
            write!(f, " {mv}")?;
        }
        Ok(())
    }
}

/// The outcome of a search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchResult {
    /// The move to play, or `None` if the side to move has no legal move.
    pub best_move: Option<Move>,
    /// Score of the last completed iteration.
    pub score: i32,
    /// Depth of the last completed iteration.
    pub depth: i32,
    /// Total nodes searched.
    pub nodes: u64,
}

/// One search thread: everything a search mutates lives here (SRC-7), so
/// that several can run side by side sharing only the transposition table
/// and the stop flag.
pub struct Searcher<'a> {
    /// The position being searched; moves are made and unmade on it.
    position: Position,
    /// Evaluator following `position`.
    evaluator: PstEvaluator,
    /// Shared transposition table.
    table: &'a TranspositionTable,
    /// Shared flag set from outside to end the search.
    stop: &'a AtomicBool,
    /// When the search began; time limits are measured from here.
    start: Instant,
    /// Time limits.
    budget: TimeBudget,
    /// Node limit, if any.
    node_limit: Option<u64>,
    /// Deepest iteration to run.
    max_depth: i32,
    /// Nodes searched so far.
    nodes: u64,
    /// Greatest ply reached in the current iteration.
    seldepth: usize,
    /// Whether limits may interrupt the search yet. Off during the first
    /// iteration, so that there is always a move to play.
    may_abort: bool,
    /// Set once a limit has interrupted the search; every node then returns
    /// immediately and its score is meaningless.
    aborted: bool,
    /// Triangular principal-variation table: `pv[ply]` holds the best line
    /// found from the node at `ply`, of length `pv_len[ply]`.
    pv: Vec<[Move; MAX_PLY]>,
    /// Length of each line in `pv`.
    pv_len: [usize; MAX_PLY + 1],
    /// Killer moves: for each ply, the two quiet moves that most recently
    /// caused a cutoff there, newest first.
    killers: [[Move; 2]; MAX_PLY + 1],
    /// History scores of quiet moves, indexed by side to move, origin and
    /// destination: how often the move has caused a cutoff, anywhere in
    /// the tree, against how often it was tried and failed to.
    history: Box<[[[i32; 64]; 64]; 2]>,
    /// For each ply, whether the node there is currently searching a null
    /// move, so that its child does not try one straight after.
    null_moved: [bool; MAX_PLY + 1],
    /// Best line from the root that is safe to play: taken from the last
    /// completed iteration, or from the current one once a root move has
    /// been fully searched and found best (SRC-6).
    root_pv: Vec<Move>,
}

impl<'a> Searcher<'a> {
    /// Prepares a search of `position` under `limits`.
    ///
    /// `start` is the moment the `go` command arrived; `overhead_ms` is the
    /// time to hold back on each move (see [`TimeBudget::new`]).
    pub fn new(
        position: Position,
        table: &'a TranspositionTable,
        stop: &'a AtomicBool,
        limits: &Limits,
        start: Instant,
        overhead_ms: u64,
    ) -> Searcher<'a> {
        let max_ply = MAX_PLY as i32 - 1;
        Searcher {
            evaluator: PstEvaluator::new(&position),
            budget: TimeBudget::new(limits, position.side_to_move(), overhead_ms),
            position,
            table,
            stop,
            start,
            node_limit: limits.nodes,
            max_depth: limits
                .depth
                .map_or(max_ply, |depth| depth.clamp(1, max_ply)),
            nodes: 0,
            seldepth: 0,
            may_abort: false,
            aborted: false,
            pv: vec![[Move::NULL; MAX_PLY]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            killers: [[Move::NULL; 2]; MAX_PLY + 1],
            history: Box::new([[[0; 64]; 64]; 2]),
            null_moved: [false; MAX_PLY + 1],
            root_pv: Vec::with_capacity(MAX_PLY),
        }
    }

    /// Runs the search by iterative deepening: depth 1, then 2, and so on
    /// until a limit is reached. `on_iteration` is called with a report
    /// after each completed depth.
    ///
    /// Searching every shallower depth first costs little, because each
    /// iteration is several times larger than the one before, and pays for
    /// itself: the previous iteration's best line is searched first, which
    /// makes cutoffs far more frequent, and there is always a finished
    /// result to fall back on when time runs out.
    pub fn run(&mut self, on_iteration: &mut dyn FnMut(&SearchInfo)) -> SearchResult {
        let mut result = SearchResult {
            best_move: None,
            score: DRAW,
            depth: 0,
            nodes: 0,
        };

        for depth in 1..=self.max_depth {
            self.seldepth = 0;
            self.may_abort = depth > 1;
            let score = self.search_root(depth, result.score);
            if self.aborted {
                break;
            }
            if self.root_pv.is_empty() {
                // No legal move: checkmate or stalemate at the root.
                result.score = score;
                break;
            }
            result.score = score;
            result.depth = depth;
            on_iteration(&SearchInfo {
                depth,
                seldepth: self.seldepth,
                score,
                nodes: self.nodes,
                time: self.start.elapsed(),
                hashfull: self.table.hashfull(),
                pv: &self.root_pv,
            });

            let out_of_time = self
                .budget
                .soft
                .is_some_and(|soft| self.start.elapsed() >= soft);
            if out_of_time || self.stop.load(Ordering::Relaxed) {
                break;
            }
        }

        result.best_move = self.root_pv.first().copied();
        result.nodes = self.nodes;
        result
    }

    /// Searches the root to `depth` and returns its score, given `guess`,
    /// the score of the previous iteration.
    ///
    /// From [`ASPIRATION_MIN_DEPTH`] on, the search is first tried with a
    /// narrow *aspiration window* around the previous score. The score
    /// rarely moves far from one iteration to the next, and a narrow
    /// window makes every node cut off sooner, so the iteration is much
    /// cheaper when the guess holds. When it does not, the search fails
    /// outside the window and is repeated with that side of the window
    /// widened, doubling the margin each time, until it either returns a
    /// score inside the window or the window has grown to the full range.
    /// Only the failing side moves, so the other bound keeps its
    /// cutoffs. A mate score is far from any guess, so the window is
    /// opened fully as soon as a bound reaches the mate range.
    ///
    /// Shallow iterations use the full window: they are cheap, and their
    /// scores are too unsettled for a guess to be worth much.
    fn search_root(&mut self, depth: i32, guess: i32) -> i32 {
        if depth < ASPIRATION_MIN_DEPTH {
            return self.negamax(depth, 0, -INFINITE, INFINITE);
        }
        let mut delta = ASPIRATION_WINDOW;
        let mut alpha = (guess - delta).max(-INFINITE);
        let mut beta = (guess + delta).min(INFINITE);
        loop {
            let score = self.negamax(depth, 0, alpha, beta);
            if self.aborted {
                return score;
            }
            if score <= alpha {
                alpha = if score.abs() >= MATE_BOUND {
                    -INFINITE
                } else {
                    (alpha - delta).max(-INFINITE)
                };
            } else if score >= beta {
                beta = if score.abs() >= MATE_BOUND {
                    INFINITE
                } else {
                    (beta + delta).min(INFINITE)
                };
            } else {
                return score;
            }
            delta *= 2;
        }
    }

    /// Counts a node and reports whether the search has been interrupted.
    ///
    /// The node limit is compared at every node so that node-limited
    /// searches are exactly reproducible. The clock and the stop flag are
    /// consulted only every [`NODE_POLL_MASK`] + 1 nodes, because reading
    /// them at every node would cost more than the search itself (SRC-6).
    #[inline]
    fn enter_node(&mut self) -> bool {
        // Once interrupted, the unwinding calls are not nodes searched.
        if self.aborted {
            return true;
        }
        self.nodes += 1;
        if !self.may_abort {
            return false;
        }
        if self.node_limit.is_some_and(|limit| self.nodes >= limit) {
            self.aborted = true;
        } else if self.nodes & NODE_POLL_MASK == 0 {
            let out_of_time = self
                .budget
                .hard
                .is_some_and(|hard| self.start.elapsed() >= hard);
            self.aborted = out_of_time || self.stop.load(Ordering::Relaxed);
        }
        self.aborted
    }

    /// Plays `mv` on the position, keeping the evaluator in step.
    #[inline]
    fn make(&mut self, mv: Move) {
        self.evaluator.make_move(&self.position, mv);
        self.position.make_move(mv);
    }

    /// Takes back `mv`, keeping the evaluator in step.
    #[inline]
    fn unmake(&mut self, mv: Move) {
        self.position.unmake_move(mv);
        self.evaluator.unmake_move();
    }

    /// Returns the ordering score of `mv`: higher is searched earlier.
    ///
    /// The earlier a good move is tried, the sooner a cutoff comes and the
    /// smaller the tree. The order is the hash move, then queen promotions,
    /// then captures that do not lose material by static exchange
    /// evaluation, by most valuable victim and least valuable attacker,
    /// then the two `killers`, then the remaining quiet moves by their
    /// history score (SRC-3), and last the captures that lose material,
    /// again by victim and attacker.
    ///
    /// A losing capture is the least promising move in most positions: it
    /// gives up material, and the quiet moves have a chance of keeping
    /// it. Searching it after every quiet move means it is reached only
    /// when nothing else has cut off, which is rare.
    ///
    /// A killer is a quiet move that refuted a sibling position at the same
    /// ply. Sibling positions differ by one move of the opponent, so the
    /// same reply often refutes many of them; trying it early finds those
    /// cutoffs without searching the other quiet moves first.
    fn order_score(&self, mv: Move, hash_move: Move, killers: [Move; 2]) -> i32 {
        if mv == hash_move {
            return ORDER_HASH_MOVE;
        }
        if mv == killers[0] {
            return ORDER_KILLER_FIRST;
        }
        if mv == killers[1] {
            return ORDER_KILLER_SECOND;
        }
        let kind_on = |square| {
            self.position
                .piece_on(square)
                .map_or(PieceKind::Pawn, |piece| piece.kind())
        };
        let mut score = 0;
        if mv.is_capture() {
            // The en-passant victim is not on the destination square.
            let victim = if mv.is_en_passant() {
                PieceKind::Pawn
            } else {
                kind_on(mv.to())
            };
            let base = if see_ge(&self.position, mv, 0) {
                ORDER_CAPTURE
            } else {
                ORDER_BAD_CAPTURE
            };
            score += base + victim.index() as i32 * ORDER_VICTIM_WEIGHT
                - kind_on(mv.from()).index() as i32;
        }
        match mv.promotion() {
            Some(PieceKind::Queen) => score += ORDER_QUEEN_PROMOTION,
            Some(kind) => score += kind.index() as i32,
            None if !mv.is_capture() => {
                let side = self.position.side_to_move().index();
                score = self.history[side][mv.from().index()][mv.to().index()];
            }
            None => {}
        }
        score
    }

    /// Adjusts the history score of the quiet move `mv` for the side to
    /// move by `bonus`, which is negative for a move that failed.
    ///
    /// The update is `bonus - score * |bonus| / HISTORY_MAX`. The second
    /// term pulls the score back towards zero in proportion to its size, so
    /// it can never pass `HISTORY_MAX` in either direction, and a move's
    /// recent record outweighs its distant past.
    fn update_history(&mut self, mv: Move, bonus: i32) {
        let side = self.position.side_to_move().index();
        let score = &mut self.history[side][mv.from().index()][mv.to().index()];
        *score += bonus - *score * bonus.abs() / HISTORY_MAX;
    }

    /// Returns the ordering scores of every move in `list`. Pass
    /// [`Move::NULL`] for a hash move or killer that is not available.
    fn score_moves(
        &self,
        list: &MoveList,
        hash_move: Move,
        killers: [Move; 2],
    ) -> [i32; MoveList::CAPACITY] {
        let mut scores = [0; MoveList::CAPACITY];
        for (score, &mv) in scores.iter_mut().zip(list.iter()) {
            *score = self.order_score(mv, hash_move, killers);
        }
        scores
    }

    /// Moves the highest-scored move among `list[index..]` to `index` and
    /// returns it.
    ///
    /// Sorting is done one move at a time because most nodes end in a
    /// cutoff after the first move or two, so a full sort would be wasted.
    /// Ties keep their original order, which keeps the search deterministic.
    #[inline]
    fn pick_move(list: &mut MoveList, scores: &mut [i32], index: usize) -> Move {
        let mut best = index;
        for candidate in index + 1..list.len() {
            if scores[candidate] > scores[best] {
                best = candidate;
            }
        }
        // Rotate rather than swap, so the skipped moves keep their order.
        list[index..=best].rotate_right(1);
        scores[index..=best].rotate_right(1);
        list[index]
    }

    /// Records `mv` followed by the child's best line as the best line at
    /// `ply`.
    fn update_pv(&mut self, ply: usize, mv: Move) {
        let child_len = self.pv_len[ply + 1].min(MAX_PLY - 1);
        let (parents, children) = self.pv.split_at_mut(ply + 1);
        let line = &mut parents[ply];
        line[0] = mv;
        line[1..=child_len].copy_from_slice(&children[0][..child_len]);
        self.pv_len[ply] = child_len + 1;
    }

    /// Searches the current position to `depth` plies and returns its score
    /// for the side to move, given the window `(alpha, beta)`. `ply` is the
    /// distance from the root.
    ///
    /// If the search has been interrupted the return value is meaningless
    /// and must be discarded; callers check `self.aborted`.
    fn negamax(&mut self, mut depth: i32, ply: usize, mut alpha: i32, mut beta: i32) -> i32 {
        if depth <= 0 {
            return self.quiescence(ply, alpha, beta);
        }
        self.pv_len[ply] = 0;
        if self.enter_node() {
            return DRAW;
        }
        self.seldepth = self.seldepth.max(ply);

        if ply > 0 {
            // Draws that need no move generation (SRC-4).
            if self.position.is_repetition(ply) || self.position.has_insufficient_material() {
                return DRAW;
            }
            if ply >= MAX_PLY {
                return self.evaluator.evaluate(&self.position);
            }

            // Mate-distance pruning. No line from here can score better
            // than mating on the next move, or worse than being mated on
            // the next move, so the window is clamped to those values. If
            // a shorter mate has already been found elsewhere, the window
            // becomes empty and the node is cut off at once: nothing it
            // could find would be preferred. This only prunes lines that
            // cannot change the result, so it is exact.
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        // Transposition table cutoff. The same position is often reached by
        // different move orders; if it has already been searched at least as
        // deeply as is needed now, the stored result can be reused instead
        // of searching again. An exact score is always usable. A lower
        // bound is enough when it already reaches beta, and an upper bound
        // when it does not exceed alpha, because in both cases the node's
        // outcome relative to the window is settled. The root is excluded,
        // since it must produce a move, not just a score.
        let key = self.position.key();
        let entry = self.table.probe(key, ply);
        if ply > 0
            && let Some(entry) = entry
            && entry.depth >= depth
        {
            let usable = match entry.bound {
                Bound::Exact => true,
                Bound::Lower => entry.score >= beta,
                Bound::Upper => entry.score <= alpha,
                Bound::None => false,
            };
            if usable {
                return entry.score;
            }
        }

        // Internal iterative reduction. A node with no hash move is one
        // the search has not seen before at a useful depth, or whose
        // entry has been overwritten, and without a first move to try it
        // is expensive to search: the ordering falls back on killers and
        // history alone. Searching it one ply shallower costs little
        // accuracy and fills the table, so that if the node is reached
        // again (as it usually is in the next iteration) it has a hash
        // move. Only deep enough nodes are reduced, where the saving is
        // worth the lost ply.
        if ply > 0 && depth >= IIR_MIN_DEPTH && entry.is_none_or(|entry| entry.mv == Move::NULL) {
            depth -= 1;
        }

        let in_check = self.position.in_check();
        // Not meaningful in check, where it is never used.
        let static_eval = if in_check {
            -INFINITE
        } else {
            self.evaluator.evaluate(&self.position)
        };

        // Reverse futility pruning. Near the horizon, if the static
        // evaluation is above beta by more than the opponent could
        // plausibly win back in the remaining depth, the node is assumed to
        // fail high without searching it. The margin grows with depth
        // because more can change in a deeper search. It is skipped in
        // check, in PV nodes, which need exact scores, and when beta is a
        // mate score, which no evaluation margin can justify.
        if !in_check
            && beta - alpha == 1
            && depth <= RFP_MAX_DEPTH
            && beta.abs() < MATE_BOUND
            && static_eval - RFP_MARGIN * depth >= beta
        {
            return static_eval;
        }

        // Null-move pruning. Having the move is almost always an advantage,
        // so if the side to move can pass and a reduced search still scores
        // at least beta, a real move would do so too and the node can be
        // cut off without searching any. The reduced search is much cheaper
        // than the moves it replaces.
        //
        // It is skipped where the assumption fails or the test is
        // pointless:
        // * in check, where passing is illegal;
        // * in PV nodes (a window wider than one), which need exact scores;
        // * when the static evaluation is already below beta, since
        //   passing is then unlikely to reach it;
        // * with only pawns and a king, where zugzwang, a position in which
        //   every move is worse than passing, is common;
        // * directly after another null move, which would just hand the
        //   move back and search the same position at lower depth.
        //
        // A mate score from the null search is not trusted, because the
        // mate was found with the opponent's help; beta is returned
        // instead.
        let side = self.position.side_to_move();
        if !in_check
            && beta - alpha == 1
            && depth >= NULL_MOVE_MIN_DEPTH
            && ply > 0
            && !self.null_moved[ply - 1]
            && self.position.has_non_pawn_material(side)
            && static_eval >= beta
        {
            let reduction = NULL_MOVE_REDUCTION + depth / NULL_MOVE_DEPTH_DIVISOR;
            self.null_moved[ply] = true;
            self.position.make_null_move();
            let score = -self.negamax(depth - 1 - reduction, ply + 1, -beta, -beta + 1);
            self.position.unmake_null_move();
            self.null_moved[ply] = false;
            if self.aborted {
                return DRAW;
            }
            if score >= beta {
                return if score >= MATE_BOUND { beta } else { score };
            }
        }

        let mut list = MoveList::new();
        generate(&self.position, GenKind::All, &mut list);
        if list.is_empty() {
            // No legal moves: checkmate if in check, otherwise stalemate.
            return if in_check { -MATE + ply as i32 } else { DRAW };
        }
        // The fifty-move rule is tested only now because a checkmate
        // delivered by the hundredth halfmove takes precedence over it.
        if ply > 0 && self.position.halfmove_clock() >= 100 {
            return DRAW;
        }

        // The hash move comes from the table and may belong to another
        // position, so it is only ever used to reorder moves that were
        // generated for this one, never played directly (SRC-8). At the
        // root the previous iteration's best move is used instead, which
        // cannot have been overwritten.
        let hash_move = match self.root_pv.first() {
            Some(&mv) if ply == 0 => mv,
            _ => entry.map_or(Move::NULL, |entry| entry.mv),
        };
        let mut scores = self.score_moves(&list, hash_move, self.killers[ply]);

        // Futility pruning. Near the horizon, a quiet move is unlikely to
        // raise the score above alpha when the static evaluation is below
        // alpha by more than a quiet move could plausibly gain in the
        // remaining depth, so such moves are skipped. The margin grows
        // with depth because more can change in a deeper search. Captures
        // and promotions are always searched, since they can change the
        // material balance at once, and so are moves that give check,
        // whose consequences a static evaluation cannot see. Only moves
        // are pruned, never the node: the first move is always searched,
        // so the node still has a real score and a move to store. The
        // pruning is skipped in check, where every move is an evasion, in
        // PV nodes, which need exact scores, and when alpha is a mate
        // score, which no evaluation margin can relate to.
        let futile = !in_check
            && beta - alpha == 1
            && depth <= FUTILITY_MAX_DEPTH
            && alpha.abs() < MATE_BOUND
            && static_eval + FUTILITY_MARGIN_BASE + FUTILITY_MARGIN_PER_PLY * depth <= alpha;

        let mut best_score = -INFINITE;
        let mut best_move = Move::NULL;
        // Quiet moves searched so far that did not cause a cutoff.
        let mut failed_quiets = [Move::NULL; 64];
        let mut failed_count = 0;
        // Late move pruning. Near the horizon in a non-PV node, once a
        // number of quiet moves have been searched and none has beaten
        // alpha, the remaining quiet moves, which the ordering ranks
        // lower still, are skipped outright. The number grows with the
        // square of the depth, since a deeper search has more to lose
        // from a wrong skip. Captures and promotions are never skipped,
        // and nothing is skipped in check. It is the quiet moves
        // *searched* that are counted, not the position in the list, so
        // moves skipped for other reasons do not use up the allowance.
        let prune_late = !in_check && beta - alpha == 1 && depth <= LMP_MAX_DEPTH;
        let late_limit = LMP_BASE + LMP_DEPTH_SCALE * depth * depth;
        let mut quiets_searched = 0;
        for index in 0..list.len() {
            let mv = Searcher::pick_move(&mut list, &mut scores, index);
            let quiet = !mv.is_capture() && mv.promotion().is_none();
            if prune_late && quiet && quiets_searched >= late_limit {
                continue;
            }
            self.make(mv);
            // Whether the move gives check is known only once it is made.
            let gives_check = self.position.in_check();
            if futile && quiet && index > 0 && !gives_check {
                self.unmake(mv);
                continue;
            }
            // Check extension. A move that gives check without hanging the
            // checking piece is searched one ply deeper. The reply is
            // forced, so the extra ply costs little, and it keeps a
            // forcing sequence from being cut off at the horizon one move
            // short of its point. Checks that lose material by static
            // exchange are left alone: most are spite checks that only
            // delay the inevitable, and extending every one of them cost
            // more depth than it found (see docs/EXPERIMENTS.md, run 11).
            // The exchange is evaluated in the position before the move,
            // so the move is briefly taken back; checks are rare enough
            // for that to be cheap.
            let extend = gives_check && {
                self.unmake(mv);
                let safe = see_ge(&self.position, mv, 0);
                self.make(mv);
                safe
            };
            let new_depth = depth - 1 + i32::from(extend);
            // Principal variation search. With good move ordering the
            // first move is usually best, so the others only need to be
            // shown to be no better. That is asked with a *null window*
            // `(alpha, alpha + 1)`, which no score can fall inside: the
            // search can only answer "at most alpha" or "more than alpha",
            // and it cuts off far sooner than with a real window. If a
            // later move does beat alpha (and the window is not already
            // null), its true score is needed, so it is searched again with
            // the full window. The re-searches cost less than the null
            // windows save.
            //
            // Late move reductions go a step further for quiet moves late
            // in the ordering: the null-window search is also made
            // shallower. Almost all such moves fail low and are dismissed
            // cheaply. One that unexpectedly beats alpha is searched again
            // at full depth before it is believed, so a reduction can cost
            // time but cannot by itself change the result. Moves are not
            // reduced when in check or when they give check, where a
            // shallow search is most likely to miss something, and the
            // reduced depth never drops below one ply, so the move still
            // gets a real search rather than quiescence only.
            let score = if index == 0 {
                -self.negamax(new_depth, ply + 1, -beta, -alpha)
            } else {
                let mut reduction = 0;
                if quiet
                    && depth >= LMR_MIN_DEPTH
                    && index >= LMR_FULL_DEPTH_MOVES
                    && !in_check
                    && !gives_check
                {
                    let table = LMR_TABLE[depth.min(63) as usize][index.min(63)] as i32;
                    reduction = table.min(depth - 2);
                }
                let mut probe = -self.negamax(new_depth - reduction, ply + 1, -alpha - 1, -alpha);
                if reduction > 0 && probe > alpha {
                    probe = -self.negamax(new_depth, ply + 1, -alpha - 1, -alpha);
                }
                if probe > alpha && probe < beta {
                    -self.negamax(new_depth, ply + 1, -beta, -alpha)
                } else {
                    probe
                }
            };
            self.unmake(mv);
            if quiet {
                quiets_searched += 1;
            }
            if self.aborted {
                return DRAW;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mv;
                    self.update_pv(ply, mv);
                    if ply == 0 {
                        self.root_pv.clear();
                        self.root_pv
                            .extend_from_slice(&self.pv[0][..self.pv_len[0]]);
                    }
                    if alpha >= beta {
                        // Remember a quiet refutation. Captures and
                        // promotions are already ordered early.
                        if quiet {
                            if self.killers[ply][0] != mv {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = mv;
                            }
                            // Reward the move that cut off and penalise
                            // the quiet moves that were tried before it
                            // and did not, so that next time, in any
                            // position, it is tried ahead of them.
                            let bonus =
                                (HISTORY_BONUS_SCALE * depth * depth).min(HISTORY_BONUS_MAX);
                            self.update_history(mv, bonus);
                            for &failed in &failed_quiets[..failed_count] {
                                self.update_history(failed, -bonus);
                            }
                        }
                        break;
                    }
                }
            }
            if quiet && failed_count < failed_quiets.len() {
                failed_quiets[failed_count] = mv;
                failed_count += 1;
            }
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_move != Move::NULL {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.table
            .store(key, best_move, best_score, depth, bound, ply);
        best_score
    }

    /// Resolves captures and checks at the search horizon and returns the
    /// score of the resulting quiet position (SRC-2).
    ///
    /// Stopping the search at a fixed depth and evaluating would misjudge
    /// any position in the middle of an exchange: a queen that has just
    /// captured a defended pawn looks a pawn up. So captures and promotions
    /// are searched on until none remain. The side to move may also decline
    /// to capture at all: the static evaluation (the *stand-pat* score) is
    /// a lower bound on its score, since it can usually make some quiet
    /// move that keeps it.
    ///
    /// That reasoning fails in check, where there may be no safe quiet
    /// move, so in check every evasion is searched and there is no stand
    /// pat; a position with no evasion is checkmate.
    fn quiescence(&mut self, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.pv_len[ply] = 0;
        if self.enter_node() {
            return DRAW;
        }
        self.seldepth = self.seldepth.max(ply);
        if ply >= MAX_PLY {
            return self.evaluator.evaluate(&self.position);
        }

        let in_check = self.position.in_check();
        let mut best_score = -INFINITE;
        let mut list = MoveList::new();
        if in_check {
            generate(&self.position, GenKind::All, &mut list);
            if list.is_empty() {
                return -MATE + ply as i32;
            }
        } else {
            best_score = self.evaluator.evaluate(&self.position);
            if best_score >= beta {
                return best_score;
            }
            alpha = alpha.max(best_score);
            generate(&self.position, GenKind::Noisy, &mut list);
        }

        let mut scores = self.score_moves(&list, Move::NULL, [Move::NULL; 2]);
        for index in 0..list.len() {
            let mv = Searcher::pick_move(&mut list, &mut scores, index);
            // A capture that loses material by static exchange evaluation
            // is almost never the move that raises the score here: the
            // stand-pat score already offers what declining it gives.
            // Skipping such captures cuts the quiescence tree sharply at
            // little cost. Evasions are never skipped, since in check
            // there is no stand pat to fall back on.
            if !in_check && mv.is_capture() && !see_ge(&self.position, mv, 0) {
                continue;
            }
            self.make(mv);
            let score = -self.quiescence(ply + 1, -beta, -alpha);
            self.unmake(mv);
            if self.aborted {
                return DRAW;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Searches `fen` to `depth` and returns the result and the last report.
    fn search(fen: &str, depth: i32) -> (SearchResult, String) {
        let table = TranspositionTable::new(1).unwrap();
        let stop = AtomicBool::new(false);
        let limits = Limits {
            depth: Some(depth),
            ..Limits::default()
        };
        let position = Position::from_fen(fen).unwrap();
        let mut searcher = Searcher::new(position, &table, &stop, &limits, Instant::now(), 0);
        let mut last = String::new();
        let result = searcher.run(&mut |info| last = info.to_string());
        (result, last)
    }

    fn best(fen: &str, depth: i32) -> String {
        search(fen, depth).0.best_move.unwrap().to_string()
    }

    #[test]
    fn finds_mate_in_one() {
        let (result, info) = search("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 2);
        assert_eq!(result.best_move.unwrap().to_string(), "a1a8");
        assert_eq!(result.score, MATE - 1);
        assert!(info.contains("score mate 1 "), "{info}");
    }

    #[test]
    fn finds_mate_in_two_and_reports_being_mated() {
        // 1. Ra7 Kg8 2. Rb8#.
        let (result, info) = search("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1", 4);
        assert_eq!(result.score, MATE - 3);
        assert!(info.contains("score mate 2 "), "{info}");

        // The defender sees the same mate from the other side.
        let (result, info) = search("7k/R7/8/8/8/8/8/1R4K1 b - - 0 1", 4);
        assert_eq!(result.score, -(MATE - 2));
        assert!(info.contains("score mate -1 "), "{info}");
    }

    #[test]
    fn wins_hanging_material() {
        assert_eq!(best("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", 3), "d2d5");
    }

    #[test]
    fn quiescence_avoids_losing_exchanges() {
        // Taking the defended pawn loses the queen for a pawn.
        let (result, _) = search("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1", 1);
        assert_ne!(result.best_move.unwrap().to_string(), "d2d5");
    }

    #[test]
    fn no_legal_moves_gives_no_best_move() {
        let (result, _) = search("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3);
        assert_eq!(result.best_move, None);
        let (result, _) = search("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", 3);
        assert_eq!((result.best_move, result.score), (None, -MATE));
    }

    #[test]
    fn insufficient_material_is_a_draw() {
        assert_eq!(search("4k3/8/8/8/8/8/8/4KN2 w - - 0 1", 4).0.score, DRAW);
    }

    #[test]
    fn fixed_depth_search_is_deterministic() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let first = search(fen, 4).0;
        let second = search(fen, 4).0;
        assert_eq!(first, second);
        assert!(first.nodes > 0);
    }

    #[test]
    fn node_limit_stops_the_search() {
        let table = TranspositionTable::new(1).unwrap();
        let stop = AtomicBool::new(false);
        let limits = Limits {
            nodes: Some(5_000),
            ..Limits::default()
        };
        let mut searcher = Searcher::new(
            Position::startpos(),
            &table,
            &stop,
            &limits,
            Instant::now(),
            0,
        );
        let result = searcher.run(&mut |_| {});
        assert!(result.best_move.is_some());
        assert_eq!(result.nodes, 5_000);
    }

    #[test]
    fn stop_flag_ends_an_unlimited_search() {
        let table = TranspositionTable::new(1).unwrap();
        let stop = AtomicBool::new(true);
        let mut searcher = Searcher::new(
            Position::startpos(),
            &table,
            &stop,
            &Limits::default(),
            Instant::now(),
            0,
        );
        // The first iteration always completes, so there is a move to play.
        let result = searcher.run(&mut |_| {});
        assert!(result.best_move.is_some());
        assert_eq!(result.depth, 1);
    }
}
