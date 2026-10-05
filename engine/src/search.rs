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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::{Evaluator, PstEvaluator};
use crate::movegen::{GenKind, generate};
use crate::moves::{Move, MoveList};
use crate::params::{
    NODE_POLL_MASK, ORDER_CAPTURE, ORDER_HASH_MOVE, ORDER_QUEEN_PROMOTION, ORDER_VICTIM_WEIGHT,
};
use crate::position::Position;
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
            let score = self.negamax(depth, 0, -INFINITE, INFINITE);
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

    /// Counts a node and reports whether the search has been interrupted.
    ///
    /// The node limit is compared at every node so that node-limited
    /// searches are exactly reproducible. The clock and the stop flag are
    /// consulted only every [`NODE_POLL_MASK`] + 1 nodes, because reading
    /// them at every node would cost more than the search itself (SRC-6).
    #[inline]
    fn enter_node(&mut self) -> bool {
        self.nodes += 1;
        if self.aborted || !self.may_abort {
            return self.aborted;
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
    /// then captures by most valuable victim and least valuable attacker,
    /// then everything else in generation order (SRC-3).
    fn order_score(&self, mv: Move, hash_move: Move) -> i32 {
        if mv == hash_move {
            return ORDER_HASH_MOVE;
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
            score += ORDER_CAPTURE + victim.index() as i32 * ORDER_VICTIM_WEIGHT
                - kind_on(mv.from()).index() as i32;
        }
        match mv.promotion() {
            Some(PieceKind::Queen) => score += ORDER_QUEEN_PROMOTION,
            Some(kind) => score += kind.index() as i32,
            None => {}
        }
        score
    }

    /// Returns the ordering scores of every move in `list`.
    fn score_moves(&self, list: &MoveList, hash_move: Move) -> [i32; MoveList::CAPACITY] {
        let mut scores = [0; MoveList::CAPACITY];
        for (score, &mv) in scores.iter_mut().zip(list.iter()) {
            *score = self.order_score(mv, hash_move);
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
    fn negamax(&mut self, depth: i32, ply: usize, mut alpha: i32, beta: i32) -> i32 {
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
        }

        let mut list = MoveList::new();
        generate(&self.position, GenKind::All, &mut list);
        if list.is_empty() {
            // No legal moves: checkmate if in check, otherwise stalemate.
            return if self.position.in_check() {
                -MATE + ply as i32
            } else {
                DRAW
            };
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
        let key = self.position.key();
        let hash_move = match self.root_pv.first() {
            Some(&mv) if ply == 0 => mv,
            _ => self
                .table
                .probe(key, ply)
                .map_or(Move::NULL, |entry| entry.mv),
        };
        let mut scores = self.score_moves(&list, hash_move);

        let mut best_score = -INFINITE;
        let mut best_move = Move::NULL;
        for index in 0..list.len() {
            let mv = Searcher::pick_move(&mut list, &mut scores, index);
            self.make(mv);
            let score = -self.negamax(depth - 1, ply + 1, -beta, -alpha);
            self.unmake(mv);
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
                        break;
                    }
                }
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

        let mut scores = self.score_moves(&list, Move::NULL);
        for index in 0..list.len() {
            let mv = Searcher::pick_move(&mut list, &mut scores, index);
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
