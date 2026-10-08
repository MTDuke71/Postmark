//! The Universal Chess Interface: the text protocol through which a GUI or
//! match runner drives the engine (UCI-1 to UCI-7).
//!
//! Commands arrive one per line on standard input and are handled on the
//! thread that reads them. A search runs on its own thread, so the reader
//! stays free to answer `isready` and to act on `stop` while the engine is
//! thinking (UCI-5).

use std::fmt::Display;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::bench;
use crate::eval::{Eval, Evaluator};
use crate::movegen::{GenKind, generate};
use crate::moves::{Move, MoveList};
use crate::nnue::Network;
use crate::params::{DEFAULT_HASH_MB, DEFAULT_MOVE_OVERHEAD_MS};
use crate::perft::divide;
use crate::position::Position;
use crate::search::Searcher;
use crate::timeman::Limits;
use crate::tt::TranspositionTable;
use crate::types::Color;

/// Largest transposition table the `Hash` option accepts, in megabytes.
const MAX_HASH_MB: usize = 65_536;

/// Largest value the `Move Overhead` option accepts, in milliseconds.
const MAX_MOVE_OVERHEAD_MS: u64 = 5_000;

/// Stack size of the search thread. The search recurses once per ply and
/// each level keeps a move list and its ordering scores on the stack, so it
/// is given more room than the default.
const SEARCH_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Writes one line to standard output.
///
/// Write errors are ignored: if the GUI has closed the pipe there is nobody
/// left to tell, and the engine must not crash because of it.
fn send(line: impl Display) {
    let _ = writeln!(io::stdout(), "{line}");
}

/// Returns the legal move in `position` whose UCI notation is `text`, or
/// `None` if there is no such move.
///
/// Matching against the generated moves both validates the move and
/// recovers the details the notation leaves out (capture, castling, en
/// passant).
pub fn parse_move(position: &Position, text: &str) -> Option<Move> {
    let mut list = MoveList::new();
    generate(position, GenKind::All, &mut list);
    list.iter().copied().find(|mv| mv.to_string() == text)
}

/// The engine as seen through UCI: the current position, the options, and
/// the search in progress, if any.
pub struct Engine {
    /// Position set by the last `position` command.
    position: Position,
    /// Transposition table, shared with the search thread.
    table: Arc<TranspositionTable>,
    /// Set to ask the search thread to finish.
    stop: Arc<AtomicBool>,
    /// The running search thread, if one has been started and not joined.
    search: Option<JoinHandle<()>>,
    /// Whether the running search was started with `go infinite`, and so
    /// will not end by itself.
    infinite: bool,
    /// Value of the `Move Overhead` option, in milliseconds.
    move_overhead: u64,
    /// The network loaded by the `EvalFile` option, if any. Searches use
    /// it when present and the hand-crafted tables otherwise.
    network: Option<Arc<Network>>,
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

impl Engine {
    /// Creates an engine at the start position with default options.
    ///
    /// # Panics
    ///
    /// Panics if even a one-megabyte transposition table cannot be
    /// allocated.
    pub fn new() -> Engine {
        let table = TranspositionTable::new(DEFAULT_HASH_MB)
            .or_else(|| TranspositionTable::new(1))
            .expect("not enough memory for a transposition table");
        Engine {
            position: Position::startpos(),
            table: Arc::new(table),
            stop: Arc::new(AtomicBool::new(false)),
            search: None,
            infinite: false,
            move_overhead: DEFAULT_MOVE_OVERHEAD_MS,
            network: None,
        }
    }

    /// Reads and executes commands from standard input until `quit` or end
    /// of input.
    ///
    /// At end of input a search that will finish by itself is allowed to,
    /// so that commands can be piped in from a file; an infinite search is
    /// stopped.
    pub fn run(&mut self) {
        for line in io::stdin().lock().lines() {
            match line {
                Ok(line) => {
                    if !self.execute(&line) {
                        return;
                    }
                }
                // A line that is not valid UTF-8 cannot be a command.
                Err(error) if error.kind() == io::ErrorKind::InvalidData => {}
                Err(_) => break,
            }
        }
        if self.infinite {
            self.stop_search();
        } else {
            self.wait_for_search();
        }
    }

    /// Executes one command line. Returns `false` if the command was `quit`
    /// and `true` otherwise. Unknown or malformed commands are ignored and
    /// never cause a panic (UCI-6).
    pub fn execute(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&command, arguments)) = tokens.split_first() else {
            return true;
        };
        match command {
            "uci" => self.identify(),
            "isready" => send("readyok"),
            "ucinewgame" => {
                self.stop_search();
                self.table.clear();
                self.position = Position::startpos();
            }
            "position" => {
                self.stop_search();
                self.set_position(arguments);
            }
            "setoption" => {
                self.stop_search();
                self.set_option(arguments);
            }
            "go" => self.go(arguments),
            "stop" => self.stop_search(),
            "quit" => {
                self.stop_search();
                return false;
            }
            "bench" => {
                self.stop_search();
                let depth = arguments
                    .first()
                    .and_then(|depth| depth.parse().ok())
                    .unwrap_or(bench::DEFAULT_DEPTH);
                let (nodes, time) = bench::run(depth, self.network.as_ref());
                let nps = nodes as u128 * 1000 / time.as_millis().max(1);
                send(format_args!("{nodes} nodes {nps} nps"));
            }
            "eval" => {
                self.stop_search();
                self.print_eval();
            }
            "perft" => {
                self.stop_search();
                match arguments.first().and_then(|depth| depth.parse().ok()) {
                    Some(depth) => self.perft(depth),
                    None => send("info string usage: perft <depth>"),
                }
            }
            "d" => send(&self.position),
            _ => {}
        }
        true
    }

    /// Answers the `uci` command with the engine's identity and options.
    fn identify(&self) {
        send(format_args!("id name {} {}", crate::NAME, crate::VERSION));
        send("id author MTDuke71");
        send(format_args!(
            "option name Hash type spin default {DEFAULT_HASH_MB} min 1 max {MAX_HASH_MB}"
        ));
        // Accepted so that GUIs can set it; one thread until Lazy SMP.
        send("option name Threads type spin default 1 min 1 max 1");
        send(format_args!(
            "option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max {MAX_MOVE_OVERHEAD_MS}"
        ));
        send("option name EvalFile type string default <empty>");
        send("uciok");
    }

    /// Handles the `eval` command: prints the static evaluation of the
    /// current position, in centipawns for the side to move, from the
    /// hand-crafted tables and, if a network is loaded, from the network.
    fn print_eval(&self) {
        let pst = Eval::new(None, &self.position).evaluate(&self.position);
        send(format_args!("pst {pst}"));
        if let Some(network) = &self.network {
            let nnue = Eval::new(Some(network), &self.position).evaluate(&self.position);
            send(format_args!("nnue {nnue}"));
        }
    }

    /// Handles `setoption name EvalFile value <path>`: loads the network at
    /// `path`, or unloads the current one if `path` is empty or `<empty>`.
    /// A file that cannot be loaded is reported and leaves the current
    /// network in place.
    fn set_eval_file(&mut self, path: &str) {
        if path.is_empty() || path == "<empty>" {
            self.network = None;
            return;
        }
        match Network::from_file(path) {
            Ok(network) => {
                send(format_args!(
                    "info string loaded network {path} (hidden size {})",
                    network.hidden()
                ));
                self.network = Some(Arc::new(network));
            }
            Err(error) => send(format_args!("info string cannot load {path}: {error}")),
        }
    }

    /// Handles `position (startpos | fen <fen>) [moves <move>...]`.
    ///
    /// The moves are played through [`Position::make_move`], so the position
    /// remembers the game so far, which repetition detection relies on. If
    /// the FEN is invalid or any move is illegal the whole command is
    /// ignored and the current position is kept.
    fn set_position(&mut self, arguments: &[&str]) {
        let moves_at = arguments
            .iter()
            .position(|&token| token == "moves")
            .unwrap_or(arguments.len());
        let (setup, moves) = arguments.split_at(moves_at);

        let mut position = match setup.split_first() {
            Some((&"startpos", _)) => Position::startpos(),
            Some((&"fen", fields)) => match Position::from_fen(&fields.join(" ")) {
                Ok(position) => position,
                Err(error) => {
                    send(format_args!("info string invalid FEN: {error}"));
                    return;
                }
            },
            _ => return,
        };

        // `moves` still starts with the "moves" keyword itself.
        for text in moves.iter().skip(1) {
            let Some(mv) = parse_move(&position, text) else {
                send(format_args!("info string illegal move: {text}"));
                return;
            };
            position.make_move(mv);
        }
        self.position = position;
    }

    /// Handles `setoption name <name> [value <value>]`. Option names are
    /// matched without regard to case, as the protocol requires.
    fn set_option(&mut self, arguments: &[&str]) {
        let value_at = arguments
            .iter()
            .position(|&token| token == "value")
            .unwrap_or(arguments.len());
        let (name, value) = arguments.split_at(value_at);
        if name.first() != Some(&"name") {
            return;
        }
        let name = name[1..].join(" ").to_ascii_lowercase();
        // A file path may contain spaces, so the value keeps all its tokens.
        let value = value.get(1..).unwrap_or(&[]).join(" ");
        let value = value.as_str();

        match name.as_str() {
            "hash" => {
                let Ok(megabytes) = value.parse::<usize>() else {
                    return;
                };
                match TranspositionTable::new(megabytes.clamp(1, MAX_HASH_MB)) {
                    Some(table) => self.table = Arc::new(table),
                    None => send("info string not enough memory for that hash size"),
                }
            }
            "move overhead" => {
                if let Ok(overhead) = value.parse::<u64>() {
                    self.move_overhead = overhead.min(MAX_MOVE_OVERHEAD_MS);
                }
            }
            "evalfile" => self.set_eval_file(value),
            // "threads" is accepted and has no effect yet.
            _ => {}
        }
    }

    /// Handles `go`, starting a search on its own thread.
    ///
    /// The thread prints an `info` line for each completed depth and
    /// finally `bestmove`. For `go infinite` the best move is withheld
    /// until `stop` arrives, as the protocol requires, even if the search
    /// itself has run out of depth.
    fn go(&mut self, arguments: &[&str]) {
        // Taken first, so that everything below counts against the clock.
        let start = Instant::now();
        self.stop_search();

        let mut limits = Limits::default();
        let mut tokens = arguments.iter();
        while let Some(&token) = tokens.next() {
            // Clock values can be negative when a GUI reports a flag fall.
            let mut number = || {
                let value = tokens.next().and_then(|text| text.parse::<i64>().ok());
                value.map(|value| value.max(0) as u64)
            };
            match token {
                "wtime" => limits.time[Color::White.index()] = number(),
                "btime" => limits.time[Color::Black.index()] = number(),
                "winc" => limits.increment[Color::White.index()] = number().unwrap_or(0),
                "binc" => limits.increment[Color::Black.index()] = number().unwrap_or(0),
                "movestogo" => limits.moves_to_go = number(),
                "movetime" => limits.movetime = number(),
                "nodes" => limits.nodes = number(),
                "depth" => limits.depth = number().map(|depth| depth.min(i32::MAX as u64) as i32),
                "infinite" => limits.infinite = true,
                "perft" => {
                    if let Some(depth) = number() {
                        self.perft(depth.min(u32::MAX as u64) as u32);
                    }
                    return;
                }
                _ => {}
            }
        }

        self.stop.store(false, Ordering::Relaxed);
        self.table.new_search();
        self.infinite = limits.infinite;

        let position = self.position.clone();
        let table = Arc::clone(&self.table);
        let stop = Arc::clone(&self.stop);
        let overhead = self.move_overhead;
        let network = self.network.clone();
        let search = move || {
            let mut searcher = Searcher::new(
                position,
                &table,
                &stop,
                &limits,
                start,
                overhead,
                network.as_ref(),
            );
            let result = searcher.run(&mut |info| send(info));
            while limits.infinite && !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(1));
            }
            match result.best_move {
                Some(mv) => send(format_args!("bestmove {mv}")),
                // The protocol's notation for "no move": mate or stalemate.
                None => send("bestmove 0000"),
            }
        };
        match thread::Builder::new()
            .name("search".to_string())
            .stack_size(SEARCH_STACK_BYTES)
            .spawn(search)
        {
            Ok(handle) => self.search = Some(handle),
            Err(error) => send(format_args!("info string cannot start search: {error}")),
        }
    }

    /// Asks the running search, if any, to finish, and waits until it has
    /// printed its best move.
    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait_for_search();
    }

    /// Waits for the running search, if any, to finish by itself. Must not
    /// be called while an infinite search is running, which never does.
    pub fn wait_for_search(&mut self) {
        if let Some(handle) = self.search.take() {
            // A panic in the search thread has already been reported.
            let _ = handle.join();
        }
        self.infinite = false;
    }

    /// Runs perft on the current position to `depth`, printing the count
    /// below each move, then the total, the time and the speed (MGN-5).
    fn perft(&mut self, depth: u32) {
        let start = Instant::now();
        let parts = divide(&mut self.position, depth);
        // At depth 0 there are no moves to list; the only node is the root.
        let nodes = if depth == 0 {
            1
        } else {
            parts.iter().map(|&(_, nodes)| nodes).sum()
        };
        let elapsed = start.elapsed();

        for (mv, nodes) in &parts {
            send(format_args!("{mv}: {nodes}"));
        }
        send("");
        send(format_args!("Nodes: {nodes}"));
        send(format_args!("Time: {} ms", elapsed.as_millis()));
        let nps = nodes as u128 * 1_000_000 / elapsed.as_micros().max(1);
        send(format_args!("NPS: {nps}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::START_FEN;

    #[test]
    fn position_command_sets_up_the_board() {
        let mut engine = Engine::new();
        engine.execute("position startpos moves e2e4 e7e5 g1f3");
        assert_eq!(
            engine.position.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        );
        engine.execute("position fen 4k3/8/8/8/8/8/8/4K2R w K - 0 1 moves e1g1");
        assert_eq!(engine.position.to_fen(), "4k3/8/8/8/8/8/8/5RK1 b - - 1 1");
        engine.execute("position startpos");
        assert_eq!(engine.position.to_fen(), START_FEN);
    }

    #[test]
    fn bad_position_commands_are_ignored() {
        let mut engine = Engine::new();
        engine.execute("position startpos moves e2e4");
        let before = engine.position.to_fen();
        for command in [
            "position",
            "position nonsense",
            "position fen",
            "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
            "position startpos moves e2e5",
            "position startpos moves e2e4 e2e4",
            "position startpos moves garbage",
        ] {
            assert!(engine.execute(command));
            assert_eq!(engine.position.to_fen(), before, "{command}");
        }
    }

    #[test]
    fn promotion_and_castling_moves_are_parsed() {
        let position = Position::from_fen("r3k2r/1P6/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let promotion = parse_move(&position, "b7a8n").unwrap();
        assert!(promotion.is_capture() && promotion.promotion().is_some());
        assert!(parse_move(&position, "e1c1").unwrap().is_castle());
        assert_eq!(parse_move(&position, "b7b8"), None);
        assert_eq!(parse_move(&position, "e1e3"), None);
    }

    #[test]
    fn options_are_applied_and_bad_values_ignored() {
        let mut engine = Engine::new();
        engine.execute("setoption name Move Overhead value 50");
        assert_eq!(engine.move_overhead, 50);
        engine.execute("setoption name move overhead value 999999");
        assert_eq!(engine.move_overhead, MAX_MOVE_OVERHEAD_MS);
        for command in [
            "setoption",
            "setoption name",
            "setoption name Hash",
            "setoption name Hash value lots",
            "setoption name Hash value -4",
            "setoption name Threads value 8",
            "setoption name Unknown value 1",
            "setoption value 3",
        ] {
            assert!(engine.execute(command), "{command}");
        }
        engine.execute("setoption name Hash value 2");
        assert_eq!(engine.move_overhead, MAX_MOVE_OVERHEAD_MS);
    }

    #[test]
    fn eval_file_option_loads_and_unloads_a_network() {
        let path = std::env::temp_dir().join("postmark-uci-test.nnue");
        let w1 = vec![1; crate::nnue::FEATURES * 16];
        let network = Network::new(16, w1, vec![0; 16], vec![3; 32], 0);
        std::fs::write(&path, network.to_bytes()).unwrap();
        let path = path.to_string_lossy().into_owned();

        let mut engine = Engine::new();
        engine.execute("setoption name EvalFile value no/such/file.nnue");
        assert!(engine.network.is_none());
        engine.execute(&format!("setoption name EvalFile value {path}"));
        assert_eq!(engine.network.as_deref(), Some(&network));
        assert!(engine.execute("eval"));
        assert!(engine.execute("go depth 2"));
        engine.wait_for_search();
        engine.execute("setoption name EvalFile value <empty>");
        assert!(engine.network.is_none());
    }

    #[test]
    fn go_runs_a_search_and_stop_ends_an_infinite_one() {
        let mut engine = Engine::new();
        assert!(engine.execute("go depth 3"));
        engine.wait_for_search();
        assert!(engine.search.is_none());

        assert!(engine.execute("go infinite"));
        assert!(engine.execute("isready"));
        assert!(engine.execute("stop"));
        assert!(engine.search.is_none());
    }

    #[test]
    fn junk_input_is_ignored_and_quit_ends_the_loop() {
        let mut engine = Engine::new();
        for command in [
            "",
            "   ",
            "hello",
            "go depth x wtime",
            "go wtime -5 btime -5",
            "perft",
        ] {
            assert!(engine.execute(command), "{command:?}");
            engine.stop_search();
        }
        assert!(!engine.execute("quit"));
    }
}
