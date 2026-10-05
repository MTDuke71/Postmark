//! Postmark engine binary.
//!
//! Until the UCI loop arrives with milestone M2, the binary takes its command
//! from the command line:
//!
//! * no arguments — print the identification banner;
//! * `perft <depth> [fen]` — run perft from the given position (the start
//!   position if no FEN is given), listing the count below each move (MGN-5).

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use postmark::perft::divide;
use postmark::position::{Position, START_FEN};

/// Exit status for a command line that could not be understood.
const USAGE_ERROR: u8 = 2;

/// Program entry point: dispatches on the first command-line argument.
fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => {
            println!("{}", postmark::banner());
            ExitCode::SUCCESS
        }
        Some("perft") => run_perft(&args[1..]),
        Some(other) => {
            eprintln!("unknown command '{other}'");
            eprintln!("usage: postmark [perft <depth> [fen]]");
            ExitCode::from(USAGE_ERROR)
        }
    }
}

/// Runs the `perft` command.
///
/// `args` holds the depth followed, optionally, by a FEN, which may arrive
/// either as one quoted argument or split into its space-separated fields.
/// Prints one `move: count` line per legal move, then the total node count,
/// the elapsed time and the speed in nodes per second. Returns a failure
/// status if the depth or the FEN is invalid.
fn run_perft(args: &[String]) -> ExitCode {
    let Some(depth) = args.first().and_then(|depth| depth.parse::<u32>().ok()) else {
        eprintln!("usage: postmark perft <depth> [fen]");
        return ExitCode::from(USAGE_ERROR);
    };
    let fen = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        START_FEN.to_string()
    };
    let mut position = match Position::from_fen(&fen) {
        Ok(position) => position,
        Err(error) => {
            eprintln!("invalid FEN: {error}");
            return ExitCode::from(USAGE_ERROR);
        }
    };

    let start = Instant::now();
    let parts = divide(&mut position, depth);
    // At depth 0 there are no moves to list and the only node is the root.
    let nodes = if depth == 0 {
        1
    } else {
        parts.iter().map(|&(_, nodes)| nodes).sum()
    };
    let elapsed = start.elapsed();

    for (mv, nodes) in &parts {
        println!("{mv}: {nodes}");
    }
    println!();
    println!("Nodes: {nodes}");
    println!("Time: {} ms", elapsed.as_millis());
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        println!("NPS: {:.0}", nodes as f64 / seconds);
    }
    ExitCode::SUCCESS
}
