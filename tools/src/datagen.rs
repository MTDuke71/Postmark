//! `datagen`: generates NNUE training data by self-play (spec EVL-4).
//!
//! Each thread plays games against itself from book openings at a fixed
//! search depth and writes one line per recorded position:
//!
//! ```text
//! <score> <fen> <result>
//! ```
//!
//! `score` is the search score in centipawns from White's point of view,
//! `fen` the position before the move, and `result` the outcome of the
//! game the position came from (`1-0`, `0-1` or `1/2-1/2`), so that the
//! trainer can decide later whether to blend it into the label. Lines
//! starting with `#` are provenance: the build, the labelling evaluator,
//! the depth and the settings that produced the file, as EVL-4 requires.
//!
//! The recipe follows the one that worked for Huginn and FableR: a few
//! random moves early in each game for variety; only quiet positions
//! recorded (not in check, best move neither a capture nor a promotion,
//! score within bounds, past the opening); the label is the search score,
//! so that the network learns what the search found, not what the static
//! evaluator thought.
//!
//! Usage: `datagen --book <epd> --out <prefix> [options]`; run with
//! `--help` for the options. Output goes to `<prefix>.<thread>.txt`.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{self, Command};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use postmark::movegen::{GenKind, generate};
use postmark::moves::MoveList;
use postmark::nnue::Network;
use postmark::position::Position;
use postmark::rng::splitmix64;
use postmark::search::Searcher;
use postmark::timeman::Limits;
use postmark::tt::TranspositionTable;
use postmark::types::Color;

/// The settings of a run, from the command line.
#[derive(Clone, Debug)]
struct Settings {
    /// Opening book: one FEN or EPD record per line.
    book: PathBuf,
    /// Output path prefix; thread `n` writes `<prefix>.<n>.txt`.
    out: PathBuf,
    /// Network to label with, or none for the hand-crafted evaluator.
    net: Option<PathBuf>,
    /// Search depth for every labelled move.
    depth: i32,
    /// Number of game-playing threads.
    threads: usize,
    /// Stop once this many positions have been recorded in total.
    positions: u64,
    /// Stop once this many games have been started in total.
    games: u64,
    /// Seed of the pseudo-random choices (openings and random moves).
    seed: u64,
    /// Transposition table size per thread, in megabytes.
    hash: usize,
    /// Plies from the opening during which a move may be random.
    random_plies: usize,
    /// Chance, in percent, that a move within `random_plies` is random.
    random_percent: u64,
    /// Positions before this ply are not recorded.
    min_ply: usize,
    /// Positions with |score| at or above this are not recorded.
    max_score: i32,
    /// A side whose score is at or below minus this for `resign_moves`
    /// consecutive moves of its own resigns.
    resign: i32,
    /// See `resign`.
    resign_moves: u32,
    /// Games are adjudicated drawn at this many plies.
    max_ply: usize,
}

impl Settings {
    /// Parses the command line, exiting with a message on error.
    fn parse() -> Settings {
        let mut settings = Settings {
            book: PathBuf::new(),
            out: PathBuf::new(),
            net: None,
            depth: 7,
            threads: thread::available_parallelism()
                .map_or(1, |n| n.get().saturating_sub(1).max(1)),
            positions: 1_000_000,
            games: u64::MAX,
            seed: 1,
            hash: 16,
            random_plies: 20,
            random_percent: 8,
            min_ply: 10,
            max_score: 1500,
            resign: 2000,
            resign_moves: 4,
            max_ply: 400,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            if flag == "--help" || flag == "-h" {
                usage(0);
            }
            let Some(value) = arguments.next() else {
                eprintln!("{flag} needs a value");
                usage(2);
            };
            // Each option is parsed into its field; a value that does not
            // parse is reported with the flag it was given for.
            macro_rules! number {
                () => {
                    match value.parse() {
                        Ok(number) => number,
                        Err(_) => {
                            eprintln!("{flag}: bad value {value:?}");
                            usage(2);
                        }
                    }
                };
            }
            match flag.as_str() {
                "--book" => settings.book = PathBuf::from(value),
                "--out" => settings.out = PathBuf::from(value),
                "--net" => settings.net = Some(PathBuf::from(value)),
                "--depth" => settings.depth = number!(),
                "--threads" => settings.threads = number!(),
                "--positions" => settings.positions = number!(),
                "--games" => settings.games = number!(),
                "--seed" => settings.seed = number!(),
                "--hash" => settings.hash = number!(),
                "--random-plies" => settings.random_plies = number!(),
                "--random-percent" => settings.random_percent = number!(),
                "--min-ply" => settings.min_ply = number!(),
                "--max-score" => settings.max_score = number!(),
                "--resign" => settings.resign = number!(),
                "--resign-moves" => settings.resign_moves = number!(),
                "--max-ply" => settings.max_ply = number!(),
                _ => {
                    eprintln!("unknown option {flag}");
                    usage(2);
                }
            }
        }
        if settings.book.as_os_str().is_empty() || settings.out.as_os_str().is_empty() {
            eprintln!("--book and --out are required");
            usage(2);
        }
        if settings.threads == 0 || settings.depth < 1 {
            eprintln!("--threads and --depth must be at least 1");
            usage(2);
        }
        settings
    }
}

/// Prints the usage text and exits with `code`.
fn usage(code: i32) -> ! {
    eprintln!(
        "usage: datagen --book <epd> --out <prefix> [--net <file>] [--depth N] [--threads N]
               [--positions N] [--games N] [--seed N] [--hash MB]
               [--random-plies N] [--random-percent N] [--min-ply N] [--max-score N]
               [--resign N] [--resign-moves N] [--max-ply N]"
    );
    process::exit(code);
}

/// Everything the threads share.
struct Shared {
    /// The run's settings.
    settings: Settings,
    /// Opening positions.
    book: Vec<String>,
    /// The labelling network, if any.
    network: Option<Arc<Network>>,
    /// Games started so far.
    games: AtomicU64,
    /// Positions recorded so far.
    positions: AtomicU64,
    /// Set once the limits are reached; threads finish their game and stop.
    done: AtomicBool,
}

/// The outcome of a game, from White's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// White won.
    WhiteWins,
    /// Black won.
    BlackWins,
    /// Drawn.
    Draw,
}

impl Outcome {
    /// The outcome in which `loser` lost.
    fn loss_for(loser: Color) -> Outcome {
        match loser {
            Color::White => Outcome::BlackWins,
            Color::Black => Outcome::WhiteWins,
        }
    }

    /// The PGN result string.
    fn as_str(self) -> &'static str {
        match self {
            Outcome::WhiteWins => "1-0",
            Outcome::BlackWins => "0-1",
            Outcome::Draw => "1/2-1/2",
        }
    }
}

/// One recorded position, before the game's outcome is known.
struct Record {
    /// Search score in centipawns from White's point of view.
    score: i32,
    /// The position.
    fen: String,
}

/// Program entry point.
fn main() {
    let settings = Settings::parse();
    let book = match read_book(&settings.book) {
        Ok(book) if !book.is_empty() => book,
        Ok(_) => {
            eprintln!("{}: no openings", settings.book.display());
            process::exit(1);
        }
        Err(error) => {
            eprintln!("{}: {error}", settings.book.display());
            process::exit(1);
        }
    };
    let (network, evaluator) = match &settings.net {
        Some(path) => {
            let bytes = std::fs::read(path).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                process::exit(1);
            });
            let network = Network::from_bytes(&bytes).unwrap_or_else(|error| {
                eprintln!("{}: {error}", path.display());
                process::exit(1);
            });
            let description = format!(
                "nnue {} (hidden {}, {} bytes, fnv1a {:016x})",
                path.display(),
                network.hidden(),
                bytes.len(),
                fnv1a(&bytes)
            );
            (Some(Arc::new(network)), description)
        }
        None => (None, "pst (material + piece-square tables)".to_string()),
    };

    let header = provenance(&settings, &evaluator, book.len());
    let shared = Shared {
        settings,
        book,
        network,
        games: AtomicU64::new(0),
        positions: AtomicU64::new(0),
        done: AtomicBool::new(false),
    };
    let start = Instant::now();

    thread::scope(|scope| {
        for index in 0..shared.settings.threads {
            let shared = &shared;
            let header = &header;
            scope.spawn(move || {
                if let Err(error) = worker(shared, index, header) {
                    eprintln!("thread {index}: {error}");
                    shared.done.store(true, Ordering::Relaxed);
                }
            });
        }
        // Progress, from this thread, until the workers are done.
        let mut last = 0;
        while !shared.done.load(Ordering::Relaxed) {
            // Short naps, so that the end of the run is noticed promptly.
            let until = Instant::now() + Duration::from_secs(10);
            while Instant::now() < until && !shared.done.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
            if shared.done.load(Ordering::Relaxed) {
                break;
            }
            let positions = shared.positions.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs().max(1);
            eprintln!(
                "games {} positions {} ({}/s now, {}/s overall) elapsed {}m{:02}s",
                shared.games.load(Ordering::Relaxed),
                positions,
                (positions - last) / 10,
                positions / elapsed,
                elapsed / 60,
                elapsed % 60
            );
            last = positions;
        }
    });

    let positions = shared.positions.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs().max(1);
    eprintln!(
        "done: games {} positions {} in {}m{:02}s ({}/s)",
        shared.games.load(Ordering::Relaxed),
        positions,
        elapsed / 60,
        elapsed % 60,
        positions / elapsed
    );
}

/// Reads the opening book: one position per line, FEN or EPD (the first
/// four fields are used; the move counters are optional).
fn read_book(path: &PathBuf) -> io::Result<Vec<String>> {
    let reader = BufReader::new(File::open(path)?);
    let mut book = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let fields: Vec<&str> = line.split_whitespace().take(6).collect();
        if fields.len() < 4 {
            continue;
        }
        // EPD operations follow the fourth field; only numeric fifth and
        // sixth fields are move counters.
        let mut fen = fields[..4].join(" ");
        if fields.len() == 6 && fields[4].parse::<u16>().is_ok() && fields[5].parse::<u16>().is_ok()
        {
            fen = fields.join(" ");
        }
        if Position::from_fen(&fen).is_ok() {
            book.push(fen);
        }
    }
    Ok(book)
}

/// The FNV-1a hash of `bytes`, used to identify a network file.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Builds the provenance header written at the top of every output file.
fn provenance(settings: &Settings, evaluator: &str, openings: usize) -> String {
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_string(),
        );
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    format!(
        "# postmark datagen
# build: {} commit {commit}
# evaluator: {evaluator}
# depth: {}
# book: {} ({openings} openings)
# seed: {}
# random moves: {}% of moves before ply {}
# recorded: not in check, best move not a capture or promotion, |score| < {}, ply >= {}
# adjudication: resign at <= -{} for {} own moves, draw at ply {}
# started: {started} (unix seconds)
# format: <score cp, White's view> <fen> <result>
",
        postmark::banner(),
        settings.depth,
        settings.book.display(),
        settings.seed,
        settings.random_percent,
        settings.random_plies,
        settings.max_score,
        settings.min_ply,
        settings.resign,
        settings.resign_moves,
        settings.max_ply,
    )
}

/// One game-playing thread: plays games until the shared limits are
/// reached, writing its positions to its own file.
fn worker(shared: &Shared, index: usize, header: &str) -> io::Result<()> {
    let settings = &shared.settings;
    let path = settings.out.with_extension(format!("{index}.txt"));
    let path = match settings.out.extension() {
        // `with_extension` would replace an existing extension.
        Some(_) => PathBuf::from(format!("{}.{index}.txt", settings.out.display())),
        None => path,
    };
    let mut out = BufWriter::new(File::create(&path)?);
    out.write_all(header.as_bytes())?;
    writeln!(out, "# thread: {index}")?;

    let table = TranspositionTable::new(settings.hash)
        .ok_or_else(|| io::Error::other(format!("cannot allocate a {} MB table", settings.hash)))?;
    let stop = AtomicBool::new(false);
    let limits = Limits {
        depth: Some(settings.depth),
        ..Limits::default()
    };
    // Each thread has its own stream; the mix of seed and index keeps
    // them apart.
    let mut rng = settings
        .seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(index as u64 * 0xD1B5_4A32_D192_ED03);
    let mut records: Vec<Record> = Vec::new();
    let mut keys: Vec<u64> = Vec::new();

    while !shared.done.load(Ordering::Relaxed) {
        if shared.games.fetch_add(1, Ordering::Relaxed) >= settings.games {
            shared.done.store(true, Ordering::Relaxed);
            break;
        }
        let opening = &shared.book[(splitmix64(&mut rng) % shared.book.len() as u64) as usize];
        let mut position = Position::from_fen(opening).expect("book positions were checked");
        records.clear();
        keys.clear();
        keys.push(position.key());
        let mut low_scores = [0u32; 2];

        let outcome = loop {
            let ply = keys.len() - 1;
            let us = position.side_to_move();
            let mut list = MoveList::new();
            generate(&position, GenKind::All, &mut list);
            if list.is_empty() {
                break if position.in_check() {
                    Outcome::loss_for(us)
                } else {
                    Outcome::Draw
                };
            }
            if position.has_insufficient_material()
                || position.halfmove_clock() >= 100
                || keys.iter().filter(|&&key| key == position.key()).count() >= 3
                || ply >= settings.max_ply
            {
                break Outcome::Draw;
            }

            let random =
                ply < settings.random_plies && splitmix64(&mut rng) % 100 < settings.random_percent;
            let mv = if random {
                list[(splitmix64(&mut rng) % list.len() as u64) as usize]
            } else {
                let mut searcher = Searcher::new(
                    position.clone(),
                    &table,
                    &stop,
                    &limits,
                    Instant::now(),
                    0,
                    shared.network.as_ref(),
                );
                let result = searcher.run(&mut |_| {});
                let Some(best) = result.best_move else {
                    break Outcome::Draw;
                };
                let score = result.score;
                if !position.in_check()
                    && !best.is_capture()
                    && best.promotion().is_none()
                    && score.abs() < settings.max_score
                    && ply >= settings.min_ply
                {
                    records.push(Record {
                        score: match us {
                            Color::White => score,
                            Color::Black => -score,
                        },
                        fen: position.to_fen(),
                    });
                }
                // Resignation: a side that has judged itself lost for
                // several moves running is not going to recover, and the
                // positions from here on would fail the score filter.
                let low = &mut low_scores[us.index()];
                *low = if score <= -settings.resign {
                    *low + 1
                } else {
                    0
                };
                if *low >= settings.resign_moves {
                    break Outcome::loss_for(us);
                }
                best
            };
            position.make_move(mv);
            keys.push(position.key());
        };

        for record in &records {
            writeln!(out, "{} {} {}", record.score, record.fen, outcome.as_str())?;
        }
        let total = shared
            .positions
            .fetch_add(records.len() as u64, Ordering::Relaxed)
            + records.len() as u64;
        if total >= settings.positions {
            shared.done.store(true, Ordering::Relaxed);
        }
    }
    out.flush()
}
