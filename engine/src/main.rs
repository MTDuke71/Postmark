//! Postmark engine binary.
//!
//! With no arguments the engine speaks UCI on standard input and output.
//! With arguments, they are executed as a single command and the engine
//! then exits, which is how test tools run `postmark bench`.

use std::env;

use postmark::uci::Engine;

/// Program entry point: runs one command from the command line if given,
/// otherwise the UCI loop.
fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let mut engine = Engine::new();
    if arguments.is_empty() {
        println!("{}", postmark::banner());
        engine.run();
    } else {
        engine.execute(&arguments.join(" "));
        engine.wait_for_search();
    }
}
