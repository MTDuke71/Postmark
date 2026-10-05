//! Postmark engine binary.
//!
//! At milestone M0 this only prints the identification banner. The UCI loop
//! arrives with milestone M2.

/// Program entry point: prints the engine banner and exits.
fn main() {
    println!("{}", postmark::banner());
}
