//! Postmark chess engine library.
//!
//! The engine is built as a library so that the UCI binary, the test suite and
//! later tooling all drive the same code. See `docs/SPEC.md` for the
//! requirements this crate implements; requirement IDs such as `BLD-2` in
//! comments refer to that document.

/// Engine name, as shown to the user and reported over UCI.
pub const NAME: &str = "Postmark";

/// Engine version, taken from the crate manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Label of the CPU tier this binary was compiled for (BLD-2).
///
/// The tier is decided by the target features the compiler was given, not by
/// the machine the binary runs on: `"v3"` when AVX2, BMI2 and POPCNT are all
/// enabled (the `cargo v3` build), `"generic"` otherwise. Hot paths select
/// their implementation with the same `cfg(target_feature = ...)` tests, so
/// this label always matches the code that was actually compiled in.
pub const BUILD_TIER: &str = if cfg!(all(
    target_feature = "avx2",
    target_feature = "bmi2",
    target_feature = "popcnt"
)) {
    "v3"
} else {
    "generic"
};

/// Returns the one-line identification banner, e.g. `Postmark 0.1.0 (v3)`.
///
/// The banner names the engine, its version and its CPU tier so that a binary
/// can always be matched to the build that produced it.
pub fn banner() -> String {
    format!("{NAME} {VERSION} ({BUILD_TIER})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_names_engine_version_and_tier() {
        assert_eq!(banner(), format!("Postmark {VERSION} ({BUILD_TIER})"));
    }

    #[test]
    fn build_tier_matches_enabled_target_features() {
        let has_v3 = cfg!(target_feature = "avx2")
            && cfg!(target_feature = "bmi2")
            && cfg!(target_feature = "popcnt");
        assert_eq!(BUILD_TIER == "v3", has_v3);
    }
}
