//! Search limits and time management (UCI-2, TIM-1).

use std::time::Duration;

use crate::params::{
    TIME_HARD_MAX_PERCENT, TIME_HARD_MULTIPLIER, TIME_INCREMENT_PERCENT, TIME_MOVES_TO_GO,
    TIME_SOFT_PERCENT,
};
use crate::types::Color;

/// The conditions under which a search must end, as given by a `go` command.
/// Every field is optional; with none set the search runs to the maximum
/// depth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    /// Stop after completing this depth.
    pub depth: Option<i32>,
    /// Stop after searching this many nodes.
    pub nodes: Option<u64>,
    /// Spend exactly this many milliseconds on the move.
    pub movetime: Option<u64>,
    /// Remaining clock time in milliseconds, indexed by colour.
    pub time: [Option<u64>; 2],
    /// Increment per move in milliseconds, indexed by colour.
    pub increment: [u64; 2],
    /// Moves left until the next time control, if the control has one.
    pub moves_to_go: Option<u64>,
    /// Keep searching until told to stop, even after reaching the maximum
    /// depth.
    pub infinite: bool,
}

/// How long a search may run (TIM-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBudget {
    /// Once this much time has passed, no new iteration is started.
    pub soft: Option<Duration>,
    /// Once this much time has passed, the search is aborted mid-iteration.
    pub hard: Option<Duration>,
}

impl TimeBudget {
    /// Works out the time budget for `side` to move under `limits`, holding
    /// back `overhead_ms` milliseconds for communication delays.
    ///
    /// * With `movetime`, both limits are that time less the overhead: the
    ///   whole of it should be used.
    /// * With a clock, the remaining time (less the overhead) is divided by
    ///   the number of moves it must cover, which is `movestogo` if given
    ///   and [`TIME_MOVES_TO_GO`] otherwise, and part of the increment is
    ///   added. The soft limit is a fraction of that allotment and the hard
    ///   limit a multiple of it, the latter capped at a share of the
    ///   remaining time so that one long think can never flag.
    /// * Otherwise there is no time limit.
    pub fn new(limits: &Limits, side: Color, overhead_ms: u64) -> TimeBudget {
        if let Some(movetime) = limits.movetime {
            let time = Duration::from_millis(movetime.saturating_sub(overhead_ms).max(1));
            return TimeBudget {
                soft: Some(time),
                hard: Some(time),
            };
        }

        let Some(remaining) = limits.time[side.index()] else {
            return TimeBudget {
                soft: None,
                hard: None,
            };
        };
        let remaining = remaining.saturating_sub(overhead_ms).max(1);
        let moves = limits.moves_to_go.unwrap_or(TIME_MOVES_TO_GO).max(1);
        let increment = limits.increment[side.index()] * TIME_INCREMENT_PERCENT / 100;
        // The allotment may not exceed what is actually on the clock.
        let allotment = (remaining / moves + increment).min(remaining);

        let cap = (remaining * TIME_HARD_MAX_PERCENT / 100).max(1);
        let hard = (allotment * TIME_HARD_MULTIPLIER).min(cap);
        let soft = (allotment * TIME_SOFT_PERCENT / 100).clamp(1, hard);
        TimeBudget {
            soft: Some(Duration::from_millis(soft)),
            hard: Some(Duration::from_millis(hard)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(duration: Option<Duration>) -> u64 {
        duration.unwrap().as_millis() as u64
    }

    #[test]
    fn no_clock_means_no_limit() {
        let budget = TimeBudget::new(&Limits::default(), Color::White, 10);
        assert_eq!((budget.soft, budget.hard), (None, None));
    }

    #[test]
    fn movetime_is_used_in_full_less_overhead() {
        let limits = Limits {
            movetime: Some(1000),
            ..Limits::default()
        };
        let budget = TimeBudget::new(&limits, Color::Black, 10);
        assert_eq!((ms(budget.soft), ms(budget.hard)), (990, 990));
    }

    #[test]
    fn clock_budget_uses_the_right_colour_and_stays_within_the_clock() {
        let limits = Limits {
            time: [Some(60_000), Some(1_000)],
            increment: [600, 0],
            ..Limits::default()
        };
        let white = TimeBudget::new(&limits, Color::White, 10);
        let black = TimeBudget::new(&limits, Color::Black, 10);
        assert!(ms(white.soft) > ms(black.soft));
        assert!(ms(white.soft) <= ms(white.hard));
        assert!(ms(white.hard) < 60_000);
        assert!(ms(black.hard) < 1_000);
    }

    #[test]
    fn tiny_clocks_never_produce_a_zero_or_overlong_budget() {
        for remaining in [0, 1, 5, 11, 50] {
            let limits = Limits {
                time: [Some(remaining), None],
                increment: [1000, 0],
                moves_to_go: Some(1),
                ..Limits::default()
            };
            let budget = TimeBudget::new(&limits, Color::White, 10);
            assert!(ms(budget.soft) >= 1);
            assert!(ms(budget.hard) <= remaining.max(1));
        }
    }
}
