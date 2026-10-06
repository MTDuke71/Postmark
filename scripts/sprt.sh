#!/usr/bin/env bash
# Runs the SPRT merge gate (spec TST-4, TST-5) between two engine binaries.
#
#   scripts/sprt.sh <new-binary> <base-binary> [elo0] [elo1]
#
# The bounds default to [0, 5], the gate for a strength change. Use -5 0 for a
# refactor or simplification that only has to show it does not lose strength.
#
# fastchess stops by itself when the test is decided: "H1 was accepted" means
# the change passes, "H0 was accepted" means it fails.
#
# Two files are written to the current directory, named after the new binary:
#   <name>.pgn            every game, with the score, depth and time per move
#   <name>.fastchess.log  fastchess's own warnings and errors
# Check the log, and the game endings in the console output, for time losses,
# illegal moves and unresponsive engines before trusting the result.
#
# Environment:
#   FASTCHESS    path to the fastchess binary
#   BOOK         path to the opening book (EPD)
#   CONCURRENCY  games played at once (default 8: one per physical core on
#                the test machine, leaving the second hardware threads free)
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <new-binary> <base-binary> [elo0] [elo1]" >&2
    exit 2
fi

new=$1
base=$2
elo0=${3:-0}
elo1=${4:-5}
tools=${FASTCHESS_DIR:-$HOME/Documents/fastchess-windows-x86-64}
fastchess=${FASTCHESS:-$tools/fastchess.exe}
book=${BOOK:-$tools/UHO_2024/UHO_2024_+090_+099/UHO_2024_8mvs_+090_+099.epd}
name=$(basename "$new")
name=${name%.*}

"$fastchess" \
    -engine cmd="$new" name=new \
    -engine cmd="$base" name=base \
    -each tc=8+0.08 option.Hash=16 option.Threads=1 \
    -openings file="$book" format=epd order=random \
    -rounds 50000 -games 2 -repeat \
    -concurrency "${CONCURRENCY:-8}" -recover \
    -sprt elo0="$elo0" elo1="$elo1" alpha=0.05 beta=0.05 model=logistic \
    -pgnout file="$name.pgn" \
    -log file="$name.fastchess.log" level=warn \
    -ratinginterval 50
