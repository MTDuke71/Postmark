#!/usr/bin/env python3
"""Trains a Postmark NNUE network from `datagen` output (spec EVL-4, EVL-6).

The network is the layout shared with Huginn and FableR: 768 input
features (colour, piece kind, square), two perspective accumulators of
hidden size H with shared first-layer weights, clipped ReLU, and one
output neuron that sees the side to move's accumulator first. The output
is in sixteenths of a pawn so that the quantised weights stay small while
the centipawn range stays open.

Two steps:

    train_nnue.py preprocess <prefix> --out <data.npz>
        Reads <prefix>.*.txt (one `<score> <fen> <result>` line per
        position, `#` lines ignored), parses every FEN once and stores the
        feature indices, labels and results compactly.

    train_nnue.py train <data.npz> --out <net.nnue> [--hidden 128] ...
        Trains on the GPU if there is one, holds out 1% for validation,
        writes the network after every epoch in Postmark's file format
        (PMNN header, then i16 weights), and finishes with a few probe
        positions evaluated in floating point, to compare with the
        engine's `eval` command after loading the file.

The loss is the Texel loss, a mean squared error between the sigmoids of
the prediction and the label (scale 400 cp), plus a small anchor term on
the raw centipawns, which keeps the net's scale honest where the sigmoid
saturates. The label is the search score; `--result-weight` blends the
game result in, in sigmoid space, for experiments.
"""

import argparse
import glob
import multiprocessing
import os
import struct
import sys
import time

import numpy as np

FEATURES = 768
PAD = FEATURES  # index of the all-zero padding row
MAX_PIECES = 32
PIECES = {c: i for i, c in enumerate("PNBRQKpnbrqk")}
RESULTS = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}
MAGIC = b"PMNN"
VERSION = 1

PROBES = [
    ("start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("queen odds, White to move", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"),
    ("queen odds, Black to move", "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"),
    ("KB vs K (drawn)", "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1"),
    ("rook endgame", "8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
]


def white_features(fen):
    """Returns the White-perspective feature indices of `fen` and whether
    White is to move. The Black perspective is derived later from these."""
    board, stm = fen.split(" ")[:2]
    features = []
    rank, file = 7, 0
    for ch in board:
        if ch == "/":
            rank, file = rank - 1, 0
        elif ch.isdigit():
            file += int(ch)
        else:
            piece = PIECES[ch]
            features.append(piece * 64 + rank * 8 + file)
            file += 1
    return features, stm == "w"


def parse_file(path):
    """Parses one datagen file into arrays: features [n, 32] (White view,
    padded), white-to-move flags, scores (White view) and results."""
    rows, stms, scores, results = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                score = int(parts[0])
            except ValueError:
                continue
            fen = " ".join(parts[1:7])
            result = RESULTS.get(parts[7])
            if result is None:
                continue
            features, white_to_move = white_features(fen)
            if len(features) > MAX_PIECES:
                continue
            rows.append(features + [PAD] * (MAX_PIECES - len(features)))
            stms.append(white_to_move)
            scores.append(score)
            results.append(result)
    return (
        np.array(rows, dtype=np.int16).reshape(-1, MAX_PIECES),
        np.array(stms, dtype=np.bool_),
        np.array(scores, dtype=np.int16),
        np.array(results, dtype=np.int8),
    )


def preprocess(prefix, out_path, workers):
    files = sorted(glob.glob(prefix + ".*.txt"))
    if not files:
        sys.exit(f"no files match {prefix}.*.txt")
    print(f"reading {len(files)} files with {workers} workers", flush=True)
    started = time.time()
    with multiprocessing.Pool(workers) as pool:
        parts = pool.map(parse_file, files)
    features = np.concatenate([p[0] for p in parts])
    stm = np.concatenate([p[1] for p in parts])
    scores = np.concatenate([p[2] for p in parts])
    results = np.concatenate([p[3] for p in parts])
    print(f"{len(scores)} positions in {time.time() - started:.0f}s", flush=True)
    print(
        f"score mean {scores.mean():.1f} sd {scores.std():.1f}; "
        f"results 1-0 {np.mean(results == 1):.1%} draw {np.mean(results == 0):.1%} "
        f"0-1 {np.mean(results == -1):.1%}; white to move {stm.mean():.1%}"
    )
    np.savez(out_path, features=features, stm=stm, scores=scores, results=results)
    print(f"saved {out_path}", flush=True)


def black_view(features):
    """Maps White-perspective feature indices to the Black perspective:
    colours swapped and ranks mirrored. Padding stays padding."""
    import torch

    pad = features == PAD
    colour = features // 384
    rest = features % 384
    kind = rest // 64
    square = rest % 64
    flipped = ((1 - colour) * 6 + kind) * 64 + (square ^ 56)
    return torch.where(pad, features, flipped)


def build_net(hidden):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = nn.Embedding(FEATURES + 1, hidden, padding_idx=PAD)
            self.ft_bias = nn.Parameter(torch.zeros(hidden))
            self.out = nn.Linear(2 * hidden, 1)
            nn.init.uniform_(self.ft.weight, -0.05, 0.05)
            with torch.no_grad():
                self.ft.weight[PAD].zero_()

        def accumulate(self, rows):
            return torch.clamp(self.ft(rows).sum(dim=1) + self.ft_bias, 0, 1)

        def forward(self, us, them):
            hidden = torch.cat([self.accumulate(us), self.accumulate(them)], dim=1)
            # The output is in sixteenths of a pawn (EVL-6).
            return self.out(hidden).squeeze(1) * 16.0

        def clamp_for_quantisation(self):
            # Keeps every weight inside the range the i16 file format and
            # the engine's 32-bit output sum can hold.
            with torch.no_grad():
                self.ft.weight.clamp_(-0.49, 0.49)
                self.ft.weight[PAD].zero_()
                self.ft_bias.clamp_(-0.49, 0.49)
                self.out.weight.clamp_(-4.0, 4.0)

    return Net()


def perspectives(features, stm):
    """Returns (side to move's features, the other side's) for a batch."""
    import torch

    black = black_view(features)
    stm = stm.unsqueeze(1)
    return torch.where(stm, features, black), torch.where(stm, black, features)


def evaluate_fen(net, fen, device):
    """The net's evaluation of `fen` in centipawns for the side to move."""
    import torch

    features, white_to_move = white_features(fen)
    rows = torch.tensor(
        [features + [PAD] * (MAX_PIECES - len(features))], dtype=torch.long, device=device
    )
    stm = torch.tensor([white_to_move], device=device)
    with torch.no_grad():
        us, them = perspectives(rows, stm)
        return float(net(us, them)[0])


def export(net, path, hidden):
    """Writes the network in Postmark's file format."""
    w1 = net.ft.weight.detach().cpu().numpy()[:FEATURES]  # [768, H], feature-major
    b1 = net.ft_bias.detach().cpu().numpy()
    w2 = net.out.weight.detach().cpu().numpy().ravel()  # [2H], side to move first
    b2 = float(net.out.bias.detach().cpu().numpy()[0])
    scale = 256.0
    w1q = np.clip(np.round(w1 * scale), -32767, 32767).astype("<i2")
    b1q = np.clip(np.round(b1 * scale), -32767, 32767).astype("<i2")
    w2q = np.clip(np.round(w2 * scale), -32767, 32767).astype("<i2")
    b2q = int(round(b2 * scale * scale))
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, hidden))
        f.write(w1q.tobytes())
        f.write(b1q.tobytes())
        f.write(w2q.tobytes())
        f.write(struct.pack("<i", b2q))


def train(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    data = np.load(args.data)
    # int32 indices are accepted by the embedding and halve the memory.
    features = torch.from_numpy(data["features"].astype(np.int32))
    stm = torch.from_numpy(data["stm"])
    scores = torch.from_numpy(data["scores"].astype(np.float32))
    results = torch.from_numpy(data["results"].astype(np.float32))
    n = len(scores)

    # The last 1% is the validation set; the files were written by
    # independent threads, so it is not the tail of one game sequence.
    split = n - max(1, n // 100)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    train_idx, val_idx = order[:split], order[split:]
    print(f"{split} training positions, {n - split} validation", flush=True)

    # Everything lives on the device; 20M positions take about 2.7 GB.
    features = features.to(device)
    stm = stm.to(device)
    scores = scores.to(device)
    results = results.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    scale = 400.0
    # Labels are from White's view; the net predicts for the side to move.
    sign = torch.where(stm, 1.0, -1.0)
    score_stm = scores * sign
    result_stm = results * sign
    target = (1 - args.result_weight) * torch.sigmoid(score_stm / scale) + args.result_weight * (
        result_stm + 1
    ) / 2
    anchor_target = score_stm.clamp(-1200, 1200)

    net = build_net(args.hidden).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=max(1, args.epochs // 3), gamma=args.lr_decay
    )

    def loss_on(index):
        us, them = perspectives(features[index], stm[index])
        prediction = net(us, them)
        texel = torch.mean((torch.sigmoid(prediction / scale) - target[index]) ** 2)
        anchor = torch.mean(((prediction - anchor_target[index]) / 800) ** 2)
        return texel + 0.15 * anchor, texel

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        net.train()
        perm = train_idx[torch.randperm(split, generator=generator, device=device)]
        total, steps = 0.0, 0
        for start in range(0, split, args.batch):
            loss, _ = loss_on(perm[start : start + args.batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            net.clamp_for_quantisation()
            total += loss.item()
            steps += 1
        scheduler.step()
        net.eval()
        with torch.no_grad():
            val_loss, val_texel = 0.0, 0.0
            val_steps = 0
            for start in range(0, len(val_idx), args.batch):
                loss, texel = loss_on(val_idx[start : start + args.batch])
                val_loss += loss.item()
                val_texel += texel.item()
                val_steps += 1
        export(net, args.out, args.hidden)
        print(
            f"epoch {epoch}: train {total / steps:.6f} val {val_loss / val_steps:.6f} "
            f"(texel {val_texel / val_steps:.6f}) lr {scheduler.get_last_lr()[0]:.2e} "
            f"{time.time() - started:.0f}s -> {args.out}",
            flush=True,
        )

    print("probes (centipawns, side to move):")
    for name, fen in PROBES:
        print(f"  {evaluate_fen(net, fen, device):8.1f}  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("preprocess", help="parse datagen files into an npz")
    p.add_argument("prefix", help="datagen output prefix: reads <prefix>.*.txt")
    p.add_argument("--out", required=True, help="npz to write")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    t = sub.add_parser("train", help="train a network from an npz")
    t.add_argument("data", help="npz from preprocess")
    t.add_argument("--out", required=True, help="network file to write")
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=16384)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--lr-decay", type=float, default=0.5, help="factor applied every epochs/3")
    t.add_argument("--result-weight", type=float, default=0.0, help="0 = search score only")
    t.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "preprocess":
        preprocess(args.prefix, args.out, args.workers)
    else:
        train(args)


if __name__ == "__main__":
    main()
