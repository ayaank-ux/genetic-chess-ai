# Genetic Chess AI

A Python experiment in evolving chess evaluation heuristics. Each candidate has 12 numerical weights. Candidates play games using a shared minimax search with alpha-beta pruning; selection, crossover, and mutation produce the next generation.

`python-chess` supplies board state and chess rules. Our code controls evaluation, search, and evolution. An optional Stockfish executable supplies external benchmark opponents.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run genetic-chess-ai --help
```

The dependency lockfile is included with the project. Stockfish is optional and installed separately. No UI or GPU is required.

## Quick start

Inspect the starting position and the baseline's preferred move:

```bash
uv run genetic-chess-ai evaluate --depth 2
```

Play a baseline game and save its PGN and JSON:

```bash
uv run genetic-chess-ai play --depth 2 --output runs/baseline-game
```

Run a small training experiment:

```bash
uv run genetic-chess-ai train \
  --output runs/first \
  --population 4 --elites 1 --generations 2 \
  --depth 1 --max-plies 160 --benchmark-openings 1 --seed 42
```

For a fast plumbing check, use `--max-plies 8`. Such short games will usually be move-limit draws and are not evidence of learning. The normal defaults use 12 candidates, 5 generations, depth 2, and a 160-ply limit; measure a small run before scaling up.

Continue the experiment with two **additional** generations:

```bash
uv run genetic-chess-ai resume runs/first --generations 2
```

Benchmark the latest generation's champion against the fixed baseline:

```bash
uv run genetic-chess-ai benchmark \
  --genome runs/first/champion.json \
  --depth 1 --openings 4 --output runs/baseline-benchmark
```

New runs require a new output directory, so existing results are preserved. Progress goes to stderr; command results are JSON on stdout. `python -m genetic_chess_ai` is an alternative entry point.

## Stockfish

Obtain a binary for your platform from the [official Stockfish downloads](https://stockfishchess.org/download/). Keep the same binary for a series of experiments. The project does not bundle or automatically install it.

Inspect the engine and the supported skill/Elo ranges:

```bash
uv run genetic-chess-ai stockfish-info /path/to/stockfish
```

Run paired games against a limited Stockfish:

```bash
uv run genetic-chess-ai benchmark \
  --genome runs/first/champion.json --depth 1 \
  --stockfish /path/to/stockfish --sf-nodes 1000 \
  --openings 4 --output runs/stockfish-1000
```

Choose exactly one of `--sf-nodes`, `--sf-depth`, or `--sf-time`. With no explicit limit, Stockfish gets 1,000 nodes per move. Optionally add `--sf-skill` or `--sf-elo`, using a value supported by your binary. These options are mutually exclusive; Elo enables `UCI_LimitStrength`.

The same Stockfish options work with `train`, adding a Stockfish benchmark for each generation's champion. They do not affect training fitness. Create a benchmark ladder by running separate benchmarks at fixed limits and keeping earlier configurations for comparison.

The adapter uses one thread, 16 MiB of hash, no pondering, and a fresh game identity plus a hash clear between games. Reports record the engine identity, binary SHA-256, and settings. Limited Stockfish can still be very strong; its search depth and configured Elo do not measure our engine's depth or rating.

## How training works

1. Include the fixed baseline genome and initialize the remaining candidates with perturbed weights.
2. Shuffle candidates into balanced pairings. Each pairing plays both colors from the same opening.
3. Play every candidate against the baseline and the bounded archive of prior champions.
4. Score wins as 1, draws as 0.5, and losses as 0. Every candidate gets the same number of games.
5. Benchmark the generation champion on separate openings. These results never select parents.
6. Preserve elites, choose parents by tournament selection, use per-weight crossover, and mutate offspring.
7. Save results and a checkpoint containing the next population and random-number-generator state.

`--mutation-only` disables crossover for comparison experiments. Additional controls include `--mutation-rate`, `--mutation-scale`, `--tournament-size`, `--pairing-rounds`, and `--archive-size`. Population size must be even and at least four.

Pairing rounds sample opponents without a full round robin; opponents may recur in different rounds. All matches in a generation share one training opening, rotating across four openings over generations. Prior champions are deduplicated by genome and bounded by the archive size.

Ties are resolved using the experiment's seeded random generator. If all fitness scores are equal, training reports that there is no selection signal that generation. Mutation still proceeds, but equal scores are not evidence of improvement.

## Evaluation and search

Scores are White-relative. White maximizes and Black minimizes. The pawn weight is fixed at 100; other weights evolve within bounded ranges.

| Feature | Definition |
| --- | --- |
| Pawn, knight, bishop, rook, queen | Difference in piece counts; five weights |
| Activity | Attacked destinations of knights, bishops, rooks, and queens, excluding friendly pieces; not legal mobility |
| Center control | Number of attackers on d4, e4, d5, and e5 |
| Bishop pair | Whether a side has at least two bishops |
| Doubled pawns | Extra pawns on occupied files |
| Isolated pawns | Pawns with no friendly pawns on adjacent files |
| Passed pawns | Pawns with no enemy pawn ahead on the same or neighboring files, weighted by advancement |
| King shield | Friendly pawns directly ahead of the king or on the adjacent files |

Each feature is computed as White minus Black. Baseline doubled/isolated pawn weights are negative. Checkmate scores are separate and dominate all normal evaluations; mate distance favors quicker wins and delayed losses.

The engine uses deterministic capture/promotion ordering and fixed-depth alpha-beta search. There is no transposition table or quiescence search yet, so shallow tactical mistakes and horizon effects are expected. All in-house agents use the same implementation and depth.

## Results and reproducibility

```text
runs/first/
  config.json                 # Search and evolution settings
  metadata.json               # Versions, source hashes, openings, scoring policy
  checkpoint.json             # Next population, RNG state, archive, history
  champion.json               # Latest generation champion, not an all-time winner
  history.json                # Per-generation metrics
  generation-0001/
    champion.json
    report.json               # Population, fitness, diversity, benchmark results
    training-games.json
    training-games.pgn
    benchmark-games.json
    benchmark-games.pgn
```

The game runner automatically claims available repetition and fifty-move draws. `--max-plies` counts individual agent moves after the supplied opening. Games hitting that limit score as draws and retain a distinct `move_limit` termination; inspect that count when interpreting scores.

Illegal moves and agent exceptions are recorded as explicit forfeits. Training and benchmarks fail after saving the game records if an agent fails, preventing broken agents from silently distorting fitness. Engine startup failures abort immediately.

Resume starts at the last completed generation; interrupted work in the next generation is replayed. For in-house agents, unchanged code/dependencies and the same seed reproduce genomes, moves, and scores, excluding elapsed times. Resume checks Python, chess dependency, source hashes, and any Stockfish binary/settings. External engine strength reduction and time-based limits may introduce additional nondeterminism.

Compare score `(wins + 0.5 * draws) / games`, game count, termination reasons, runtime, and normalized population diversity. Repeat promising runs with different seeds and more opening positions. The small included benchmark set supports development; reserve a larger untouched test set before claiming general strength gains. A champion beating its own training population is insufficient evidence on its own.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover search against plain minimax, evaluation symmetry, mates/draws, board restoration, PGN replay, fair schedules, elitism, resume equivalence, interruptions, and CLI workflows. Optional real Stockfish tests run when it is on `PATH`, or with:

```bash
STOCKFISH_EXECUTABLE=/path/to/stockfish uv run pytest
```

The architecture and milestones are in [docs/plan.md](docs/plan.md). Implementation modules follow that plan: `evaluation`, `search`, `agent`, `game`, `tournament`, `evolution`, and `experiments`, with `cli` providing the command interface. The engine integration uses the [python-chess UCI API](https://python-chess.readthedocs.io/en/stable/engine.html).
