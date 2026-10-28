# Genetic Chess AI Project Plan

## Objective

Explore whether a genetic algorithm can improve a chess evaluation heuristic under a controlled search budget. Build the training, evaluation, search, and experiment tooling in Python. No UI is needed for the initial project.

The first success criterion is that evolved candidates outperform a fixed, hand-written baseline on held-out games using the same search implementation and budget. Stockfish provides an additional external benchmark.

## Core approach

Each candidate is a vector of numerical evaluation weights, called its genome. All candidates use the same feature definitions and minimax search with alpha-beta pruning.

For board position `s` and weights `w`:

`evaluation(s, w) = sum(w[i] * feature[i](s))`

Features express White's advantage over Black; positive evaluations favor White. Search maximizes for White and minimizes for Black. Checkmate and draws are handled explicitly, outside the weighted evaluation, with mate scores dominating ordinary evaluations.

Begin with material and a few inexpensive positional features. Possible additions include mobility, center control, pawn structure, king safety, and piece placement. Add features gradually so their value and computational cost can be measured.

Evolution adjusts the importance of existing features. It does not invent new chess concepts or generate arbitrary evaluation code. Expanding the feature set is a separate experiment.

## Library boundaries

Use `python-chess` for board representation, legal moves, making and undoing moves, game outcomes, PGN records, and communication with external engines.

Our code owns evaluation, search, move selection, evolution, match scheduling, and experiment reporting. Do not use Stockfish to select moves for evolved agents or evaluate their search leaves.

Stockfish runs as a separate executable controlled through the library's UCI support. The application remains Python, but the benchmark engine is an external dependency.

Python and library board operations may eventually limit search throughput. Profile before considering a replacement; keep board access separate enough that this layer can be changed later.

## Architecture

Proposed modules under `src/genetic_chess_ai/`:

| Module | Responsibility |
| --- | --- |
| `evaluation` | Extract features and evaluate positions using a genome. |
| `search` | Select moves using minimax and alpha-beta pruning. |
| `agent` | Provide a common move-selection interface for evolved, baseline, and Stockfish agents. |
| `game` | Run games, enforce limits, detect outcomes, and record moves. |
| `tournament` | Schedule opponents, starting positions, and games with colors swapped. |
| `evolution` | Initialize populations, select parents, preserve elites, crossover, and mutate. |
| `experiments` | Load configuration, orchestrate runs, benchmark champions, and save results. |

Dependency flow:

```text
Experiments
  ├── Evolution → Training tournament
  └── Benchmark tournament
                    ↓
                Game runner
                    ↓
              Agent interface
                ├── Evolved / baseline → Search → Evaluation
                └── Stockfish → UCI engine process

Board state and chess rules: python-chess
```

Keep genomes as plain serializable data, independent of agents and board state. Give feature definitions a stable ordering and schema version so saved genomes cannot silently acquire a different meaning.

## Search design

Start with a small fixed depth and a shared implementation for all in-house agents. Keep evaluation conventions explicit and restore the board after every explored move.

Later improvements may include move ordering, transposition tables, and quiescence search. Apply these consistently to all candidates and re-establish benchmark results when search behavior changes. Cached evaluations must not leak between different genomes.

Stockfish depth is not directly comparable to our depth because the engines use different search techniques. Treat a Stockfish configuration as an opponent definition, not as an equal computational budget.

## Evolution loop

1. Initialize a population around sensible hand-written weights, with random variation.
2. Schedule a balanced sample of matches for each candidate.
3. Compute fitness from game results: win = 1, draw = 0.5, loss = 0.
4. Preserve a small number of elite candidates unchanged.
5. Select parents with a method such as tournament selection.
6. Create offspring using per-weight inheritance or numerical blending.
7. Mutate selected weights with small random perturbations.
8. Save the generation, benchmark its champion, and repeat.

Use a fixed pawn value or another documented scale constraint to remove redundant weight scaling. Define feature scales and reasonable mutation ranges; large raw feature counts should not dominate accidentally.

Compare crossover plus mutation against mutation-only evolution. Keep population size, search budget, and match schedule comparable so the comparison is meaningful.

## Fitness and fair matches

Population matches provide the initial fitness signal. Include a fixed baseline and a bounded archive of earlier champions to reduce dependence on the current population's weaknesses.

- Play each selected opening position twice, swapping agent colors.
- Give candidates comparable opponent and opening schedules.
- Use fixed search settings across in-house candidates.
- Separate training openings from held-out benchmark openings.
- Handle repetition and move-count draws consistently.
- Record games stopped by an experiment move limit separately from natural draws; document how they contribute to fitness.
- Record failures such as illegal moves or engine timeouts explicitly rather than silently scoring them as ordinary chess results.

Do not use final material advantage as the main fitness measure. Winning games is the target behavior, and material alone can reward the wrong objective.

## Stockfish benchmark ladder

Pin one Stockfish version and define several named configurations rather than installing multiple versions initially.

Available controls include depth, node count, time per move, `Skill Level`, and `UCI_LimitStrength` with `UCI_Elo`. Inspect the installed engine's supported options and ranges. Skill and Elo settings intentionally weaken play; the configured Elo is not a measured rating for our agent.

Start with a shallow or low-node configuration and measure its actual difficulty. Even limited Stockfish may beat every early candidate, so it should initially be an external benchmark rather than the only source of training fitness.

Benchmark champions against fixed configurations over time. Add stronger opponents as performance improves while retaining earlier configurations for comparable progress measurements. Stockfish matches can later contribute to training if they produce a useful, nonuniform fitness signal.

Record the binary version, limits, skill settings, thread count, hash size, and engine reset/cache policy. Node limits reduce dependence on machine speed, but do not guarantee identical results across versions or configurations.

## Experiment records

Save enough information to reproduce and interpret each run:

- Experiment configuration, random seeds, code revision when available, and dependency versions.
- Feature schema, baseline weights, and search settings.
- Population genomes, fitness scores, champion history, and checkpoints for resuming training.
- Opponent identities, opening positions, colors, results, termination reasons, and PGN games.
- Benchmark wins/draws/losses, game count, runtime, and score percentage: `(wins + 0.5 * draws) / games`.
- Population diversity and performance against the baseline and earlier champions.

A single lucky champion or small match sample is not sufficient evidence of improvement. Validate final candidates on held-out openings and repeat training with multiple random seeds when feasible.

## Implementation milestones

### 1. Play a complete baseline game

Integrate the board library, implement a small fixed evaluation, and add minimax with alpha-beta pruning. Run baseline agents against each other and record games.

Validate evaluation signs, terminal scoring, legal move selection, and board restoration. Compare alpha-beta against plain minimax on small positions to check that pruning preserves the result.

### 2. Establish repeatable matches

Add the shared agent interface, paired opening schedules, explicit termination policies, and result persistence. Establish the fixed baseline and training/benchmark opening split.

### 3. Run the first genetic experiment

Add genome serialization, population initialization, selection, elitism, crossover, mutation, and checkpoints. Use a small population and shallow search to complete an end-to-end run before scaling up.

Initial sizing to measure and adjust: approximately 8–15 weights and 12–20 candidates. Start with fewer weights if the initial feature set is smaller. Estimate runtime from actual games before choosing a generation count.

### 4. Add Stockfish benchmarks

Implement the UCI-backed agent, pin the engine version, and define fixed opponent configurations. Benchmark generation champions using paired held-out openings.

### 5. Evaluate and extend

Compare evolved candidates with the baseline, earlier champions, and Stockfish configurations. Repeat promising experiments across seeds. Add features or search improvements one at a time and measure their effects.

## Scope and constraints

The first version excludes a UI, distributed training, neural networks, and evolution of arbitrary program structures. Parallel game execution can be added after the sequential training loop is correct and measured.

The main compute cost is the number of games multiplied by moves per game and search cost per move. Avoid a full population round robin by default; use balanced sampled schedules when necessary.

The project succeeds first by demonstrating measurable improvement in an understandable heuristic, not by matching optimized engines such as Stockfish.

## Implementation status

The initial Python implementation supports the planned baseline engine, repeatable games, evolutionary training with resume, optional Stockfish benchmarks, and persisted experiment metrics. See the [README](../README.md) for commands, feature definitions, result formats, and current limitations.

Search currently uses fixed depth and capture/promotion ordering. Quiescence search, transposition tables, parallel execution, larger opening suites, and evidence of general strength improvement remain future experiments.

## References

- [python-chess documentation](https://python-chess.readthedocs.io/en/stable/)
- [Python UCI engine integration and search limits](https://python-chess.readthedocs.io/en/stable/engine.html)
- [Stockfish FAQ: strength controls](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html)
