"""Persisted training and benchmarks, with generation-boundary resume."""

import hashlib
import json
import platform
import random
import subprocess
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from genetic_chess_ai.agent import HeuristicAgent, StockfishAgent, StockfishConfig
from genetic_chess_ai.evaluation import FEATURES, Genome
from genetic_chess_ai.evolution import EvolutionConfig, diversity, initialize, next_generation
from genetic_chess_ai.game import BENCHMARK_OPENINGS, TRAIN_OPENINGS, GameResult
from genetic_chess_ai.tournament import paired_match, sampled_pairs, summarize

CHECKPOINT_VERSION = 1


def runtime_identity() -> dict:
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {"python": platform.python_version(), "chess": version("chess"), "sources": sources}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    depth: int = 2
    max_plies: int = 160
    pairing_rounds: int = 1
    archive_size: int = 2
    benchmark_openings: int = 2
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    stockfish: StockfishConfig | None = None

    def __post_init__(self) -> None:
        if self.depth < 1 or self.max_plies < 1 or self.pairing_rounds < 1:
            raise ValueError("Depth, max plies, and pairing rounds must be positive")
        if self.archive_size < 0:
            raise ValueError("Archive size cannot be negative")
        if not 1 <= self.benchmark_openings <= len(BENCHMARK_OPENINGS):
            raise ValueError(f"Benchmark openings must be between 1 and {len(BENCHMARK_OPENINGS)}")

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        values = dict(data)
        values["evolution"] = EvolutionConfig(**values["evolution"])
        if values.get("stockfish") is not None:
            values["stockfish"] = StockfishConfig(**values["stockfish"])
        return cls(**values)


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metadata() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "runtime_identity": runtime_identity(),
        "python": platform.python_version(),
        "chess": version("chess"),
        "project": version("genetic-chess-ai"),
        "git_revision": revision,
        "git_dirty": dirty,
        "feature_names": list(FEATURES),
        "baseline": Genome().to_dict(),
        "training_openings": [asdict(opening) for opening in TRAIN_OPENINGS],
        "benchmark_openings": [asdict(opening) for opening in BENCHMARK_OPENINGS],
        "draw_policy": "Automatically claim all available draws",
        "move_limit_policy": "Score as 0.5 each; record termination separately",
    }


def save_games(directory: Path, stem: str, games: list[GameResult]) -> None:
    write_json(directory / f"{stem}.json", [game.to_dict() for game in games])
    target = directory / f"{stem}.pgn"
    temporary = target.with_suffix(".pgn.tmp")
    temporary.write_text("\n\n".join(game.pgn for game in games) + "\n")
    temporary.replace(target)


def _tuple_tree(value: object) -> object:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _ensure_success(games: list[GameResult]) -> None:
    failures = [game for game in games if game.error]
    if failures:
        raise RuntimeError(f"Match failed; inspect saved game records: {failures[0].error}")


def run_training(
    output: Path,
    generations: int,
    config: TrainingConfig | None = None,
    *,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run `generations` additional generations. Checkpoints contain the NEXT population."""
    if generations < 1:
        raise ValueError("Generations must be positive")
    identity = runtime_identity()
    emit = progress or (lambda message: None)
    if resume:
        checkpoint = json.loads((output / "checkpoint.json").read_text())
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Incompatible checkpoint version")
        if checkpoint.get("runtime_identity") != identity:
            raise ValueError(
                "Python, chess dependency, or source code changed; start a new experiment"
            )
        config = TrainingConfig.from_dict(checkpoint["config"])
        rng = random.Random()
        rng.setstate(_tuple_tree(checkpoint["rng_state"]))
        population = [Genome.from_dict(item) for item in checkpoint["population"]]
        if len(population) != config.evolution.population_size:
            raise ValueError("Checkpoint population size does not match its configuration")
        archive = checkpoint["archive"]
        start = checkpoint["next_generation"]
        history = checkpoint["history"]
    else:
        config = config or TrainingConfig()
        rng = random.Random(config.seed)
        population = initialize(config.evolution, rng)
        archive, history, start = [], [], 1
        checkpoint = {}

    with ExitStack() as stack:
        stockfish = (
            stack.enter_context(StockfishAgent(config.stockfish)) if config.stockfish else None
        )
        engine_metadata = stockfish.metadata if stockfish else None
        if resume and checkpoint.get("stockfish") != engine_metadata:
            raise ValueError("Stockfish identity/settings changed; start a new experiment")
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "config.json", asdict(config))
            write_json(output / "metadata.json", metadata())

        def save_checkpoint(next_number: int) -> None:
            write_json(
                output / "checkpoint.json",
                {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "runtime_identity": identity,
                    "config": asdict(config),
                    "next_generation": next_number,
                    "population": [g.to_dict() for g in population],
                    "archive": archive,
                    "rng_state": rng.getstate(),
                    "history": history,
                    "stockfish": engine_metadata,
                },
            )

        if not resume:
            save_checkpoint(start)
        for number in range(start, start + generations):
            began = perf_counter()
            emit(f"Generation {number}: evaluating {len(population)} candidates")
            directory = output / f"generation-{number:04d}"
            directory.mkdir(exist_ok=True)
            agents = [
                HeuristicAgent(f"g{number:04d}-c{i:03d}", genome, config.depth)
                for i, genome in enumerate(population)
            ]
            baseline = HeuristicAgent("baseline", depth=config.depth)
            old_champions = [
                HeuristicAgent(item["name"], Genome.from_dict(item["genome"]), config.depth)
                for item in archive
            ]
            training_games = []
            opening = TRAIN_OPENINGS[(number - 1) % len(TRAIN_OPENINGS)]
            pairs = sampled_pairs(len(agents), config.pairing_rounds, rng)
            for i, j in pairs:
                training_games.extend(
                    paired_match(agents[i], agents[j], [opening], config.max_plies)
                )
            for candidate in agents:
                for opponent in [baseline, *old_champions]:
                    training_games.extend(
                        paired_match(candidate, opponent, [opening], config.max_plies)
                    )
            save_games(directory, "training-games", training_games)
            _ensure_success(training_games)
            standings = [summarize(training_games, agent.name) for agent in agents]
            fitness = [standing["score"] for standing in standings]
            if len(set(fitness)) == 1:
                emit(
                    f"Generation {number}: all candidates tied; no selection signal this generation"
                )
            best = max(fitness)
            champion_index = rng.choice([i for i, score in enumerate(fitness) if score == best])
            champion = HeuristicAgent(
                f"champion-g{number:04d}", population[champion_index], config.depth
            )

            emit(f"Generation {number}: benchmarking champion (training score {best:.1%})")
            benchmark_games = []
            benchmark_results = {}
            for opponent in [baseline, *old_champions, *([stockfish] if stockfish else [])]:
                games = paired_match(
                    champion,
                    opponent,
                    BENCHMARK_OPENINGS[: config.benchmark_openings],
                    config.max_plies,
                )
                benchmark_games.extend(games)
                benchmark_results[opponent.name] = summarize(games, champion.name)
            save_games(directory, "benchmark-games", benchmark_games)
            _ensure_success(benchmark_games)
            report = {
                "generation": number,
                "champion": champion.genome.to_dict(),
                "champion_candidate": agents[champion_index].name,
                "training_score": best,
                "mean_fitness": sum(fitness) / len(fitness),
                "diversity": diversity(population),
                "population": [
                    {"name": agent.name, "genome": agent.genome.to_dict(), "fitness": stats}
                    for agent, stats in zip(agents, standings, strict=True)
                ],
                "benchmarks": benchmark_results,
                "training_games": len(training_games),
                "seconds": perf_counter() - began,
            }
            write_json(directory / "report.json", report)
            write_json(directory / "champion.json", champion.genome.to_dict())
            if config.archive_size:
                archive = [item for item in archive if item["genome"] != champion.genome.to_dict()]
                archive.append({"name": champion.name, "genome": champion.genome.to_dict()})
                archive = archive[-config.archive_size :]
            population = next_generation(population, fitness, config.evolution, rng)
            history.append(
                {
                    key: report[key]
                    for key in (
                        "generation",
                        "training_score",
                        "mean_fitness",
                        "diversity",
                        "benchmarks",
                        "seconds",
                    )
                }
            )
            save_checkpoint(number + 1)
            write_json(output / "champion.json", champion.genome.to_dict())
            write_json(output / "history.json", history)
            emit(
                f"Generation {number}: baseline score "
                f"{benchmark_results['baseline']['score']:.1%}; "
                f"saved to {directory}"
            )
    return history[-1]


def run_benchmark(
    genome: Genome,
    output: Path,
    *,
    depth: int = 2,
    max_plies: int = 160,
    openings: int = 4,
    stockfish_config: StockfishConfig | None = None,
) -> dict:
    if not 1 <= openings <= len(BENCHMARK_OPENINGS):
        raise ValueError(f"Openings must be between 1 and {len(BENCHMARK_OPENINGS)}")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    candidate = HeuristicAgent("candidate", genome, depth)
    with ExitStack() as stack:
        opponent = (
            stack.enter_context(StockfishAgent(stockfish_config))
            if stockfish_config
            else HeuristicAgent("baseline", depth=depth)
        )
        output.mkdir(parents=True, exist_ok=False)
        games = paired_match(candidate, opponent, BENCHMARK_OPENINGS[:openings], max_plies)
        save_games(output, "games", games)
        report = {
            "candidate": genome.to_dict(),
            "depth": depth,
            "max_plies": max_plies,
            "openings": openings,
            "opponent": opponent.name,
            "results": summarize(games, candidate.name),
            "stockfish": opponent.metadata if isinstance(opponent, StockfishAgent) else None,
            "metadata": metadata(),
        }
        write_json(output / "report.json", report)
        _ensure_success(games)
    return report
