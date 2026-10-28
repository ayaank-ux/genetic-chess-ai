"""Command-line entry points for playing, evolving, and benchmarking."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine

from genetic_chess_ai.agent import HeuristicAgent, StockfishAgent, StockfishConfig
from genetic_chess_ai.evaluation import Genome, evaluate, features
from genetic_chess_ai.evolution import EvolutionConfig
from genetic_chess_ai.experiments import (
    TrainingConfig,
    run_benchmark,
    run_training,
    save_games,
    write_json,
)
from genetic_chess_ai.game import Opening, play_game
from genetic_chess_ai.search import search


def _stockfish_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stockfish", metavar="PATH", help="Stockfish executable; omit for baseline"
    )
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--sf-depth", type=int, help="Stockfish depth limit")
    limits.add_argument("--sf-nodes", type=int, help="Stockfish node limit (default: 1000)")
    limits.add_argument("--sf-time", type=float, help="Stockfish seconds per move")
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument("--sf-skill", type=int, help="Stockfish Skill Level")
    strength.add_argument(
        "--sf-elo", type=int, help="Enable Stockfish UCI_LimitStrength and UCI_Elo"
    )


def _stockfish_config(args: argparse.Namespace) -> StockfishConfig | None:
    settings = (args.sf_depth, args.sf_nodes, args.sf_time, args.sf_skill, args.sf_elo)
    if args.stockfish is None:
        if any(value is not None for value in settings):
            raise ValueError("Stockfish settings require --stockfish PATH")
        return None
    nodes = args.sf_nodes
    if args.sf_depth is None and args.sf_time is None and nodes is None:
        nodes = 1000
    return StockfishConfig(
        path=args.stockfish,
        depth=args.sf_depth,
        nodes=nodes,
        time=args.sf_time,
        skill=args.sf_skill,
        elo=args.sf_elo,
    )


def _genome(path: Path | None) -> Genome:
    return Genome.load(path) if path else Genome()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolve chess evaluation heuristics in Python")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate", help="Inspect features and search a position")
    evaluate_parser.add_argument("--fen", default=chess.STARTING_FEN)
    evaluate_parser.add_argument("--genome", type=Path)
    evaluate_parser.add_argument("--depth", type=int, default=2)

    play = commands.add_parser("play", help="Play a game and save JSON and PGN")
    play.add_argument("--white", type=Path, help="White genome JSON (default: baseline)")
    play.add_argument("--black", type=Path, help="Black genome JSON (default: baseline)")
    play.add_argument("--depth", type=int, default=2)
    play.add_argument("--max-plies", type=int, default=160)
    play.add_argument("--fen", default=chess.STARTING_FEN)
    play.add_argument("--output", type=Path, required=True, help="New output directory")

    train = commands.add_parser("train", help="Start an evolutionary experiment")
    train.add_argument("--output", type=Path, required=True, help="New experiment directory")
    train.add_argument("--generations", type=int, default=5)
    train.add_argument("--population", type=int, default=12)
    train.add_argument("--elites", type=int, default=2)
    train.add_argument("--tournament-size", type=int, default=3)
    train.add_argument("--mutation-rate", type=float, default=0.3)
    train.add_argument("--mutation-scale", type=float, default=1.0)
    train.add_argument("--mutation-only", action="store_true")
    train.add_argument("--depth", type=int, default=2)
    train.add_argument("--max-plies", type=int, default=160)
    train.add_argument("--pairing-rounds", type=int, default=1)
    train.add_argument("--archive-size", type=int, default=2)
    train.add_argument("--benchmark-openings", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    _stockfish_arguments(train)

    resume = commands.add_parser("resume", help="Continue from a generation checkpoint")
    resume.add_argument("run", type=Path)
    resume.add_argument("--generations", type=int, default=1, help="Additional generations")

    benchmark = commands.add_parser("benchmark", help="Play paired held-out benchmark games")
    benchmark.add_argument("--genome", type=Path, help="Candidate genome (default: baseline)")
    benchmark.add_argument("--output", type=Path, required=True, help="New output directory")
    benchmark.add_argument("--depth", type=int, default=2)
    benchmark.add_argument("--max-plies", type=int, default=160)
    benchmark.add_argument("--openings", type=int, default=4)
    _stockfish_arguments(benchmark)

    inspect = commands.add_parser(
        "stockfish-info", help="Inspect UCI identity and supported settings"
    )
    inspect.add_argument("path", nargs="?", default="stockfish")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "evaluate":
                board = Opening("custom", fen=args.fen).board()
                genome = _genome(args.genome)
                result = search(board, genome, args.depth)
                report = {
                    "fen": board.fen(),
                    "evaluation": evaluate(board, genome),
                    "features": dict(
                        zip(genome.to_dict()["features"], features(board), strict=True)
                    ),
                    "move": result.move.uci() if result.move else None,
                    "search_score": result.score,
                    "nodes": result.nodes,
                }
            case "play":
                white = HeuristicAgent("white", _genome(args.white), args.depth)
                black = HeuristicAgent("black", _genome(args.black), args.depth)
                opening = Opening("custom", fen=args.fen)
                opening.board()
                if args.max_plies < 1:
                    raise ValueError("max_plies must be positive")
                args.output.mkdir(parents=True, exist_ok=False)
                game = play_game(white, black, opening, args.max_plies)
                save_games(args.output, "game", [game])
                write_json(
                    args.output / "config.json",
                    {
                        "white": white.genome.to_dict(),
                        "black": black.genome.to_dict(),
                        "depth": args.depth,
                        "max_plies": args.max_plies,
                        "opening": asdict(opening),
                    },
                )
                if game.error:
                    raise RuntimeError(f"Game failed; inspect {args.output}: {game.error}")
                report = game.to_dict()
            case "train":
                config = TrainingConfig(
                    seed=args.seed,
                    depth=args.depth,
                    max_plies=args.max_plies,
                    pairing_rounds=args.pairing_rounds,
                    archive_size=args.archive_size,
                    benchmark_openings=args.benchmark_openings,
                    evolution=EvolutionConfig(
                        args.population,
                        args.elites,
                        args.tournament_size,
                        args.mutation_rate,
                        args.mutation_scale,
                        not args.mutation_only,
                    ),
                    stockfish=_stockfish_config(args),
                )
                report = run_training(
                    args.output,
                    args.generations,
                    config,
                    progress=lambda text: print(text, file=sys.stderr, flush=True),
                )
            case "resume":
                report = run_training(
                    args.run,
                    args.generations,
                    resume=True,
                    progress=lambda text: print(text, file=sys.stderr, flush=True),
                )
            case "benchmark":
                report = run_benchmark(
                    _genome(args.genome),
                    args.output,
                    depth=args.depth,
                    max_plies=args.max_plies,
                    openings=args.openings,
                    stockfish_config=_stockfish_config(args),
                )
            case "stockfish-info":
                with StockfishAgent(StockfishConfig(path=args.path)) as engine:
                    report = {
                        "id": engine.metadata["id"],
                        "options": {
                            name: {
                                "type": option.type,
                                "default": option.default,
                                "min": option.min,
                                "max": option.max,
                                "var": option.var,
                            }
                            for name, option in engine.engine.options.items()
                        },
                    }
        print(json.dumps(report, indent=2, allow_nan=False))
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        RuntimeError,
        chess.engine.EngineError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Training can resume from the last completed generation.\n")
