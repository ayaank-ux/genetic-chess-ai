"""Interchangeable agents; only the external benchmark talks to Stockfish."""

import hashlib
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import chess
import chess.engine

from genetic_chess_ai.evaluation import Genome
from genetic_chess_ai.search import search


class Agent(Protocol):
    name: str

    def new_game(self) -> None: ...

    def choose_move(self, board: chess.Board) -> chess.Move | None: ...


@dataclass
class HeuristicAgent:
    name: str = "baseline"
    genome: Genome = field(default_factory=Genome)
    depth: int = 2

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("Search depth must be at least 1")

    def new_game(self) -> None:
        pass

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        return search(board, self.genome, self.depth).move


@dataclass(frozen=True)
class StockfishConfig:
    path: str = "stockfish"
    depth: int | None = None
    nodes: int | None = 1000
    time: float | None = None
    skill: int | None = None
    elo: int | None = None
    threads: int = 1
    hash_mb: int = 16
    timeout: float = 10.0

    def __post_init__(self) -> None:
        limits = (self.depth, self.nodes, self.time)
        if sum(value is not None for value in limits) != 1:
            raise ValueError("Choose exactly one Stockfish depth, node, or time limit")
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in limits):
            raise ValueError("Stockfish search limits must be positive and finite")
        if self.skill is not None and self.elo is not None:
            raise ValueError("Choose either Stockfish skill or Elo, not both")
        if (
            self.threads < 1
            or self.hash_mb < 1
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("Threads, hash size, and timeout must be positive")


class StockfishAgent:
    def __init__(self, config: StockfishConfig):
        self.config = config
        self.name = "stockfish"
        self.engine = chess.engine.SimpleEngine.popen_uci(config.path, timeout=config.timeout)
        self._game = object()
        options = {"Threads": config.threads, "Hash": config.hash_mb}
        if config.elo is not None:
            options.update({"UCI_LimitStrength": True, "UCI_Elo": config.elo})
        else:
            options["UCI_LimitStrength"] = False
            if config.skill is not None:
                options["Skill Level"] = config.skill
        try:
            for key, value in options.items():
                if key not in self.engine.options:
                    raise ValueError(f"Engine does not support {key}")
                self.engine.options[key].parse(value)
            self.engine.configure(options)
        except BaseException:
            self.close()
            raise
        self.metadata = {
            "id": dict(self.engine.id),
            "config": asdict(config),
            "options": options,
            "reset_policy": "ucinewgame per game; Clear Hash when supported; no pondering",
            "binary_path": str(Path(config.path).resolve())
            if Path(config.path).exists()
            else config.path,
        }
        binary = shutil.which(config.path)
        if binary:
            with Path(binary).open("rb") as stream:
                self.metadata["binary_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            self.metadata["binary_path"] = str(Path(binary).resolve())

    def new_game(self) -> None:
        self._game = object()
        if "Clear Hash" in self.engine.options:
            self.engine.configure({"Clear Hash": None})

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        limit = chess.engine.Limit(
            depth=self.config.depth, nodes=self.config.nodes, time=self.config.time
        )
        return self.engine.play(board, limit, game=self._game, ponder=False).move

    def close(self) -> None:
        try:
            self.engine.quit()
        except (chess.engine.EngineError, TimeoutError):
            self.engine.close()

    def __enter__(self) -> "StockfishAgent":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
