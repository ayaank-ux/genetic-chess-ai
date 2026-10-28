"""A small, explicit feature vocabulary. Scores are always White-relative."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import chess

SCHEMA_VERSION = 1
FEATURES = (
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "activity",
    "center_control",
    "bishop_pair",
    "doubled_pawns",
    "isolated_pawns",
    "passed_pawns",
    "king_shield",
)
BASELINE_WEIGHTS = (100.0, 320.0, 330.0, 500.0, 900.0, 4.0, 10.0, 30.0, -15.0, -12.0, 25.0, 10.0)
# Pawn value stays fixed. Bounds also keep ordinary scores far below mate scores.
BOUNDS = ((100, 100), (150, 600), (150, 600), (250, 900), (500, 1500)) + ((-100, 100),) * 7
MUTATION_SCALES = (0.0, 30.0, 30.0, 50.0, 90.0, 3.0, 5.0, 10.0, 5.0, 5.0, 10.0, 5.0)
CENTER = (chess.D4, chess.E4, chess.D5, chess.E5)


@dataclass(frozen=True)
class Genome:
    weights: tuple[float, ...] = BASELINE_WEIGHTS

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} weights")
        for name, value, (low, high) in zip(FEATURES, self.weights, BOUNDS, strict=True):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "features": list(FEATURES),
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        if data.get("schema_version") != SCHEMA_VERSION or data.get("features") != list(FEATURES):
            raise ValueError("Incompatible genome feature schema")
        return cls(tuple(float(value) for value in data["weights"]))

    @classmethod
    def load(cls, path: Path) -> "Genome":
        return cls.from_dict(json.loads(path.read_text()))


def _side_features(board: chess.Board, color: chess.Color) -> list[float]:
    values = [float(len(board.pieces(piece, color))) for piece in range(chess.PAWN, chess.KING)]
    # Activity is attacked destinations excluding friendly pieces, not legal mobility.
    # This definition is symmetric and does not change board.turn or castling state.
    activity = sum(
        len(board.attacks(square) & ~chess.SquareSet(board.occupied_co[color]))
        for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for square in board.pieces(piece, color)
    )
    center = sum(len(board.attackers(color, square)) for square in CENTER)
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    files = [sum(chess.square_file(square) == file for square in pawns) for file in range(8)]
    doubled = sum(max(0, count - 1) for count in files)
    isolated = sum(
        count
        for file, count in enumerate(files)
        if (file == 0 or files[file - 1] == 0) and (file == 7 or files[file + 1] == 0)
    )
    passed = 0.0
    for square in pawns:
        rank, file = chess.square_rank(square), chess.square_file(square)
        if not any(
            abs(chess.square_file(enemy) - file) <= 1
            and (chess.square_rank(enemy) > rank if color else chess.square_rank(enemy) < rank)
            for enemy in enemy_pawns
        ):
            advancement = rank - 1 if color else 6 - rank
            passed += (advancement + 1) / 6
    king = board.king(color)
    shield = 0
    if king is not None:
        next_rank = chess.square_rank(king) + (1 if color else -1)
        shield = sum(
            chess.square_rank(square) == next_rank
            and abs(chess.square_file(square) - chess.square_file(king)) <= 1
            for square in pawns
        )
    values.extend((activity, center, int(values[2] >= 2), doubled, isolated, passed, shield))
    return values


def features(board: chess.Board) -> tuple[float, ...]:
    return tuple(
        w - b
        for w, b in zip(
            _side_features(board, chess.WHITE), _side_features(board, chess.BLACK), strict=True
        )
    )


def evaluate(board: chess.Board, genome: Genome) -> float:
    return sum(
        weight * feature for weight, feature in zip(genome.weights, features(board), strict=True)
    )
