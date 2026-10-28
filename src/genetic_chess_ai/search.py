"""Deterministic minimax with alpha-beta pruning and fixed move ordering."""

from dataclasses import dataclass
from math import inf

import chess

from genetic_chess_ai.evaluation import Genome, evaluate

MATE_SCORE = 1_000_000.0
PIECE_ORDER = (0, 100, 320, 330, 500, 900, 20_000)


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: float
    nodes: int


def terminal_score(board: chess.Board, ply: int = 0) -> float | None:
    # Both agents automatically claim available draws, including a claim by intended move.
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return MATE_SCORE - ply if outcome.winner else -MATE_SCORE + ply


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    def priority(move: chess.Move) -> tuple[int, str]:
        victim = board.piece_type_at(move.to_square)
        if board.is_en_passant(move):
            victim = chess.PAWN
        capture = (
            0
            if victim is None
            else (
                10 * PIECE_ORDER[victim] - PIECE_ORDER[board.piece_type_at(move.from_square) or 0]
            )
        )
        promotion = PIECE_ORDER[move.promotion or 0]
        return (-(capture + promotion), move.uci())

    return sorted(board.legal_moves, key=priority)


def search(board: chess.Board, genome: Genome, depth: int = 2) -> SearchResult:
    if depth < 1:
        raise ValueError("Search depth must be at least 1")
    nodes = 0

    def visit(
        remaining: int, alpha: float, beta: float, ply: int
    ) -> tuple[float, chess.Move | None]:
        nonlocal nodes
        nodes += 1
        terminal = terminal_score(board, ply)
        if terminal is not None:
            return terminal, None
        if remaining == 0:
            return evaluate(board, genome), None
        maximizing = board.turn == chess.WHITE
        best_score = -inf if maximizing else inf
        best_move = None
        for move in ordered_moves(board):
            board.push(move)
            try:
                score, _ = visit(remaining - 1, alpha, beta, ply + 1)
            finally:
                board.pop()
            if best_move is None or (score > best_score if maximizing else score < best_score):
                best_score, best_move = score, move
            if maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_move

    score, move = visit(depth, -inf, inf, 0)
    return SearchResult(move, score, nodes)
