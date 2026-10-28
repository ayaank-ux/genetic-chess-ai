"""Game execution with explicit draw, truncation, and failure policies."""

from dataclasses import asdict, dataclass
from time import perf_counter

import chess
import chess.pgn

from genetic_chess_ai.agent import Agent


@dataclass(frozen=True)
class Opening:
    name: str
    moves: tuple[str, ...] = ()
    fen: str = chess.STARTING_FEN

    def board(self) -> chess.Board:
        board = chess.Board(self.fen)
        if not board.is_valid():
            raise ValueError(f"Invalid opening position: {self.name}")
        for move in self.moves:
            board.push_uci(move)
        return board


TRAIN_OPENINGS = (
    Opening("open-game", ("e2e4", "e7e5", "g1f3", "b8c6")),
    Opening("queens-pawn", ("d2d4", "d7d5", "c2c4", "e7e6")),
    Opening("sicilian", ("e2e4", "c7c5", "g1f3", "d7d6")),
    Opening("english", ("c2c4", "e7e5", "b1c3", "g8f6")),
)
BENCHMARK_OPENINGS = (
    Opening("caro-kann", ("e2e4", "c7c6", "d2d4", "d7d5")),
    Opening("indian", ("d2d4", "g8f6", "c2c4", "g7g6")),
    Opening("french", ("e2e4", "e7e6", "d2d4", "d7d5")),
    Opening("reti", ("g1f3", "d7d5", "g2g3", "g8f6")),
)


@dataclass
class GameResult:
    white: str
    black: str
    opening: str
    initial_fen: str
    result: str
    termination: str
    plies: int
    seconds: float
    moves: list[str]
    pgn: str
    error: str | None = None

    def score(self, color: chess.Color) -> float:
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[self.result]
        return white_score if color else 1 - white_score

    def to_dict(self) -> dict:
        data = asdict(self)
        del data["pgn"]
        return data


def play_game(
    white: Agent, black: Agent, opening: Opening | None = None, max_plies: int = 160
) -> GameResult:
    if max_plies < 1:
        raise ValueError("max_plies must be at least 1")
    opening = opening or Opening("start")
    board = opening.board()
    initial_fen = board.fen()
    initial_stack = len(board.move_stack)
    started = perf_counter()
    error = None
    result, termination = "1/2-1/2", "move_limit"
    # Initialization failures are infrastructure errors and abort the caller.
    white.new_game()
    black.new_game()
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result, termination = outcome.result(), outcome.termination.name.lower()
            break
        if len(board.move_stack) - initial_stack >= max_plies:
            break
        agent = white if board.turn else black
        try:
            # Protect the authoritative board from agent mutation, preserving repetition history.
            move = agent.choose_move(board.copy(stack=True))
            if not isinstance(move, chess.Move) or move not in board.legal_moves:
                termination = "illegal_move"
                error = f"{agent.name} returned an illegal move: {move}"
            else:
                board.push(move)
                continue
        except Exception as exc:
            termination = "agent_error"
            error = f"{agent.name}: {type(exc).__name__}: {exc}"
        result = "0-1" if board.turn else "1-0"
        break
    game = chess.pgn.Game.from_board(board)
    game.headers.update(
        {
            "Event": "Genetic Chess AI",
            "White": white.name,
            "Black": black.name,
            "Result": result,
            "Opening": opening.name,
            "Termination": termination,
        }
    )
    if error:
        game.end().comment = error
    return GameResult(
        white.name,
        black.name,
        opening.name,
        initial_fen,
        result,
        termination,
        len(board.move_stack) - initial_stack,
        perf_counter() - started,
        [move.uci() for move in board.move_stack[initial_stack:]],
        str(game),
        error,
    )
