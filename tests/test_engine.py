import io
from math import inf

import chess
import chess.pgn
import pytest

from genetic_chess_ai.agent import HeuristicAgent
from genetic_chess_ai.evaluation import FEATURES, Genome, evaluate, features
from genetic_chess_ai.game import BENCHMARK_OPENINGS, TRAIN_OPENINGS, Opening, play_game
from genetic_chess_ai.search import MATE_SCORE, search, terminal_score


def plain_minimax(board, depth, ply=0):
    terminal = terminal_score(board, ply)
    if terminal is not None:
        return terminal
    if depth == 0:
        return evaluate(board, Genome())
    maximize = board.turn
    best = -inf if maximize else inf
    for move in board.legal_moves:
        board.push(move)
        try:
            value = plain_minimax(board, depth - 1, ply + 1)
        finally:
            board.pop()
        best = max(best, value) if maximize else min(best, value)
    return best


def test_initial_evaluation_and_color_symmetry():
    assert evaluate(chess.Board(), Genome()) == 0
    board = TRAIN_OPENINGS[0].board()
    board.push_san("Bb5")
    board.push_san("a6")
    assert features(board.mirror()) == pytest.approx(tuple(-v for v in features(board)))
    assert evaluate(board.mirror(), Genome()) == pytest.approx(-evaluate(board, Genome()))


def test_pawn_features():
    board = chess.Board("7k/8/8/2P5/2P5/P7/8/7K w - - 0 1")
    values = dict(zip(FEATURES, features(board), strict=True))
    assert values["pawn"] == 3
    assert values["doubled_pawns"] == 1
    assert values["isolated_pawns"] == 3
    assert values["passed_pawns"] == pytest.approx((4 + 3 + 2) / 6)


@pytest.mark.parametrize("opening", TRAIN_OPENINGS[:2])
def test_alpha_beta_matches_minimax_and_restores_board(opening):
    board = opening.board()
    before = board.fen(), list(board.move_stack)
    result = search(board, Genome(), 2)
    assert result.score == pytest.approx(plain_minimax(board, 2))
    assert result.move in board.legal_moves
    assert (board.fen(), board.move_stack) == before


@pytest.mark.parametrize("mirror", [False, True])
def test_finds_mate_for_either_color(mirror):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    if mirror:
        board = board.mirror()
    result = search(board, Genome(), 2)
    assert abs(result.score) == MATE_SCORE - 1
    assert (result.score > 0) == board.turn
    board.push(result.move)
    assert board.is_checkmate()


@pytest.mark.parametrize(
    "fen",
    [
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # Stalemate.
        "7k/8/6K1/8/8/8/8/8 w - - 0 1",  # Insufficient material.
    ],
)
def test_terminal_draws(fen):
    result = search(chess.Board(fen), Genome(), 2)
    assert result.move is None
    assert result.score == 0


def test_repetition_history_is_preserved():
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)
    before = list(board.move_stack)
    assert search(board, Genome(), 1).score == 0
    assert board.move_stack == before


def test_search_restores_board_on_evaluation_failure(monkeypatch):
    def fail(*args):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr("genetic_chess_ai.search.evaluate", fail)
    board = chess.Board()
    with pytest.raises(RuntimeError):
        search(board, Genome(), 1)
    assert board.fen() == chess.STARTING_FEN
    assert board.move_stack == []


def test_genome_schema_and_bounds():
    assert Genome.from_dict(Genome().to_dict()) == Genome()
    invalid = Genome().to_dict()
    invalid["features"].reverse()
    with pytest.raises(ValueError, match="schema"):
        Genome.from_dict(invalid)
    for value in (float("nan"), float("inf"), -100, 101):
        with pytest.raises(ValueError):
            Genome((value, *Genome().weights[1:]))


def test_opening_sets_are_legal_and_disjoint():
    train = {opening.board().fen() for opening in TRAIN_OPENINGS}
    held_out = {opening.board().fen() for opening in BENCHMARK_OPENINGS}
    assert train.isdisjoint(held_out)


def test_move_limit_and_pgn_replay():
    opening = TRAIN_OPENINGS[0]
    game = play_game(HeuristicAgent("a", depth=1), HeuristicAgent("b", depth=1), opening, 4)
    assert game.termination == "move_limit"
    assert game.result == "1/2-1/2"
    assert game.plies == 4
    parsed = chess.pgn.read_game(io.StringIO(game.pgn))
    board = parsed.board()
    for move in parsed.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
    assert len(board.move_stack) == len(opening.moves) + 4


def test_game_reaches_checkmate():
    opening = Opening("mate", fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    game = play_game(HeuristicAgent("a", depth=1), HeuristicAgent("b", depth=1), opening, 1)
    assert game.result == "1-0"
    assert game.termination == "checkmate"


@pytest.mark.parametrize("raises", [False, True])
def test_bad_agent_forfeits_explicitly(raises):
    class BadAgent:
        name = "bad"

        def new_game(self):
            pass

        def choose_move(self, board):
            board.clear()
            if raises:
                raise RuntimeError("broken")
            return chess.Move.from_uci("e2e5")

    game = play_game(BadAgent(), HeuristicAgent("good", depth=1), max_plies=2)
    assert game.result == "0-1"
    assert game.termination == ("agent_error" if raises else "illegal_move")
    assert game.error
    assert game.plies == 0


def test_invalid_search_and_game_settings():
    with pytest.raises(ValueError):
        search(chess.Board(), Genome(), 0)
    with pytest.raises(ValueError):
        Opening("invalid", fen="8/8/8/8/8/8/8/8 w - - 0 1").board()
