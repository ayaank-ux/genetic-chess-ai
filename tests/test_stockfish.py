"""Optional real-engine integration tests; core tests need no Stockfish download."""

import os
import shutil

import chess
import chess.engine
import pytest

from genetic_chess_ai.agent import StockfishAgent, StockfishConfig
from genetic_chess_ai.evaluation import Genome
from genetic_chess_ai.experiments import run_benchmark

ENGINE = os.environ.get("STOCKFISH_EXECUTABLE") or shutil.which("stockfish")
pytestmark = pytest.mark.skipif(not ENGINE, reason="Set STOCKFISH_EXECUTABLE for real UCI tests")


def test_uci_move_reset_and_metadata():
    with StockfishAgent(StockfishConfig(path=ENGINE, nodes=100)) as agent:
        for _ in range(2):
            agent.new_game()
            board = chess.Board()
            assert agent.choose_move(board) in board.legal_moves
        assert "Stockfish" in agent.metadata["id"]["name"]
        assert len(agent.metadata["binary_sha256"]) == 64


def test_real_benchmark(tmp_path):
    report = run_benchmark(
        Genome(),
        tmp_path / "stockfish",
        depth=1,
        max_plies=4,
        openings=1,
        stockfish_config=StockfishConfig(path=ENGINE, nodes=100),
    )
    assert report["results"]["games"] == 2
    assert report["stockfish"]["options"]["Threads"] == 1
    assert report["results"]["terminations"] == {"move_limit": 2}


def test_engine_rejects_unsupported_strength():
    with pytest.raises(chess.engine.EngineError):
        StockfishAgent(StockfishConfig(path=ENGINE, skill=1000))
