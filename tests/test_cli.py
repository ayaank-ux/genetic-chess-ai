import json
import subprocess
import sys

from genetic_chess_ai.cli import main


def test_module_cli():
    result = subprocess.run(
        [sys.executable, "-m", "genetic_chess_ai", "evaluate", "--depth", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["move"]


def test_training_cli_and_resume(tmp_path, capsys):
    output = tmp_path / "training"
    main(
        [
            "train",
            "--output",
            str(output),
            "--population",
            "4",
            "--generations",
            "1",
            "--depth",
            "1",
            "--max-plies",
            "1",
            "--benchmark-openings",
            "1",
        ]
    )
    assert json.loads(capsys.readouterr().out)["generation"] == 1
    main(["resume", str(output), "--generations", "1"])
    assert json.loads(capsys.readouterr().out)["generation"] == 2


def test_missing_stockfish_is_actionable(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "genetic_chess_ai",
            "benchmark",
            "--output",
            str(tmp_path / "unused"),
            "--stockfish",
            str(tmp_path / "missing-engine"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "unused").exists()
