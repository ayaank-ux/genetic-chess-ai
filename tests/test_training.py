import json
import random
from collections import Counter

import pytest

from genetic_chess_ai.agent import HeuristicAgent, StockfishConfig
from genetic_chess_ai.evaluation import Genome
from genetic_chess_ai.evolution import EvolutionConfig, diversity, initialize, next_generation
from genetic_chess_ai.experiments import TrainingConfig, run_benchmark, run_training
from genetic_chess_ai.game import TRAIN_OPENINGS
from genetic_chess_ai.tournament import paired_match, sampled_pairs, summarize


def tiny_config():
    return TrainingConfig(
        depth=1,
        max_plies=2,
        benchmark_openings=1,
        evolution=EvolutionConfig(population_size=4, elites=1),
    )


def read(path):
    return json.loads(path.read_text())


def test_schedule_and_color_balance():
    pairs = sampled_pairs(12, 3, random.Random(42))
    assert Counter(index for pair in pairs for index in pair) == dict.fromkeys(range(12), 3)
    assert all(a != b for a, b in pairs)
    games = paired_match(
        HeuristicAgent("a", depth=1), HeuristicAgent("b", depth=1), TRAIN_OPENINGS[:1], 1
    )
    assert [(game.white, game.black) for game in games] == [("a", "b"), ("b", "a")]
    assert games[0].initial_fen == games[1].initial_fen
    assert summarize(games, "a")["score"] == 0.5


@pytest.mark.parametrize("crossover", [False, True])
def test_elitism_mutation_and_seed(crossover):
    config = EvolutionConfig(population_size=4, elites=1, mutation_rate=1, crossover=crossover)
    rng = random.Random(7)
    population = initialize(config, rng)
    assert population == initialize(config, random.Random(7))
    children = next_generation(population, [0, 0.25, 0.5, 1], config, rng)
    assert children[0] == population[3]
    assert len(children) == 4
    assert all(genome.weights[0] == 100 for genome in children)
    assert diversity(children) > 0
    assert any(child not in population for child in children[1:])


def test_resume_matches_uninterrupted_training(tmp_path):
    full, split = tmp_path / "full", tmp_path / "split"
    run_training(full, 2, tiny_config())
    run_training(split, 1, tiny_config())
    run_training(split, 1, resume=True)
    full_checkpoint = read(full / "checkpoint.json")
    split_checkpoint = read(split / "checkpoint.json")
    for key in ("population", "archive", "rng_state", "next_generation", "config"):
        assert full_checkpoint[key] == split_checkpoint[key]
    for number in (1, 2):
        name = f"generation-{number:04d}"
        full_report, split_report = (
            read(full / name / "report.json"),
            read(split / name / "report.json"),
        )
        full_report.pop("seconds")
        split_report.pop("seconds")
        assert full_report == split_report
        assert (full / name / "training-games.pgn").read_text() == (
            split / name / "training-games.pgn"
        ).read_text()
    assert Genome.load(split / "champion.json")
    assert len(read(split / "history.json")) == 2
    standings = read(split / "generation-0001" / "report.json")["population"]
    assert {item["fitness"]["games"] for item in standings} == {4}


def test_interruption_replays_unfinished_generation(tmp_path, monkeypatch):
    import genetic_chess_ai.experiments as experiments

    output = tmp_path / "run"
    original = experiments.paired_match

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(experiments, "paired_match", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_training(output, 1, tiny_config())
    assert read(output / "checkpoint.json")["next_generation"] == 1
    monkeypatch.setattr(experiments, "paired_match", original)
    run_training(output, 1, resume=True)
    assert read(output / "checkpoint.json")["next_generation"] == 2


def test_existing_output_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):
        run_training(tmp_path, 1, tiny_config())


def test_failed_agent_saves_evidence_without_advancing_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / "failure"
    monkeypatch.setattr(HeuristicAgent, "choose_move", lambda self, board: None)
    with pytest.raises(RuntimeError, match="Match failed"):
        run_training(output, 1, tiny_config())
    assert read(output / "checkpoint.json")["next_generation"] == 1
    records = read(output / "generation-0001" / "training-games.json")
    assert all(record["termination"] == "illegal_move" for record in records)
    assert not (output / "champion.json").exists()


def test_resume_rejects_changed_runtime(tmp_path):
    output = tmp_path / "runtime"
    run_training(output, 1, tiny_config())
    checkpoint_path = output / "checkpoint.json"
    checkpoint = read(checkpoint_path)
    checkpoint["runtime_identity"]["chess"] = "different-version"
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="changed"):
        run_training(output, 1, resume=True)


def test_benchmark_artifacts(tmp_path):
    report = run_benchmark(Genome(), tmp_path / "benchmark", depth=1, max_plies=2, openings=1)
    assert report["results"]["games"] == 2
    assert report["results"]["terminations"] == {"move_limit": 2}
    assert (tmp_path / "benchmark" / "games.pgn").is_file()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 3},
        {"elites": 12},
        {"mutation_rate": 1.1},
        {"mutation_scale": float("nan")},
        {"tournament_size": 20},
    ],
)
def test_invalid_evolution_config(kwargs):
    with pytest.raises(ValueError):
        EvolutionConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 0},
        {"nodes": -1},
        {"time": float("nan"), "nodes": None},
        {"nodes": None},
        {"skill": 1, "elo": 1500},
    ],
)
def test_invalid_stockfish_config(kwargs):
    with pytest.raises(ValueError):
        StockfishConfig(**kwargs)
