"""Color-balanced matches and balanced sampled population schedules."""

import random
from collections import Counter
from collections.abc import Sequence

from genetic_chess_ai.agent import Agent
from genetic_chess_ai.game import GameResult, Opening, play_game


def paired_match(
    first: Agent, second: Agent, openings: Sequence[Opening], max_plies: int
) -> list[GameResult]:
    games = []
    for opening in openings:
        games.append(play_game(first, second, opening, max_plies))
        games.append(play_game(second, first, opening, max_plies))
    return games


def sampled_pairs(size: int, rounds: int, rng: random.Random) -> list[tuple[int, int]]:
    if size < 2 or size % 2:
        raise ValueError("Population must have an even number of candidates")
    if rounds < 1:
        raise ValueError("At least one pairing round is required")
    pairs = []
    for _ in range(rounds):
        indices = list(range(size))
        rng.shuffle(indices)
        pairs.extend(zip(indices[::2], indices[1::2], strict=True))
    return pairs


def summarize(games: Sequence[GameResult], agent_name: str) -> dict:
    scores = []
    terminations: Counter[str] = Counter()
    for game in games:
        if agent_name not in (game.white, game.black):
            continue
        if game.white == game.black:
            raise ValueError("Summary requires distinct agent names")
        scores.append(game.score(game.white == agent_name))
        terminations[game.termination] += 1
    return {
        "games": len(scores),
        "wins": scores.count(1.0),
        "draws": scores.count(0.5),
        "losses": scores.count(0.0),
        "score": sum(scores) / len(scores) if scores else 0.0,
        "terminations": dict(terminations),
    }
