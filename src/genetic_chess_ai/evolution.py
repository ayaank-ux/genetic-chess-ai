"""Real-valued genomes, tournament selection, elitism, and bounded mutation."""

import math
import random
from dataclasses import dataclass
from statistics import pstdev

from genetic_chess_ai.evaluation import BOUNDS, MUTATION_SCALES, Genome


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 12
    elites: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.3
    mutation_scale: float = 1.0
    crossover: bool = True

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("Population size must be even and at least 4")
        if not 1 <= self.elites < self.population_size:
            raise ValueError("Elites must be between 1 and population size - 1")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("Tournament size must be between 1 and population size")
        if not math.isfinite(self.mutation_rate) or not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be in [0, 1]")
        if not math.isfinite(self.mutation_scale) or self.mutation_scale < 0:
            raise ValueError("Mutation scale must be nonnegative and finite")


def mutate(genome: Genome, rng: random.Random, rate: float, scale: float) -> Genome:
    values = list(genome.weights)
    for i in range(1, len(values)):
        if rng.random() < rate:
            low, high = BOUNDS[i]
            values[i] = min(high, max(low, values[i] + rng.gauss(0, MUTATION_SCALES[i] * scale)))
    return Genome(tuple(values))


def initialize(config: EvolutionConfig, rng: random.Random) -> list[Genome]:
    return [Genome()] + [mutate(Genome(), rng, 1.0, 2.0) for _ in range(config.population_size - 1)]


def next_generation(
    population: list[Genome], fitness: list[float], config: EvolutionConfig, rng: random.Random
) -> list[Genome]:
    if len(population) != config.population_size or len(fitness) != len(population):
        raise ValueError("Population and fitness must match the configured population size")
    # Randomize equal-fitness order so draws do not systematically favor early indices.
    ranking = list(range(len(population)))
    rng.shuffle(ranking)
    ranking.sort(key=lambda index: fitness[index], reverse=True)
    children = [population[i] for i in ranking[: config.elites]]

    def parent() -> Genome:
        competitors = rng.sample(range(len(population)), config.tournament_size)
        return population[max(competitors, key=lambda i: fitness[i])]

    while len(children) < config.population_size:
        first = parent()
        if config.crossover:
            second = parent()
            first = Genome(
                tuple(
                    a if rng.random() < 0.5 else b
                    for a, b in zip(first.weights, second.weights, strict=True)
                )
            )
        children.append(mutate(first, rng, config.mutation_rate, config.mutation_scale))
    return children


def diversity(population: list[Genome]) -> float:
    """Mean standard deviation normalized by each evolvable weight's allowed range."""
    return sum(
        pstdev(genome.weights[i] for genome in population) / (high - low)
        for i, (low, high) in enumerate(BOUNDS)
        if high > low
    ) / (len(BOUNDS) - 1)
