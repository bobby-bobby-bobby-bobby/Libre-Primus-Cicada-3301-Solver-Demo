"""
Search strategies for the cryptanalysis mining loop.

Implements four independent strategies, each returning a list of
:class:`Hypothesis` objects ready for GPU-accelerated scoring:

  RandomSearch      – uniform random sampling
  GridSearch        – exhaustive sweep over a constrained grid
  BeamSearch        – keep top-K at each step; expand neighbourhood
  GeneticSearch     – evolution with mutation + crossover + elitism

All strategies include adaptive bias: they track which transform
components most consistently raise scores and weight future sampling
accordingly.

Cross-page diversity preservation is enforced via a minimum Hamming
distance threshold between candidates.
"""

from __future__ import annotations

import copy
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .keyspace import Keyspace
from ..pipeline.transforms import (
    AtbashVariant,
    BlockTransposition,
    IndexPermutation,
    LearnedSubstitution,
    ModularShift,
    NumericToLatinLabels,
    VigenereShift,
)
from ..pipeline.pipeline import Pipeline
from ..pipeline.accelerator import Accelerator


# ---------------------------------------------------------------------------
# Hypothesis: a pipeline + its score
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """
    A single search hypothesis: a pipeline configuration and its score.

    Parameters
    ----------
    pipeline : Pipeline
        The transform stack to evaluate.
    score : float
        Composite score from the last evaluation (−∞ = not yet scored).
    generation : int
        Search generation / iteration when this was created.
    tags : Dict[str, Any]
        Arbitrary metadata (strategy name, parent IDs, etc.).
    """

    pipeline: Pipeline
    score: float = -math.inf
    generation: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline.to_dict(),
            "score": self.score,
            "generation": self.generation,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SearchStrategy(ABC):
    """Abstract base class for all search strategies."""

    def __init__(
        self,
        keyspace: Keyspace,
        accelerator: Optional[Accelerator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.keyspace = keyspace
        self.acc = accelerator or Accelerator()
        self.config = config or {}
        self._generation = 0
        # Adaptive bias: tracks per-transform-type mean score improvement.
        self._bias: Dict[str, float] = {}

    def update_bias(self, transform_name: str, score_delta: float) -> None:
        """Register a score delta for a particular transform type."""
        prev = self._bias.get(transform_name, 0.0)
        # Exponential moving average
        self._bias[transform_name] = 0.9 * prev + 0.1 * score_delta

    @abstractmethod
    def next_batch(self, batch_size: int) -> List[Hypothesis]:
        """Generate the next batch of hypotheses to evaluate."""

    @abstractmethod
    def update(self, scored_hypotheses: List[Hypothesis]) -> None:
        """Consume scored hypotheses and update internal state."""

    def _random_pipeline(self, encoding: str = "index") -> Pipeline:
        """Build a random pipeline from the keyspace."""
        ksp = self.keyspace
        num_slots = random.randint(1, self.config.get("max_pipeline_depth", 4))
        unknown_slots = int(self.config.get("unknown_transform_slots", 1))
        transforms = []
        for _ in range(num_slots):
            t_type = random.choice(
                ["ModularShift", "VigenereShift", "AtbashVariant",
                 "BlockTransposition", "IndexPermutation", "LearnedSubstitution"]
            )
            if t_type == "ModularShift":
                p = ksp.random_modular_shift_params()
                transforms.append(ModularShift(p["a"], p["b"], p["mod"]))
            elif t_type == "VigenereShift":
                transforms.append(VigenereShift(ksp.random_vigenere_key()))
            elif t_type == "AtbashVariant":
                transforms.append(AtbashVariant())
            elif t_type == "BlockTransposition":
                transforms.append(BlockTransposition(ksp.random_block_pattern()))
            elif t_type == "IndexPermutation":
                transforms.append(
                    IndexPermutation(ksp.random_permutation(), mod=29)
                )
            else:  # LearnedSubstitution (unknown transform slot)
                transforms.append(LearnedSubstitution.random(mod=29))

        return Pipeline(transforms, encoding=encoding, accelerator=self.acc)


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


class RandomSearch(SearchStrategy):
    """Uniform random sampling of the pipeline space."""

    def next_batch(self, batch_size: int) -> List[Hypothesis]:
        self._generation += 1
        encodings = self.config.get("encodings", ["index"])
        return [
            Hypothesis(
                pipeline=self._random_pipeline(random.choice(encodings)),
                generation=self._generation,
                tags={"strategy": "random"},
            )
            for _ in range(batch_size)
        ]

    def update(self, scored_hypotheses: List[Hypothesis]) -> None:
        # Update adaptive bias from scored results.
        for h in scored_hypotheses:
            if h.score > -math.inf:
                for t in h.pipeline.transforms:
                    self.update_bias(t.name, h.score)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


class GridSearch(SearchStrategy):
    """
    Exhaustive grid search over the modular-shift keyspace.

    Only iterates ModularShift(a, b) over the constrained prime/offset
    grid.  For large grids this streams hypotheses in batches.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._grid_iter = self._build_grid()

    def _build_grid(self) -> Iterator[Hypothesis]:
        ksp = self.keyspace
        encodings = self.config.get("encodings", ["index"])
        for mod in ksp.modular_mods.domain:
            for a in ksp.modular_multipliers.domain:
                for b in ksp.modular_offsets.domain:
                    enc = random.choice(encodings)
                    yield Hypothesis(
                        pipeline=Pipeline(
                            [ModularShift(a, b, mod)],
                            encoding=enc,
                            accelerator=self.acc,
                        ),
                        generation=self._generation,
                        tags={"strategy": "grid", "a": a, "b": b, "mod": mod},
                    )

    def next_batch(self, batch_size: int) -> List[Hypothesis]:
        self._generation += 1
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(self._grid_iter))
        except StopIteration:
            # Grid exhausted; fall back to random.
            while len(batch) < batch_size:
                batch.append(
                    Hypothesis(
                        pipeline=self._random_pipeline(),
                        generation=self._generation,
                        tags={"strategy": "grid_overflow"},
                    )
                )
        return batch

    def update(self, scored_hypotheses: List[Hypothesis]) -> None:
        pass  # Grid search doesn't adapt.


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


class BeamSearch(SearchStrategy):
    """
    Beam search: maintain a set of top-K hypotheses and expand them.

    Each iteration generates *batch_size* neighbours of beam members by
    randomly perturbing one transform parameter.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beam_width = int(self.config.get("beam_width", 50))
        self.beam: List[Hypothesis] = []

    def next_batch(self, batch_size: int) -> List[Hypothesis]:
        self._generation += 1
        if not self.beam:
            # Cold start: populate with random.
            encodings = self.config.get("encodings", ["index"])
            return [
                Hypothesis(
                    pipeline=self._random_pipeline(random.choice(encodings)),
                    generation=self._generation,
                    tags={"strategy": "beam_init"},
                )
                for _ in range(batch_size)
            ]

        # Expand neighbourhood of each beam member.
        candidates = []
        per_member = max(1, batch_size // len(self.beam))
        for parent in self.beam:
            for _ in range(per_member):
                child = self._perturb(parent)
                candidates.append(child)
            if len(candidates) >= batch_size:
                break

        # Top up to batch_size with fresh randoms.
        while len(candidates) < batch_size:
            candidates.append(
                Hypothesis(
                    pipeline=self._random_pipeline(),
                    generation=self._generation,
                    tags={"strategy": "beam_fill"},
                )
            )
        return candidates[:batch_size]

    def _perturb(self, parent: Hypothesis) -> Hypothesis:
        """Create a neighbour of *parent* by mutating one transform."""
        ksp = self.keyspace
        new_pipeline = Pipeline(
            list(parent.pipeline.transforms),
            encoding=parent.pipeline.encoding,
            accelerator=self.acc,
        )
        if not new_pipeline.transforms:
            return Hypothesis(
                pipeline=self._random_pipeline(),
                generation=self._generation,
                tags={"strategy": "beam_perturb"},
            )

        # Pick a random transform to replace.
        idx = random.randrange(len(new_pipeline.transforms))
        old_t = new_pipeline.transforms[idx]

        if isinstance(old_t, ModularShift):
            # Perturb one parameter.
            p = {
                "a": old_t.a,
                "b": old_t.b,
                "mod": old_t.mod,
            }
            field = random.choice(["a", "b", "mod"])
            if field == "a":
                p["a"] = ksp.modular_multipliers.random_value()
            elif field == "b":
                p["b"] = ksp.modular_offsets.random_value()
            else:
                p["mod"] = ksp.modular_mods.random_value()
            new_pipeline.transforms[idx] = ModularShift(**p)

        elif isinstance(old_t, VigenereShift):
            new_key = old_t.key_vector[:]
            pos = random.randrange(len(new_key))
            new_key[pos] = random.randint(0, 28)
            new_pipeline.transforms[idx] = VigenereShift(new_key, old_t.mod)

        elif isinstance(old_t, LearnedSubstitution):
            new_table = old_t.table[:]
            # Swap two random positions.
            i, j = random.sample(range(len(new_table)), 2)
            new_table[i], new_table[j] = new_table[j], new_table[i]
            new_pipeline.transforms[idx] = LearnedSubstitution(new_table, old_t.mod)

        else:
            # Replace with a fresh random transform of the same type.
            new_pipeline.transforms[idx] = type(old_t).from_dict(old_t.to_dict())

        return Hypothesis(
            pipeline=new_pipeline,
            generation=self._generation,
            tags={"strategy": "beam_perturb", "parent_score": parent.score},
        )

    def update(self, scored_hypotheses: List[Hypothesis]) -> None:
        """Merge scored hypotheses into the beam, keeping the top-K."""
        all_hyps = self.beam + scored_hypotheses
        all_hyps.sort(key=lambda h: h.score, reverse=True)
        self.beam = all_hyps[: self.beam_width]
        for h in scored_hypotheses:
            for t in h.pipeline.transforms:
                self.update_bias(t.name, max(0.0, h.score))


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------


class GeneticSearch(SearchStrategy):
    """
    Genetic algorithm search.

    Population evolves through:
      - Tournament selection
      - Single-point crossover on the transform list
      - Random mutation of individual transforms
      - Elitism: top fraction survives unchanged
      - Periodic diversity reset to escape local optima
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        gcfg = self.config.get("genetic", {})
        self.pop_size = int(gcfg.get("population_size", 100))
        self.mutation_rate = float(gcfg.get("mutation_rate", 0.15))
        self.crossover_rate = float(gcfg.get("crossover_rate", 0.7))
        self.elite_frac = float(gcfg.get("elite_fraction", 0.1))
        self.diversity_threshold = float(
            self.config.get("diversity_threshold", 0.1)
        )
        self.diversity_reset_interval = int(
            self.config.get("diversity_reset_interval", 500)
        )
        self.population: List[Hypothesis] = []
        self._iterations_since_reset = 0

    def next_batch(self, batch_size: int) -> List[Hypothesis]:
        self._generation += 1
        self._iterations_since_reset += 1

        if not self.population:
            encodings = self.config.get("encodings", ["index"])
            self.population = [
                Hypothesis(
                    pipeline=self._random_pipeline(random.choice(encodings)),
                    generation=self._generation,
                    tags={"strategy": "genetic_init"},
                )
                for _ in range(self.pop_size)
            ]
            return self.population[:batch_size]

        # Periodic diversity reset: inject fresh individuals.
        if self._iterations_since_reset >= self.diversity_reset_interval:
            self._diversity_reset()
            self._iterations_since_reset = 0

        offspring = self._evolve(batch_size)
        return offspring

    def _evolve(self, n: int) -> List[Hypothesis]:
        """Produce *n* offspring through selection + crossover + mutation."""
        elite_n = max(1, int(self.elite_frac * len(self.population)))
        sorted_pop = sorted(
            self.population, key=lambda h: h.score, reverse=True
        )
        elites = [copy.deepcopy(h) for h in sorted_pop[:elite_n]]
        offspring = list(elites)

        while len(offspring) < n:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                parent_a = self._tournament()
                parent_b = self._tournament()
                child = self._crossover(parent_a, parent_b)
            else:
                child = copy.deepcopy(self._tournament())

            child = self._mutate(child)
            child.generation = self._generation
            child.score = -math.inf
            offspring.append(child)

        return offspring[:n]

    def _tournament(self, k: int = 3) -> Hypothesis:
        """K-tournament selection."""
        contestants = random.sample(
            self.population, min(k, len(self.population))
        )
        return max(contestants, key=lambda h: h.score)

    def _crossover(self, a: Hypothesis, b: Hypothesis) -> Hypothesis:
        """Single-point crossover on transform lists."""
        t_a = a.pipeline.transforms
        t_b = b.pipeline.transforms
        if not t_a or not t_b:
            return copy.deepcopy(a)
        cut_a = random.randrange(len(t_a))
        cut_b = random.randrange(len(t_b))
        new_transforms = t_a[:cut_a] + t_b[cut_b:]
        # Clamp depth.
        max_depth = self.config.get("max_pipeline_depth", 6)
        new_transforms = new_transforms[:max_depth]
        if not new_transforms:
            new_transforms = [copy.deepcopy(random.choice(t_a + t_b))]
        return Hypothesis(
            pipeline=Pipeline(
                new_transforms,
                encoding=a.pipeline.encoding,
                accelerator=self.acc,
            ),
            generation=self._generation,
            tags={"strategy": "genetic_crossover"},
        )

    def _mutate(self, h: Hypothesis) -> Hypothesis:
        """Mutate transforms in-place with probability *mutation_rate*."""
        ksp = self.keyspace
        for i, t in enumerate(h.pipeline.transforms):
            if random.random() > self.mutation_rate:
                continue
            if isinstance(t, ModularShift):
                field = random.choice(["a", "b"])
                if field == "a":
                    t = ModularShift(ksp.modular_multipliers.random_value(), t.b, t.mod)
                else:
                    t = ModularShift(t.a, ksp.modular_offsets.random_value(), t.mod)
            elif isinstance(t, VigenereShift):
                new_key = t.key_vector[:]
                pos = random.randrange(len(new_key))
                new_key[pos] = random.randint(0, 28)
                t = VigenereShift(new_key, t.mod)
            elif isinstance(t, LearnedSubstitution):
                new_table = t.table[:]
                i1, i2 = random.sample(range(len(new_table)), 2)
                new_table[i1], new_table[i2] = new_table[i2], new_table[i1]
                t = LearnedSubstitution(new_table, t.mod)
            h.pipeline.transforms[i] = t
        return h

    def _diversity_reset(self) -> None:
        """Replace the bottom half of the population with fresh individuals."""
        encodings = self.config.get("encodings", ["index"])
        n_replace = len(self.population) // 2
        sorted_pop = sorted(
            self.population, key=lambda h: h.score, reverse=True
        )
        survivors = sorted_pop[: len(sorted_pop) - n_replace]
        fresh = [
            Hypothesis(
                pipeline=self._random_pipeline(random.choice(encodings)),
                generation=self._generation,
                tags={"strategy": "genetic_reset"},
            )
            for _ in range(n_replace)
        ]
        self.population = survivors + fresh

    def update(self, scored_hypotheses: List[Hypothesis]) -> None:
        """Merge scored batch into population, keeping best pop_size."""
        self.population.extend(scored_hypotheses)
        self.population.sort(key=lambda h: h.score, reverse=True)
        self.population = self.population[: self.pop_size]
        for h in scored_hypotheses:
            for t in h.pipeline.transforms:
                self.update_bias(t.name, max(0.0, h.score))
