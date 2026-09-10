from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any, Self

from evedesign.model import (
    BaseModel,
    MutationScorer,
    Scorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import model_param_context, status_done, status_start

try:
    from promb import init_db  # type: ignore[import-not-found,import-untyped]

    IMPORT_AVAILABLE = True
except ImportError:
    init_db = None
    IMPORT_AVAILABLE = False


Aggregation = str | Callable[[Sequence[float]], float]
AGGREGATIONS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": fmean,
    "min": min,
    "max": max,
}


class OASisHumanness(BaseModel, Scorer, MutationScorer):
    """Score proteins by the fraction of 9-mers found in human OAS."""

    available = IMPORT_AVAILABLE
    name: str = "OASisHumanness"
    citations: list[str] = ["doi:10.1080/19420862.2021.2020203"]

    requires_target: bool = True
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_cpu_parallel: bool = False
    supports_gpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        aggregation: Aggregation = "weighted_mean",
    ):
        """
        Parameters
        ----------
        aggregation
            Combines scores across proteins. ``"weighted_mean"`` weights each
            protein by its number of 9-mers. ``"mean"``, ``"min"``, and ``"max"``
            combine protein scores directly. A callable receives the protein
            scores and returns one value.
        """
        if not self.available:
            raise ImportError(
                "promb could not be imported. Install evedesign with the "
                "'promb' optional dependency."
            )

        if not callable(aggregation) and aggregation not in (
            "weighted_mean",
            *AGGREGATIONS,
        ):
            raise ValueError(
                "aggregation must be 'weighted_mean', 'mean', 'min', 'max', "
                "or a callable"
            )

        self.aggregation = aggregation
        self._system: System | None = None
        self._db: Any = None

    @property
    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if not system:
            return False, "System must contain at least one protein entity"

        for entity in system:
            if entity.type != "protein":
                return False, "Can only handle protein entities"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    def _load_db(self) -> None:
        if self._db is None:
            self._db = init_db("human-oas", verbose=False)

    def _delete_db(self) -> None:
        self._db = None

    def _aggregate(
        self,
        entity_scores: Sequence[float],
        peptide_counts: Sequence[int],
    ) -> float:
        if callable(self.aggregation):
            return float(self.aggregation(entity_scores))

        if self.aggregation == "weighted_mean":
            total_peptides = sum(peptide_counts)
            if total_peptides == 0:
                return 0.0
            weighted_score = sum(
                score * count for score, count in zip(entity_scores, peptide_counts)
            )
            return weighted_score / total_peptides

        return float(AGGREGATIONS[self.aggregation](entity_scores))

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        self.ready_or_raise()
        self._validate_instances(instances)

        if not instances:
            return []

        status_start(status_callback, "Scoring human peptide content")

        scores: list[float] = []
        with model_param_context(self._load_db, self._delete_db, keep_model=False):
            for instance in instances:
                entity_scores = []
                peptide_counts = []
                for entity in instance:
                    sequence = "".join(entity.normalized_rep())
                    peptide_count = max(len(sequence) - 8, 0)
                    entity_scores.append(
                        self._db.compute_peptide_content(sequence)
                        if peptide_count
                        else 0.0
                    )
                    peptide_counts.append(peptide_count)

                scores.append(self._aggregate(entity_scores, peptide_counts))

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, scores)
