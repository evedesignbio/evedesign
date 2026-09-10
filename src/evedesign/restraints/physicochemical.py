import abc
from abc import ABC
from typing import Sequence, Any, Self

import numpy as np
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback, Site
from evedesign.utils import status_start, status_done


class RangeRestraint(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, ABC):
    """
    Restraint on maintaining a defined isoelectric point range
    """
    available = True

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        exponent: int = 1,
    ):
        if lower_bound > upper_bound:
            raise ValueError("Lower bound must be <= upper bound")

        self._system = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.exponent = exponent

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        # for now only handle single-protein systems but can trivially extend to multiple proteins
        for entity_idx, entity in enumerate(system):
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

    @abc.abstractmethod
    def _calculate_property(self, entity_seqs: Sequence[str]) -> float:
        pass

    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        self.ready_or_raise()
        self._validate_instances(instances)

        status_start(status_callback, "Starting scoring")

        scores = np.zeros(len(instances))
        for instance_idx, instance in enumerate(instances):
            # compile all entity sequences in normalized form
            entity_seqs = [
                "".join(entity.normalized_rep()) for entity_idx, entity in enumerate(instance)
            ]

            # compute property for current instance
            prop = self._calculate_property(entity_seqs)

            # compute deviation from specified bounds and scale by exponent
            if prop < self.lower_bound:
                score = (self.lower_bound - prop) ** self.exponent
            elif prop > self.upper_bound:
                score = (prop - self.upper_bound) ** self.exponent
            else:
                score = 0

            scores[instance_idx] = score

        status_done(status_callback, "Finished scoring")

        # returns cores
        return assign_scores_to_instances(
            instances, scores
        )


class IsoelectricPointRestraint(RangeRestraint):
    name: str = "IsoelectricPointRestraint"
    citations: list[str] = ["doi.org/10.1093/bioinformatics/btp163"]

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        exponent: int = 1,
    ):
        """
        Restraint on isoelectric point of protein entities in system

        Parameters
        ----------
        lower_bound
            Lower acceptable bound of pI
        upper_bound
            Upper acceptable bound of pI
        exponent
            Exponent applied to difference to closest bound, if outside of [lower_bound, upper_bound]
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            exponent=exponent,
        )
        self._system = None

    def _complex_pi(
        self,
        sequences: Sequence[str],
        ph: float = 7.775,
        min_: float = 4.05,  # minimum achievable pI
        max_: float = 12.0,  # maximum achievable pI
    ):
        chain_ips = [IsoelectricPoint(seq) for seq in sequences]

        def complex_charge(ph: float):
            return sum(ip.charge_at_pH(ph) for ip in chain_ips)

        charge = complex_charge(ph)
        if max_ - min_ > 0.0001:
            if charge > 0:
                min_ = ph
            else:
                max_ = ph
            return self._complex_pi(sequences, (min_ + max_) / 2, min_, max_)

        return ph

    def _calculate_property(self, entity_seqs: Sequence[str]) -> float:
        return self._complex_pi(entity_seqs)


class MolecularWeightRestraint(RangeRestraint):
    name: str = "MolecularWeightRestraint"
    citations: list[str] = ["doi.org/10.1093/bioinformatics/btp163"]

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        exponent: int = 1,
    ):
        """
        Restraint on molecular weight of protein entities in system

        Parameters
        ----------
        lower_bound
            Lower acceptable bound of mW (in Da)
        upper_bound
            Upper acceptable bound of mW (in Da)
        exponent
            Exponent applied to difference to closest bound, if outside of [lower_bound, upper_bound]
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            exponent=exponent,
        )
        self._system = None

    def _calculate_property(self, entity_seqs: Sequence[str]) -> float:
        return sum(
            ProteinAnalysis(seq).molecular_weight() for seq in entity_seqs
        )