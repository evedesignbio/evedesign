from abc import ABC, abstractmethod
from typing import Any, Sequence
from evedesign.system import System, SystemInstance


class Analyzer(ABC):
    """
    Abstract base class for analysis methods stacked on top of generation/scoring/transform pipelines
    (these may use all information in system and instances, and also enhance the system with extra
    metadata)
    """
    @abstractmethod
    def analyze(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        data: Any,
        entity: int | None = None,
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Perform analysis on system/instances, returning a shallow copy with updated
        metadata fields for any analysis results

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which analysis should be performed
        data
            Arbitrary additional data specific to analysis that is not a descriptive property of system
        entity
            Index of entity that analysis should be applied to (if None, use all entities)

        Returns
        -------
        Tuple containing results from analysis in
        (i) System
        (ii) SystemInstances
        """
        pass
