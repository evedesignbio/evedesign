import re
from abc import ABC
from typing import Sequence, Any, Self, TypedDict, Literal

import numpy as np

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance, StructureChainMap
from evedesign.types import StatusCallback, Site
from evedesign.utils import status_start, status_done, ensure_sequence


class Motif(TypedDict):
    name: str
    regex: str
    exposure_offsets: list[int]


MOTIFS: dict[str, Motif] = {
    # high risk
    "deamidation_ngs": {"name": "Deamidation (high risk)", "regex": r"N[GS]", "exposure_offsets": [0, 1]},
    "asp_isomerization": {"name": "Asp isomerization", "regex": r"D[DGHST]", "exposure_offsets": [0, 1]},
    "fragmentation_dp": {"name": "Fragmentation (Asp-Pro)", "regex": r"DP", "exposure_offsets": [0, 1]},

    # medium/low risk or context-dependent
    "n-glycosylation": {"name": "N-glycosylation", "regex": r"N[^P][ST]", "exposure_offsets": [0]},
    "deamidation_nahnt": {"name": "Deamidation (moderate risk)", "regex": r"N[AHNT]", "exposure_offsets": [0, 1]},
    "asn_hydrolysis": {"name": "Asn hydrolysis (Asn-Pro)", "regex": r"NP", "exposure_offsets": [0]},
    "fragmentation_ts": {"name": "Fragmentation (Thr-Ser)", "regex": r"TS", "exposure_offsets": [0, 1]},
    "met_oxidation": {"name": "Met oxidation", "regex": r"M", "exposure_offsets": [0]},
    "trp_oxidation": {"name": "Trp oxidation", "regex": r"W", "exposure_offsets": [0]},
    "n-terminal_pyroglutamate": {"name": "N-terminal pyroglutamate", "regex": r"^[QE]", "exposure_offsets": [0]},
    "integrin-binding_motif": {"name": "Integrin-binding motif (RGD)", "regex": r"RGD", "exposure_offsets": [0, 1, 2]},
}


class MotifRestraint(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, ABC):
    """
    Base restraint for rSASA-based sequence liabilities
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
        exposure_cutoff: float = 0.2,
        exposure_source: Literal["system", "instance"] | None = "system",
        exposure_agg: Literal["mean", "median", "min", "max"] = "max",
        missing_exposure: Literal["exclude", "include"] = "include",
    ):
        """
        Create new sequence motif restraint

        Parameters
        ----------
        exposure_cutoff
            Minimal exposure cutoff to consider a motif a hit
        exposure_source
            If "system", approximate from structures on system (may have different side chains),
            if "instance", retrieve from structural model on instance (needs prior transform with
            structure prediction model applied). If None, perform purely sequence-based lookup.
        exposure_agg
            Use this function to aggregate multiple rSASA values per residue
        missing_exposure
            If "include", consider hits for which no 3D structure information is available
        """
        if exposure_agg == "mean":
            self._agg_func = np.mean
        elif exposure_agg == "median":
            self._agg_func = np.median
        elif exposure_agg == "min":
            self._agg_func = np.min
        elif exposure_agg == "max":
            self._agg_func = np.max
        else:
            raise ValueError(f"Invalid aggregation function: {exposure_agg}")


        self.exposure_cutoff = exposure_cutoff
        self.exposure_source = exposure_source
        self.exposure_agg = exposure_agg
        self.missing_exposure = missing_exposure

        self._system = None
        self._sasa_map = None

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

    def _entity_sasa_map(self, id_to_chains: StructureChainMap):
        if id_to_chains is None:
            raise ValueError(f"Missing structure chain for exposure_source={self.exposure_source}")

        residue_sasa_map = {}
        for id_, chains in id_to_chains.items():
            chain_list = ensure_sequence(chains)
            for chain in chain_list:
                chain_map = chain.res_df(
                    sasa=True
                ).set_index(
                    "res_id"
                )["rel_sasa_residue"].to_dict()

                for res_id, sasa in chain_map.items():
                    residue_sasa_map.setdefault(res_id, [])
                    residue_sasa_map[res_id].append(sasa)

        # aggregate to one value per residue
        residue_sasa_map_agg = {
            res_id: self._agg_func(sasas) for res_id, sasas in residue_sasa_map.items()
        }

        return residue_sasa_map_agg

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system

        if self.exposure_source == "system":
            self._sasa_map = {}
            for entity_idx, entity in enumerate(self._system):
                self._sasa_map[entity_idx] = self._entity_sasa_map(
                    entity.structures
                )

        return self


class ExposedMotifRestraint(MotifRestraint):
    """
    Restraint on occurrence of solvent-exposed sequence motifs
    """
    available = True
    name: str = "ExposedMotifRestraint"
    citations: list[str] = []

    def __init__(
        self,
        motifs: list[Motif],
        exposure_cutoff: float = 0.2,
        exposure_source: Literal["system", "instance"] | None = "system",
        exposure_agg: Literal["mean", "median", "min", "max"] = "max",
        missing_exposure: Literal["exclude", "include"] = "include",
    ):
        """
        Create new sequence motif scanner

        Parameters
        ----------
        motifs
            List of sequence-based motifs
        exposure_cutoff
            Minimal exposure cutoff to consider a motif a hit
        exposure_source
            If "system", approximate from structures on system (may have different side chains),
            if "instance", retrieve from structural model on instance (needs prior transform with
            structure prediction model applied). If None, perform purely sequence-based lookup.
        exposure_agg
            Use this function to aggregate multiple rSASA values per residue
        missing_exposure
            If "include", consider hits for which no 3D structure information is available
        """
        super().__init__(
            exposure_cutoff=exposure_cutoff,
            exposure_source=exposure_source,
            exposure_agg=exposure_agg,
            missing_exposure=missing_exposure
        )

        # compile motif regexes
        self.motifs = [
            {
                **motif_spec,
                # lookahead wraps a capturing group to not consume match
                "re_compiled": re.compile(f"(?=({motif_spec['regex']}))")
            } for motif_spec in motifs
        ]

    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score immunogenicity for full sequence (higher = more immunogenic)
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        status_start(status_callback, "Starting scoring")

        scores = np.zeros(len(instances))
        metadata = []

        for instance_idx, instance in enumerate(instances):
            hits: list[Site] = []
            score = 0

            # iterate each entity
            for entity_idx, entity in enumerate(instance):
                if self.exposure_source == "instance":
                    sasa_map = self._entity_sasa_map(entity.models)
                elif self.exposure_source == "system":
                    sasa_map = self._sasa_map[entity_idx]
                else:
                    sasa_map = None

                seq_norm = "".join(entity.normalized_rep())

                # iterate different motifs - all occurrences
                for motif in self.motifs:
                    for m in motif["re_compiled"].finditer(seq_norm):
                        match_str = m.group(1)
                        start = m.start() + self._system[entity_idx].first_index

                        # verify the hit passes the rSASA cutoff
                        if sasa_map is not None:
                            sasas = [
                                sasa_map.get(start + offset) for offset in motif["exposure_offsets"]  # noqa
                            ]

                            passes = [
                                (
                                    (sasa is not None and sasa > self.exposure_cutoff) or
                                    (sasa is None and self.missing_exposure == "include")
                                )
                                for sasa in sasas
                            ]

                            if not all(passes):
                                continue

                        # if we made it here, count the hit
                        score += 1
                        hits.append({
                            "entity": entity_idx,
                            "pos": start,
                            "length": len(match_str),
                            "type": "liability_motif",
                            "subtype": str(motif["name"]),
                            "score": 1.0,
                            "weight": None,
                        })

            metadata.append(hits)
            scores[instance_idx] = score

        status_done(status_callback, "Finished scoring")

        # returns shallow copy
        instances_with_scores =  assign_scores_to_instances(
            instances, scores
        )

        for idx, cur_metadata in enumerate(metadata):
            if instances_with_scores[idx].metadata is None:
                 instances_with_scores[idx].metadata = {}

            if "sites" not in instances_with_scores[idx].metadata:
                instances_with_scores[idx].metadata["sites"] = []

            instances_with_scores[idx].metadata["sites"].append(cur_metadata)

        return instances_with_scores
