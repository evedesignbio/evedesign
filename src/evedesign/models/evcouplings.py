"""
Wrapper classes around the EVcouplings/EVmutation Potts model

This wrapper deliberately keeps only the two stages of the evcouplings pipeline that
are relevant for modelling already-aligned protein or nucleotide families inside
evedesign:

MSA construction is out of scope: the provided System is expected to already carry a MSA

Two concrete engines are provided as subclasses of the abstract EVcouplings base:

EVcouplingsMeanField runs mean-field DCA and has no external dependency

EVcouplingsPLM runs the pseudo-likelihood solver via the plmc binary, a program 
that is not on PyPI (see https://github.com/debbiemarkslab/plmc). Use
EVcouplingsMeanField to avoid the external dependency
"""
import tempfile
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Self, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.sequence import Sequence as EvedesignSequence, Sequences
from evedesign.system import System, SystemInstance, Entity
from evedesign.constants import GAP, MASK
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start

if not hasattr(np, "in1d"):
    # EVcouplings 0.2.1 still calls np.in1d, which was removed in NumPy 2.
    np.in1d = np.isin

try:
    from evcouplings.align.alignment import (
        Alignment,
        ALPHABET_DNA,
        ALPHABET_PROTEIN,
        ALPHABET_RNA,
    )
    from evcouplings.couplings.mapping import Segment, SegmentIndexMapper
    from evcouplings.couplings.mean_field import MeanFieldDCA
    from evcouplings.couplings.model import (
        CouplingsModel,
        _single_mutant_hamiltonians,
        _delta_hamiltonian,
        FULL,
    )
    from evcouplings.couplings.tools import run_plmc
    from evcouplings.utils.system import ExternalToolError
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class EVcouplings(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Abstract base wrapper around EVcouplings/EVmutation

    Holds all logic shared between the inference engines. Subclasses implement 
    _fit() for a specific engine. Instantiate EVcouplingsPLM or EVcouplingsMeanField 
    directly
    """
    available = IMPORT_AVAILABLE
    name: str = "EVcouplings"
    citations: list[str] = [
        # EVmutation
        "10.1038/nbt.3769",
        # EVcouplings
        "10.1093/bioinformatics/bty862"]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    _honors_precomputed_weights: bool = True
    _supported_entity_types: set[str] = {"protein", "dna", "rna"}

    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        theta: float | None = None,
    ):
        """
        Initialise the shared EVcouplings state.

        Parameters
        ----------
        max_gap_fraction
            Alignment columns whose (unweighted) gap frequency strictly exceeds this
            threshold are excluded from the fitted model and from positions()
        theta
            Sequence reweighting identity threshold; sequences with pairwise identity
            >= theta are clustered and down-weighted. Only used when the provided
            Sequences carry no precomputed weights. If theta is None and no precomputed
            weights are available, build() raises a ValueError
        """
        if not self.available:
            raise ImportError(
                "evcouplings package could not be imported. Install w/"
                "pip install evedesign[evcouplings]"
            )

        if not 0.0 < max_gap_fraction <= 1.0:
            raise ValueError("max_gap_fraction must be in (0, 1]")

        self.max_gap_fraction = max_gap_fraction
        self.theta = theta

        self._system: System | None = None

        # parsed Potts model produced by build(); pickles directly with the wrapper
        self.model: CouplingsModel | None = None
        # residue indices of the positions w/ enough coverage
        self._index_list: np.ndarray | None = None
        self._index_mapper: "SegmentIndexMapper | None" = None
        self._segment_id_to_entity: dict[str, int] = {}
        self._model_position_map: list[tuple[int, int]] = []
        self._entity_pos_to_model: dict[tuple[int, int], int] = {}
        self._modelled_entities: list[int] = []
        self._alphabet: str | None = None


    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self.model is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        if len(system) == 0:
            return False, "System must contain at least one entity"

        entity_types = {entity.type for entity in system}
        if not entity_types <= cls._supported_entity_types:
            return False, "Can only handle protein, DNA, or RNA biopolymer entities"

        if len(entity_types) != 1:
            return False, "All modelled entities must have the same biopolymer type"

        aligned_records = []
        for entity_idx, entity in enumerate(system):
            if not entity.defined_sequence():
                return False, f"Entity {entity_idx} must have a defined rep sequence"

            if entity.sequences is None or len(entity.sequences.seqs) == 0:
                return False, (
                    f"Entity {entity_idx} must provide an MSA "
                    "(entity.sequences) for model inference"
                )

            if entity.sequences.type_ != entity.type:
                return False, (
                    f"Entity {entity_idx} sequence type {entity.sequences.type_!r} "
                    f"does not match entity type {entity.type!r}"
                )

            if not entity.sequences.aligned:
                return False, f"Entity {entity_idx} sequences must be aligned"

            try:
                seqs = entity.sequences.to_a3m().remove_inserts().seqs
            except NotImplementedError as exc:
                return False, f"Entity {entity_idx} sequences must be convertible to A3M: {exc}"

            target_rep = "".join(entity.rep)
            bad = [i for i, seq in enumerate(seqs) if len(seq.seq) != len(target_rep)]
            if bad:
                return False, (
                    f"Entity {entity_idx} MSA match-state length does not match "
                    f"target length ({len(target_rep)}) for sequence index {bad[0]}"
                )

            if seqs[0].seq != target_rep:
                return False, (
                    f"Entity {entity_idx} first MSA sequence must equal the "
                    "target/focus sequence"
                )

            aligned_records.append(seqs)

        row_counts = {len(records) for records in aligned_records}
        if len(row_counts) != 1:
            return False, "Paired MSAs must contain the same number of rows for each entity"

        if len(system) > 1:
            row_count = row_counts.pop()
            for row_idx in range(row_count):
                keys = [
                    records[row_idx].key
                    for records in aligned_records
                    if records[row_idx].key is not None
                ]
                if len(keys) > 1 and len(set(keys)) != 1:
                    return False, (
                        "Paired MSA sequence keys must match across entities "
                        f"at row {row_idx}"
                    )

        return True, ""


    @classmethod
    def _alphabet_for_entity(cls, entity: Entity) -> str:
        if entity.type == "protein":
            return ALPHABET_PROTEIN
        if entity.type == "dna":
            return ALPHABET_DNA
        if entity.type == "rna":
            return ALPHABET_RNA
        raise ValueError(f"Unsupported EVcouplings entity type: {entity.type!r}")

    @classmethod
    def _segment_type_for_entity(cls, entity: Entity) -> str:
        if entity.type == "protein":
            return "aa"
        if entity.type in {"dna", "rna"}:
            return entity.type
        raise ValueError(f"Unsupported EVcouplings entity type: {entity.type!r}")

    @staticmethod
    def _segment_id(entity_idx: int) -> str:
        if entity_idx < 26:
            return chr(ord("A") + entity_idx)
        return f"E{entity_idx}"

    @staticmethod
    def _sequence_id(seq: EvedesignSequence, fallback: str) -> str:
        if seq.id_ is None:
            return fallback
        return str(seq.id_).split()[0]

    def _entity_match_records(
        self,
        entity: Entity,
        entity_idx: int,
    ) -> tuple[list[EvedesignSequence], list[str]]:
        target_rep = "".join(entity.rep)
        length = len(target_rep)

        seqs = entity.sequences.to_a3m().remove_inserts().seqs
        match_seqs = [s.seq for s in seqs]

        bad = [i for i, m in enumerate(match_seqs) if len(m) != length]
        if bad:
            raise ValueError(
                f"Entity {entity_idx} MSA match-state length does not match "
                f"target length ({length}) for {len(bad)} sequence(s), "
                f"e.g. sequence index {bad[0]}"
            )

        if match_seqs[0] != target_rep:
            raise ValueError(
                f"Entity {entity_idx} first MSA sequence (match states) must "
                "equal the target/focus sequence. EVcouplings requires the "
                "target sequence as the first alignment record"
            )

        return seqs, match_seqs

    def _build_alignment(
        self,
        system: System,
    ) -> tuple["Alignment", "Sequences", str, int]:
        """
        Build an evcouplings Alignment from system MSAs and lower-case
        high-gap columns so they are excluded from the fit.

        Multi-entity systems are represented as paired MSAs concatenated in entity
        order. The alignment is numbered from a fixed first index of 1; mapping back
        to evedesign entity positions is handled with SegmentIndexMapper.

        Returns the possibly modified Alignment, the concatenated sequence record
        used for fallback weighting, the focus id, and the total target length.
        """
        alphabet = self._alphabet_for_entity(system[0])
        seqs_by_entity = []
        match_seqs_by_entity = []
        segments = []
        self._segment_id_to_entity = {}
        self._modelled_entities = list(range(len(system)))

        for entity_idx, entity in enumerate(system):
            seqs, match_seqs = self._entity_match_records(entity, entity_idx)
            seqs_by_entity.append(seqs)
            match_seqs_by_entity.append(match_seqs)

            segment_id = self._segment_id(entity_idx)
            self._segment_id_to_entity[segment_id] = entity_idx
            segments.append(
                Segment(
                    self._segment_type_for_entity(entity),
                    entity.id or f"entity_{entity_idx}",
                    int(entity.first_index),
                    int(entity.first_index) + len(entity.rep) - 1,
                    segment_id=segment_id,
                )
            )

        row_counts = {len(records) for records in seqs_by_entity}
        if len(row_counts) != 1:
            raise ValueError("Paired MSAs must contain the same number of rows for each entity")
        row_count = row_counts.pop()

        for row_idx in range(row_count):
            keys = [
                records[row_idx].key
                for records in seqs_by_entity
                if records[row_idx].key is not None
            ]
            if len(keys) > 1 and len(set(keys)) != 1:
                raise ValueError(
                    "Paired MSA sequence keys must match across entities "
                    f"at row {row_idx}"
                )

        match_seqs = [
            "".join(entity_match_seqs[row_idx] for entity_match_seqs in match_seqs_by_entity)
            for row_idx in range(row_count)
        ]
        total_length = len(match_seqs[0])

        focus_id = self._sequence_id(seqs_by_entity[0][0], "focus")
        ids = (
            [f"{focus_id}/1-{total_length}"]
            + [
                self._sequence_id(seqs_by_entity[0][row_idx], f"seq{row_idx}")
                for row_idx in range(1, row_count)
            ]
        )

        matrix = np.array([list(m) for m in match_seqs])
        alignment = Alignment(matrix, sequence_ids=ids, alphabet=alphabet)

        # unweighted per-column gap frequency, exclude columns above the threshold
        gap_freq = alignment.count(GAP, axis="pos", normalize=True)
        excluded = gap_freq > self.max_gap_fraction

        if excluded.all():
            raise ValueError(
                "All positions exceed max_gap_fraction, what are you aligning...?"
            )

        if excluded.any():
            # lower-casing turns these into fake insert columns, which get stripped later
            alignment = alignment.lowercase_columns(np.where(excluded)[0])

        self._index_mapper = SegmentIndexMapper(True, 1, *segments)

        concatenated_sequences = Sequences(
            seqs=[
                EvedesignSequence(
                    seq=seq,
                    id=ids[row_idx].split("/")[0],
                    key=seqs_by_entity[0][row_idx].key,
                    type=system[0].type,
                )
                for row_idx, seq in enumerate(match_seqs)
            ],
            aligned=True,
            type=system[0].type,
            format="a3m",
        )

        return alignment, concatenated_sequences, focus_id, total_length


    @abstractmethod
    def _fit(
        self,
        alignment: "Alignment",
        focus_id: str,
        num_model_positions: int,
        weights: Sequence[float] | None = None,
    ) -> "CouplingsModel":
        """
        Fit the engine-specific Potts model and return the parsed CouplingsModel.
        """
        raise NotImplementedError

    def _sequence_weights(
        self,
        system: System,
        concatenated_sequences: Sequences,
    ) -> Sequence[float] | None:
        if len(system) == 1:
            weights = system[0].sequences.weights
        else:
            weights_by_entity = [entity.sequences.weights for entity in system]
            provided_weights = [weights for weights in weights_by_entity if weights is not None]
            if len(provided_weights) == 0:
                weights = None
            elif len(provided_weights) != len(system):
                raise ValueError(
                    "Precomputed weights for paired MSAs must be provided for all "
                    "entities, or for none of them"
                )
            elif not all(np.allclose(provided_weights[0], weights) for weights in provided_weights[1:]):
                raise ValueError("Precomputed weights for paired MSAs must match across entities")
            else:
                weights = provided_weights[0]

        return weights

    def _entity_position_for_model_pos(self, model_pos: int) -> tuple[int, int]:
        segment_id, pos = self._index_mapper.to_target(int(model_pos))
        return self._segment_id_to_entity[segment_id], int(pos)

    def _set_model_position_maps(self) -> None:
        self._model_position_map = [
            self._entity_position_for_model_pos(int(pos)) for pos in self._index_list
        ]
        self._entity_pos_to_model = {
            entity_pos: model_idx
            for model_idx, entity_pos in enumerate(self._model_position_map)
        }

    def _model_subsequence(self, instance: SystemInstance) -> np.ndarray:
        symbols = []
        for entity_idx, pos in self._model_position_map:
            entity = self.system[entity_idx]
            offset = int(pos) - int(entity.first_index)
            symbols.append(str(np.asarray(instance[entity_idx].rep)[offset]))
        return np.asarray(symbols, dtype="U1")


    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Fitting EVcouplings model")

        self._system = system

        # reset any previous fit
        self.model = None
        self._index_list = None
        self._index_mapper = None
        self._model_position_map = []
        self._entity_pos_to_model = {}
        self._alphabet = self._alphabet_for_entity(system[0])

        alignment, concatenated_sequences, focus_id, _ = self._build_alignment(system)

        # number of modelled positions = match columns
        num_model_positions = int(
            np.array([c.isupper() and c != GAP for c in alignment[0]]).sum()
        )

        weights = self._sequence_weights(system, concatenated_sequences)
        needs_theta = weights is None or not self._honors_precomputed_weights
        if needs_theta and self.theta is None:
            raise ValueError(
                "Sequence reweighting requires either precomputed weights on the "
                "provided Sequences or an explicit theta"
            )

        if weights is None and self._honors_precomputed_weights:
            weights = concatenated_sequences.compute_weights(theta=self.theta).weights

        self.model = self._fit(alignment, focus_id, num_model_positions, weights)
        self._index_list = np.asarray(self.model.index_list, dtype=int)
        self._set_model_position_maps()

        status_done(status_callback, "EVcouplings model finished fitting")

        return self


    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        """
        Return the modelled positions. Positions excluded due to high gap
        content are not part of the fitted model and are therefore not returned
        """
        self.ready_or_raise()
        return list(self._model_position_map)


    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score full sequences by their statistical energy

        Only the modelled (non-excluded) positions contribute, excluded positions are
        ignored
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        subseqs = []
        for instance in instances:
            subseq = "".join(self._model_subsequence(instance))
            subseqs.append(subseq)

        # column 0 is the total Hamiltonian (J_ij + h_i sub-sums in columns 1, 2)
        hamiltonians = self.model.hamiltonians(subseqs)[:, 0]

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(
            instances, np.asarray(hamiltonians, dtype=float)
        )

    def _instance_background(
        self,
        instance: SystemInstance,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map an instance onto the model-internal integer background
        over the modelled (non-excluded) positions.

        Returns
        -------
        bg
            Integer array of length L with the modelled-position symbols (0 for invalid)
        subseq
            Char array of length L with the raw modelled-position symbols (for ref labelling)
        invalid
            Boolean array of length L, True where the symbol is not in the model alphabet
        """
        model = self.model
        subseq = self._model_subsequence(instance)

        bg = np.zeros(model.L, dtype=int)
        invalid = np.zeros(model.L, dtype=bool)
        for i, sym in enumerate(subseq):
            sym = str(sym)
            if sym in model.alphabet_map:
                bg[i] = model.alphabet_map[sym]
            else:
                invalid[i] = True

        return bg, subseq, invalid


    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None,
    ) -> pd.DataFrame:
        """
        Compute all single substitutions to an instance via the model single-mutant matrix.

        This wraps _single_mutant_hamiltonians kernel (same as CouplingsModel.single_mut_mat) 
        but evaluated relative to the given instance rather than the model query seq

        Scores are delta-Hamiltonians (mutant - instance)
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        if entity is not None and entity not in self._modelled_entities:
            raise ValueError(f"EVcouplings does not model entity {entity}")

        if positions is not None:
            if entity is None:
                raise ValueError(
                    "Parameter entity must be explicitly specified if using parameter positions"
                )
            self.valid_positions(positions, instance, entity, raise_invalid=True)

        status_start(status_callback, "Computing single mutation scan")

        model = self.model

        # map the instance onto the integer background over modelled positions
        bg, subseq, invalid = self._instance_background(instance)

        # full L x num_symbols delta-Hamiltonian matrix relative to this background
        delta = _single_mutant_hamiltonians(bg, model.J_ij, model.h_i)[:, :, FULL]

        pos_filter = set(positions) if positions is not None else None

        # columns of delta are in model alphabet order
        model_alphabet = list(model.alphabet)

        rows = []
        index_tuples = []
        for i in range(model.L):
            evc_entity, evc_pos = self._model_position_map[i]
            if entity is not None and evc_entity != entity:
                continue
            if pos_filter is not None and evc_pos not in pos_filter:
                continue
            rows.append(delta[i])
            index_tuples.append((evc_entity, evc_pos, str(subseq[i])))

        df = pd.DataFrame(
            rows,
            columns=model_alphabet,
            index=pd.MultiIndex.from_tuples(
                index_tuples, names=["entity", "pos", "ref"]
            ),
        )

        # reorder/select columns into the evedesign alphabet
        alphabet_entities = [entity] if entity is not None else self._modelled_entities
        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[entity_idx].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            ) for entity_idx in alphabet_entities
        ])
        df = df.reindex(merged_alphabet, axis=1)

        status_done(status_callback, "Single mutation scan complete")

        return df


    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None,
    ) -> pd.DataFrame:
        """
        Compute conditional substitution scores P(x_i | x_\\i) for one position per instance

        Conditional energy of symbol A at position i given the rest of
        the sequence fixed is h_i[i, A] + sum_{j != i} J_ij[i, j, A, x_j]
        """
        self.ready_or_raise()

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length"
            )

        self._validate_instances(instances)

        # validate entity/position per instance
        for instance, entity, pos in zip(instances, entities, positions):
            self.valid_positions(
                positions=[pos], instance=instance, entities=[entity], raise_invalid=True
            )

        status_start(status_callback, "Computing conditional scores")

        model = self.model
        num_symbols = model.num_symbols

        model_alphabet = list(model.alphabet)

        rows = []
        index_tuples = []
        for inst_idx, (instance, entity, pos) in enumerate(zip(instances, entities, positions)):
            bg, _, invalid = self._instance_background(instance)
            mi = self._entity_pos_to_model[(int(entity), int(pos))]

            bg_invalid = invalid.copy()
            bg_invalid[mi] = False

            logits = np.array([
                _delta_hamiltonian(
                    np.array([mi]), np.array([symbol]), bg, model.J_ij, model.h_i
                )[FULL]
                for symbol in range(num_symbols)
            ])

            rows.append(logits)
            index_tuples.append((inst_idx, int(entity), int(pos)))

        df = pd.DataFrame(
            rows,
            columns=model_alphabet,
            index=pd.MultiIndex.from_tuples(
                index_tuples, names=["instance", "entity", "pos"]
            ),
        )

        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[entity_idx].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            ) for entity_idx in set(entities)
        ])
        df = df.reindex(merged_alphabet, axis=1)

        status_done(status_callback, "Conditional scoring complete")

        return df


class EVcouplingsMeanField(EVcouplings):
    """
    EVcouplings model fitted with mean-field DCA
    """
    name: str = "EVcouplingsMeanField"
    _honors_precomputed_weights: bool = False

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        theta: float | None = None,
        pseudo_count: float = 0.5,
    ):
        """
        Instantiate a mean-field DCA EVcouplings model.

        Parameters
        ----------
        max_gap_fraction
            Alignment columns whose (unweighted) gap frequency strictly exceeds this
            threshold are excluded from the fitted model and from positions()
        theta
            Sequence reweighting identity threshold. Sequences with pairwise identity
            >= theta are clustered and down-weighted
        pseudo_count
            Pseudo-count for frequency regularization
        """
        super().__init__(max_gap_fraction=max_gap_fraction, theta=theta)
        self.pseudo_count = pseudo_count

    def _fit(
        self,
        alignment: "Alignment",
        focus_id: str,
        num_model_positions: int,
        weights: Sequence[float] | None = None,
    ) -> "CouplingsModel":
        if weights is not None:
            logger.warning(
                "EVcouplingsMeanField does not support precomputed sequence weights. "
                "The provided weights will be ignored and weights will be recomputed "
                "from theta."
            )
        fit_kwargs = {"pseudo_count": self.pseudo_count}
        if self.theta is not None:
            fit_kwargs["theta"] = self.theta
        return MeanFieldDCA(alignment).fit(**fit_kwargs)


class EVcouplingsPLM(EVcouplings):
    """
    EVcouplings model fitted with the plmc pseudo-likelihood solver

    Requires the external plmc binary (https://github.com/debbiemarkslab/plmc)
    """
    name: str = "EVcouplingsPLM"
    # plmc can be compiled w/ multi-core fitting
    supports_cpu_parallel: bool = True

    @property
    def handles_deletions(self) -> bool:
        return not self.ignore_gaps

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        theta: float | None = None,
        lambda_h: float = 0.01,
        lambda_J: float = 0.01,
        lambda_J_times_Lq: bool = True,
        lambda_group: float | None = None,
        scale_clusters: float | None = None,
        iterations: int | None = 100,
        ignore_gaps: bool = False,
        independent_model: bool = False,
        plmc_binary: str | PathLike = "plmc",
        cpu: int | Literal["max"] | None = None,
    ):
        """
        Instantiate a plmc (pseudo-likelihood) EVcouplings model.

        Parameters
        ----------
        max_gap_fraction
            Alignment columns whose (unweighted) gap frequency strictly exceeds this
            threshold are excluded from the fitted model and from positions()
        theta
            Sequence reweighting identity threshold. Sequences with pairwise identity
            >= theta are clustered and down-weighted. Used as a fallback when the
            provided Sequences carry no precomputed weights
        lambda_h
            L2 regularisation strength on fields h_i
        lambda_J
            L2 regularisation strength on couplings J_ij. If lambda_J_times_Lq is True,
            this base value is scaled by (num_symbols - 1)*(L - 1) (as in standard
            evcouplings)
        lambda_J_times_Lq
            Scale lambda_J by the number of states and modelled positions
        lambda_group
            Group L1 regularisation strength on couplings (None = plmc default)
        scale_clusters
            Scale weights of sequence clusters by this value (None = plmc default)
        iterations
            Maximum L-BFGS iterations. Defaults to 100 (the standard EVcouplings
            cap?) None = plmc default
        ignore_gaps
            If True, exclude gaps from parameter inference. Note that this also implies
            gaps cannot be scored--the default (False) keeps gap as a model symbol
        independent_model
            If True, fit an independent-site (no couplings) model
        plmc_binary
            Path to / name of the plmc binary
        cpu
            Number of cores for plmc (requires OpenMP-compiled plmc) - or "max"
        """
        super().__init__(max_gap_fraction=max_gap_fraction, theta=theta)
        # most of these params are just defaults inherited from EVcouplings, leaving here
        # in case users want to mess with them for some reason
        self.lambda_h = lambda_h
        self.lambda_J = lambda_J
        self.lambda_J_times_Lq = lambda_J_times_Lq
        self.lambda_group = lambda_group
        self.scale_clusters = scale_clusters
        self.iterations = iterations
        self.ignore_gaps = ignore_gaps
        self.independent_model = independent_model
        self.plmc_binary = plmc_binary
        self.cpu = cpu

    def _fit(
        self,
        alignment: "Alignment",
        focus_id: str,
        num_model_positions: int,
        weights: Sequence[float] | None = None,
    ) -> "CouplingsModel":
        # scale lambda_J as in the standard couplings protocol
        lambda_J = self.lambda_J
        alphabet = self._alphabet or ALPHABET_PROTEIN
        if self.lambda_J_times_Lq:
            num_symbols = len(alphabet) - (1 if self.ignore_gaps else 0)
            lambda_J = lambda_J * (num_symbols - 1) * (num_model_positions - 1)

        iterations = 1 if self.independent_model else self.iterations

        # writing temp files will be unavoidable if we want to limit
        # the amount of code we copy from couplings
        # although writing the alignment to file is super annoying
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            aln_file = tmp / "alignment.fasta"
            couplings_file = tmp / "couplings.txt"
            param_file = tmp / "model.params"

            with open(aln_file, "w") as f:
                alignment.write(f, format="fasta")

            # prefer precomputed sequence weights 
            weight_file = None
            if weights is not None:
                weight_file = tmp / "weights.txt"
                with open(weight_file, "w") as f:
                    f.write("\n".join(str(w) for w in weights) + "\n")

            try:
                run_plmc(
                    str(aln_file),
                    str(couplings_file),
                    param_file=str(param_file),
                    focus_seq=focus_id,
                    alphabet=alphabet,
                    theta=self.theta if weights is None else None,
                    scale=self.scale_clusters,
                    ignore_gaps=self.ignore_gaps,
                    iterations=iterations,
                    lambda_h=self.lambda_h,
                    lambda_J=lambda_J,
                    lambda_g=self.lambda_group,
                    cpu=self.cpu,
                    binary=str(self.plmc_binary),
                    weight_file=str(weight_file) if weight_file is not None else None,
                )
            except ExternalToolError as e:
                if isinstance(e.__cause__, OSError):
                    raise FileNotFoundError(
                        f"plmc binary not found or not executable: {self.plmc_binary!r}. "
                        "Pass the path to the compiled plmc executable (ex. "
                        "/path/to/plmc/bin/plmc)"
                    ) from e
                raise

            model = CouplingsModel(str(param_file), file_format="plmc_v2")

        if self.independent_model:
            model = model.to_independent_model()

        return model
