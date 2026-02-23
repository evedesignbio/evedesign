"""
Restraining generated sequence distance to reference sequences
"""
from typing import Self, Sequence

import numpy as np

from evedesign.model import BaseModel, Scorer, ConditionalMutationScorer, MutationScorer
from evedesign.system import Entity, System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import str_to_np_char_view, map_array

EntityToReferenceSeqs = dict[int, list[str]]


class LinearSeqDistRestraint(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Linear distance restraint between generated sequences and a set of reference sequences.
    For simplicity, assumes all compared sequences (i.e. on a per-entity basis) have the same
    length and are aligned.

    # TODO: not yet optimized for performance (when using large sequence sets, bring in numba)
    # TODO: constructor param for number of CPUs to use (when parallelizing)?

    Note on sign convention:
    Scoring methods return distance (or delta of distance) to reference sequences; i.e. a positive
    weight on this restraint during sampling will enforce sequences to become more dissimilar
    to the reference sequence(s); a negative weight will enforce designs to become more similar.
    """
    available = True
    name: str = "LinearSeqDistRestraint"
    citations: list[str] = ["doi:10.1038/s41467-024-49119-x"]

    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True
    handles_insertions: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        exclude_gaps_from_distance: bool = True,
    ):
        """
        Create new linear sequence distance restrain

        Parameters
        ----------
        exclude_gaps_from_distance
            If True, do not count positions where either of two compared sequences
            has a gap symbol
        """
        self._system = None

        # will hold mapped and verified reference sequences for comparison
        self._ref_seqs = None
        self._alphabets = None
        self._alphabet_mapping = None
        self._ref_seqs_mapped = None

        self.exclude_gaps_from_distance = exclude_gaps_from_distance

    @property
    def ready(self):
        return (
            self.system is not None and
            self._ref_seqs is not None and
            self._alphabets is not None and
            self._alphabet_mapping is not None and
            self._ref_seqs_mapped is not None
        )

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(
        cls,
        system: System,
        data: EntityToReferenceSeqs,
    ) -> tuple[bool, str]:
        # core requirements: we need at least one restrained biopolymer sequence,
        # and length of all sequences per entity must agree with reference sequence

        # determine valid sequence entities that could be restrained
        valid_entities_to_len = {
            entity_idx: len(entity.rep) for entity_idx, entity in enumerate(system) if entity.defined_sequence()
        }

        if len(data) == 0:
            return False, "Must specify at least one entity with reference sequences"

        # iterate through all specified reference sequences on each of the entities and verify
        for entity_idx, ref_seqs in data.items():
            if entity_idx not in valid_entities_to_len:
                return False, (
                    f"Restraint specified on entity {entity_idx} but valid "
                    f"entities with defined representation are {list(valid_entities_to_len.keys())}"
                )

            cur_entity_length = valid_entities_to_len[entity_idx]
            invalid = [seq for seq in ref_seqs if len(seq) != cur_entity_length]
            if len(invalid) > 0:
                return False, f"Reference sequence(s) do not have correct length of {cur_entity_length}: {invalid}"

        return True, ""

    def build(
        self,
        system: System,
        data: EntityToReferenceSeqs,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        # verify if we can model the system
        self.can_model_or_raise(system, data)

        # store system with this instance
        self._system = system

        # store reference sequences for comparison (already checked validity via can_model() above)
        self._ref_seqs = {
            entity_idx: str_to_np_char_view(
                entity_ref_seqs
            ) for entity_idx, entity_ref_seqs in data.items()
        }

        # store alphabets for each entity
        self._alphabets = {
            entity_idx: entity.alphabet(include_gap=True)
            for entity_idx, entity in enumerate(self._system)
            if entity.defined_sequence()
        }

        # merge alphabet symbols across all constrained entities and create mapping to numerical indices
        merged_symbols = Entity.merge_alphabet_symbols([
            alphabet for entity_idx, alphabet in self._alphabets.items() if entity_idx in self._ref_seqs
        ])

        self._alphabet_mapping = {
            symbol: idx for idx, symbol in enumerate(merged_symbols)
        }

        # map to numerical indices
        try:
            self._ref_seqs_mapped = {
                entity_idx: map_array(
                    entity_ref_seqs, self._alphabet_mapping
                )
                for entity_idx, entity_ref_seqs in self._ref_seqs.items()
            }
        except KeyError as e:
            raise ValueError("Invalid symbol in reference sequences") from e

        return self

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # validate instance sequences with specific requirements for this class
        self._validate_instances(instances)

        # for accumulating distances across all entities and instances
        dists = np.zeros(len(instances), dtype="int")

        # loop through target entities
        for entity_idx, cur_ref_seqs in self._ref_seqs.items():
            # extract sequences for current entity from instances as numpy array
            # (do not use np.array(list) as this is way slower)
            x = np.array(
                [inst[entity_idx].rep for inst in instances]
            )

            # iterate through references one by one;
            # TODO: optimize with numba or scipy cdist if large reference sequence sets
            #  (e.g. comparing to MSA) become relevant
            for ref in cur_ref_seqs:
                # silence type warnings by wrapping in array()
                diff = np.array(ref != x)

                if self.exclude_gaps_from_distance:
                    diff = diff & (ref != "-") & (x != "-")

                dists += diff.sum(axis=1)

        assert len(dists) == len(instances)
        return dists

    # Note: following implementation breaks on latest pandas versions due to direct assignment to .values;
    #  but can now be replaced with the ConditionalScorer mixin for simplicity
    #
    # def score_conditional(
    #     self,
    #     instances: Sequence[SystemInstance],
    #     entities: Sequence[int],
    #     positions: Sequence[int],
    #     status_callback: StatusCallback | None = None
    # ) -> pd.DataFrame:
    #     self.ready_or_raise()
    #
    #     if not len(instances) == len(entities) == len(positions):
    #         raise ValueError(
    #             "Sequences for instances, entities and positions must all have same length"
    #         )
    #
    #     # validate instance sequences with specific requirements for this class
    #     self._validate_instances(instances)
    #
    #     # validate entities / positions
    #     self.valid_positions(
    #         positions=positions, entities=entities, raise_invalid=True
    #     )
    #
    #     # initialize table of instance/entity/pos triplets and add current
    #     # instance symbol for later comparison to restraint sequences
    #     entity_to_first_index = {
    #         entity_idx: entity.first_index for entity_idx, entity in enumerate(self._system)
    #     }
    #
    #     # prepare empty scoring matrix
    #     res = pd.DataFrame({
    #         "instance": np.arange(len(instances)),
    #         "entity": entities,
    #         "pos": positions,
    #     }).set_index(
    #         ["instance", "entity", "pos"]
    #     ).reindex(
    #         self._alphabet_mapping, axis=1, fill_value=np.nan
    #     )
    #
    #     # determine instance symbol for each row, this allows to reuse the scores for single_mutation_scan()
    #     inst_symbol = np.array([
    #         instance[entity_idx].rep[
    #             pos - entity_to_first_index[entity_idx]
    #         ]
    #         for (instance, entity_idx, pos) in zip(instances, entities, positions)
    #     ])
    #
    #     inst_symbol_idx = map_array(inst_symbol, self._alphabet_mapping)
    #     gap_idx = self._alphabet_mapping[GAP]
    #
    #     # compare sequences entity by entity and accumulate updated subgroup dataframes
    #     groups = res.groupby("entity", sort=False)
    #
    #     for entity_idx, all_row_idx in groups.indices.items():
    #         entity_idx = int(entity_idx)  # noqa
    #
    #         # get current alphabet for initializing relevant entries in array to 0, keep all others as nan
    #         alphabet = self._alphabets[entity_idx]
    #         for symbol in alphabet:
    #             res.values[
    #                 all_row_idx, self._alphabet_mapping[symbol]
    #             ] = 0
    #
    #         # keep neutral scores to positions in entities that are restrained
    #         if entity_idx not in self._ref_seqs:
    #             continue
    #
    #         # map requested position for each instance
    #         cur_positions = (
    #             res.iloc[all_row_idx].index.get_level_values("pos").values - entity_to_first_index[entity_idx]
    #         )
    #
    #         # compare to all reference sequences for current entity
    #         # (use version mapped to indices for direct fancy indexing into numpy array)
    #         cur_ref_seqs = self._ref_seqs_mapped[entity_idx]
    #
    #         # iterate through individual reference sequences
    #         # TODO: may need to make this more efficient for larger sets of restraint sequences
    #         #  (e.g. comparing against entire MSA)
    #         for i in range(len(cur_ref_seqs)):
    #             # extract symbols at different positions in this reference sequence
    #             ref_symbols = cur_ref_seqs[i, cur_positions]
    #
    #             # treat gap special case
    #             if self.exclude_gaps_from_distance:
    #                 # determine if reference has a gap at specified positions
    #                 ref_not_gap = ref_symbols != gap_idx
    #
    #                 # only update positions where reference is not a gap, as we can change
    #                 # symbols in reference gap positions arbitrarily without changing restraint;
    #                 # if instance is gap and reference is not, handle just like regular symbol exchanges
    #                 res.values[
    #                     all_row_idx[ref_not_gap], ref_symbols[ref_not_gap]
    #                 ] -= 1
    #             else:
    #                 # otherwise treat all symbols equally
    #                 res.values[all_row_idx, ref_symbols] -= 1
    #
    #     # retrieve value for instance symbol across all rows, then subtract from full matrix to normalize
    #     inst_symbol_val = res.values[np.arange(len(res)), inst_symbol_idx]
    #     res.values[:, :] -= inst_symbol_val[:, None]
    #
    #     assert len(res) == len(instances)
    #     return res

