"""
Sequence generation with Gibbs sampling.

Implementation assumes fixed length of sequences (no inserts, deletions can be sampled if part of alphabet).
"""
from typing import Sequence, Literal, Callable
import numpy as np
import pandas as pd
import torch
from loguru import logger
from evedesign.constants import GAP
from evedesign.model import Generator, ConditionalMutationScorer, Scorer
from evedesign.system import System, Entity, SystemInstance, EntityInstance
from evedesign.types import StatusCallback, CHAIN_COMPONENT_KEY, SCORE_COMPONENT_KEY, EntityPosList
from evedesign.utils import status_progress, ensure_sequence, map_array

ScanOrder = Literal[
    "random", "sequential"
]

InitStrategy = Literal[
    "random", "system"
]

# maps from initial temperature, current sweep and total number of sweeps to current temperature for sweep
TemperatureSchedule = Callable[
    [
        float,  # initial temperature (via generate() parameter)
        int,  # current sweep
        int,  # total number of sweeps
        int,  # current step
        int,  # total number of steps per sweep
    ],
    float   # current temperature
]

_ENTITY = "entity"
_POS = "pos"
_FROM = "from"
_TO = "to"
_SCORE_DIFF = "score_diff"
_TEMPERATURE = "temperature"


class GibbsSampler(Generator):
    """
    Gibbs sampling from linear combination of Scorers.

    Uses inverse sign convention to usual implementations, i.e. high scores
    correspond to high probabilities.

    Notes and design choices:
    1. This sampler does not parallelize individual chains, as this does not play nicely with
     parallelized GPU-based computations (better to batch individual steps), and as this precludes
     interactions between the different chains right away (e.g. library diversity constraints)
     At this point, parallelization / device choice is entirely up to individual scorers so
     each scorer can optimize individually for its bottlenecks (e.g. number of GPUs, available CPUs,
     memory, ...), and to keep the sampler implementation as lean as possible.

     TODO: may decouple/parallelize GPU-based and CPU-based computations with multiprocessing,
      so the CPU-based computations happen in parallel to heavy GPU-based computations

    2. Current implementation assumes for simplicity that all chains have the same length;
     this requirement could be relaxed eventually to each chain can have its own length (however fixed
     along chain!)

    3. Current implementation can only sample entities of same type (protein or nucleotide entities only,
     but not combinations of types, e.g. design protein and nucleotide entities simultaneously)
    """
    citations: list[str] = ["doi:10.1038/s41467-024-49119-x"]
    name: str = "GibbsSampler"

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    def __init__(
        self,
        scorers: Sequence[ConditionalMutationScorer],
        weights: Sequence[float] | None = None,
        num_sweeps: int = 1000,
        init_strategy: InitStrategy = "random",
        scan_order: ScanOrder = "random",
        temperature_schedule: TemperatureSchedule | None = None,
        require_strict_pos : bool = True,
        record_full_chain : bool = True,
        rng: np.random.Generator | None = None,
    ):
        """
        Create new Gibbs sampler

        Parameters
        ----------
        scorers
            Scores to combine into joint score for optimization
        weights
            Weight each scorer will be multiplied with (weight_1 * score_1 + ... + weight_n * score_n).
            If specified, needs to have same length as scorers parameter. If None, all weights will be set to 1.0.
            Use negative weights to invert the semantics of a scorer (e.g. to design against or in favor
            of the occurrence of a certain sequence motif)
        num_sweeps
            Number of Gibbs sweeps across entire system (number of total sampling steps will be
            number of sites x number of sweeps)
        init_strategy
            Create starting samples by randomly sampling designed positions from available alphabet ("random"),
            or use the representations associated with the entities in the system ("system"). For fixed positions,
            will always use representation from system.
        scan_order
            Strategy to determine scan order for each chain. Will either sample randomly (with or without replacement)
            from intersection of positions available from all scorers, or iterate through these sequentially
            as specified by scorers.
        temperature_schedule
            Function mapping from starting temperature, current step and num_steps to temperature for the
            current step. Set to None for constant temperature (specified in generate() function call).
        require_strict_pos
            If True, verify that all scorers model the same set of positions in the system or raise
            a ValueError
        record_full_chain
            If True, record updates performed in each chain for each Gibbs step and attach to instance metadata.
            If False, only final samples at end of each chain will be returned.
        rng
            Numpy random number generator for random sampling. If None, will create own generator inside
            the sampler.
        """
        # must have at least one scorer
        if len(scorers) == 0:
            raise ValueError("Must provide at least one scorer")

        # assume all weights to be 1.0 if weights is None, otherwise check number of weights matches scorers
        if weights is None:
            weights = np.ones(len(scorers))

        if len(scorers) != len(weights):
            raise ValueError("Number of scorers must match number of weights")

        # verify all scorers and store available positions
        for i, scorer in enumerate(scorers):
            if scorer.system is None:
                raise ValueError(
                    f"Scorer {i} does not have an associated system"
                )

            if scorer.system != scorers[0].system:
                raise ValueError(
                    f"Scorer {i} system is not equal to first system (all systems must be identical across scorers)"
                )

            for entity_idx, entity in enumerate(scorers[0].system):
                if bool(entity.deletions) and not scorer.handles_deletions:
                    raise ValueError(
                        f"Scorer {i} does not handle deletions but entity {entity_idx} has deletions = True"
                    )

        if scan_order not in ["random", "sequential"]:
            raise ValueError("Invalid scan order")

        if init_strategy not in ["random", "system"]:
            raise ValueError("Invalid initialization strategy")

        # make a copy of system for easier access
        self._system = scorers[0].system

        self.scorers = scorers
        self.weights = weights
        self.num_sweeps = num_sweeps
        self.temperature_schedule = temperature_schedule
        self.scan_order = scan_order
        self.init_strategy = init_strategy

        # store available positions for each scorer based on constructing a temporary
        # instance (to also retrieve position list from variable-length models,
        # which we will drive in fixed length mode here)
        _mock_instance = SystemInstance([
            EntityInstance(rep=ent.rep) for ent in self._system
        ])
        self.scorer_to_pos = {
            idx: scorer.positions(instance=_mock_instance)
            for idx, scorer in enumerate(self.scorers)
        }

        # determine shared positions by intersection, will only be able to design those
        self.shared_pos = set.intersection(
            *(set(v) for v in self.scorer_to_pos.values())
        )

        # require at least one position to model
        if len(self.shared_pos) == 0:
            raise ValueError(
                "No shared positions between scorers, will not be able to sample from system. " +
                f"Positions per scorer: {self.scorer_to_pos}"
            )

        # if strict position requirement is enabled, check that all scorers model the same positions
        if require_strict_pos:
            all_pos = set.union(
                *(set(v) for v in self.scorer_to_pos.values())
            )

            if len(all_pos) != len(self.shared_pos):
                raise ValueError(
                    "Inconsistent position lists between scorers"
                )

        self.record_full_chain = record_full_chain
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def system(self) -> System | None:
        return self._system

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        # ignore instance specification due to restriction to fixed-length design
        # in the implementation provided by this class
        return sorted(self.shared_pos)

    def _design_params(
        self,
        entities: Sequence[int] | None,
        fixed_pos: EntityPosList | None,
    ) -> tuple[list[int], list[str], list[tuple[int, int]]]:
        """
        Helper method to verify specified entities and fixed positions, and compute
        list of positions in entities that are used for design

        Parameters
        ----------
        entities
            Cf generate() method documentation
        fixed_pos
            Cf generate() method documentation

        Returns
        -------
        Entity type and list of (entity_idx, position_in_entity) tuples that
        are selected for design
        """
        entities_to_type = {
            idx: entity.type for idx, entity in enumerate(self._system)
        }

        # determine and verify entities and positions to design
        if entities is not None:
            entities = ensure_sequence(entities)
            if (set(entities) & set(entities_to_type)) != set(entities):
                raise ValueError(
                    f"Invalid entity selection {entities}, available entities are {list(entities_to_type)}"
                )
        else:
            # otherwise, use all entities
            entities = sorted(entities_to_type)

        # verify all designed entities have an existing representation (so we can assume length)
        for entity in entities:
            if not self._system[entity].defined_sequence():
                raise ValueError(
                    "All designed entities must have a specified representation with nonzero length"
                )

        # verify fixed position specification
        if fixed_pos is not None:
            # verify fixed positions
            if set(fixed_pos) & set(entities) != set(fixed_pos):
                raise ValueError(
                    "Entities specified in fixed_pos must be included in entities to design"
                )

            # turn into flat tuple representation
            fixed_pos = set([
                (entity_idx, pos) for entity_idx, pos_list in fixed_pos.items() for pos in pos_list
            ])

            # verify fixed positions are all available in joint model used for scoring
            invalid_fixed_pos = fixed_pos - self.shared_pos

            if len(invalid_fixed_pos) > 0:
                raise ValueError(f"Invalid fixed positions not available for sampling detected: {invalid_fixed_pos}")
        else:
            fixed_pos = set()

        # remove fixed positions from all available positions, we need at least one to design
        design_pos = sorted(
            (entity, pos) for  (entity, pos) in self.shared_pos
            if (entity, pos) not in fixed_pos and entity in entities
        )

        if len(design_pos) == 0:
            raise ValueError("No positions left to design after removing fixed positions")

        # set up joint alphabet, merging across all entity types that are designed
        alphabet = Entity.merge_alphabet_symbols([
            self._system[entity_idx].alphabet(
                include_gap=bool(self._system[entity_idx].deletions)
            ) for entity_idx in entities
        ])

        return entities, alphabet, design_pos

    def _init_samples(
        self,
        num_designs: int,
        entities: list[int],
        pos_to_design: list[tuple[int, int]],
    ) -> tuple[np.ndarray, dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
        """
        Initialize samples based on system and random sampling

        Parameters
        ----------
        num_designs
            Number of initialized samples to build
        entities
            Indices of designed entities
        pos_to_design
            Variable positions that should be initialized

        Returns
        -------
        Mapping from entity index to samples
        """
        # prepare auxiliary mappings around array
        # (e.g. designed entities [1,3] -> [0, 1] in design matrix)
        entity_to_array_idx = {
            entity_idx: array_idx for array_idx, entity_idx in enumerate(entities)
        }

        # mapping from entity index to length of each entity (use to slice
        # design matrix) - both designed and fixed positions
        entity_to_len = {
            entity_idx: len(self._system[entity_idx].rep) for entity_idx in entities
        }

        # array-based maps for fancy indexing (populated further down)
        entity_to_array_idx_linear = np.zeros((max(entities) + 1), dtype="int")
        entity_to_first_index_linear = np.zeros((max(entities) + 1), dtype="int")

        # initialize empty design matrix for number to num_designs x designed_entity x max_num_positions
        # (longest designed entity determines size of array in last dimension)
        samples = np.empty(
            (num_designs, len(entities), max(entity_to_len.values())),
            dtype="<U1"
        )

        for array_idx, entity_idx in enumerate(entities):
            entity = self._system[entity_idx]
            alphabet = entity.alphabet(
                include_gap=bool(self._system[entity_idx].deletions)
            )
            alphabet_set = set(alphabet)

            # initialize array-based mappings
            entity_to_array_idx_linear[entity_idx] = array_idx
            entity_to_first_index_linear[entity_idx] = entity.first_index

            # randomize full sequence across all chains
            seq_len = len(entity.rep)

            # initialize relevant slice of array for each entity across all chains/samples randomly;
            # in case of using fixed starting sequence (init_strategy == 'system') we will overwrite
            # this random init further down for simplicity
            samples[
                :, array_idx, :seq_len
            ] = self.rng.choice(
                alphabet, size=(num_designs, seq_len), replace=True
            )

            # set fixed positions based on system representation (this will be redundant for init_strategy == "system")
            for pos, symbol in enumerate(entity.rep, start=entity.first_index):
                # set to fixed symbol
                if (entity_idx, pos) not in pos_to_design or self.init_strategy == "system":
                    if symbol not in alphabet_set:
                        raise ValueError(
                            "Fixed position in system representation is not part of alphabet" +
                            f" (entity: {entity_idx}, pos: {pos}, symbol: {symbol}, valid alphabet: {alphabet})"
                        )

                    samples[:, array_idx, pos - entity.first_index] = symbol

        return samples, entity_to_array_idx, entity_to_len, entity_to_array_idx_linear, entity_to_first_index_linear

    @classmethod
    def _init_scan_order(
        cls,
        num_designs: int,
        pos_to_design: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Initialize sequential scan order

        Parameters
        ----------
        num_designs
            Cf. generate()
        pos_to_design
            List of positions that are sampled (determines length of sweep)

        Returns
        -------
        2D array with sweep indices along columns (each chain has its own row)
        """
        # number of positions to design defines length of one sweep
        num_pos = len(pos_to_design)

        # turn into numpy array of tuples and repeat once per chain
        pos_array = np.array(
            pos_to_design, dtype=[(_ENTITY, "int"), (_POS, "int")]
        )

        sequential_order = np.tile(
            pos_array, num_designs
        ).reshape(
            num_designs, num_pos
        )

        return sequential_order

    def _verify_and_update_scores(
        self,
        scores: pd.DataFrame,
        scorer_idx: int,
        alphabet: Sequence[str],
        num_designs: int,
    ):
        assert len(scores) == num_designs, "Invalid length of scoring dataframe"

        for entity_idx, entity in enumerate(self.system):
            try:
                # if we want deletions, make sure the score column is present
                if bool(entity.deletions):
                    entity_rows = scores.loc[pd.IndexSlice[:, entity_idx, :]]
                    if GAP not in entity_rows.columns:
                        raise ValueError(
                            f"Scorer {scorer_idx} did not provide values for gap, but deletions=True"
                        )
                    else:
                        if entity_rows[GAP].isnull().any():
                            raise ValueError(
                                f"Scorer {scorer_idx} returned NA values for entity {entity_idx} where deletions=True"
                            )
                else:
                    # if we do not want deletions, but the column is present, blank it out
                    if GAP in scores.columns:
                        scores.loc[pd.IndexSlice[:, entity_idx, :], GAP] = np.nan
            except KeyError:
                # if entity not found in current score table, we can simply skip it
                continue

        # make sure dataframe has all columns for target alphabet
        # (predictor may return more columns than needed for designed entities, or fewer if an entity leading
        # to extended alphabet is not included in current sampled entity positions)
        return scores.reindex(alphabet, axis=1)

    def _build_or_update_instances(
        self,
        instances: list[SystemInstance] | None,
        num_designs: int,
        entities: Sequence[int],
        samples: np.ndarray,
        entity_to_len: dict[int, int],
        entity_to_array_idx: dict[int, int],
        updated_entities: np.ndarray[int] | None,
    ) -> tuple[list[SystemInstance], dict[int, np.ndarray]]:
        """
        Helper method to initialize or update samples from
        current state of sample array
        """
        # copy instances just to be 100% on safe side so instances cannot be mutated by accident
        samples_copy = {
            entity_idx: samples[
                :, array_idx, :entity_to_len[entity_idx]
            ].copy()
            for entity_idx, array_idx in entity_to_array_idx.items()
        }

        # if instances not yet initialized, do so
        if instances is None:
            # build molecular system instance list; will update in-place with
            # new sequences after each Gibbs step
            instances = [
                SystemInstance([
                    EntityInstance(
                        rep=(
                            samples_copy[entity_idx][design_idx]
                            if entity_idx in entities
                            else self._system[entity_idx].rep
                        )
                    ) for entity_idx, entity in enumerate(self._system)
                ]) for design_idx in range(num_designs)
            ]
        else:
            # otherwise update in place based on latest sample matrix;
            # only update the changed entity instance for efficiency
            for design_idx in range(num_designs):
                # entity updated for the current design
                updated_ent_idx = updated_entities[design_idx]

                # assign updated representation
                instances[design_idx][updated_ent_idx].rep = samples_copy[updated_ent_idx][design_idx]

        return instances, samples_copy

    def _score_designs(self, instances: list[SystemInstance]):
        """
        Compute final design scores as far as possible, modifying instances in place

        If all scorers implement Scorer interface, will attach weighted sum of scores
        to score attribute; individual unweighted scores will be attached to metadata attribute
        in any case.
        """
        # if models support generic score computation, attach composite and individual scores
        # to generated instances
        all_scores_applied = True

        # accumulator for weighted scores
        weighted_score_sum = np.zeros(len(instances))

        # check if we can use system rep to normalize the scores to a shared reference
        try:
            ref_instance = self.system.rep_to_instance()
            instances_with_ref = [ref_instance] + instances
        except ValueError:
            ref_instance = None
            instances_with_ref = instances

        for scorer_idx, (scorer, weight) in enumerate(
            zip(self.scorers, self.weights)
        ):
            if isinstance(scorer, Scorer):
                scorer_name = type(scorer).__name__

                # score instances with current model
                scores_with_ref = scorer.score(instances_with_ref)

                # normalize scores to target if possible, otherwise take as is
                if ref_instance is not None:
                    scores = scores_with_ref[1:] - scores_with_ref[0]
                else:
                    scores = scores_with_ref

                assert len(scores) == len(instances) == len(weighted_score_sum)

                weighted_score_sum += weight * scores

                # attach score component to instance
                for idx, score in enumerate(scores.tolist()):
                    # initialize metadata (careful about any values that may already be present)
                    if instances[idx].metadata is None:
                        instances[idx].metadata = {}

                    if SCORE_COMPONENT_KEY not in instances[idx].metadata:
                        instances[idx].metadata[SCORE_COMPONENT_KEY] = []

                    instances[idx].metadata[SCORE_COMPONENT_KEY].append({
                        "index": scorer_idx,
                        "name": scorer_name,
                        "weight": weight,
                        "score": score,
                        "ref_score": float(scores_with_ref[0]) if ref_instance is not None else None,
                    })
            else:
                # need to write it this slightly ugly way or linter complains about type of scorer in if clause
                all_scores_applied = False

        # attach full score to instances (in-place) if we could compute all components
        if all_scores_applied:
            for idx, score in enumerate(weighted_score_sum):
                instances[idx].score = score

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        # verify/update entity selection and extract positions to design
        entities, alphabet, pos_to_design = self._design_params(
            entities, fixed_pos
        )

        # auxiliary variables for fancy indexing into design array
        alphabet_array = np.array(alphabet)
        alphabet_map = {
            symbol: idx for idx, symbol in enumerate(alphabet_array)
        }
        design_idx_all = np.arange(num_designs)

        # initialize samples for all designed chains, we represent these as numpy arrays
        # internally since we assume entity representations that all have the same length;
        # we will assemble these into strings for passing into individual scorers
        (
            samples, entity_to_array_idx, entity_to_len,
            entity_to_array_idx_linear, entity_to_first_index_linear
        ) = self._init_samples(
            num_designs, entities, pos_to_design
        )

        # initialize full instances to pass to scorers from sample array
        instances, initial_samples_joined = self._build_or_update_instances(
            instances=None,
            num_designs=num_designs,
            entities=entities,
            samples=samples,
            entity_to_len=entity_to_len,
            entity_to_array_idx=entity_to_array_idx,
            updated_entities=None
        )

        # initialize sequential scan order for all positions to be designed (will be reshuffled
        # per sweep in case random scan order is chosen)
        order = self._init_scan_order(num_designs, pos_to_design)

        # number of steps per sweep is the number of positions we want to design
        num_steps = len(pos_to_design)

        # accumulate updates at each Gibbs step so all chains be traced stepwise
        if self.record_full_chain:
            updates = np.empty(
                (self.num_sweeps * num_steps, num_designs),
                dtype=[(_ENTITY, int), (_POS, int), (_TO, "<U1"), (_SCORE_DIFF, float), (_TEMPERATURE, float)]
            )
        else:
            updates = None

        # iterate through sweeps (sweep = one full scan of all designed positions)
        for sweep in range(self.num_sweeps):
            # update status (fraction of sweeps completed)
            status_progress(status_callback, sweep / self.num_sweeps)

            # permute the current sweep scan order if using random order
            # (we always sample without replacement for now for simplicity);
            # note that rng.shuffle is not applicable here since all chains
            # would be shuffled in same way
            if self.scan_order == "random":
                order = self.rng.permuted(order, axis=1)

            assert (
                order.shape[0] == num_designs and order.shape[1] == num_steps
            ), "Scan order array has wrong shape"

            # iterate through all steps for current sweep
            for step in range(num_steps):
                # determine temperature for current sweep/step if we have a temperature schedule in place
                if self.temperature_schedule is not None:
                    step_temp = self.temperature_schedule(
                        temperature, sweep, self.num_sweeps, step, num_steps
                    )
                else:
                    step_temp = temperature

                # extract entity and position to sample for each chain in current step as flat arrays
                step_ent = order[_ENTITY][:, step]
                step_pos = order[_POS][:, step]
                assert len(step_ent) == len(step_pos) == num_designs

                # apply all scorers to current instances and compute weighted sum of scores;
                # we could decouple GPU and CPU-based computations here with multiprocessing
                # eventually to increase speed
                agg_scores = None
                for scorer_idx, (scorer, weight) in enumerate(
                    zip(self.scorers, self.weights)
                ):
                    # compute weighted score
                    s = scorer.score_conditional(
                        instances, step_ent, step_pos
                    ) * weight

                    s = self._verify_and_update_scores(
                        s, scorer_idx=scorer_idx, alphabet=alphabet, num_designs=num_designs,
                    )

                    # ensure nothing bad happened to row index
                    assert np.all(s.index.get_level_values(0) == design_idx_all)
                    assert np.all(s.index.get_level_values(1) == step_ent).all()
                    assert np.all(s.index.get_level_values(2) == step_pos).all()

                    if agg_scores is None:
                        agg_scores = s
                    else:
                        agg_scores = agg_scores.add(s, axis=0)

                    assert (
                        len(agg_scores) == num_designs
                    ), f"Invalid length of aggregated scoring matrix after scorer {scorer_idx}"

                # Gibbs step

                # keep track of token before update for chain tracking, and remap to numeric indices
                current_tokens = samples[
                    design_idx_all,
                    entity_to_array_idx_linear[step_ent],
                    step_pos - entity_to_first_index_linear[step_ent],
                ].copy()

                current_token_idx = map_array(current_tokens, alphabet_map)

                # replace any missing values to exclude from sampling, and scale by temperature for current step;
                # Note we are using an inverted scale here (e.g. not -E/T but E/T where higher E means "better");
                # we go through pytorch here to use the parallelized multinomial implementation which is much
                # more suitable here
                scores_scaled = torch.from_numpy(
                    agg_scores.replace(np.nan, -np.inf).to_numpy(copy=True)
                ) / step_temp

                p = scores_scaled.softmax(dim=-1)

                sampled_token_idx = torch.multinomial(
                    p, num_samples=1
                ).flatten()

                sampled_tokens = alphabet_array[sampled_token_idx.numpy()]

                # compute temperature-scaled score difference to current token
                score_diff = (
                    scores_scaled[design_idx_all, sampled_token_idx] - scores_scaled[design_idx_all, current_token_idx]
                ).numpy()

                # update sample matrix and instances for next step
                assert len(design_idx_all) == len(step_ent) == len(step_pos) == len(sampled_tokens)

                # update design matrix
                samples[
                    design_idx_all,
                    entity_to_array_idx_linear[step_ent],
                    step_pos - entity_to_first_index_linear[step_ent],
                ] = sampled_tokens

                logger.info(
                    f"Gibbs sweep={sweep + 1}/{self.num_sweeps} step={step + 1}/{num_steps} T={step_temp:.3f}"
                )

                # update instances based on new design matrix
                instances, _ = self._build_or_update_instances(
                    instances=instances,
                    num_designs=num_designs,
                    entities=entities,
                    samples=samples,
                    entity_to_len=entity_to_len,
                    entity_to_array_idx=entity_to_array_idx,
                    updated_entities=step_ent
                )

                # record chain information
                if updates is not None:
                    cur_iter = sweep * num_steps + step
                    updates[_ENTITY][cur_iter, :] = step_ent
                    updates[_POS][cur_iter, :] = step_pos
                    updates[_TO][cur_iter, :] = sampled_tokens
                    updates[_SCORE_DIFF][cur_iter, :] = score_diff
                    updates[_TEMPERATURE][cur_iter, :] = step_temp

        # attach metadata to instances
        if updates is not None:
            for design_idx in range(num_designs):
                instances[design_idx].metadata = {
                    CHAIN_COMPONENT_KEY: {
                        "init": {
                            entity_idx: "".join(initial_samples_joined[entity_idx][design_idx])
                            for entity_idx in entities
                        },
                        "chain": updates[:, design_idx].tolist()
                    }
                }

        # score final designs (modifying in place)
        self._score_designs(instances)

        return instances
