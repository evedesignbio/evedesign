from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, Sequence, Any
import numpy as np
import pandas as pd
from evedesign.dataset import LabeledInstanceDataset
from evedesign.system import System, SystemInstance, Entity, EntityInstance, Mutant, Mutation
from evedesign.types import StatusCallback, ModelStats, BioPolymers, EntityPosList


class _Core(ABC):
    """
    Minimal core functionality required by any modular class used for sequence design.

    Note: this class should not be implemented directly but rather through one of its
     more specific subclasses like Generator
    """
    @property
    @abstractmethod
    # plain-text name of method
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    # citation strings for method
    def citations(self) -> list[str]:
        pass

    @property
    @abstractmethod
    # must return system modelled by the current instance, or None if not yet defined
    def system(self) -> System | None:
        pass

    @property
    @abstractmethod
    # whether model needs a specified target sequence in system
    def requires_target(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model requires fixed-length sequences
    # (implies insertions cannot be modeled, and deletions need to be modelled by GAP symbol)
    def requires_fixed_length(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model is able to model deletions (may be possible for models
    # with required fixed length depending on alphabet)
    def handles_deletions(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model is able to model insertions (implies requires_fixed_length to be False)
    def handles_insertions(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *must* be run on GPU
    def requires_gpu(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *can* be run on GPU (implies this is an advantage, otherwise set this to False)
    def supports_gpu(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *can* be parallelized on CPU (implies this is an advantage, otherwise set this to False)
    def supports_cpu_parallel(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *can* be parallelized on GPU (implies this is an advantage, otherwise set this to False)
    def supports_gpu_parallel(self) -> bool:
        pass

    def positions(
        self,
        instance: SystemInstance | None,
    ) -> list[tuple[int, int]]:
        """
        Return list of all available modelled positions per entity *instance* that are explicitly
        captured by the model

        This method is a default implementation that returns all biopolymer positions in different
        entities; if a method is not able to handle all positions, it must overwrite this method.

        Notes:
        1. Positions that are not modelled (e.g. excluded positions for EVmutation) should not
         be returned by this method

        2. For fixed-length models, entity instance positions will by definition be the same
         as entity representation positions. These models can opt to set the instance argument to
         a default value of None.

        3. Models able to handle insertions should *not* return first_index - 1 coding; the ability
         to handle this position for insertions is implied by self.handles_insertions

        4. Returned positions should be ordered in ascending order
         by i) entity index, ii) position index in entity

        Returns
        -------
        List of position tuples (entity_idx, position)
        """
        if self.system is None:
            raise ValueError(
                "No system present on model"
            )

        if instance is None:
            source = self.system
        else:
            source = instance

        return [
            (entity_idx, pos)
            for entity_idx, entity in enumerate(self.system)
            for pos, _ in enumerate(
                source[entity_idx].rep,
                start=entity.first_index
            )
            if (
                entity.type in BioPolymers and
                entity.first_index is not None and
                source[entity_idx].rep is not None
            )
        ]

    def valid_positions(
        self,
        positions: Sequence[int],
        instance: SystemInstance | None = None,
        entities: int | Sequence[int] = 0,
        raise_invalid: bool = False,
    ) -> list[tuple[int, int]]:
        """
        Helper method to verify if a list of positions for a given entity instance in system is valid
        (via positions()).

        Parameters
        ----------
        positions
            List of unique positions to check
        instance
            System instance to verify positions against. Can be set to None for
            fixed-length models, otherwise will raise a ValueError if not specified.
        entities
            List of entities corresponding to each position (if sequence);
            or can be fixed to one entity which will be applied to all positions (if int)
        raise_invalid
            If invalid position contained in input list, raise a ValueError

        Returns
        -------
        List of valid position tuples
        """
        if instance is None and not self.requires_fixed_length:
            raise ValueError(
                "Need to specify instance since not a fixed-length model"
            )

        if isinstance(entities, int):
            given_pos = [
                (entities, pos) for pos in positions
            ]
        else:
            if len(positions) != len(entities):
                raise ValueError("Length of entities and positions must agree")

            given_pos = [
                (entity, pos) for entity, pos in zip(entities, positions)
            ]

        available_pos = set(
            self.positions(instance=instance)
        )

        valid_pos = [
            entity_pos for entity_pos in given_pos if entity_pos in available_pos
        ]

        if raise_invalid and len(valid_pos) != len(positions):
            raise ValueError(
                f"Invalid positions given, valid options are {sorted(available_pos)}"
                f" but given are {sorted(given_pos)}"
            )

        return valid_pos

    def _validate_instances(
        self,
        instances: Sequence[SystemInstance],
        raise_invalid: bool = True,
    ) -> bool:
        valid = [
            self.system.valid_instance(
                instance,
                validate_reps=True,
                require_reps=True,
                fixed_length=self.requires_fixed_length,
                allow_deletions=self.handles_deletions,
                raise_invalid=raise_invalid,
            ) for instance in instances
        ]

        return all(valid)

class Generator(_Core):
    """
    Interface implemented by classes that can generate new samples
    (e.g. generative models or samplers on top of scoring models)

    TODO: check whether it makes sense to add more designs parameters shared
     across most methods here, or whether it is better to add additional parameters
     to individual methods (with default arguments) based on the functionality
     of each method, or whether these should all go into System specification
    """
    @abstractmethod
    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None
    ) -> list[SystemInstance]:
        """
        Sample new sequences from generative model

        Note: Implementation should raise ValueError if any of the specified design options are not supported

        Note: Method must always return at least num_designs elements in the output list,
         but may also return more designs than requested e.g. if beneficial due to batch size

        Note: Any position specification numbering (e.g. of fixed positions with fixed_pos) must match
         sequence numbering of *system* entity representation (with corresponding value of first_index,
         by default 1; i.e. one-based indexing of positions!), cannot use the entity instance index here as it may
         vary in variable-length designs. Implementations for designing variable lengths are responsible for
         correctly mapping positions to instance positions internally, making use of insert/deletion coding
         in the respective instance (see EntityInstance documentation for more detail)

        Parameters
        ----------
        num_designs
            Number of designs to generate
        entities
            Indices of entities in system that should be designed during generation (others will be kept fixed).
            If None, will attempt to design all entities.
        fixed_pos
            Mapping from entity index to positions that should be fixed during design. Any entity referenced
            in the mapping must be also included in the "entities" parameter.
        temperature
            Sampling temperature (higher values generate more diversity)
        status_callback
            Callback function to track computation status

        Returns
        -------
        Designed instances (sequences/structures) of system (guaranteed to contain at least num_design instances)
        """
        pass


class Scorer(_Core):
    """
    Interface implemented by classes that can score (e.g. density/log likelihood/arbitrary unit score) for
    entire designs (scalar value per system instance).
    """
    @abstractmethod
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        """
        Score different realizations of the modelled system (e.g. different sequences
        generated from a model)

        Note:
        1. Scores returned by this function should be raw logits comparable between all instances
         scored in the same call. Scores between multiple calls do not have to be comparable (user
         is responsible for including a reference instance for normalization in these cases)

        2. Implementation is responsible for verifying if the provided instances can be modelled,
         and to extract all information needed (e.g. deletions marked by GAP for models handling deletions,
         insertions marked with lowercase symbols for models handling insertions, etc.)

        Parameters
        ----------
        instances
            Designs to score with model
        status_callback
            Callback function to track computation status

        Returns
        -------
        Vector of scores (one per instance, in same order as instances input parameter)
        """
        pass


class ConditionalMutationScorer(_Core, ABC):
    """
    Interface implemented by classes that can compute conditional probabilities
    P(x_i | x_\\i) to be used e.g. for Gibbs sampling even if not
    able to compute full P(x_1, ..., x_n)

    This class also serves as a mixin to provide a default implementation using the Scorer scorer() method
    *if* available (which may not be always the case if the method only allows relative scoring to a target);
    note that this default implementation should be overwritten with custom implementations that are either
    more accurate or more efficient if possible by exploiting the special structure of the problem
    (e.g. computing all substitutions in one forward pass of the model, or conditioning the prediction on all
    known positions in order-invariant autoregressive models; check the EVmutation2 implementation as a
    reference example)
    """
    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Compute scores for all substitutions in a single position
        across a batch of sequences (single position can differ between instances), e.g.
        for Gibbs sampling-based generation of multiple designs in parallel.

        Note:
        1. This function allows to exploit the fact that often single mutations for
         one position can be computed more efficiently than arbitrary full sequences
         (e.g. in Potts model hamiltonian). If no customized implementation is available,
         this method should still wrap around score() for applications like Gibbs sampling.

        2. Logits are not relative to any particular sequence (e.g. "wildtype"), but
         meant to be interpreted relative to each other (i.e. should be treated as raw logits)
         across possible symbols *per* sampled instance/entity/position combination

        3. Return dataframe row index is over instance index/entity index/position triplets;
         columns index over different symbols (amino acids etc.). Guaranteed to have same length as instance,
         entities and positions. Rows must be in the same order as input instance/entity/position triplets.
         Columns must be in same order as returned by Entity.alphabet() (or union thereof if multiple types
         of entities in system), missing predictions must be encoded by np.nan

        4. Optional insertion handling: Models able to provide scores for insertions should include these
         by requesting an alphabet including insertion symbols: Entity.alphabet(..., include_inserts=True).
         Insertions are implied to occur immediately after the position in the dataframe index, an insertion
         before the first sequence position should be coded by pos=entity.first_index - 1
         (with all uppercase symbols), with all uppercase/non-insert symbol values set to NaN.

        5. Methods returning predictions across entities with more than one alphabet should use
         Entity.merge_alphabet_symbols() to determine the mixed alphabet/column order. The alphabet of each
         dataframe row is implied by the type of the respective entity, all symbols from other alphabets
         not relevant for current row should be set to NaN)

        Parameters
        ----------
        instances
            Target instances/sequences for which scores should be calculated. Must
            have same length as entities and positions.
        entities
            List of entity indexes which selects exactly one entity per instance for scoring.
            Must have same length as instances and positions.
        positions
            List of positions which selects exactly one position per instance/entity pair.
            Must have same length as instances and entities.
        status_callback
            Callback function to track computation status

        Returns
        -------
        Dataframe with raw logit scores (seq x symbols);
        """
        if not isinstance(self, Scorer):
            raise NotImplementedError(
                "Model does not implemented Scorer interface, cannot use this default implementation"
            )

        if hasattr(self, "ready_or_raise"):
            self.ready_or_raise()

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length"
            )

        # validate instance sequences with specific requirements for this class
        self._validate_instances(instances)

        # validate entity / position per instance as we cannot assume fixed length here
        for instance, entity, pos in zip(instances, entities, positions):
            self.valid_positions(
                positions=[pos], instance=instance, entities=[entity], raise_invalid=True
            )

        # accumulate mutated instances for scoring, and accumulate index information for easy dataframe construction
        all_instances = []
        all_mutants = []
        instance_indices = []
        all_refs = []

        for instance_idx, (instance, entity, position) in enumerate(zip(instances, entities, positions)):
            # get alphabet for current entity in instance
            alphabet = self.system[entity].alphabet(
                include_inserts=self.handles_insertions, include_gap=self.handles_deletions
            )

            # assemble all single mutations for the current triplet (subject to entity type)
            pos_norm = position - self.system[entity].first_index
            cur_ref = str(instance[entity].rep[pos_norm])
            all_refs.append(cur_ref)

            mutants = [
                [
                    Mutation(
                        entity=entity, pos=position, ref=cur_ref, to=to
                    )
                ] for to in alphabet
            ]

            # create mutated instances (this includes self-mutant for normalization as well)
            all_mutants += mutants
            all_instances += self.system.mutate(instance, mutants)
            instance_indices += [instance_idx] * len(mutants)

        # pass through scoring method (note we could also use score_mutants per instance above
        # if MutationScorer interface implemented but for now default to this simpler solution)
        scores = self.score(all_instances, status_callback)

        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[entity_idx].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            ) for entity_idx in set(entities)
        ])

        # assemble into dataframe
        series = pd.Series(scores)
        series.index = pd.MultiIndex.from_tuples(
            ((instance_idx, mutant[0].entity, mutant[0].pos, mutant[0].to)
            for (instance_idx, mutant)
            in zip(instance_indices, all_mutants)),
            names=["instance", "entity", "pos", "to"]
        )

        df = (series.unstack(
            level="to"
        ).reindex(
            merged_alphabet, axis=1)
        )

        # apply row-wise normalization to reference (even if not strictly needed according to specification)
        assert len(all_refs) == len(df)
        ref_scores = np.array([
            row[ref_symbol] for ref_symbol, (idx, row) in zip(all_refs, df.iterrows())
        ])

        return df.sub(ref_scores, axis=0)


class MutationScorer(_Core, ABC):
    """
    Interface for methods that allow to score mutations to an instance

    This class also serves as a mixin to provide a default implementation using the Scorer scorer() method
    *if* available (which may not be always the case if the method only allows relative scoring to a target);
    note that this default implementation should be overwritten with custom implementations that are either
    more accurate or more efficient if possible by exploiting the special structure of the problem
    (e.g. computing all substitutions in one forward pass of the model, or conditioning the prediction on all
    known positions in order-invariant autoregressive models; check the EVmutation2 implementation as a
    reference example)
    """
    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Compute all single substitutions to one particular instance (aka "single mutation scan")
        batching across different positions. This is different to score_conditional() which
        batches substitutions to exactly one single position across many different instances.

        Note:
        1. Mutation logits should be *relative* to the given instance (like a log-odds ratio),
         so that self-substitutions are assigned are score of 0, beneficial substitutions are score > 0,
         and damaging substitutions a score < 0. This differs from score_conditional, where there is
         no notion of a "wildtype" sequence to compute relative scores to.

        2. The implementation of this function can draw on score(), score_conditional(), score_mutants()
         or any method-specific implementations as needed to provide the most efficient/accurate way
         to single mutant effect calculation

        3. Optional insertion handling: Models able to provide scores for insertions should include these
         by requesting an alphabet including insertion symbols: Entity.alphabet(..., include_inserts=True).
         Insertions are implied to occur immediately after the position in the dataframe index, an insertion
         before the first sequence position should be coded by pos=entity.first_index - 1 *and* ref = "" in the
         dataframe index, with all uppercase/non-insert symbol values set to NaN.

        4. Methods returning predictions across entities with more than one alphabet should use
         Entity.merge_alphabet_symbols() to determine the mixed alphabet/column order. The alphabet of
         each dataframe row is implied by the type of the respective entity, all symbols from other
         alphabets not relevant for current row should be set to NaN)

        Parameters
        ----------
        instance
            Target system instance specification to mutate
        entity
            Index of entity for which mutation scan should be computed. If None, score all entities in system.
            Must be specified as int if using positions parameter.
        positions
            Subset of positions to score. If None, scores for all positions will be computed across all entities;
            if specified, must also specify entity.
        status_callback
            Callback function to track computation status

        Returns
        -------
        Dataframe with log-odds scores (seq x symbol) relative to instance; rows index over
        entity/position/ref triplets, columns index over different symbols (amino acids etc.).
        Columns must be in same order as returned by Entity.alphabet(); missing predictions must
        be coded by np.nan.
        """
        if positions is not None:
            if entity is None:
                raise ValueError(
                    "Parameter entity must be explicitly specified if using parameter positions"
                )
            else:
                self.valid_positions(positions, instance, entity, raise_invalid=True)

        if entity is not None and not 0 <= entity < len(self.system):
            raise ValueError(
                "Invalid entity selection"
            )

        # create list of all single mutants (all positions)
        mutants = self.system.single_mutants(
            instance, deletions=self.handles_deletions, insertions=self.handles_insertions
        )

        # filter mutant list by entity/position if specified
        if entity is not None or positions is not None:
            mutants = [
                mutant for mutant in mutants
                if mutant[0].entity == entity and (positions is None or mutant[0].pos in positions)
            ]

        # compute scores through score_mutants - might already be a more efficient custom implementation
        # than score() (which itself can fallback on score() if not implemented); note these
        # scores will already be normalized to target so no need to normalize here
        scores = self.score_mutants(
            instance, mutants, status_callback
        )

        # build into dataframe and return
        series = pd.Series(scores)

        series.index = pd.MultiIndex.from_tuples(
            [mutant[0] for mutant in mutants], names=["entity", "pos", "ref", "to"]
        )

        # make sure column index (symbols) has right order
        all_entities = set(series.index.get_level_values("entity"))
        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[entity_idx].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            ) for entity_idx in all_entities
        ])

        return series.unstack(
            level="to"
        ).reindex(
            merged_alphabet, axis=1
        )

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        """
        Compute logit scores for a list of mutations to a specified system instance
        (can be any single or higher-order mutants); this method is to allow specialized, more efficient
        or accurate implementations of mutant calculations than computing the full score of the WT and
        mutant sequence. In case no such specialization is possible or needed for a method, it can simply
        call out to the score() function.

        Note:
        1. Mutation logits should be *relative* to the given instance (like a log-odds score),
         so that self-substitutions are assigned are score of 0, beneficial substitutions are score > 0,
         and damaging substitutions  a score < 0. This differs from score_conditional, where there is
         no notion of a "wildtype" sequence to compute relative scores to.

        2. Implementations of this method may either compute mutant and reference scores for substraction
         with the score() method or draw on any specialized implementations of single and higher-order mutation
         scoring that are more accurate / efficient.

        Parameters
        ----------
        instance
            Target system instance specification to mutate
        mutants
            List of mutations of any order to compute
        status_callback
            Callback function to track computation status

        Returns
        -------
        1D array of scores, guaranteed to be in the same order as mutants list
        """
        if not isinstance(self, Scorer):
            raise NotImplementedError(
                "Model does not implemented Scorer interface, cannot use this default implementation"
            )

        if hasattr(self, "ready_or_raise"):
            self.ready_or_raise()

        # check instance against molecular system
        self._validate_instances([instance])

        # validate mutants
        self.system.valid_mutants(
            instance,
            mutants,
            deletions=self.handles_deletions,
            insertions=self.handles_insertions,
            raise_invalid=True
        )

        # create mutants, add target to compute score for relative normalization
        instances = [instance] + self.system.mutate(instance, mutants)

        # score instances
        scores = self.score(instances, status_callback)

        # normalize scores to target instance, remove target from list
        scores_norm = scores[1:] - scores[0]

        return scores_norm


class Transformer(_Core):
    """
    Interface implemented by models that transform instances from one representation to another
    (e.g. from sequence to embeddings or structures, or vice versa).

    Note: Implementations may transform to any representation attribute present on SystemInstance
     (rep, embedding, structure)

    Note: Implementations must verify that all relevant input attributes on instances are specified

    Note: implementations should also set the "score" attribute on the SystemInstance to simultaneously
     score and transform instances for increased computational efficiency (e.g. compute likelihood
     score and embed) if it is able to compute both at the same time

    Note: Implementation must not mutate the provided instance list (references to embeddings and structures
     can be reused for efficiency when copying, i.e. a shallow copy of SystemInstance and EntityInstance objects
     is sufficient)

    TODO: eventually revisit if beneficial to add specialized methods for single-mutant embeddings
     (like for scoring)
    """
    @abstractmethod
    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None
    ) -> list[SystemInstance]:
        """
        Transform system instances from one representation to another

        Parameters
        ----------
        instances
            List of system instances to be transformed
        entity:
            The index of the entity to transform. If None, transform all entities in system.
        status_callback
            Callback function to track computation status

        Returns
        -------
        Transformed instances (copy, not modified in place), with updated attributes and/or score
        """
        pass


@dataclass
class RequiredResources:
    """
    All memory resources in megabytes, times in minutes
    """
    min_gpu_cores: int | None
    min_gpu_memory_per_core: int | None

    min_cpu_cores: int | None
    min_cpu_memory_per_core: int | None

    max_batch_size: int | None

    time: int | None


class BaseModel(_Core):
    """
    Core definition of models operating directly on molecular systems with sequences, structures, data, ...
    (not to be used for higher-level implementations like samplers etc.)
    """
    @property
    @abstractmethod
    # required attributes on Entity that must be specified; type, rep, id and first_index are always mandatory on System
    # and can be left out here. If attributes have no direct relevance to model, should be set to None.
    def required_entity_attributes(self) -> list[str] | None:
        pass

    @property
    @abstractmethod
    # optional attributes on Entity that can but do not have to be specified.  If attributes have no direct
    # relevance to model, should be set to None.
    def optional_entity_attributes(self) -> list[str] | None:
        pass

    @property
    @abstractmethod
    # indicates if model was built and is ready for scoring/generation
    def ready(self) -> str:
        pass

    def ready_or_raise(self) -> None:
        """
        Verifies if model is ready for predictions by checking ready property,
        or raises a ValueError otherwise
        """
        if not self.ready:
            raise ValueError("Must call build() first to use model")

    @classmethod
    @abstractmethod
    def can_model(
        cls,
        system: System,
        data: Any,
    ) -> tuple[bool, str]:
        """
        Check if the model is able to perform computations on the specified
        molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (cf. documentation for build() method)

        Returns
        -------
        bool
            True if model is able to handle the system, False otherwise
        str
            Message specifying why model is not able to handle the system
        """
        pass

    @classmethod
    def can_model_or_raise(
        cls,
        system: System,
        data: Any,
    ) -> None:
        """
        Check if the model is able to perform computations on the specified
        molecular system via can_model(), raise a ValueError otherwise

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (cf. documentation for build() method)

        Returns
        -------
        bool
            True if model is able to handle the system, False otherwise
        str
            Message specifying why model is not able to handle the system
        """
        can_model, can_model_msg = cls.can_model(system, data)
        if not can_model:
            raise ValueError(can_model_msg)

    @abstractmethod
    def build(
        self,
        system: System,
        data: Any,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        Prepare model for calculations on a given molecular system (e.g. scoring or sampling).
        Conditional approaches will typically perform computations here whereas unconditional approaches
        may simply do nothing other than return self.
        In the case of inference-only conditional models, implementations of this method will be very
        light (e.g. compute an encoding), whereas for other conditional models this method may be
        compute-heavy (e.g. EVE VAE models trained on a family-specific MSA)

        Notes re implementation:
        1) Should always verify if the system can
        be modelled using self.can_model() or raise a ValueError instead

        2) Should always assign system to self.system

        3) Should always return self to allow method chaining

        4) Should pay careful attention whether any external model parameters
        (e.g. PyTorch model) are stored inside the class to avoid potential problems and inflated
        memory usage if instances of the class are serialized; use the available context managers
        to handle this behavior reliably

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (could be labelled data points, external sequences to compare to, etc.)
        status_callback
            Callback function to receive progress updates

        Returns
        -------
        self
            Reference to the instance for method chaining
        """
        pass

    def stats(self) -> ModelStats | None:
        """
        Return summary statistics from model building (cross-validation performance etc.).

        Default behaviour is to not return statistics

        Returns
        -------
        Model-dependent statistics
        """
        return None


class SupervisedBaseModel(BaseModel):
    @abstractmethod
    def build(
        self,
        system: System,
        data: LabeledInstanceDataset,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        Cf documentation for BaseModel, except that model receives
        a set of labeled instances as data for (semi-)supervised model training.

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Labeled instance dataset
        status_callback
            Callback function to receive progress updates

        Returns
        -------
        self
            Reference to the instance for method chaining
        """
        pass


def system_subset_model(model_class):
    """
    Factory function to dynamically modify a model so it can seamlessly operate on a subset
    of entities in a system (e.g. to apply single-protein LLM to binder designed in complex
    of a target structure)

    TODO: need to add proper type hinting here; updated signature change of build()
      not compatible with system_subset_model(model_class: T) -> T

    Parameters
    ----------
    model_class
        Model class to wrap

    Returns
    -------
    Updated model class
    """
    # functionality added on top of model_class only makes sense if it is a model operating on systems
    if not issubclass(model_class, BaseModel):
        raise TypeError("model_class must inherit from BaseModel")

    class SubsetModel(model_class):
        def __init__(self, **args):
            # note: we store model as attribute rather than really subclassing it
            # as this creates headaches with method resolution between child and parent class
            # super().__init__(**args)

            self.model = model_class(**args)
            self.entity_subset = None
            self.entity_subset_map = None
            self.system_full = None
            self.system_subset = None

        # _Core properties
        @property
        def name(self) -> str:
            return self.model.name

        @property
        # citation strings for method
        def citations(self) -> list[str]:
            return self.model.citations

        @property
        def system(self) -> System | None:
            # to outside world, we need to claim we model full system
            return self.system_full

        @property
        def requires_target(self) -> bool:
            return self.model.requires_target

        @property
        def requires_fixed_length(self) -> bool:
            return self.model.requires_fixed_length

        @property
        def handles_deletions(self) -> bool:
            return self.model.handles_deletions

        @property
        def handles_insertions(self) -> bool:
            return self.model.handles_insertions

        @property
        def requires_gpu(self) -> bool:
            return self.model.handles_gpu

        @property
        def supports_gpu(self) -> bool:
            return self.model.supports_gpu

        @property
        def supports_cpu_parallel(self) -> bool:
            return self.model.supports_cpu_parallel

        @property
        def supports_gpu_parallel(self) -> bool:
            return self.model.supports_gpu_parallel

        # BaseModel properties
        @property
        def required_entity_attributes(self) -> list[str] | None:
            return self.model.required_entity_attributes

        @property
        def optional_entity_attributes(self) -> list[str] | None:
            return self.model.optional_entity_attributes

        @property
        def ready(self) -> str:
            # implies all other relevant attributes were set by build()
            return self.system_full is not None

        def stats(self) -> ModelStats | None:
           return self.model.stats()

        @staticmethod
        def _filter_system(
            system: System,
            entity_subset: Sequence[int] | None = None,
        ) -> System:
            for idx in entity_subset:
                if idx < 0 or idx >= len(system):
                    raise ValueError(f"Invalid entity index: {idx}")

            return System([
                system[entity_idx] for entity_idx in sorted(entity_subset)
            ])

        def _filter_instance(self, instance) -> SystemInstance:
            # filter entity instance to subset of entities, must keep order
            return SystemInstance([
                instance[idx] for idx in sorted(self.entity_subset)
            ])

        def _map_entity(self, entity: int | None):
            # check and map selected entity
            if entity is not None:
                if entity not in self.entity_subset_map:
                    raise ValueError(
                        f"Entity {entity} not covered by subset map: {self.entity_subset_map}"
                    )

                entity_mapped = self.entity_subset_map[entity]
            else:
                entity_mapped = None

            return entity_mapped

        @classmethod
        def can_model(
            cls,
            system: System,
            data: Any,
            entity_subset: Sequence[int] | None = None,
        ) -> tuple[bool, str]:
            return model_class.can_model(
                system, data, cls._filter_system(system, entity_subset)
            )

        def build(
            self,
            system: System,
            data: Any,
            status_callback: StatusCallback | None = None,
            entity_subset: Sequence[int] | None = None,
        ) -> Self:
            # filter system and store, also retain full system and indices for remapping
            self.system_subset = self._filter_system(system, entity_subset)
            self.entity_subset = sorted(entity_subset)
            self.entity_subset_map = {
                full_index: filt_index for filt_index, full_index in enumerate(self.entity_subset)
            }
            self.system_full = system

            # build wrapped model on filtered system
            self.model.build(
                self.system_subset, data, status_callback
            )

            return self

        def positions(
            self,
            instance: SystemInstance | None,
        ) -> list[tuple[int, int]]:
            self.ready_or_raise()
            if instance is not None:
                self._validate_instances([instance])

            if instance is not None:
                instance_filt = self._filter_instance(instance)
            else:
                instance_filt = None

            # get positions on mapped instance
            positions = self.model.positions(instance_filt)

            # remap entity indices and return
            return [
                (self.entity_subset[entity], pos) for (entity, pos) in positions
            ]

        def score(
            self,
            instances: Sequence[SystemInstance],
            status_callback: StatusCallback | None = None
        ) -> np.ndarray[tuple[int], np.dtype[float]]:
            self.ready_or_raise()
            self._validate_instances(instances)

            instances_filt = [
                self._filter_instance(instance) for instance in instances
            ]

            return self.model.score(
                instances_filt, status_callback
            )

        def score_conditional(
            self,
            instances: Sequence[SystemInstance],
            entities: Sequence[int],
            positions: Sequence[int],
            status_callback: StatusCallback | None = None
        ) -> pd.DataFrame:
            self.ready_or_raise()
            self._validate_instances(instances)

            # check all entity selections
            try:
                entities_mapped = [
                    self.entity_subset_map[entity_idx] for entity_idx in entities
                ]
            except KeyError as e:
                raise ValueError("Invalid entity index") from e

            instances_filt = [
                self._filter_instance(instance) for instance in instances
            ]

            # score
            scores = self.model.score_conditional(
                instances_filt, entities_mapped, positions, status_callback
            )

            # remap entity index in dataframe
            assert scores.index.names[1] == "entity"
            scores.index = scores.index.set_levels(
                scores.index.levels[1].map(
                    lambda entity_idx: self.entity_subset[entity_idx]
                ).values, level=1
            )

            return scores

        def single_mutation_scan(
            self,
            instance: SystemInstance,
            entity: int | None = None,
            positions: Sequence[int] | None = None,
            status_callback: StatusCallback | None = None
        ) -> pd.DataFrame:
            self.ready_or_raise()
            self._validate_instances([instance])

            # filter instance to subsystem, and map entity indices
            instance_filt = self._filter_instance(instance)
            entity_mapped = self._map_entity(entity)

            # compute mutation matrix with model on instance limited to system subset
            scores = self.model.single_mutation_scan(
                instance=instance_filt, entity=entity_mapped, positions=positions, status_callback=status_callback
            )

            # remap entity index in dataframe
            assert scores.index.names[0] == "entity"
            scores.index = scores.index.set_levels(
                scores.index.levels[0].map(
                    lambda entity_idx: self.entity_subset[entity_idx]
                ).values, level=0
            )

            return scores

        def score_mutants(
            self,
            instance: SystemInstance,
            mutants: Sequence[Mutant],
            status_callback: StatusCallback | None = None
        ) -> np.ndarray[tuple[int], np.dtype[float]]:
            self.ready_or_raise()
            self._validate_instances([instance])

            # map mutants to filtered indices
            mutants_mapped = [
                [
                    Mutation(
                        entity=self._map_entity(mutation.entity), pos=mutation.pos, ref=mutation.ref,to=mutation.to
                    ) for mutation in mutant
                ]
                for mutant in mutants
            ]

            instance_filt = self._filter_instance(instance)

            return self.model.score_mutants(
                instance_filt, mutants_mapped, status_callback
            )

        def transform(
            self,
            instances: Sequence[SystemInstance],
            entity: int | None = None,
            status_callback: StatusCallback | None = None
        ) -> list[SystemInstance]:
            self.ready_or_raise()
            self._validate_instances(instances)

            # filter instances to subsystem, and map entity indices
            instances_filt = [
                self._filter_instance(instance) for instance in instances
            ]
            entity_mapped = self._map_entity(entity)

            # transform the filtered instances
            instances_transformed = self.model.transform(
                instances_filt, entity_mapped, status_callback
            )

            # update transformed instances by filling in original entity instances from full instances
            # for entities not covered by subset model
            for instance_transformed, instance_full in zip(instances_transformed, instances):
                entity_instances = [
                    (
                        instance_transformed[self.entity_subset_map[entity_idx]]
                        if entity_idx in self.entity_subset_map
                        else instance_full[entity_idx]
                    )
                    for entity_idx, _ in enumerate(self.system_full)
                ]
                instance_transformed.data = entity_instances

            return instances_transformed

        def generate(
            self,
            num_designs: int,
            entities: Sequence[int] | None = None,
            fixed_pos: EntityPosList | None = None,
            temperature: float = 1.0,
            status_callback: StatusCallback | None = None
        ) -> list[SystemInstance]:
            self.ready_or_raise()

            # map designed entities if specified (will raise if invalid)
            if entities is not None:
                entities_mapped = [
                    self._map_entity(entity) for entity in entities
                ]
            else:
                entities_mapped = None

            # map fixed positions (will raise if invalid)
            if fixed_pos is not None:
                fixed_pos_mapped = {
                    self.entity_subset_map[entity]: positions for entity, positions in fixed_pos.items()
                }
            else:
                fixed_pos_mapped = None

            # design on filtered system
            designs = self.model.generate(
                num_designs=num_designs,
                entities=entities_mapped,
                fixed_pos=fixed_pos_mapped,
                temperature=temperature,
                status_callback=status_callback
            )

            # update designs by filling unspecified instances for entities not covered by subset model
            for instance in designs:
                entity_instances = [
                    (
                        instance[self.entity_subset_map[entity_idx]]
                        if entity_idx in self.entity_subset_map
                        else EntityInstance()
                    )
                    for entity_idx, _ in enumerate(self.system_full)
                ]
                instance.data = entity_instances

            return designs

    # remove methods which are not present on parent (eg transform) so it behaves
    # exactly the same to the outside world
    for attr in dir(SubsetModel):
        if not hasattr(model_class, attr) and not attr.startswith("_"):
            delattr(SubsetModel, attr)

    return SubsetModel
