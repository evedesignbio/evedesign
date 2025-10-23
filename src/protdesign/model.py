from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Self, Tuple, Sequence, Any
import numpy as np
import pandas as pd
from protdesign.entity import System, SystemInstance, EntityPosList, Mutant
from protdesign.types import StatusCallback


class _Core(ABC):
    """
    Minimal core functionality required by any modular class used for sequence design.

    Note: this class should not be implemented directly but rather through one of its
     more specific subclasses like Generator
    """
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

    @abstractmethod
    def positions(
        self,
        instance: SystemInstance | None,
    ) -> List[Tuple[int, int]]:
        """
        Return list of all available modelled positions per entity *instance* that are explicitly
        captured by the model

        Notes:
        1. Positions that are not modelled (e.g. excluded positions for EVmutation) should not
         be returned by this method

        2. For fixed-length models, entity instance positions will by definition be the same
         as entity representation positions. These models can opt to set the instance argument to
         a default value of None.

        3. Models able to handle insertions should also return first_index - 1 coding for
         an N-terminal insertion (but must not model substitution effects for this position)

        4. Returned positions should be ordered in ascending order
         by i) entity index, ii) position index in entity

        Returns
        -------
        List of position tuples (entity_idx, position)
        """
        pass

    def valid_positions(
        self,
        positions: Sequence[int],
        instance: SystemInstance | None = None,
        entities: int | Sequence[int] = 0,
        raise_invalid: bool = False,
    ) -> List[tuple[int, int]]:
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


class Generator(_Core):
    """
    Interface implemented by classes that can generate new samples
    (e.g. generative models or samplers on top of scoring models)

    TODO: check whether it makes sense to add more designs parameters shared
     across most methods here, or whether it is better to add additional parameters
     to individual methods (with default arguments) based on the functionality
     of each method
    """
    @abstractmethod
    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
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
        deletions
            If True, allow the model to sample deletions relative to the entities representation
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


class ConditionalMutationScorer(_Core):
    """
    Interface implemented by classes that can compute conditional probabilities
    P(x_i | x_\i) to be used e.g. for Gibbs sampling even if not
    able to compute full P(x_1, ..., x_n)
    """
    @abstractmethod
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
        pass


class MutationScorer(_Core):
    """
    Interface for methods that allow to score mutations to an instance
    """
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass


class Transformer(_Core):
    """
    Interface implemented by models that transform instances from one representation to another
    (e.g. from sequence to embeddings or structures, or vice versa).

    Note: Implementations may transform to any representation attribute present on SystemInstance
     (rep, embedding, structure)

    Note: Implementations must verify that all relevant input attributes on instances are specified

    Note: implementations may also set the "score" attribute on the SystemInstance to simultaneously
     score and transform instances for increased computational efficiency (e.g. compute likelihood
     score and embed).

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
    ) -> List[SystemInstance]:
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
    # plain-text name of method
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    # whether model has long-running build step (e.g. EVE VAE)
    def requires_heavy_build(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs unaligned sequences as input
    def requires_seqs(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs aligned sequences as input
    def requires_msa(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs 3D structures as input
    def requires_3d(self) -> bool:
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
    ) -> Tuple[bool, str]:
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

    @classmethod
    @abstractmethod
    def required_resources(
        cls,
        system: System,
        data: Any,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        """
        Estimate the required resources to perform computations on molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (cf. documentation for build() method)
        use_gpu
            Set to True if you want to estimate resources making use of GPU
            (only for models supporting GPU-based computations)
        build
            Set as True to estimate resources for model building. Set as False to
            estimate resources for inference (scoring / sampling).

        Returns
        -------
        RequiredResources
            CPU/GPU/RAM requirements for running computations on molecular system
        """
        pass

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

