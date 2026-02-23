"""
Functionality for reducing design dimensionality to analyze relationships between generated and natural sequences
"""
from abc import ABC, abstractmethod
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence, Literal
import numpy as np
from numba import prange, jit
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from evedesign.analysis import Analyzer
from evedesign.tools.mmseqs2 import filter_sequences_mmseqs

try:
    from umap import UMAP  # noqa
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from evedesign.system import System, SystemInstance
from evedesign.types import EntityType, SEQSPACE_PROJECTION_COMPONENT_KEY
from evedesign.utils import str_to_np_char_view, map_array, index_map

# each element is (entity, list of sequences, number of system sequences at end of list)
CollectedSequences = list[tuple[int, list[str], list[int]]]


@jit(nopython=True, parallel=True)
def hamming_distance_no_gaps(matrix, exclude_value):
    """
    Calculate pairwise sequence distance matrix for a set of sequences

    The function will by default use the available number of threads
    (as returned by numba.get_num_threads()). If a different number should
    be used, the caller is responsible to set the number of threads with
    numba.set_num_threads

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols)
    exclude_value : int
        Value >= 0 in matrix that will be exclud_ed from identity calculation, e.g. gap or lowercase character.
        Set to -1 to enable legacy behaviour num_cluster_members_legacy which includes gaps in identity calculation.

    Returns
    -------
    np.array
        Symmetric distance matrix normalized to range 0 to 1
    """
    N, L = matrix.shape  # noqa

    # minimal cluster size is 1 (self) but for parallelization we set the self-hit below inside the loop
    # and initialize to zero here
    dist_matrix = np.zeros((N, N))

    # compare all pairs of sequences; we cannot assume symmetry of the resulting matrix here due to exclusion of
    # gaps (this is also convenient for parallelizing the outer loop); no speedup from using a separate function
    # with regular range(N) in single-thread case so can always use this function

    for i in prange(N):
        # compare to all other sequences
        for j in range(i + 1, N):
            # differences
            dist = 0

            # total number of pairs compared
            pairs = 0

            # compare all positions
            for k in range(L):
                if matrix[i, k] != exclude_value and matrix[j, k] != exclude_value:
                    pairs += 1
                    if matrix[i, k] != matrix[j, k]:
                        dist += 1

            # avoid potential division by zero
            if pairs == 0:
                pairs = 1

            dist_norm = dist / pairs
            dist_matrix[i, j] = dist_norm
            dist_matrix[j, i] = dist_norm

    return dist_matrix


@jit(nopython=True, parallel=True)
def hamming_distance_no_gaps_ref_vs_comp(matrix_ref, matrix_comp, exclude_value):
    """
    Calculate pairwise sequence distance NxM matrix between a reference (N entries)
    and a comparison set (M entries) of sequences

    The function will by default use the available number of threads
    (as returned by numba.get_num_threads()). If a different number should
    be used, the caller is responsible to set the number of threads with
    numba.set_num_threads

    Parameters
    ----------
    matrix_ref : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols)
    matrix_comp : np.array
        M x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols)
    exclude_value : int
        Value >= 0 in matrix that will be excluded from identity calculation, e.g. gap or lowercase character.
        Set to -1 to enable legacy behaviour num_cluster_members_legacy which includes gaps in identity calculation.

    Returns
    -------
    np.array
        NxM distance matrix normalized to range 0 to 1
    """
    N, L = matrix_ref.shape  # noqa
    M, L_comp = matrix_comp.shape  # noqa

    if L != L_comp:
        raise ValueError(
            f"Sequences have differing lengths between reference and comparison set {L}vs {L_comp}"
        )

    # minimal cluster size is 1 (self) but for parallelization we set the self-hit below inside the loop
    # and initialize to zero here
    dist_matrix = np.zeros((N, M))

    # compare all pairs of sequences; we cannot assume symmetry of the resulting matrix here due to exclusion of
    # gaps (this is also convenient for parallelizing the outer loop); no speedup from using a separate function
    # with regular range(N) in single-thread case so can always use this function

    for i in prange(N):
        # compare to all other sequences
        for j in range(0, M):
            # differences
            dist = 0

            # total number of pairs compared
            pairs = 0

            # compare all positions
            for k in range(L):
                if matrix_ref[i, k] != exclude_value and matrix_comp[j, k] != exclude_value:
                    pairs += 1
                    if matrix_ref[i, k] != matrix_comp[j, k]:
                        dist += 1

            # avoid potential division by zero
            if pairs == 0:
                pairs = 1

            dist_norm = dist / pairs
            dist_matrix[i, j] = dist_norm

    return dist_matrix


def distance_matrix_mmseqs_ref_vs_comp(
    ref_sequences: list[str],
    comp_sequences: list[str],
    mmseqs_path: str = "mmseqs"
) -> np.ndarray:
    def write_fasta_dealigned(path: Path, seqs: list[str]):
        with path.open("w") as f:
            for i, sequence in enumerate(seqs):
                f.write(
                    f">{i}\n{sequence.replace('-', '').upper()}\n"
                )

    # 1) compute distance matrix of landmark_sequences x sequences
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Write sequences to a temporary FASTA file
        query_fa = tempdir / "ref_sequences.fasta"
        target_fa = tempdir / "comp_sequences.fasta"
        write_fasta_dealigned(query_fa, ref_sequences)
        write_fasta_dealigned(target_fa, comp_sequences)

        # MMseqs easy-search
        res_tsv = tempdir / "search.tsv"
        cmd = [
            mmseqs_path,
            "easy-search",
            str(query_fa),
            str(target_fa),
            str(res_tsv),
            str(tempdir),
            "--prefilter-mode", "2",
            "--search-type", "1",
            "--seq-id-mode", "2",
            "-e", "inf",
            "--format-output", "query,target,pident"]
        subprocess.run(
            cmd,
            capture_output=True
        )

        # Compute distance matrix
        d_matrix = np.ones((len(ref_sequences), len(comp_sequences)), dtype=np.float32)
        with res_tsv.open() as f:
            for line in f:
                q, t, pident = line.rstrip().split("\t")
                pid = float(pident)
                d_matrix[int(q), int(t)] = 1.0 - (pid / 100.0)

        return d_matrix


class SequenceSpaceProjection(Analyzer, ABC):
    """
    Project sequences into lower-dimensional space for visual inspection

    Note: may want to re-express this as implementation of Transformer interface;
     however this is not compatible with analyzing sequences in underlying System
    """
    def __init__(
        self,
        acceptable_entity_types: list[EntityType],
        num_components: int = 2,
        include_system_sequences: bool = True,
        system_sequence_fragment_filter: float | None = 0.7
    ):
        """
        Create new MDS-based sequence space projector

        Parameters
        ----------
        acceptable_entity_types
            List of entity types that projector implementation can handle
        num_components
            Number of components to project sequences down to
        include_system_sequences
            If true, include system sequences besides designed sequences
        system_sequence_fragment_filter
            Only keep system sequences that have a non-gap symbol aligned to
            the target sequence for at least the given fraction of positions.
            Will be ignored if system sequences are not aligned; use None to
            disable filtering.
        """
        self.num_components = num_components
        self.include_system_sequences = include_system_sequences
        self.acceptable_entity_types = acceptable_entity_types
        self.system_sequence_fragment_filter = system_sequence_fragment_filter

    def _select_entities(
        self,
        system: System,
        entity: int | None,
    ) -> list[int]:
        """
        Helper method to determine entities used for computation

        Parameters
        ----------
        system
            System for which instances/natural sequences will be projected
        entity
            If None, use all entities, if specified, use particular entity.

        Returns
        -------
        List of selected entities
        """
        all_entities = list(range(0, len(system)))

        # either use all entities if unspecified, or restrict to selected entity
        if entity is None:
            entities = all_entities
        else:
            if entity not in all_entities:
                raise ValueError(
                    f"Invalid entity selection, valid options are {' '.join(map(str, all_entities))}"
                )

            entities = [entity]

        # make sure only entities are selected that projection method can handle (protein, DNA, ...)
        for checked_entity in entities:
            entity_type = system[checked_entity].type
            if entity_type not in self.acceptable_entity_types:
                raise ValueError(
                    f"Entity {checked_entity} is of type {entity_type} but only the following are "
                    f"allowed: {', '.join(self.acceptable_entity_types)} "
                )

        return entities

    def _collect_sequences(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entities: list[int],
        require_aligned: bool = True,
    ) -> CollectedSequences:
        """
        Collect rep sequences for all analyzed entities

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances from which sequences will be collected
        entities
            Indices of entities for which sequences will be collected
        require_aligned
            If True, requires that all instance and system sequences are aligned (same number of match states)

        Returns
        -------
        List of collected sequence reps per entity
        """
        if self.include_system_sequences:
            if len(entities) != 1:
                raise ValueError(
                    "Must specify a single entity for inclusion of system sequences as mapping may be ambiguous; "
                    "this feature may be implemented at a later time"
                )

            system_sequences = system.data[entities[0]].sequences
            if system_sequences is not None and require_aligned and not system_sequences.aligned:
                raise ValueError(
                    "System sequences must be aligned for analysis"
                )

        # assemble sequences on per-entity basis; this allows us to look at entity-specific information
        # like alphabets when performing actual computation
        all_seqs = []
        for entity in entities:
            cur_entity_seqs = system.data[entity].sequences
            if self.include_system_sequences and cur_entity_seqs is not None:
                # if requiring alignment, remove dealigned positions (otherwise will rarely find an MSA
                # that could be handled)
                system_seqs = {
                    i: (entry.remove_insertions().seq if require_aligned else entry.dealign().seq)
                    for i, entry
                    in enumerate(cur_entity_seqs.seqs)
                    if (
                        self.system_sequence_fragment_filter is None or
                        not cur_entity_seqs.aligned or
                        len(entry.remove_insertions().dealign().seq) >= (
                            len(system[entity].rep) * self.system_sequence_fragment_filter
                        )
                    )
                }
            else:
                system_seqs = {}

            # ensure all instances have a defined rep
            if any([
                instance[entity].rep is None for instance in instances
            ]):
                raise ValueError(
                    "Entity instance contains rep that is None; "
                    "for sequence space projection all instances must have specified rep"
                )

            instance_seqs = [
                "".join(instance[entity].rep if require_aligned else instance[entity].normalized_rep())
                for instance in instances
            ]

            # if requiring alignment, need to verify all sequences now have same length
            merged_seqs = list(system_seqs.values()) + instance_seqs

            if require_aligned:
                seq_lengths = {
                    len(seq) for seq in merged_seqs
                }
                if len(seq_lengths) != 1:
                    raise ValueError(
                        f"Aligned sequences required but input sequences have differing lengths for "
                        f"entity {entity}: {seq_lengths}"
                    )

            all_seqs.append((
                entity, merged_seqs, list(system_seqs.keys())
            ))

        # return assembled sequences per entity
        return all_seqs

    def add_projections(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entities: list[int],
        projections: np.ndarray,
        sequences: CollectedSequences
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Add projections as metadata to system and instances

        Parameters
        ----------
        system
            System to which analysis results will be attached
        instances
            Instances to which analysis results will be attached
        entities:
            Indices of entities for which sequences will be collected
        projections
            Projections that will be attached to system and instances
        sequences
            Per-entity collection of sequences used to compute the projection
            (including indices of system sequences that were used)

        Returns
        -------
        Tuple containing results from analysis in
        (i) System
        (ii) SystemInstances
        """
        if self.include_system_sequences:
            system_projections = projections[:-len(instances)]
            instance_projections = projections[-len(instances):]
        else:
            system_projections = None
            instance_projections = projections

        # shallow copy of instances, then attach metadata
        updated_instances = [
            inst.copy() for inst in instances
        ]
        for idx, inst in enumerate(updated_instances):
            if inst.metadata is None:
                inst.metadata = {}

            inst.metadata[SEQSPACE_PROJECTION_COMPONENT_KEY] = instance_projections[idx, :].tolist()

        # deep copy of system, then attach metadata
        updated_system = system.copy()
        if system_projections is not None:
            # for now, only single entity projection if using system sequences
            assert len(entities) == 1

            # get indices of system sequences (may have been filtered)
            _, _, system_seq_indices = sequences[0]

            # lengths must match
            assert len(system_projections) == len(system_seq_indices)

            for (seq_idx, proj) in zip(system_seq_indices, system_projections):
                seq = updated_system[entities[0]].sequences.seqs[seq_idx]
                if seq.metadata is None:
                    seq.metadata = {}

                seq.metadata[SEQSPACE_PROJECTION_COMPONENT_KEY] = proj.tolist()

        return updated_system, updated_instances

    @abstractmethod
    def distances_and_projection(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int], CollectedSequences]:
        """
        Perform sequence space projection analysis, returning results directly
        (e.g. for interactive analysis)

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which projections should be computed
        entity
            Index of entity based on which projections should be computed
            (if None, use all applicable entities)

        Returns
        -------
        Tuple containing main results from analysis:
        (i) Distance matrix
        (ii) Projection of shape num_sequences x num_components; system sequences will be first
        (iii) Selected entities
        (iv) Collected sequences
        """
        pass

    def analyze(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        data: None = None,
        entity: int | None = None
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Perform sequence space projection analysis

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which projections should be computed
        data
            Not used, must be None
        entity
            Index of entity based on which projections should be computed
            (if None, use all applicable entities)

        Returns
        -------
        Tuple containing results from sequence space analysis in
        (i) System (only updated if include_system_sequences is True)
        (ii) SystemInstances
        """
        if data is not None:
            raise ValueError(
                "data argument must be None"
            )

        # validate entities, in particular fixed length requirement for this class
        dist_matrix, projections, entities, sequences = self.distances_and_projection(  # noqa
            system, instances, entity
        )

        # add projection to shallow copy of system and instances
        return self.add_projections(
            system, instances, entities, projections, sequences
        )


class SequenceSpaceProjectionAligned(SequenceSpaceProjection):
    """
    Sequence space projection, Assuming sequences are aligned and have same length of match states;
    will discard any inserts relative to consensus from analysis
    """
    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        system_sequence_fragment_filter: float | None = 0.7
    ):
        """
        Initialize new sequence space projector

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        system_sequence_fragment_filter
            Only keep system sequences that have a non-gap symbol aligned to
            the target sequence for at least the given fraction of positions.
            Will be ignored if system sequences are not aligned; use None to
            disable filtering.
        """
        super().__init__(
            acceptable_entity_types=["protein", "dna", "rna"],
            num_components=num_components,
            include_system_sequences=include_system_sequences,
            system_sequence_fragment_filter=system_sequence_fragment_filter
        )

    @classmethod
    def _distance_matrix(
        cls,
        system: System,
        collected_sequences: CollectedSequences,
        default_value: int = -1
    ) -> np.ndarray:
        """
        Compute distance matrix from set of instance/system sequences
        (potentially for multiple entities)

        Parameters
        ----------
        system
            System for which sequences are analyzed
        collected_sequences
            Extracted rep sequences for all entities
        default_value
            Default value to map gaps/non-standard symbols to

        Returns
        -------
        Distance matrix of shape len(collected_sequences) x num_components
        """

        # map sequences for each entity to integer array for numba computation
        entity_arrays = [
            map_array(
                str_to_np_char_view(seqs),
                # do not map gaps so we can easily exclude with same value in numba calculation
                index_map(system[entity].alphabet(include_gap=False), default_value=default_value),
            )
            for (entity, seqs, _) in collected_sequences
        ]

        # merge array together across entities
        array_merged = np.concatenate(entity_arrays + entity_arrays, axis=1)

        # compute distance matrix with numba
        dist_matrix = hamming_distance_no_gaps(array_merged, default_value)
        return dist_matrix

    @abstractmethod
    def _project(
        self,
        dist_matrix: np.ndarray[tuple[int, int], float],
    ) -> np.ndarray[tuple[int, int], float]:
        pass

    def distances_and_projection(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int], CollectedSequences]:
        # validate entities, in particular fixed length requirement for this class
        [
            system.valid_instance(
                instance,
                validate_reps=True,
                fixed_length=True,
                allow_deletions=True,
                raise_invalid=True,
            ) for instance in instances
        ]

        # determine selected entities (for regular MDS, can do all types of biopolymers)
        entities = self._select_entities(
            system, entity
        )

        # assemble instance data as needed, also verify they are aligned for all methods but landmark_mds_mmseqs
        sequences = self._collect_sequences(
            system, instances, entities, require_aligned=True
        )

        # compute distance matrix
        dist_matrix = self._distance_matrix(system, sequences)

        # perform projection
        projections = self._project(dist_matrix)

        return dist_matrix, projections, entities, sequences


class SequenceSpaceMDS(SequenceSpaceProjectionAligned):
    """
    Sequence space projection with multidimensional scaling, following https://github.com/debbiemarkslab/sequenceMDS
    """
    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        system_sequence_fragment_filter: float | None = 0.7,
        mds_kwargs: dict | None = None
    ):
        """
        Initialize new sequence space projector using multidimensional scaling (MDS)

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        system_sequence_fragment_filter
            Only keep system sequences that have a non-gap symbol aligned to
            the target sequence for at least the given fraction of positions.
            Will be ignored if system sequences are not aligned; use None to
            disable filtering.
        mds_kwargs
            Keyword arguments forwarded to constructor of sklearn.manifold.MDS
        """
        super().__init__(
            num_components=num_components,
            include_system_sequences=include_system_sequences,
            system_sequence_fragment_filter=system_sequence_fragment_filter,
        )

        self.mds_kwargs = mds_kwargs

    def _project(
        self,
        dist_matrix: np.ndarray[tuple[int, int], float]
    ) -> np.ndarray[tuple[int, int], float]:
        if self.mds_kwargs is None:
            params = {
                "normalized_stress": "auto",
            }
        else:
            params = self.mds_kwargs

        # following https://github.com/debbiemarkslab/sequenceMDS
        embedding = MDS(
            n_components=self.num_components,
            dissimilarity="precomputed",
            **params,
        )

        return embedding.fit_transform(dist_matrix)


class SequenceSpacePCA(SequenceSpaceProjectionAligned):
    """
    Sequence space projection with principal component analysis (PCA)
    """
    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        system_sequence_fragment_filter: float | None = 0.7,
        pca_kwargs: dict | None = None
    ):
        """
        Initialize new sequence space projector using multidimensional scaling (MDS)

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        system_sequence_fragment_filter
            Only keep system sequences that have a non-gap symbol aligned to
            the target sequence for at least the given fraction of positions.
            Will be ignored if system sequences are not aligned; use None to
            disable filtering.
        pca_kwargs
            Keyword arguments forwarded to constructor of sklearn.decomposition.PCA
        """
        super().__init__(
            num_components=num_components,
            include_system_sequences=include_system_sequences,
            system_sequence_fragment_filter=system_sequence_fragment_filter,
        )

        self.pca_kwargs = pca_kwargs
        self._embedder = None

    def _project(
        self,
        dist_matrix: np.ndarray[tuple[int, int], float]
    ) -> np.ndarray[tuple[int, int], float]:
        if self.pca_kwargs is None:
            params = {}
        else:
            params = self.pca_kwargs

        # following https://github.com/debbiemarkslab/sequenceMDS
        self._embedder = PCA(
            n_components=self.num_components,
            **params,
        )

        return self._embedder.fit_transform(dist_matrix)


class SequenceSpaceUMAP(SequenceSpaceProjectionAligned):
    available = UMAP_AVAILABLE

    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        system_sequence_fragment_filter: float | None = 0.7,
        umap_kwargs: dict | None = None
    ):
        """
        Initialize new sequence space projector using multidimensional scaling (MDS)

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        system_sequence_fragment_filter
            Only keep system sequences that have a non-gap symbol aligned to
            the target sequence for at least the given fraction of positions.
            Will be ignored if system sequences are not aligned; use None to
            disable filtering.
        umap_kwargs
            Keyword arguments forwarded to constructor of umap.UMAP
        """
        if not self.available:
            raise ValueError(
                "umap package is not available, please install first"
            )

        super().__init__(
            num_components=num_components,
            include_system_sequences=include_system_sequences,
            system_sequence_fragment_filter=system_sequence_fragment_filter
        )

        self.umap_kwargs = umap_kwargs

    def _project(
        self,
        dist_matrix: np.ndarray[tuple[int, int], float]
    ) -> np.ndarray[tuple[int, int], float]:
        if self.umap_kwargs is None:
            params = {
                # pronounce global structure
                "n_neighbors": 20,
                "min_dist": 0.5,
            }
        else:
            params = self.umap_kwargs

        embedding = UMAP(
            n_components=self.num_components,
            metric="precomputed",
            **params,
        )

        return embedding.fit_transform(dist_matrix)


class SequenceSpaceLandmarkMDS(SequenceSpaceProjection):
    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        system_sequence_fragment_filter: float | None = 0.7,
        num_landmarks: int = 100,
        landmark_source: Literal["all", "system", "instances"] = "all",
        landmark_selection_mode: Literal["random", "mmseqs"] = "random",
        distance_matrix_mode: Literal["aligned", "mmseqs"] = "aligned",
        mds_kwargs: dict | None = None,
        mmseqs_path: str = "mmseqs",
    ):
        """
        Initialize new sequence space projector using approximative multidimensional scaling (MDS)

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        num_landmarks
            Number of landmark points to use for approximation
        landmark_source
            Specify whether landmark points should be selected from system + entities ("all"),
            just from system sequences("system") to project designs to natural sequence space,
            or just on designs ("instances"). "system" requires include_system_sequences to be True.
        landmark_selection_mode
            Method to choose landmark points at random or by clustering.
            "mmseqs" requires MMseqs to be installed and uses clustering.
        distance_matrix_mode
            Method to compute distance matrix. "mmseqs" requires MMseqs to be installed
            and allows (potentially unaligned) sequences of varying length whereas
            "aligned" uses internal code and requires sequences to be of same length.
        mds_kwargs
            Keyword arguments forwarded to constructor of sklearn.manifold.MDS
        mmseqs_path
            Path to MMseqs executable, default assumes MMseqs is on the PATH.
        """
        self.num_landmarks = num_landmarks

        if landmark_source == "system" and not include_system_sequences:
            raise ValueError(
                "landmark_source == 'system' only valid if include_system_sequences is True"
            )

        self.landmark_source = landmark_source
        self.landmark_selection_mode = landmark_selection_mode
        self.distance_matrix_mode = distance_matrix_mode
        self._mmseqs_required = landmark_selection_mode == "mmseqs" or distance_matrix_mode == "mmseqs"
        self._fixed_length_required = distance_matrix_mode == "aligned"
        self.mmseqs_path = mmseqs_path

        acceptable_entity_types: list[EntityType]
        if self._mmseqs_required:
            acceptable_entity_types = ["protein"]
        else:
            acceptable_entity_types = ["protein", "dna", "rna"]

        super().__init__(
            num_components=num_components,
            include_system_sequences=include_system_sequences,
            acceptable_entity_types=acceptable_entity_types,
            system_sequence_fragment_filter=system_sequence_fragment_filter
        )

        self.mds_kwargs = mds_kwargs

    def _select_landmarks(self, sequences: CollectedSequences) -> np.ndarray[tuple[int], int]:
        """
        Select landmark sequences depending on mode

        Parameters
        ----------
        sequences
            Sequences from system (may not be present) and instances

        Returns
        -------
        Array of indices in sequences that were chosen as landmarks
        """
        entity_idx, entity_sequences, system_indices = sequences[0]
        num_system = len(system_indices)
        num_total = len(entity_sequences)

        # in case of random selection, we do not need to look at actual sequences,
        # just use indices (if system sequences present, we know for now we only have single entity)
        if self.landmark_source == "all":
            low = 0
            high = num_total
        elif self.landmark_source == "system":
            assert num_system > 0, "System must have at least one sequence"
            low = 0
            high = num_system
        elif self.landmark_source == "instances":
            low = num_system + 1
            high = num_total
        else:
            raise ValueError("Invalid selection")

        # make sure we do not choose more landmarks as specified datapoints
        num_landmarks_valid = min(self.num_landmarks, high - low)

        if self.landmark_selection_mode == "random":
            landmarks = np.random.default_rng().choice(
                np.arange(low, high), size=num_landmarks_valid, replace=False
            )
        elif self.landmark_selection_mode == "mmseqs":
            # we only have a single entity right now so can use sequences from above without merging
            # across entities; adjust target number of sequences by number of brackets
            # brackets = [0.2, 0.4, 0.6, 0.8, 1.0]
            brackets = [0, 1]
            landmarks = filter_sequences_mmseqs(
                entity_sequences[low:high],
                target_num_sequences=int(num_landmarks_valid / (len(brackets) - 1)),
                brackets=brackets,
                mmseqs_path=self.mmseqs_path
            )
        else:
            raise ValueError("Invalid mode")

        return landmarks

    def _distance_matrix(
        self,
        system: System,
        collected_sequences: CollectedSequences,
        landmarks: np.ndarray[tuple[int], int],
        default_value: int = -1
    ) -> np.ndarray[tuple[int, int], float]:
        """
        Compute distance matrix from set of instance/system sequences
        (potentially for multiple entities)

        Parameters
        ----------
        system
            System for which sequences are analyzed
        collected_sequences
            Extracted rep sequences for all entities
        landmarks:
            Indices of landmark sequences among collected sequences
        default_value
            Default value to map gaps/non-standard symbols to

        Returns
        -------
        Distance matrix of shape len(collected_sequences) x num_components
        """
        if self.distance_matrix_mode == "aligned":
            # map sequences for each entity to integer array for numba computation
            entity_arrays = [
                map_array(
                    str_to_np_char_view(seqs),
                    # do not map gaps so we can easily exclude with same value in numba calculation
                    index_map(system[entity].alphabet(include_gap=False), default_value=default_value),
                )
                for (entity, seqs, _) in collected_sequences
            ]

            # merge array together across entities
            array_merged = np.concatenate(entity_arrays + entity_arrays, axis=1)

            # compute distance matrix with numba
            dist_matrix = hamming_distance_no_gaps_ref_vs_comp(
                array_merged[landmarks, :], array_merged, default_value
            )
        elif self.distance_matrix_mode == "mmseqs":
            # for now only single entity supported
            _, entity_sequences, _ = collected_sequences[0]
            landmarks_set = set(landmarks)
            landmark_sequences = [
                seq for idx, seq in enumerate(entity_sequences) if idx in landmarks_set
            ]

            # sequences will be dealigned and converted to uppercase inside this function again if needed
            dist_matrix = distance_matrix_mmseqs_ref_vs_comp(
                ref_sequences=landmark_sequences,
                comp_sequences=entity_sequences,
                mmseqs_path=self.mmseqs_path
            )

            # symmetrize
            dist_matrix_lm_sym = 0.5 * ( dist_matrix[:, landmarks] + dist_matrix[:, landmarks].T)
            np.fill_diagonal(dist_matrix_lm_sym, 0.0)
            dist_matrix[:, landmarks] = dist_matrix_lm_sym
        else:
            raise ValueError("Invalid mode")

        return dist_matrix

    def _project(
        self,
        dist_matrix: np.ndarray[tuple[int, int], float],
        landmarks: np.ndarray[tuple[int], int]
    ) -> np.ndarray:
        if self.mds_kwargs is None:
            params = {
                "normalized_stress": "auto"
            }
        else:
            params = self.mds_kwargs

        # following https://github.com/debbiemarkslab/sequenceMDS
        embedding = MDS(
            n_components=self.num_components,
            dissimilarity="precomputed",
            **params,
        )

        num_landmarks, num_points = dist_matrix.shape
        all_coords = np.zeros((num_points, self.num_components))

        # project landmarks via MDS
        landmark_proj = embedding.fit_transform(dist_matrix[:, landmarks])

        # assign these coordinates directly
        all_coords[landmarks, :] = landmark_proj[:, :]

        # project non-landmarks using inverse distance weighting (note this
        # slightly different from original MDS with ues triangulation)
        landmarks_set = set(landmarks)
        for j in range(num_points):
            if j not in landmarks_set:
                distances = dist_matrix[:, j]  # distances from all landmarks to point j
                weights = 1.0 / (distances + 1e-10)
                weights = weights / weights.sum()
                all_coords[j, :] = np.average(landmark_proj, weights=weights, axis=0)

        return all_coords

    def distances_and_projection(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int], CollectedSequences]:
        """
        Perform sequence space projection analysis, returning results directly
        (e.g. for interactive analysis)

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which projections should be computed
        entity
            Index of entity based on which projections should be computed
            (if None, use all applicable entities)

        Returns
        -------
        Tuple containing main results from analysis:
        (i) Distance matrix
        (ii) Projection of shape num_sequences x num_components; system sequences will be first
        (iii) Selected entities
        (iv) Collected sequences
        """
        [
            system.valid_instance(
                instance,
                validate_reps=True,
                fixed_length=self._fixed_length_required,
                allow_deletions=True,
                raise_invalid=True,
            ) for instance in instances
        ]

        # determine selected entities (for regular MDS, can do all types of biopolymers);
        # this will also check if entity types are valid (can only do protein for MMseqs)
        entities = self._select_entities(
            system, entity
        )

        if self._mmseqs_required and len(entities) > 1:
            raise ValueError(
                "mmseqs mode can currently only handle single entity"
            )

        # assemble instance data as needed, also verify they are aligned for all methods but landmark_mds_mmseqs;
        # note we also need aligned sequences for MMseqs landmark selection for stable clustering performance
        require_aligned = self.landmark_selection_mode != "random" or self.distance_matrix_mode != "mmseqs"
        sequences = self._collect_sequences(
            system,
            instances,
            entities,
            require_aligned=require_aligned
        )

        # select landmark points (depending on specified mode)
        landmarks = self._select_landmarks(sequences)

        # compute distance matrix
        dist_matrix = self._distance_matrix(system, sequences, landmarks)

        # perform projection of landmark points
        projections = self._project(dist_matrix, landmarks)

        return dist_matrix, projections, entities, sequences