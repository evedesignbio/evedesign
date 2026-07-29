"""
Biopolymer sequence functionality (protein sequences etc.)
"""
from string import ascii_lowercase
from typing import Any, Literal, Self, TextIO
from pathlib import Path
from collections import abc

import numpy as np
from numba import jit, prange, get_num_threads, set_num_threads

from evedesign.constants import (
    MASK,
    GAP,
    VALID_AA_OR_GAP_SORTED,
    VALID_DNA_OR_GAP_SORTED,
    VALID_RNA_OR_GAP_SORTED,
)
from evedesign.types import BioPolymer, RepSequence, SequenceMetadata
from evedesign.utils import shorten, str_to_np_char_view, map_array, index_map


REMOVE_INSERTIONS_TRANSLATION = str.maketrans("", "", ascii_lowercase + ".")

ALPHABETS_OR_GAP_SORTED: dict[str, list[str]] = {
    "protein": VALID_AA_OR_GAP_SORTED,
    "dna": VALID_DNA_OR_GAP_SORTED,
    "rna": VALID_RNA_OR_GAP_SORTED,
}


@jit(nopython=True, parallel=True)
def _num_cluster_members(matrix, identity_threshold, exclude_value):
    """
    Calculate number of sequences in alignment
    within given identity_threshold of each other

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    exclude_value : int
        Value >= 0 in matrix that will be excluded from identity calculation, e.g. gap or lowercase character.
        Set to -1 to enable legacy behaviour which includes gaps in identity calculation.

    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    num_neighbors = np.zeros((N, ))
    L_seq = np.sum(matrix != exclude_value, axis=1)

    for i in prange(N):
        num_neighbors_i = 1 
        for j in range(N):
            if i == j:
                continue
            matches = 0
            for k in range(L):
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != exclude_value:
                    matches += 1
            if matches / L_seq[i] >= identity_threshold:
                num_neighbors_i += 1
        num_neighbors[i] = num_neighbors_i

    return num_neighbors


class Sequence:
    """
    Single biopolymer sequence (may include gaps and inserts in lowercase)

    # TODO: add methods for sequence verification and transformation
    # TODO: add attributes for description and any other relevant metadata
    """
    def __init__(
        self,
        seq: str,
        id: str | None = None,  # noqa
        key: str | None = None,
        type: BioPolymer = "protein",  # noqa
        metadata: SequenceMetadata | None = None,
    ):
        """
        Create new sequence object

        Parameters
        ----------
        seq
            Sequence (can contain lowercase characters and gaps)
        id
            Identifier of sequence
        key
            Key for matching sequence to other resources (e.g. paired alignment)
        type
            Type of biopolymer sequence (protein, rna, dna, ...)
        metadata
            Optional sequence metadata (embeddings, taxonomy, ...)
        """
        self.seq = seq
        self.id_ = id
        self.key = key
        self.type_ = type
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.id_} key={self.key} type={self.type_} seq={shorten(self.seq)})"
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize sequence into JSON-compatible representation

        Returns
        -------
        Serialized sequence representation
        """
        return {
            "seq": self.seq,
            "id": self.id_,
            "key": self.key,
            "type": self.type_,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, serialized_seq: dict[str, Any]) -> Self:
        """
        Deserialize JSON-compatible representation into Sequence object

        Parameters
        ----------
        serialized_seq
            Serialized sequence representation

        Returns
        -------
        Deserialized Sequence object
        """
        return cls(
            seq=serialized_seq.get("seq"),
            id=serialized_seq.get("id"),
            key=serialized_seq.get("key"),
            type=serialized_seq.get("type"),
            metadata=serialized_seq.get("metadata")
        )

    def remove_insertions(self) -> Self:
        """
        Return updated version of sequence with any insertions (lowercase letters)
        removed

        Returns
        -------
        Updated sequence without insertions
        """
        return type(self)(
            seq=self.seq.translate(REMOVE_INSERTIONS_TRANSLATION),
            id=self.id_,
            key=self.key,
            type=self.type_,
            metadata=self.metadata.copy() if self.metadata is not None else None
        )

    def dealign(self) -> Self:
        """
        Remove alignment information from sequence (removing gaps,
        converting insert positions to uppercase letters)

        Returns
        -------
        Dealigned sequence
        """
        return type(self)(
            seq=self.seq.replace(GAP, "").upper(),
            id=self.id_,
            key=self.key,
            type=self.type_,
            metadata=self.metadata.copy() if self.metadata is not None else None
        )


class Sequences:
    """
    Collection of one or more biopolymer sequences, can be aligned or unaligned

    This class only intends to be a thin wrapper around different alignment formats
    to connect input sequences to the different types of formats expected by individual methods,
    rather than a full-fledged class for computations on sequence alignments

    Note: weights are a property of sequence list (relative weights of sequences to each other),
     not of individual sequences on purpose.
    """
    def __init__(
        self,
        seqs: abc.Sequence[Sequence],
        aligned: bool = False,
        type: BioPolymer = "protein",  # noqa
        weights: abc.Sequence[float] | None = None,
        format: Literal["a3m", "a2m", "fasta", "fasta_unaligned"] | None = None,  # noqa
    ):
        self.seqs = seqs
        self.aligned = aligned
        self.type_ = type
        self.weights = weights
        self.format_ = format
        # TODO: check alignment integrity and/or autodetect properties/format

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        format: Literal["a3m", "a2m", "fasta", "fasta_unaligned"],
        type: BioPolymer = "protein"
    ) -> Self:
        """
        Load sequences from a file.
        """
        file_path = Path(path)
        seq_list = []
        expected_match_len = None

        aligned = format != "fasta_unaligned"

        with open(file_path, "r") as f:
            for seq_id, seq_str in read_fasta(f):
                # only perform match state checking for aligned formats
                if aligned:
                    match_seq = seq_str.translate(REMOVE_INSERTIONS_TRANSLATION)
                    current_match_len = len(match_seq)

                    if expected_match_len is None:
                        expected_match_len = current_match_len
                    elif current_match_len != expected_match_len:
                        raise ValueError(
                            f"Inconsistent alignment length in {file_path.name}: "
                            f"'{seq_id}' has {current_match_len} match states, expected {expected_match_len}."
                        )

                seq_list.append(
                    Sequence(seq=seq_str, id=seq_id, type=type)
                )

        return cls(
            seqs=seq_list,
            aligned=aligned,
            type=type,
            format=format,
        )
        
    def remove_inserts(self) -> Self:
        """
        Remove any insertions (lowercase letters or periods) from all sequences relative to target
        """
        
        if self.format_ == 'fasta':
            raise NotImplementedError(f"remove_inserts is not supported for format: {self.format_}")
        
        return type(self)(
            seqs=[s.remove_insertions() for s in self.seqs],
            aligned=True,
            weights=self.weights,
            format=self.format_
        )

    def compute_weights(
        self,
        theta: float = 0.8,
        method: Literal["theta_nogaps", "theta_withgaps"] = "theta_nogaps",
        cpu: int | None = None,
    ) -> Self:
        """
        Compute per-sequence weights and return a copy with the weights set

        Does not mutate the current object. The returned Sequences shares the same
        underlying Sequence objects with weights replaced

        Parameters
        ----------
        theta
            Sequence identity threshold for clustering
        method
            theta_nogaps excludes gaps from identity calculation,
            theta_withgaps includes them 
        cpu
            Number of numba threads (None uses all)

        Returns
        -------
        A copy of this Sequences object with weights set
        """
        match_seqs = [s.seq for s in self.to_a3m().remove_inserts().seqs]

        alphabet = ALPHABETS_OR_GAP_SORTED.get(
            self.type_, ALPHABETS_OR_GAP_SORTED["protein"]
        )
        mapping = index_map(list(alphabet), default_option=GAP)
        matrix = map_array(str_to_np_char_view(match_seqs), mapping)

        exclude_value = mapping[GAP] if method == "theta_nogaps" else -1

        threads_before = None
        if cpu is not None:
            threads_before = get_num_threads()
            set_num_threads(cpu)
        try:
            num_cluster_members = _num_cluster_members(matrix, theta, exclude_value)
        finally:
            if threads_before is not None:
                set_num_threads(threads_before)

        weights = [float(w) for w in 1.0 / num_cluster_members]

        return type(self)(
            seqs=self.seqs,
            aligned=self.aligned,
            type=self.type_,
            weights=weights,
            format=self.format_,
        )

    def remap_query(
        self,
        old_query: str | RepSequence,
        new_query: str | RepSequence,
        prepend_new_query: bool = True,
    ) -> "Sequences":
        """
        Remap this alignment to a new query sequence.

        Returns a new Sequences with the same hits but with columns added/removed
        to match the new query's insertions and deletions relative to the old query.

        Used to reuse one MSA across many SystemInstances without re-querying the
        MMSeqs2 server: search once on the system rep, remap per-instance.

        Parameters
        ----------
        old_query
            The query that produced the current alignment (the system rep or
            whatever was used to populate self).
        new_query
            The new query in A3M convention: uppercase for alignment columns,
            '-' for deletions, lowercase for insertions.
        prepend_new_query
            If True (default), the new query is inserted as the first sequence
            of the returned alignment. Most tools expect the query in the first
            position by convention, so this saves callers from prepending it
            themselves. The query is stored in ungapped, uppercase form (gaps
            removed, insertions uppercased) — i.e. the actual designed residues.

        Returns
        -------
        Sequences
            A new Sequences containing the same hits with columns remapped,
            optionally with the new query as the first sequence.
            Format is preserved.

        Raises
        ------
        NotImplementedError
            If self.format_ is not in {"a3m", "a2m"}.
        ValueError
            If new_query's match-column count (uppercase + gap chars) doesn't
            match old_query's match-column count.
        ValueError
            If any hit's match-column count (non-lowercase chars) doesn't match
            old_query's match-column count.
        """
        if self.format_ not in {"a3m", "a2m"}:
            raise NotImplementedError(
                f"remap_query is not supported for format: {self.format_}"
            )

        # accept both str and RepSequence (numpy U1 array)
        old_q = "".join(old_query)
        new_q = "".join(new_query)

        # match columns = uppercase (aligned) or gap; lowercase = insert (0 cols)
        def match_columns(s: str) -> int:
            return sum(1 for ch in s if not ch.islower())

        n_cols = match_columns(old_q)

        if match_columns(new_q) != n_cols:
            raise ValueError(
                f"new_query spans {match_columns(new_q)} match columns "
                f"but old_query has {n_cols}"
            )

        for hit in self.seqs:
            hit_cols = match_columns(hit.seq)
            if hit_cols != n_cols:
                raise ValueError(
                    f"hit '{hit.id_}' has {hit_cols} match columns, "
                    f"expected {n_cols} to match old_query"
                )

        def consume_match_column(hit_str: str, pos: int) -> tuple[str, str, int]:
            # carry any leading lowercase insert run, then take one match char
            start = pos
            while pos < len(hit_str) and hit_str[pos].islower():
                pos += 1
            insert_run = hit_str[start:pos]
            if pos >= len(hit_str):
                raise ValueError("hit has fewer match columns than old_query")
            return insert_run, hit_str[pos], pos + 1

        remapped = []
        for hit in self.seqs:
            hit_str = hit.seq
            out: list[str] = []
            p = 0
            for c in new_q:
                if c.islower():
                    # new-query insertion: hits have no residue here
                    out.append(GAP)
                elif c == GAP:
                    # deletion: keep the hit's insert run, drop the match residue
                    insert_run, _col, p = consume_match_column(hit_str, p)
                    out.append(insert_run)
                else:
                    # substitution/match: carry insert run + residue
                    insert_run, col, p = consume_match_column(hit_str, p)
                    out.append(insert_run)
                    out.append(col)

            # carry any trailing (C-terminal) insert run
            while p < len(hit_str) and hit_str[p].islower():
                out.append(hit_str[p])
                p += 1

            if p != len(hit_str):
                raise ValueError(
                    f"hit '{hit.id_}' has more match columns than old_query"
                )

            remapped.append(
                type(hit)(
                    seq="".join(out), id=hit.id_, key=hit.key,
                    type=hit.type_, metadata=hit.metadata,
                )
            )

        if prepend_new_query:
            # Store the query as the actual designed residues: gaps (deletions)
            # dropped and insertions uppercased, matching the match columns the
            # remapped hits are now aligned to.
            query_residues = new_q.replace(GAP, "").upper()
            seq_cls = type(remapped[0]) if remapped else Sequence
            query_type = remapped[0].type_ if remapped else "protein"
            remapped.insert(0, seq_cls(seq=query_residues, type=query_type))

        return type(self)(
            seqs=remapped,
            aligned=True,
            weights=self.weights,
            format=self.format_,
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize sequences into JSON-compatible representation

        Returns
        -------
        Serialized sequences
        """
        return {
            "seqs": [seq.serialize() for seq in self.seqs],
            "aligned": self.aligned,
            "type": self.type_,
            "weights": self.weights,
            "format": self.format_,
        }

    @classmethod
    def deserialize(cls, serialized_seqs: dict[str, Any]) -> Self:
        """
        Deserialize JSON-compatible representation of multiple sequences
        into Sequences object

        Parameters
        ----------
        serialized_seqs
            Serialized representation of sequences

        Returns
        -------
        Deserialized Sequence object
        """
        return cls(
            seqs=[Sequence.deserialize(seq) for seq in serialized_seqs["seqs"]],
            aligned=serialized_seqs.get("aligned"),
            type=serialized_seqs.get("type"),
            weights=serialized_seqs.get("weights"),
            format=serialized_seqs.get("format"),
        )

    def dealign(self) -> Self:
        # remove gaps from sequences and return new
        raise NotImplementedError(
            "Sequence dealigning not yet implemented"
        )

    def to_a3m(self) -> Self:
        # return sequences in a3m format
        if self.format_ == "a3m":
            return self
        else:
            raise NotImplementedError(
                "Conversion to a3m format not yet implemented"
            )

    def to_a2m(self) -> Self:
        # return sequences in a2m format
        # TODO: add parameter to specify strategy how to deal with inserts (drop or fully expand sequences)
        #  cf. https://github.com/debbiemarkslab/EVcouplings/blob/75bfc9677fc9412ddb7089a9f26c7a01f65bfa12/evcouplings/align/alignment.py#L236
        if self.format_ == "a2m":
            return self
        else:
            raise NotImplementedError(
                "Conversion into a2m format not yet implemented"
            )

    def to_fasta(self) -> Self:
        if self.format_ == "fasta":
            return self
        else:
            raise NotImplementedError(
                "Conversion into fasta format not yet implemented"
            )

def valid_sequence(
    seq: str | RepSequence,
    alphabet: abc.Sequence[str],
    allow_mask: bool = False,
) -> tuple[bool, list[tuple[int, str]]]:
    """
    Check if a given sequence is valid according to some alphabet

    Parameters
    ----------
    seq
        Sequence to validate
    alphabet
        Valid symbols (may contain GAP and insert symbols)
    allow_mask
        If true, allow masked positions in the sequence

    Returns
    -------
    bool
        True if valid sequence, False otherwise
    list[tuple[int, str]]
        Invalid characters and their zero-based indices in sequence
    """
    alphabet = set(alphabet)

    invalid = [
        (i, symbol) for i, symbol in enumerate(seq) if not (
            symbol in alphabet or
            (allow_mask and symbol == MASK)
        )
    ]

    return len(invalid) == 0, invalid


# TODO: following is legacy function superseded by valid_sequence(), remove eventually
# def valid_protein_sequence(
#     seq: str,
#     allow_mask: bool = False,
#     allow_gap: bool = False,
#     allow_ambiguous: bool = False,
# ) -> Tuple[bool, List[Tuple[int, str]]]:
#     """
#     Check if a given sequence is a valid protein sequence
#
#     Parameters
#     ----------
#     seq
#         Protein seqeunce
#     allow_mask
#         Consider mask character as valid symbol (default: False)
#     allow_gap
#         Consider gap character as valid symbol (default: False)
#     allow_ambiguous
#         Consider ambiguous amino acids as valid symbol (default: False)
#
#     Returns
#     -------
#     bool
#         True if valid sequence, False otherwise
#     str
#         Invalid characters and their indices in sequence
#     """
#     invalid = [
#         (i, aa) for i, aa in enumerate(seq) if not (
#             aa in AA_TO_INDEX or
#             (allow_mask and aa == MASK) or
#             (allow_gap and aa == GAP)
#         ) or (
#             not allow_ambiguous and aa in AA_TO_INDEX and INDEX_TO_AA[AA_TO_INDEX[aa]] != aa
#         )
#     ]
#
#     return len(invalid) == 0, invalid


def read_fasta(f: TextIO):
    """
    Generator function to read a FASTA-format file
    (includes aligned FASTA, A2M, A3M formats)

    Parameters
    ----------
    f : file-like object
        FASTA alignment file

    Returns
    -------
    generator of (str, str) tuples
        Returns tuples of (sequence ID, sequence)
    """
    current_sequence = ""
    current_id = None

    for line in f:
        # Start reading new entry. If we already have
        # seen an entry before, return it first.
        if line.startswith(">"):
            if current_id is not None:
                yield current_id, current_sequence

            current_id = line.rstrip()[1:]
            current_sequence = ""

        elif not line.startswith(";"):
            current_sequence += line.rstrip()

    # Also do not forget last entry in file
    yield current_id, current_sequence