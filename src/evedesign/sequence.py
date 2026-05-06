"""
Biopolymer sequence functionality (protein sequences etc.)
"""
from string import ascii_lowercase
from typing import Any, Literal, Self, TextIO
from pathlib import Path
from collections import abc
from evedesign.constants import MASK, GAP
from evedesign.types import BioPolymer, RepSequence, SequenceMetadata
from evedesign.utils import shorten


REMOVE_INSERTIONS_TRANSLATION = str.maketrans("", "", ascii_lowercase + ".")


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