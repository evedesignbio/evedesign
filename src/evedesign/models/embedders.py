from typing import Any, Literal, Self, Sequence
import numpy as np

from Bio.Align import substitution_matrices

from evedesign.constants import GAP, VALID_AA_SORTED
from evedesign.model import BaseModel, Transformer
from evedesign.system import Entity, System, SystemInstance
from evedesign.types import StatusCallback


class OneHotEmbedder(BaseModel, Transformer):
    """
    Model wrapper that transforms biopolymer sequences into per-residue one-hot embeddings.

    Each biopolymer entity is encoded against an alphabet that always includes the gap symbol:
      - protein: 20 canonical amino acids + gap (21 columns)
      - dna: 4 canonical nucleotides + gap (5 columns)
      - rna: 4 canonical nucleotides + gap (5 columns)

    Insertions
    ----------
    Controlled by independent_insertion_alphabet (applies to all biopolymer types):
      - False: insertions are treated as match states (each symbol is upper-cased before encoding)
      - True: insertion state are treated as their own independent alphabet

    Shared alphabet
    ------------------------
    Controlled by merge_alphabets:
      - False: every entity is encoded against its own alphabet
      - True: a single master alphabet is built by concatenating one block per unique
              biopolymer type present in the system, in canonical order (protein+dna+rna)
              Every entity's embedding has the full master width, but a given entity only
              populates the columns belonging to its own type's block.

    The resulting embedding is a 2D array of shape [len(rep), len(alphabet)] stored as
    EntityInstance.embedding.
    """
    name: str = "OneHotEmbedder"
    citations: list[str] = []

    # canonical ordering of biopolymer types when building a merged master alphabet
    _CANONICAL_TYPE_ORDER: list[str] = ["protein", "dna", "rna"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        merge_alphabets: bool = True,
        independent_insertion_alphabet: bool = False,
    ):
        """
        Parameters
        ----------
        merge_alphabets
            If True, encode every entity against a single master alphabet built by
            concatenating one alphabet block per unique biopolymer type present.
            If False, each entity uses its own type's alphabet.
        independent_insertion_alphabet
            If True, append insertion states to each biopolymer alphabet. 
            If False, insertions are folded into match states by
            upper-casing symbols before encoding.
        """
        self._system = None
        self._merge_alphabets = merge_alphabets
        self._independent_insertion_alphabet = independent_insertion_alphabet
        # per-entity-index alphabet and symbol to column lookup--constructed in build()
        self._entity_to_alphabet: dict[int, list[str]] = {}
        self._symbol_to_index: dict[int, dict[str, int]] = {}

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    def _entity_alphabet(self, entity: Entity) -> list[str]:
        """
        Alphabet for a single biopolymer entity: canonicals + gap, plus dedicated
        insertion states when independent_insertion_alphabet is True.
        """
        return entity.alphabet(
            include_gap=True,
            include_inserts=self._independent_insertion_alphabet,
        )

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not take data"

        if len(system) == 0:
            return False, "System must contain at least one entity"

        if not all(entity.is_biopolymer() for entity in system):
            return False, "All entities in the system must be biopolymers to encode"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,  # noqa
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system

        if self._merge_alphabets:
            self._build_merged_alphabets(system)
        else:
            self._entity_to_alphabet = {
                entity_idx: self._entity_alphabet(entity)
                for entity_idx, entity in enumerate(system)
                if entity.is_biopolymer()
            }
            self._symbol_to_index = {
                entity_idx: {symbol: col for col, symbol in enumerate(alphabet)}
                for entity_idx, alphabet in self._entity_to_alphabet.items()
            }

        return self

    def _build_merged_alphabets(self, system: System) -> None:
        """
        Build a single master alphabet by concatenating one block per unique biopolymer
        type present. Every entity is mapped to the master alphabet.
        """
        # one representative entity per unique type, preserving canonical order
        type_to_entity: dict[str, Entity] = {}
        for entity in system:
            if entity.is_biopolymer() and entity.type not in type_to_entity:
                type_to_entity[entity.type] = entity
        ordered_types = [t for t in self._CANONICAL_TYPE_ORDER if t in type_to_entity]

        master_alphabet: list[str] = []
        type_offset: dict[str, int] = {}
        type_alphabet: dict[str, list[str]] = {}
        for entity_type in ordered_types:
            block = self._entity_alphabet(type_to_entity[entity_type])
            type_offset[entity_type] = len(master_alphabet)
            type_alphabet[entity_type] = block
            master_alphabet = master_alphabet + block

        self._entity_to_alphabet = {}
        self._symbol_to_index = {}
        for entity_idx, entity in enumerate(system):
            if not entity.is_biopolymer():
                continue
            offset = type_offset[entity.type]
            block = type_alphabet[entity.type]
            # every entity shares the full master width...
            self._entity_to_alphabet[entity_idx] = master_alphabet
            # (but populates columns within its own type block)
            self._symbol_to_index[entity_idx] = {
                symbol: offset + col for col, symbol in enumerate(block)
            }

    def _one_hot(self, entity_idx: int, rep: np.ndarray) -> np.ndarray:
        """
        One-hot encode a entity representation into a [len(rep), len(alphabet)] one-hot array.
        """
        alphabet = self._entity_to_alphabet[entity_idx]
        symbol_to_index = self._symbol_to_index[entity_idx]
        encoding = np.zeros((len(rep), len(alphabet)), dtype=np.float32)

        for pos, symbol in enumerate(rep):
            symbol = str(symbol)
            # when there is no dedicated insertion alphabet, inserts uppercased for encoding
            if not self._independent_insertion_alphabet:
                symbol = symbol.upper()
            col = symbol_to_index.get(symbol)
            if col is None:
                raise ValueError(
                    f"Symbol {symbol!r} at position {pos} of entity {entity_idx} is not part of its "
                    f"one-hot alphabet {alphabet}"
                )
            encoding[pos, col] = 1.0

        return encoding

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Transform system instances by adding one-hot embeddings for their biopolymer entities.

        If entity is None, all biopolymer entities in the system are encoded; otherwise only the
        selected entity is encoded (must be a biopolymer).
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # determine entities to encode
        if entity is not None:
            if not 0 <= entity < len(self.system):
                raise ValueError(f"Invalid entity index: {entity}")
            if not self.system[entity].is_biopolymer():
                raise ValueError(
                    f"Entity {entity} is of type {self.system[entity].type!r}, can only one-hot biopolymers"
                )
            target_entities = [entity]
        else:
            target_entities = sorted(self._entity_to_alphabet)

        transformed_instances = []
        for inst_idx, instance in enumerate(instances):
            # shallow copy to avoid mutating input
            new_instance = instance.copy()

            for entity_idx in target_entities:
                entity_instance = new_instance[entity_idx]
                if entity_instance.rep is None:
                    raise ValueError(
                        f"Entity {entity_idx} of instance {inst_idx} has no rep to encode"
                    )
                entity_instance.embedding = self._one_hot(entity_idx, entity_instance.rep)

            transformed_instances.append(new_instance)

            # if people are curious
            if status_callback is not None:
                progress = ((inst_idx + 1) / len(instances)) * 500
                status_callback(
                    "running", progress, f"One-hot encoded instance {inst_idx + 1}/{len(instances)}"
                )

        return transformed_instances


class BLOSUMEmbedder(BaseModel, Transformer):
    """
    Model wrapper that transforms protein sequences into per-residue BLOSUM embeddings.

    Each residue is encoded as its row in the chosen BLOSUM substitution matrix, restricted
    to the 20 canonical amino acids: the vector of substitution scores of that residue against
    each canonical amino acid (in VALID_AA_SORTED order)

    Gaps (deletions, GAP symbol) are encoded as 20 zeros

    Insertions
    ----------
    Insertion states are turned into match states by upper-casing each
    symbol before encoding, so an inserted residue receives the same BLOSUM row as its
    upper-case match counterpart

    This embedder only supports protein entities (BLOSUM matrices are defined over amino acids)
    """
    name: str = "BLOSUMEmbedder"
    citations: list[str] = [
        "10.1073/pnas.89.22.10915"
    ]

    # Bio.Align.substitution_matrices
    _SUPPORTED_MATRICES: list[str] = ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        matrix: Literal["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90"] = "BLOSUM62",
    ):
        """
        Parameters
        ----------
        matrix
            Name of the BLOSUM substitution matrix to use (ex. "BLOSUM62", "BLOSUM50")
        """
        matrix = matrix.upper()
        if matrix not in self._SUPPORTED_MATRICES:
            raise ValueError(
                f"Unsupported BLOSUM matrix {matrix!r}, valid options are {self._SUPPORTED_MATRICES}"
            )
        self._matrix_name = matrix
        self._system = None
        # per-symbol BLOSUM embedding lookup (20 canonical AAs + gap), constructed in build()
        self._symbol_to_vector: dict[str, np.ndarray] = {}

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    def _build_lookup(self) -> dict[str, np.ndarray]:
        """
        Build a symbol to BLOSUM embedding dict over the 20 canonical amino acids (plus gap)
        """
        matrix = substitution_matrices.load(self._matrix_name)

        lookup: dict[str, np.ndarray] = {
            aa: np.array(
                [matrix[aa, other] for other in VALID_AA_SORTED], dtype=np.float32
            )
            for aa in VALID_AA_SORTED
        }
        # deletions encoded as zeros
        lookup[GAP] = np.zeros(len(VALID_AA_SORTED), dtype=np.float32)
        return lookup

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not take data"

        if len(system) == 0:
            return False, "System must contain at least one entity"

        if not any(entity.type == "protein" for entity in system):
            return False, "System must contain at least one protein entity to encode"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,  # noqa
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        self._symbol_to_vector = self._build_lookup()

        return self

    def _blosum(self, rep: np.ndarray) -> np.ndarray:
        """
        BLOSUM-encode a protein entity representation into a [len(rep), 20] array.
        """
        encoding = np.zeros((len(rep), len(VALID_AA_SORTED)), dtype=np.float32)

        for pos, symbol in enumerate(rep):
            # insertions are turned into match states by upper-casing
            symbol = str(symbol).upper()
            vector = self._symbol_to_vector.get(symbol)
            if vector is None:
                raise ValueError(
                    f"Symbol {symbol!r} at position {pos} is not a canonical amino acid or gap"
                    f"and cannot be BLOSUM-encoded"
                )
            encoding[pos] = vector

        return encoding

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Transform system instances by adding BLOSUM embeddings for their protein entities.

        If entity is None, all protein entities in the system are encoded; otherwise only the
        selected entity is encoded (must be a protein).
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # determine entities to encode
        if entity is not None:
            if not 0 <= entity < len(self.system):
                raise ValueError(f"Invalid entity index: {entity}")
            if self.system[entity].type != "protein":
                raise ValueError(
                    f"Entity {entity} is of type {self.system[entity].type!r}, can only BLOSUM-encode proteins"
                )
            target_entities = [entity]
        else:
            target_entities = [
                entity_idx
                for entity_idx, entity_obj in enumerate(self.system)
                if entity_obj.type == "protein"
            ]

        transformed_instances = []
        for inst_idx, instance in enumerate(instances):
            # shallow copy to avoid mutating input
            new_instance = instance.copy()

            for entity_idx in target_entities:
                entity_instance = new_instance[entity_idx]
                if entity_instance.rep is None:
                    raise ValueError(
                        f"Entity {entity_idx} of instance {inst_idx} has no rep to encode"
                    )
                entity_instance.embedding = self._blosum(entity_instance.rep)

            transformed_instances.append(new_instance)

            # if people are curious
            if status_callback is not None:
                progress = ((inst_idx + 1) / len(instances)) * 500
                status_callback(
                    "running", progress, f"BLOSUM encoded instance {inst_idx + 1}/{len(instances)}"
                )

        return transformed_instances
