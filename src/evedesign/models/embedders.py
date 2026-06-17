from typing import Any, Self, Sequence
import numpy as np

from evedesign.constants import GAP
from evedesign.model import BaseModel, Transformer
from evedesign.system import Entity, System, SystemInstance
from evedesign.types import StatusCallback


# Sandberg et al. (1998) J Med Chem 41(14):2481-2491. 
# z1 (hydrophobicity), z2 (steric bulk/polarizability), 
# z3 (polarity/charge), z4 and z5 (electronic effects).
ZSCALES: dict[str, tuple[float, float, float, float, float]] = {
    "A": (0.24, -2.32, 0.60, -0.14, 0.30),
    "R": (3.52, 2.50, -3.50, 1.99, -0.17),
    "N": (3.05, 1.62, 1.04, -1.15, 1.61),
    "D": (3.98, 0.93, 1.93, -2.46, 0.75),
    "C": (0.84, -1.67, 3.71, 0.18, -2.65),
    "Q": (1.75, 0.50, -1.44, -1.34, 0.66),
    "E": (3.11, 0.26, -0.11, -3.04, -0.25),
    "G": (2.05, -4.06, 0.36, -0.82, -0.38),
    "H": (2.47, 1.95, 0.26, 3.90, 0.09),
    "I": (-3.89, -1.73, -1.71, -0.84, 0.26),
    "L": (-4.28, -1.30, -1.49, -0.72, 0.84),
    "K": (2.29, 0.89, -2.49, 1.49, 0.31),
    "M": (-2.85, -0.22, 0.47, 1.94, -0.98),
    "F": (-4.22, 1.94, 1.06, 0.54, -0.62),
    "P": (-1.66, 0.27, 1.84, 0.70, 2.00),
    "S": (2.39, -1.07, 1.15, -1.39, 0.67),
    "T": (0.75, -2.18, -1.12, -1.46, -0.40),
    "W": (-4.36, 3.94, 0.59, 3.44, -1.59),
    "Y": (-2.54, 2.44, 0.43, 0.04, -1.47),
    "V": (-2.59, -2.64, -1.54, -0.85, -0.02),
}


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


class ZscaleEmbedder(BaseModel, Transformer):
    """
    Model wrapper that transforms protein sequences into per-residue z-scale embeddings.

    Deletions
    ---------
    Gap symbols are embedded as five zeros

    Insertions
    ----------
    Insertion states are turned into match states by upper-casing

    The resulting embedding is a 2D array of shape [len(rep), 5]
    """
    name: str = "ZscaleEmbedder"
    citations: list[str] = [
        "10.1021/jm9700575"
    ]

    # dimensionality of the z-scale descriptor vector per residue
    _N_SCALES: int = 5

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

    def __init__(self):
        self._system = None
        # protein entity indices to embed--constructed in build()
        self._protein_entities: list[int] = []
        # precomputed symbol to z-scale vector lookup
        self._zscale_vectors: dict[str, np.ndarray] = {
            symbol: np.asarray(vector, dtype=np.float32)
            for symbol, vector in ZSCALES.items()
        }

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not take data"

        if len(system) == 0:
            return False, "System must contain at least one entity"

        if not all(entity.type == "protein" for entity in system):
            return False, "All entities in the system must be of type 'protein' to z-scale encode"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,  # noqa
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        self._protein_entities = [
            entity_idx for entity_idx, entity in enumerate(system)
            if entity.type == "protein"
        ]
        return self

    def _zscale(self, entity_idx: int, rep: np.ndarray) -> np.ndarray:
        """
        Encode an entity representation into a [len(rep), 5] z-scale array. Gap symbols
        map to an all-zeros vector
        """
        encoding = np.zeros((len(rep), self._N_SCALES), dtype=np.float32)

        for pos, symbol in enumerate(rep):
            symbol = str(symbol)
            if symbol == GAP:
                continue
            vector = self._zscale_vectors.get(symbol.upper())
            if vector is None:
                raise ValueError(
                    f"Symbol {symbol!r} at position {pos} of entity {entity_idx} has no "
                    f"z-scale descriptor (valid symbols: {sorted(ZSCALES)})"
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
        Transform system instances by adding z-scale embeddings for their protein entities.

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
                    f"Entity {entity} is of type {self.system[entity].type!r}, can only z-scale proteins"
                )
            target_entities = [entity]
        else:
            target_entities = self._protein_entities

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
                entity_instance.embedding = self._zscale(entity_idx, entity_instance.rep)

            transformed_instances.append(new_instance)

            # if people are curious
            if status_callback is not None:
                progress = ((inst_idx + 1) / len(instances)) * 500
                status_callback(
                    "running", progress, f"Z-scale encoded instance {inst_idx + 1}/{len(instances)}"
                )

        return transformed_instances
