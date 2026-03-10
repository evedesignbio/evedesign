"""
Specification of components of molecular design system (proteins, nucleic acids, ligands, etc.)
"""
from collections import UserList
from collections.abc import Sequence
from copy import deepcopy
from io import StringIO
from math import isclose
from typing import NamedTuple, Self, Any
import numpy as np

from evedesign.sequence import valid_sequence, Sequences
from evedesign.structure import Structure, StructureFile
from evedesign.types import (
    EntityType, Metadata, BioPolymers, RepSequence, LigandRepType, SymmetryType, BondType,
    SecondaryStructureType, Embedding
)
from evedesign.constants import (
    VALID_AA_OR_GAP_SORTED, VALID_AA_SORTED,
    VALID_DNA_OR_GAP_SORTED, VALID_DNA_SORTED,
    VALID_RNA_OR_GAP_SORTED, VALID_RNA_SORTED,
    GAP
)
from evedesign.utils import ensure_sequence, shorten

# versioning scheme for System and child entities
CURRENT_SYSTEM_SPEC_VERSION = "0.2"

# versioning scheme for SystemInstance and child entity instances
CURRENT_SYSTEM_INSTANCE_SPEC_VERSION = "0.2"

"""
Data structures/types for providing mutation information in structured format

Deletions are coded by to = GAP

Insertions are coded by 
  1. ref = "",
  2. to = lowercase insert symbols as returned by Entity.alphabet()
  3. occur directly *after* the referenced position (for insertion at beginning of sequence, use first_index - 1).
"""
Mutation = NamedTuple(
    "Mutation", [("entity", int), ("pos", int), ("ref", str), ("to", str)]
)

"""
Mutant is comprised of one or more mutations; note that all individual mutations are relative to the
sequence *before* applying any of the mutations (e.g. before any numbering shifts due to insertions).
Multiple insertions are concatenated together in the order of specification in the sequence.
"""
Mutant = Sequence[Mutation]

"""
Mapping from structure identifier to one or more models (list of models implies homo-oligomers).

Conventions:
1. Each model has to contain exactly one chain
2. Numbering must map to entity/entity instance numbering the model is attached to. Entity rep positions can be
  missing if no coordinates are available, but there must not be any positions in structure that do not map to the
  entities representative
"""
StructureChainMap = dict[str, Structure | list[Structure]]

def _rep_to_np_array(rep: RepSequence | str | None) -> RepSequence | None:
    if isinstance(rep, str):
        rep = np.array(list(rep), dtype="U1")
    else:
        if rep is not None and rep.dtype != "U1":
            raise ValueError("rep must be None, str, or have dtype 'U1'")

    return rep


def _serialize_chain_map(s: StructureChainMap | None) -> dict[str, Any] | None:
    """
    Serialize StructureChainMap to JSON-encodable representation

    Parameters
    ----------
    s
        Structure chain map to serialize

    Returns
    -------
    Serialized chain map
    """
    if s is None:
        return None

    serialized_map = {}
    for key, models in s.items():
        # default to creating a sequence for simpler handling
        models = ensure_sequence(models)
        serialized_map[key] = []
        for model in models:
            assert len(model.chains()) == 1, "Only can serialize single-chain models"
            f = StringIO()
            model.to_file(f, format="cif")
            serialized_map[key].append(f.getvalue())

    return serialized_map

def _deserialize_chain_map(s: dict[str, Any] | None) -> StructureChainMap | None:
    """
    Deserialize chain map from JSON-encodable representation to StructureChainMap object

    Parameters
    ----------
    s
        Serialized chain map

    Returns
    -------
    Deserialized StructureChainMap
    """
    if s is None:
        return None

    deserialized_map = {}
    for key, models in s.items():
        models = ensure_sequence(models)
        deserialized_map[key] = []
        for model in models:
            model_deserialized = StructureFile(
                StringIO(model), format="cif"
            ).get_model()

            # extract single chain
            chains = model_deserialized.chains()
            assert (len(chains) == 1), "Only can deserialize single-chain models"
            deserialized_map[key].append(
                model_deserialized.get_chain(chains[0])
            )

    return deserialized_map

def _serialize_optional_list(x: Sequence[Any] | None):
    if x is None:
        return None
    else:
        return [
            e.serialize() for e in x
        ]

def _deserialize_optional_list(x: Sequence[Any] | None, cls):
    if x is None:
        return None
    else:
        return [
            cls.deserialize(e) for e in x
        ]


class Interaction:
    """
    Positive/negative interactions within and between entities
    """
    def __init__(
        self,
        id: str,  # noqa
        pos: Sequence[int] | None = None,
        partner_ids: Sequence[str] | None = None,
        avoid: bool = False,
    ):
        """
        Create new interaction specification

        Parameters
        ----------
        id
            Unique interaction site identifier
        pos
            List of positions for which interaction is defined, if
            None, means entire parent (either entity or insertion).
        partner_ids
            If defined, enforce specific interactions with these
            other interaction sites (referenced by their unique id).
            Self-reference id for interactions with within entity
        avoid
            If True, avoid interactions for this site (i.e. negative interaction)
            rather than enforcing them (positive interaction)
        """
        self.id = id
        self.pos = pos
        self.partner_ids = partner_ids
        self.avoid = avoid

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, Interaction):
            return False

        return (
            self.id == other.id and
            self.pos == other.pos and
            self.partner_ids == other.partner_ids and
            self.avoid == other.avoid
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize interaction to JSON-compatible representation

        Returns
        -------
        Serialized interaction represented as dict
        """
        return {
            "id": self.id,
            "pos": self.pos,
            "partner_ids": self.partner_ids,
            "avoid": self.avoid,
        }

    @classmethod
    def deserialize(cls, interaction_dict: dict[str, Any]) -> Self:
        """
        Deserialize interaction from JSON-compatible representation to object instance

        Parameters
        ----------
        interaction_dict
            Interaction attribute map

        Returns
        -------
        Deserialized Interaction object
        """
        return cls(
            id=interaction_dict.get("id"),
            pos=interaction_dict.get("pos"),
            partner_ids=interaction_dict.get("partner_ids"),
            avoid=interaction_dict.get("avoid"),
        )


class AtomBond:
    """
    Defined interaction between two atoms
    """
    def __init__(
        self,
        type: BondType,  # noqa
        source_pos: int | None,
        source_atom: str,
        target_entity_id: str,
        target_pos: int | None,
        target_atom: str,
    ):
        """
        Create new atom bond specification

        Parameters
        ----------
        type
            Type of bond: {"covalent", "hydrogen", "vdw", "ionic"}
        source_pos
            Position in source entity (None if ligand)
        source_atom
            Atom name in current entity
        target_entity_id
            id of target entity (referencing parent entity is allowed)
        target_pos
            Position in target entity (None if ligand)
        target_atom
            Atom name in target entity
        """
        self.type = type
        self.source_pos = source_pos
        self.source_atom = source_atom
        self.target_entity_id = target_entity_id
        self.target_pos = target_pos
        self.target_atom = target_atom

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, AtomBond):
            return False

        return (
            self.type == other.type and
            self.source_pos == other.source_pos and
            self.source_atom == other.source_atom and
            self.target_entity_id == other.target_entity_id and
            self.target_pos == other.target_pos and
            self.target_atom == other.target_atom
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize atom bond to JSON-compatible representation

        Returns
        -------
        Serialized atom bond represented as dict
        """
        return {
            "type": self.type,
            "source_pos": self.source_pos,
            "source_atom": self.source_atom,
            "target_entity_id": self.target_entity_id,
            "target_pos": self.target_pos,
            "target_atom": self.target_atom,
        }

    @classmethod
    def deserialize(cls, atom_bond_dict: dict[str, Any]) -> Self:
        """
        Deserialize atom bond from JSON-compatible representation to object instance

        Parameters
        ----------
        atom_bond_dict
            Atom bond attribute map

        Returns
        -------
        Deserialized AtomBond object
        """
        return cls(
            type=atom_bond_dict.get("type"),
            source_pos=atom_bond_dict.get("source_pos"),
            source_atom=atom_bond_dict.get("source_atom"),
            target_entity_id=atom_bond_dict.get("target_entity_id"),
            target_pos=atom_bond_dict.get("target_pos"),
            target_atom=atom_bond_dict.get("target_atom"),
        )

class SecondaryStructure:
    """
    Specification of secondary structure for residues in biopolymer sequences
    """
    def __init__(
        self,
        pos: int | None,
        type: SecondaryStructureType,  # noqa
    ):
        """
        Create new secondary structure specification

        Parameters
        ----------
        pos
            Apply secondary structure to this position, or apply to all positions
            in entity (if None)
        type:
            Secondary structure element type ({"H", "E", "C"})}
        """
        self.pos = pos
        self.type = type

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, SecondaryStructure):
            return False

        return self.pos == other.pos and self.type == other.type

    def serialize(self) -> dict[str, Any]:
        """
        Serialize secondary structure specification to JSON-compatible representation

        Returns
        -------
        Serialized secondary structure specification represented as dict
        """
        return {
            "pos": self.pos,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, secondary_structure_dict: dict[str, Any]) -> Self:
        """
        Deserialize residue bias from JSON-compatible representation to object instance

        Parameters
        ----------
        secondary_structure_dict
            Secondary structure specification attribute map

        Returns
        -------
        Deserialized SecondaryStructure object
        """
        return cls(
            pos=secondary_structure_dict.get("pos"),
            type=secondary_structure_dict.get("type")
        )


class ResidueBias:
    """
    Specification of positional symbol preferences for
    biopolymer sequences
    """
    def __init__(
        self,
        pos: int | None,
        bias: dict[str, float]
    ):
        """
        Define new position-specific or global symbol preference

        Parameters
        ----------
        pos
            Apply bias to this position, or apply to all positions
            in entity (if None)
        bias
            Mapping from alphabet symbol to bias logits, -inf to exclude (e.g. cysteines)
        """
        self.pos = pos
        self.bias = bias

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, ResidueBias):
            return False

        if self.pos != other.pos:
            return False

        if set(self.bias) != set(other.bias):
            return False

        # compare floats properly item by item
        for key, value in self.bias.items():
            if not isclose(value, other.bias[key]):
                return False

        return True

    def serialize(self) -> dict[str, Any]:
        """
        Serialize residue bias to JSON-compatible representation

        Returns
        -------
        Serialized residue bias represented as dict
        """
        return {
            "pos": self.pos,
            "bias": self.bias,
        }

    @classmethod
    def deserialize(cls, residue_bias_dict: dict[str, Any]) -> Self:
        """
        Deserialize residue bias from JSON-compatible representation to object instance

        Parameters
        ----------
        residue_bias_dict
            Residue bias attribute map

        Returns
        -------
        Deserialized Modification object
        """
        return cls(
            pos=residue_bias_dict.get("pos"),
            bias=residue_bias_dict.get("bias")
        )


class Modification:
    """
    Biopolymer residue modification
    """
    def __init__(
        self,
        pos: int,
        type: str  # noqa
    ):
        """
        Create new residue modification

        Parameters
        ----------
        pos
            Entity rep position of modified residue
        type
            CCD code of modification
        """
        self.pos = pos
        self.type = type

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, Modification):
            return False

        return self.pos == other.pos and self.type == other.type

    def serialize(self) -> dict[str, Any]:
        """
        Serialize modification to JSON-compatible representation

        Returns
        -------
        Serialized modification represented as dict
        """
        return {
            "pos": self.pos,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, modification_dict: dict[str, Any]) -> Self:
        """
        Deserialize modification from JSON-compatible representation to object instance

        Parameters
        ----------
        modification_dict
            Modification attribute map

        Returns
        -------
        Deserialized Modification object
        """
        return cls(
            pos=modification_dict.get("pos"),
            type=modification_dict.get("type"),
        )


class Insertion:
    """
    Variable-length insertion in base sequence
    """
    def __init__(
        self,
        pos: int | None = None,
        min_length: int = 1,
        max_length: int | None = None,
        secondary_structure: SecondaryStructureType | None = None,
        interactions: Sequence[Interaction] | None = None,
    ):
        """
        Define new insertion

        Parameters
        ----------
        pos
            Position after which insertion can occur. Use first_index - 1
            to define N-terminal extension. If None, insertion can occur
            anywhere in entity.
        min_length
            Minimum length of insertion
        max_length
            Maximum length of insertion
        secondary_structure
            Secondary structure type of designed insert residues
        interactions
            Which interactions to enforce for designed insert (must leave pos attribute
            unspecified)
        """
        self.pos = pos
        self.min_length = min_length
        self.max_length = max_length
        self.secondary_structure = secondary_structure
        self.interactions = interactions

        if interactions is not None:
            for interaction in interactions:
                if interaction.pos is not None:
                    raise ValueError("Insertions can not specify pos for Interaction")

    def __eq__(self, other):
        # only ever accept other entities for equality
        if not isinstance(other, Insertion):
            return False

        return (
            self.pos == other.pos and
            self.min_length == other.min_length and
            self.max_length == other.max_length and
            self.secondary_structure == other.secondary_structure and
            self.interactions == other.interactions
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize insertion to JSON-compatible representation

        Returns
        -------
        Serialized insertion represented as dict
        """
        return {
            "pos": self.pos,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "secondary_structure": self.secondary_structure,
            "interactions": _serialize_optional_list(self.interactions),
        }

    @classmethod
    def deserialize(cls, insertion_dict: dict[str, Any]) -> Self:
        """
        Deserialize insertion from JSON-compatible representation to object instance

        Parameters
        ----------
        insertion_dict
            Insertion attribute map

        Returns
        -------
        Deserialized Insertion object
        """
        return cls(
            pos=insertion_dict.get("pos"),
            min_length=insertion_dict.get("min_length"),
            max_length=insertion_dict.get("max_length"),
            secondary_structure=insertion_dict.get("secondary_structure"),
            interactions=_deserialize_optional_list(
                insertion_dict.get("interactions"), Interaction
            ),
        )


class Entity:
    def __init__(
        self,
        type: EntityType,  # noqa
        rep: str | RepSequence | None = None,
        id: str | None = None,  # noqa
        first_index: int | None = None,
        sequences: Sequences | None = None,
        structures: StructureChainMap | None = None,
        ligand_rep_type: LigandRepType | None = None,
        interactions: Sequence[Interaction] | None = None,
        atom_bonds: Sequence[AtomBond] | None = None,
        modifications: Sequence[Modification] | None = None,
        copies: int | None = None,
        symmetry: SymmetryType | None = None,
        secondary_structure: Sequence[SecondaryStructure] | None = None,
        cyclic: bool | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        residue_bias: Sequence[ResidueBias] | None = None,
        insertions: Sequence[Insertion] | None = None,
        deletions: bool | None = None,
    ):
        """
        Create new generic entity for molecular system.

        Note: For clarity, preferentially use subclasses for specific types
        of entities (e.g. Protein class)

        Note: Equality does not check sequences and structures

        Parameters
        ----------
        type
            Type of entity (protein, nucleotide, ligand, ...)
        id
            Unique identifier of entity
        rep
            Representation of entity (sequence, atom name, etc.)
        first_index
            Sequence index of first residue; must be specified
            for polymer types (protein, nucleotide, ...)
        sequences
            Sequence record (e.g. multiple sequence alignment of homologs) of the target
            sequence represented by this entity (only applies to proteins and nucleotides)
        structures
            Structure chains representing this entity. Use dict with structure identifiers
            as keys to supply multiple different structures; use list to supply multiple copies
            of the chain within the structure (homooligomer)
        ligand_rep_type
            Type of ligand rep specification (ligands only)
        interactions
            Positive/negative interactions within and between entities
        atom_bonds:
            Defined interactions between pairs of atoms
        modifications:
            Biopolymer residue modifications (must be None for ligand)
        copies
            Number of entity copies in molecular system. Set to None
            to leave variable.
        symmetry:
            Type of structural symmetry (cyclic, dihedral, tetrahedral, ...)
            entity should assume, must be specified together with copies attribute
        secondary_structure:
            Secondary structure assignment per position or globally
        cyclic:
            If True, make biopolymer cyclic (must be None for ligands)
        min_length:
            Minimum length of designed polymer sequence (inclusive), must be None
            for ligand entities
        max_length:
            Maximum length of designed polymer sequence (inclusive), must be None
            for ligand entities
        residue_bias:
            Modify positional or global amino acid/nucleotide base preferences. Preferences evaluated in order of list
            (later entries override earlier entries)
        insertions:
            Insertions relative to entity base sequence (must be None for ligands)
        deletions:
            If True, allow deletion of fixed-length positions from rep defined on entity.
            Must be None for ligand entities.
        """
        self.type = type
        self.rep = _rep_to_np_array(rep)
        self.id = id
        self.copies = copies

        if self.type not in BioPolymers and sequences is not None:
            raise ValueError(
                "Sequence record only supported for biopolymer entities"
            )

        self.sequences = sequences
        self.structures = structures
        self.first_index = first_index

        # extended attributes
        self.ligand_rep_type = ligand_rep_type
        self.interactions = interactions
        self.atom_bonds = atom_bonds
        self.modifications = modifications
        self.symmetry = symmetry
        self.secondary_structure = secondary_structure
        self.cyclic = cyclic
        self.min_length = min_length
        self.max_length = max_length
        self.residue_bias = residue_bias
        self.insertions = insertions
        self.deletions = deletions

        if self.type in BioPolymers:
            if self.ligand_rep_type is not None:
                raise ValueError(
                    "ligand_rep_type can only be specified for ligand entities"
                )

            if first_index is None or first_index < 1:
                raise ValueError(
                    f"first_index must be specified for type {self.type} and must be >= 1"
                )

            # verify that polymer sequence is valid if specified (including mask)
            if rep is not None:
                # allow representative to contain gaps, may want to mutate this to AA
                valid_seq, invalid = valid_sequence(
                    rep, self.alphabet(
                        include_gap=self.deletions,  # use as truth-y type
                        include_inserts=False,
                    ), allow_mask=True
                )

                if not valid_seq:
                    raise ValueError(f"Invalid sequence: {invalid}")

        elif self.type == "ligand":
            if self.ligand_rep_type is None:
                raise ValueError(
                    "ligand_rep_type must be specified for ligand entities"
                )

            if (
                self.first_index is not None or self.deletions or
                self.cyclic is not None or self.min_length is not None or
                self.max_length is not None or self.insertions is not None or
                self.modifications is not None or self.residue_bias is not None or
                self.sequences is not None
            ):
                raise ValueError(
                    "first_index, deletions, cyclic, min_length, max_length, insertions, modifications, residue_bias "
                    "can only be True/defined for biopolymer entities"
                )

        if self.symmetry is not None and self.copies is None:
            raise ValueError(
                "Attribute 'symmetry' must be specified together with 'copies'"
            )

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, Entity):
            return False

        # do not compare sequences and structures are these are auxiliary resources
        # for modeling the entity
        return (
                self.type == other.type and
                np.all(self.rep == other.rep) and
                self.id == other.id and
                self.copies == other.copies and
                self.first_index == other.first_index and
                self.ligand_rep_type == other.ligand_rep_type and
                self.interactions == other.interactions and
                self.atom_bonds == other.atom_bonds and
                self.modifications == other.modifications and
                self.symmetry == other.symmetry and
                self.secondary_structure == other.secondary_structure and
                self.cyclic == other.cyclic and
                self.min_length == other.min_length and
                self.max_length == other.max_length and
                self.residue_bias == other.residue_bias and
                self.insertions == other.insertions and
                self.deletions == other.deletions
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize entity to JSON-compatible representation

        Returns
        -------
        Serialized entity represented as dict
        """
        return {
            "id": self.id,
            "type": self.type,
            "rep": "".join(self.rep) if self.rep is not None else None,
            "copies": self.copies,
            "first_index": self.first_index,
            "sequences": self.sequences.serialize() if self.sequences is not None else None,
            "structures": _serialize_chain_map(self.structures),
            "ligand_rep_type": self.ligand_rep_type,
            "interactions": _serialize_optional_list(self.interactions),
            "atom_bonds": _serialize_optional_list(self.atom_bonds),
            "modifications": _serialize_optional_list(self.modifications),
            "symmetry": self.symmetry,
            "secondary_structure": _serialize_optional_list(self.secondary_structure),
            "cyclic": self.cyclic,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "residue_bias": _serialize_optional_list(self.residue_bias),
            "insertions": _serialize_optional_list(self.insertions),
            "deletions": self.deletions,
        }

    @classmethod
    def deserialize(cls, entity_dict: dict[str, Any]) -> Self:
        """
        Deserialize entity from JSON-compatible representation to object instance

        Parameters
        ----------
        entity_dict
            Entity attribute map

        Returns
        -------
        Deserialized Entity instance
        """
        sequences = entity_dict.get("sequences")

        return cls(
            type=entity_dict.get("type"),
            rep=entity_dict.get("rep"),
            id=entity_dict.get("id"),
            copies=entity_dict.get("copies"),
            first_index=entity_dict.get("first_index"),
            sequences=Sequences.deserialize(sequences) if sequences is not None else None,
            structures=_deserialize_chain_map(entity_dict.get("structures")),
            ligand_rep_type=entity_dict.get("ligand_rep_type"),
            interactions=_deserialize_optional_list(entity_dict.get("interactions"), Interaction),
            atom_bonds=_deserialize_optional_list(entity_dict.get("atom_bonds"), AtomBond),
            modifications=_deserialize_optional_list(entity_dict.get("modifications"), Modification),
            symmetry=entity_dict.get("symmetry"),
            secondary_structure=_deserialize_optional_list(entity_dict.get("secondary_structure"), SecondaryStructure),
            cyclic=entity_dict.get("cyclic"),
            min_length=entity_dict.get("min_length"),
            max_length=entity_dict.get("max_length"),
            residue_bias=_deserialize_optional_list(entity_dict.get("residue_bias"), ResidueBias),
            insertions=_deserialize_optional_list(entity_dict.get("insertions"), Insertion),
            deletions=entity_dict.get("deletions"),
        )

    def defined_sequence(self) -> bool:
        """
        Check if entity corresponds to a biopolymer (protein, ...)
        and has a defined representation with non-zero length

        Representation may include any valid biomolecule symbol,
        gap (coding for deletion) and mask (coding for unspecified).

        For now, not allowing inserts (lowercase symbols) in rep.

        Returns
        -------
        True if protein/nucleotide sequence with some defined length
        """
        return (
                self.type in BioPolymers and
                self.rep is not None and
                len(self.rep) > 0 and
                self.first_index is not None and
                valid_sequence(
                self.rep,
                self.alphabet(include_gap=True, include_inserts=False),
                allow_mask=True
            )
        )

    def alphabet(
        self,
        include_gap: bool=True,
        include_inserts: bool=False
    ) -> list[str]:
        """
        Return sequence alphabet for biopolymer entities

        Parameters
        ----------
        include_gap
            If true, add gap symbol to alphabet
        include_inserts
            If true, add insert symbols to alphabet (lowercase version of all symbols)

        Returns
        -------
        Alphabet for representing primary sequence of entity
        """
        if self.type == "protein":
            a = VALID_AA_OR_GAP_SORTED if include_gap else VALID_AA_SORTED
            if include_inserts:
                a = a + [symbol.lower() for symbol in VALID_AA_SORTED]
        elif self.type == "dna":
            a = VALID_DNA_OR_GAP_SORTED if include_gap else VALID_DNA_SORTED
            if include_inserts:
                a = a + [symbol.lower() for symbol in VALID_DNA_SORTED]
        elif self.type == "rna":
            a = VALID_RNA_OR_GAP_SORTED if include_gap else VALID_RNA_SORTED
            if include_inserts:
                a = a + [symbol.lower() for symbol in VALID_RNA_SORTED]
        else:
            raise NotImplementedError(
                f"Alphabet for type {self.type} not implemented"
            )

        return a

    @classmethod
    def merge_alphabet_symbols(
        cls,
        alphabets: list[list[str]]
    ) -> list[str]:
        """
        Merge symbols from different alphabets into one joint
        list of symbols. Note this does not imply a new alphabet, rather this
        method should only be used as a helper to jointly represent results for
        multiple alphabets in parallel (e.g. in ConditionalMutationScorer score_conditional()
        result dataframe)

        Parameters
        ----------
        alphabets
            List of one or more alphabets

        Returns
        -------
        Merged alphabet with each symbol occurring exactly once.
        """
        # deduplicate symbols and sort again:
        # upper-case symbols first, gap next, lowercase symbols/inserts last
        return sorted(
            {symbol for alphabet in alphabets for symbol in alphabet},
            key=lambda symbol: (symbol == symbol.lower(), symbol != GAP, symbol)
        )

    def is_biopolymer(self) -> bool:
        """
        Check if entity is a biopolymer (protein, DNA, RNA)

        Returns
        -------
        True if biopolymer, False otherwise
        """
        return self.type in BioPolymers

    def positions(self) -> list[int]:
        """
        Enumerate all positions in entity; will be empty
        if not a biopolymer or rep is None

        Returns
        -------
        List of positions
        """
        # first_index must be set for biopolymers, just include here to be 100% explicit about
        # assumptions
        if not self.type in BioPolymers or self.first_index is None or self.rep is None:
            return []

        return [
            pos for pos, _ in enumerate(self.rep, start=self.first_index)
        ]

    def expand_residue_bias(self) -> dict[int, dict[str, float]]:
        """
        Expand residue bias definition into dictionary mapping all specified positions,
        evaluating in order of residue_bias attribute (later entries overwrite
        earlier entries)

        Returns
        -------
        Expanded residue bias mapping
        """
        if self.residue_bias is None:
            return {}

        expanded = {}

        for bias_entry in self.residue_bias:
            # If pos is None, means all positions in entity
            if bias_entry.pos is None:
                entry_pos = self.positions()
            else:
                entry_pos = [bias_entry.pos]

            for cur_pos in entry_pos:
                if cur_pos not in expanded:
                    expanded[cur_pos] = {}

                for symbol, value in bias_entry.bias.items():
                    expanded[cur_pos][symbol] = value

        return expanded


class EntityInstance:
    """
    Instantiation of a single entity in a system
    """
    def __init__(
        self,
        rep: RepSequence | str | None = None,
        embedding: Embedding | None = None,
        models: StructureChainMap | None = None,
    ):
        """
        Create new instantiation of an entity in a sequence

        Notes:
        1. Under fixed-length models, length of representation in EntityInstance should always match
         length of the corresponding representation in the defining Entity

        2. Deletions relative to the Entity representation should be encoded with the GAP symbol,
         insertions with the lowercase version of the alphabet symbol (cf. Entity.alphabet()).
         This directly corresponds to how the alignment between the two representations would be encoded
         in the A3M alignment format. This encoding will allow implementations to map positions back to the
         system instance numbering (e.g. to evaluate constraints on fixed positions)

        Parameters
        ----------
        rep
            Uniquely defining representation (e.g. primary sequence) of entity. Set to None if no
            representation is yet available (e.g. just structural backbone but no sequence).
            See notes above regarding encoding of insertions and deletions.
        embedding
            Transformation of entity instance into per-residue embedding (2D array) or
            per-entity embedding (1D array) space
        models
            Structural models associated with each of the entities in the system.
            Set to None if no structural models are available.
        """
        self.rep = _rep_to_np_array(rep)
        self.embedding = embedding
        self.models = models

    def __repr__(self):
        if self.models is not None:
            structure_info = len(self.models)
        else:
            structure_info = self.models

        if self.rep is not None:
            short_rep = shorten("".join(self.rep))
        else:
            short_rep = "n/a"

        return f"EntityInstance(rep={short_rep}, models={structure_info})"

    def copy(self) -> Self:
        """
        Create a shallow copy of the entity instance (rep, embedding and models
        will still point to same objects as before)

        Returns
        -------
        Shallow copy
        """
        return type(self)(
            rep=self.rep,
            embedding=self.embedding,
            models=self.models
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize entity instance to JSON-compatible representation

        Returns
        -------
        Serialized entity instance represented as dict
        """
        return {
            "rep": "".join(self.rep) if self.rep is not None else None,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "models": _serialize_chain_map(self.models),
        }

    @classmethod
    def deserialize(cls, entity_inst_dict: dict[str, Any]) -> Self:
        """
        Deserialize entity instance from JSON-compatible representation
        to object instance

        Parameters
        ----------
        entity_inst_dict
            Entity instance attribute map

        Returns
        -------
        Deserialized EntityInstance object
        """
        embedding = entity_inst_dict.get("embedding")
        return cls(
            rep=entity_inst_dict.get("rep"),
            embedding=np.array(embedding) if embedding is not None else None,
            models=_deserialize_chain_map(entity_inst_dict.get("models")),
        )

    def normalized_rep(self) -> RepSequence:
        """
        Return representation without insert and deletion coding
        (all uppercase symbols, no gaps)

        Returns
        -------
        Normalized entity representation
        """
        return np.char.upper(self.rep[self.rep != GAP])  # noqa

    @staticmethod
    def normalize_rep_str(rep: str) -> str:
        """
        Helper method to normalize representations that are in string format

        Parameters
        ----------
        rep
            String version of representation

        Returns
        -------
        Normalized representation (inserts uppercased, gaps removed)
        """
        return rep.replace("-", "").upper()


class SystemInstance(UserList[EntityInstance]):
    """
    Result designing the representations of the entity/entities
    in a system, comprised of individual EntityInstances (one per entity),
    mirroring the "System" class comprised of entities
    """
    def __init__(
        self,
        entity_instances: EntityInstance | Sequence[EntityInstance],
        score: float | None = None,
        confidence: float | None = None,
        metadata: Metadata | None = None,
        id: str | None = None  # noqa
    ):
        """
        Create new entity system instance

        Parameters
        ----------
        entity_instances
            One or more entity instances (must match entities in corresponding System)
        score
            Score describing quality/likelihood of the designed system instance
            (higher is better, ideally in logits)
        confidence
            Reliability of model score from 0 (lowest confidence) to 1 (highest confidence)
        """
        # turn single instance into list of instances
        entity_instances = ensure_sequence(entity_instances)
        super().__init__(entity_instances)

        self.score = score
        self.confidence = confidence
        self.metadata = metadata
        self.id = id

    def __repr__(self):
        return f"SystemInstance({self.data} id={self.id} score={self.score})"

    def copy(self) -> Self:
        """
        Create a shallow copy of the system instance, making
        shallow copies of each contained entity instance as well.

        Returns
        -------
        Shallow copy
        """
        return type(self)(
            entity_instances=[ei.copy() for ei in self.data],
            score=self.score,
            confidence=self.confidence,
            metadata=self.metadata.copy() if self.metadata is not None else None,
            id=self.id
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize system instance into JSON-compatible representation

        Returns
        -------
        List of serialized EntityInstance objects
        """
        return {
            "entity_instances": [
                entity_instance.serialize() for entity_instance in self.data
            ],
            "score":self.score,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "id": self.id,
            "schema_version": CURRENT_SYSTEM_INSTANCE_SPEC_VERSION
        }

    @classmethod
    def deserialize(cls, serialized_system_instance: dict[str, Any]) -> Self:
        """
        Deserialize system instance from JSON-compatible representation into
        object instance

        Parameters
        ----------
        serialized_system_instance
            SystemInstance representation as output by serialize() method

        Returns
        -------
        List of deserialized EntityInstance objects
        """
        # if not specified, assume first version
        version = serialized_system_instance.get("schema_version", "0.2")
        if version != CURRENT_SYSTEM_INSTANCE_SPEC_VERSION:
            raise ValueError(
                f"Unable to handle SystemInstance version {version}"
            )

        return cls(
            [
                EntityInstance.deserialize(entity_instance)
                for entity_instance in serialized_system_instance["entity_instances"]
            ],
            score=serialized_system_instance.get("score"),
            confidence=serialized_system_instance.get("confidence"),
            metadata=serialized_system_instance.get("metadata"),
            id=serialized_system_instance.get("id")
        )


class System(UserList[Entity]):
    def __init__(self, entities: Entity | Sequence[Entity]):
        """
        Create new biomolecular system for modeling/design

        Parameters
        ----------
        entities
            One or more entities comprising the system
        """
        # turn single entity into list of entities
        entities = ensure_sequence(entities)

        if len(entities) == 0:
            raise ValueError(
                "Valid system must contain at least one entity"
            )

        super().__init__(entities)

    def __eq__(self, other):
        # only ever accept same class for equality
        if not isinstance(other, System):
            return False

        # systems must have same length
        if not len(self) == len(other):
            return False

        # two systems are equal if all contained entities are equal
        # (in same order)
        for ent_self, ent_other in zip(self, other):
            if ent_self != ent_other:
                return False

        return True

    def serialize(self) -> dict[str, Any]:
        """
        Serialize system into JSON-compatible representation

        Returns
        -------
        Serialized System with individual entities
        """
        return {
            "entities": [
                entity.serialize() for entity in self.data
            ],
            "schema_version": CURRENT_SYSTEM_SPEC_VERSION,
        }

    @classmethod
    def deserialize(cls, serialized_system: dict[str, Any]) -> Self:
        """
        Deserialize system from JSON-compatible representation into object instance

        Parameters
        ----------
        serialized_system
            System representation as output by serialize() method

        Returns
        -------
        List of deserialized Entity objects
        """
        # if not specified, assume first version
        version = serialized_system.get("schema_version", "0.2")
        if version != CURRENT_SYSTEM_SPEC_VERSION:
            raise ValueError(
                f"Unable to handle System version {version}"
            )

        return cls([
            Entity.deserialize(entity) for entity in serialized_system.get("entities", [])
        ])

    def copy(self) -> Self:
        """
        Create deep copy (for simplicity, usually system parts will not use too many resources) of system

        Returns
        -------
        Deep copy of system
        """
        return deepcopy(self)

    @classmethod
    def from_structure(cls, structure_model: Structure) -> Self:
        """
        Build a system from entities in a protein 3D structure

        Parameters
        ----------
        structure_model
            Input structure (all chains present in structure will be used)

        Returns
        -------
        System corresponding to entities in structure
        """
        raise NotImplementedError()

    def valid_instance(
        self,
        instance: SystemInstance,
        validate_reps: bool = True,
        require_reps: bool = False,
        validate_embeddings: bool = True,
        fixed_length: bool = True,
        allow_deletions: bool = False,
        raise_invalid: bool = False,
    ) -> bool:
        """
        Verify if instance is valid representation of this biomolecular system

        Parameters
        ----------
        instance
            System instance to validate
        fixed_length
            If True, require that length of instance sequence matches the system entity representation length
            (only sensible for fixed-length models and biopolymers)
        validate_reps
            If True, verify if *specified* sequence representations are comprised of valid amino acids/nucleotides
        require_reps
            If True, ensure that all reps are specified/not None and valid (stricter than validate_reps)
        validate_embeddings
            If True, verify if specified sequence embeddings are valid (correct shape)
        allow_deletions
            If True, allow deletions (coded by gap symbols) to be present in representation
        raise_invalid
            If True, raise ValueError if instance is invalid w.r.t. system

        Returns
        -------
        True if valid instance, False otherwise
        """
        # instance representations always must have same length as number of entities
        # in system by convention
        valid = len(self.data) == len(instance)

        for entity, entity_instance in zip(self.data, instance):
            if entity.type in BioPolymers:
                if fixed_length:
                    valid = valid and (
                        entity.rep is None or (
                            entity_instance.rep is not None and len(entity.rep) == len(entity_instance.rep)
                         )
                    )

                if (validate_reps and entity_instance.rep is not None) or require_reps:
                    is_valid_seq, _ = entity_instance.rep is not None and valid_sequence(
                        entity_instance.rep,
                        entity.alphabet(
                            include_gap=allow_deletions,
                            include_inserts=not fixed_length
                        ),
                        allow_mask=False,
                    )

                    valid = valid and is_valid_seq

                    # if we have 3D structure models, verify these against primary rep too
                    # (but only if valid sequence)
                    if is_valid_seq and entity_instance.models is not None:
                        # enumerate positions for current sequence
                        positions = np.arange(
                            entity.first_index, entity.first_index + len(entity_instance.rep)
                        )

                        # validate all models attached to current EntityInstance
                        for models in entity_instance.models.values():  # noqa
                            models = ensure_sequence(models)
                            for model in models:
                                valid = valid and model.represents(
                                    positions, entity_instance.rep, allow_missing=True
                                )

                                # do not continue with comparison if we have at least one invalid structure
                                if not valid:
                                    break

                if validate_embeddings and entity_instance.embedding is not None:
                    # check if embedding is a per-entity vector or per-position matrix
                    emb_shape = entity_instance.embedding.shape
                    valid = valid and len(emb_shape) in (1, 2)

                    # if matrix, length must match entity instance rep (if latter is specified),
                    # and also must match entity rep if fixed length is required
                    if len(emb_shape) == 2:
                        if entity_instance.rep is not None:
                            valid = valid and emb_shape[0] == len(entity_instance.rep)

                        if fixed_length and entity.rep is not None:
                            valid = valid and emb_shape[0] == len(entity.rep)

        if not valid and raise_invalid:
            raise ValueError("Provided instance is not valid for biomolecular system")

        return valid

    def _entity_to_pos_and_subs(
        self,
        instance: SystemInstance,
        deletions: bool = False,
        insertions: bool = False,
    ) -> tuple[
        dict[int, dict[int, str]],
        dict[int, list[str]],
        dict[int, set[int]]
    ]:
        """
        Helper method to determine mutable positions and available mutations in system

        Parameters
        ----------
        instance
            System instance to check against; assuming this has been previously validated with valid_instance().
        deletions
            If True, consider gap symbol a valid substitution coding for a deletion at the given position
        insertions
            If True, allow insertions (coded as lowercase symbol returned by Entity.alphabet())

        Returns
        -------
        entity_to_pos
            Mapping from entity to valid positions to ref symbol at that position
        entity_to_valid_subs
            Mapping from entity to valid substitutions for that entity
        entity_to_ins_pos
            Mapping from entity to all positions where an insertion can be made
        """
        # create mapping of valid position and reference symbol in each biopolymer entity instance with defined
        # sequence and first_index
        entity_to_pos = {
            entity_idx: {
                pos: str(ref_symbol) for (pos, ref_symbol) in enumerate(
                    instance[entity_idx].rep, start=entity.first_index
                )
            } for entity_idx, entity in enumerate(self.data)
            # only iterate defined reps for biopolymer sequences
            if entity.type in BioPolymers and entity.first_index is not None and instance[entity_idx].rep is not None
        }

        # also record possible positions for insertion including N-terminal of first_index
        entity_to_ins_pos: dict[int, set[int]]
        if insertions:
            entity_to_ins_pos = {
                entity_idx: (set(pos) | {min(pos) - 1}) for entity_idx, pos in entity_to_pos.items()
            }
        else:
            entity_to_ins_pos = {
                entity_idx: set() for entity_idx, pos in entity_to_pos.items()
            }

        entity_to_valid_subs = {
            entity_idx: self.data[entity_idx].alphabet(include_gap=deletions, include_inserts=insertions)
            for entity_idx in entity_to_pos
        }

        return entity_to_pos, entity_to_valid_subs, entity_to_ins_pos

    def valid_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        deletions: bool = False,
        insertions: bool = False,
        raise_invalid: bool = False,
    ) -> tuple[bool, list[tuple[int, Mutation]]]:
        """
        Validate mutants against a system instance

        Parameters
        ----------
        instance
            System instance to check against; assuming this has been previously validated with valid_instance().
        mutants
            Verify these mutants against system instance
        deletions
            If True, consider gap symbol a valid substitution coding for a deletion at the given position
        insertions
            If True, allow insertions (coded as lowercase symbol returned by Entity.alphabet())
        raise_invalid
            Raise ValueError if any invalid mutants are detected

        Returns
        -------
        valid
            True if all mutants are valid, False otherwise
        invalid_subs
            Tuple of mutant indies and invalid mutations in these mutants (empty if all mutants are valid)

        """
        entity_to_pos, entity_to_valid_subs, entity_to_ins_pos = self._entity_to_pos_and_subs(
            instance, deletions=deletions, insertions=insertions
        )

        # turn into set for more efficient lookup in loop
        entity_to_valid_subs = {
            entity: set(subs) for entity, subs in entity_to_valid_subs.items()
        }

        invalid_subs = [
            (i, subs) for (i, mutant) in enumerate(mutants) for subs in mutant if (
                (subs.entity not in entity_to_pos) or  # valid entity index
                 # generally invalid specification if "to" not in target alphabet
                subs.to not in entity_to_valid_subs[subs.entity] or
                # check insertions
                subs.ref == "" and (
                    (subs.pos not in entity_to_ins_pos[subs.entity]) or
                    subs.to == GAP or
                    subs.to.lower() != subs.to
                ) or

                # validate mutations/deletions
                subs.ref != "" and (
                    (subs.pos not in entity_to_pos[subs.entity]) or
                    (subs.ref != entity_to_pos[subs.entity][subs.pos]) or
                    (subs.to.upper() != subs.to)
                )
            )
        ]

        valid = len(invalid_subs) == 0

        if not valid and raise_invalid:
            raise ValueError(f"Invalid mutants: {invalid_subs}")

        return valid, invalid_subs

    def single_mutants(
        self,
        instance: SystemInstance,
        deletions: bool = False,
        insertions: bool = False,
    ) -> list[Mutant]:
        """
        Enumerate all possible single mutants for a given instance

        Parameters
        ----------
        instance
            Instance to mutate (assumed to be validated)
        deletions
            If True, include deletions for each position as mutant
        insertions
            If True, include insertions for each position as mutants

        Returns
        -------
        List of single mutants
        """
        entity_to_pos, entity_to_valid_subs, entity_to_ins_pos = self._entity_to_pos_and_subs(
            instance, deletions=deletions, insertions=insertions
        )

        # build mutations (including self mutation)
        mutants = [
            [
                Mutation(
                    entity=entity,
                    pos=pos,
                    to=subs,
                    ref=("" if subs != subs.upper() else entity_to_pos[entity][pos])
                )
            ]
            for entity in entity_to_pos
            for pos in set(entity_to_pos[entity]) | entity_to_ins_pos[entity]
            for subs in entity_to_valid_subs[entity]
            if subs != subs.upper() or pos in entity_to_pos[entity]
        ]

        return mutants

    def apply_instance(
        self,
        instance: SystemInstance
    ) -> Self:
        """
        Create new system with updated representations from given instance
        (as shallow copy). The representation of each entity instance
        will be normalized, i.e. deletions are removed and insertions
        are converted into regular uppercase symbols.

        Sequences attached to system will not be attached to new system,
        structural models will be added.

        Assumes instance has been previously validated with valid_instance()

        Parameters
        ----------
        instance
            Apply representations of this instance

        Returns
        -------
        Updated molecular system
        """
        assert len(instance) == len(self.data)

        return type(self)([
            Entity(
                type=entity.type,
                rep=entity_instance.normalized_rep(),
                id=entity.id,
                copies=entity.copies,
                first_index=entity.first_index,
                sequences=None,  # do not copy sequences as we would need to realign them
                structures=entity_instance.models
            ) for entity, entity_instance in zip(self.data, instance)
        ])

    def rep_to_instance(self) -> SystemInstance:
        """
        Transform system into its own system instance
        (e.g. for scoring WT sequence that design was started from),
        using primary rep only

        Note: Not all systems can be transformed into a valid
         system instance, e.g. if mask or gap characters are present.
         In these cases, a ValueError will be raised via valid_instance().

        Returns
        -------
        System instance derived from system representation
        """
        instance = SystemInstance([
            EntityInstance(rep=entity.rep.copy()) for entity in self.data
        ])

        self.valid_instance(instance, raise_invalid=True)

        return instance

    def mutate(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant]
    ) -> list[SystemInstance]:
        """
        Create different mutant versions of a given instance.
        Assumes mutants have been previously validated with valid_mutants()

        Parameters
        ----------
        instance
            Starting instance to be mutated
        mutants
            Different mutants to create from the instance (each supplied
            mutant, potentially comprised of multiple mutations, will
            lead to the creation of a new instance in output)

        Returns
        -------
        Mutated versions of instance (one per mutant). Will have same
        length as mutants parameter
        """
        # shallow copy system instance and entity instance
        instances = [
            instance.copy() for _ in range(len(mutants))
        ]

        # apply mutations for each mutant instance, align to instances copied above with instance_idx
        for instance_idx, mutant in enumerate(mutants):
            # create editable new copies of entities modified by mutation (assumed to be editable and correct
            # based on prior validation)
            mutated_entities = set(
                mutation.entity for mutation in mutant
            )

            entity_to_rep = {
                entity_idx: list(
                    map(str, instances[instance_idx][entity_idx].rep)
                ) for entity_idx in mutated_entities
            }

            # sort mutations in mutant by descending positions, this will allow us to apply
            # any insertions without breaking position indexing;
            # as insertions are made after substitution/deletion with the same position, do
            # not need to worry about their relative ordering
            mutant_sorted = sorted(
                mutant, key=lambda m: (m.entity, m.pos)
            )

            # iterate mutations and update
            for mutation in reversed(mutant_sorted):
                pos_adj = mutation.pos - self.data[mutation.entity].first_index
                if mutation.ref != "":
                    entity_to_rep[mutation.entity][pos_adj] = mutation.to
                else:
                    entity_to_rep[mutation.entity].insert(pos_adj + 1, mutation.to)

            # reassign updated reps to current instance
            for entity_idx, rep in entity_to_rep.items():
                instances[instance_idx][entity_idx].rep = np.array(rep, dtype="U1")

        return instances


class _BiopolymerEntity(Entity):
    """
    Helper class for syntactic sugar classes Protein, DNA, RNA,
    should never be instantiated directly.
    """
    _entity_type = None
    def __init__(
        self,
        rep: str | RepSequence | None = None,
        id: str | None = None,  # noqa
        first_index: int = 1,
        sequences: Sequences | None = None,
        structures: StructureChainMap | None = None,
        interactions: Sequence[Interaction] | None = None,
        atom_bonds: Sequence[AtomBond] | None = None,
        modifications: Sequence[Modification] | None = None,
        copies: int | None = None,
        symmetry: SymmetryType | None = None,
        secondary_structure: Sequence[SecondaryStructure] | None = None,
        cyclic: bool = False,
        min_length: int | None = None,
        max_length: int | None = None,
        residue_bias: Sequence[ResidueBias] | None = None,
        insertions: Sequence[Insertion] | None = None,
        deletions: bool = False,
    ):
        """
        Create new biopolymer entity. Syntactic sugar for instantiating Entity class directly,
        cf. to this class for parameter documentation.
        """
        if self._entity_type is None:
            raise ValueError(
                "Should not instantiate this class, use Protein, DNA, RNA instead"
            )

        super().__init__(
            type=self._entity_type,
            rep=rep,
            id=id,
            first_index=first_index,
            sequences=sequences,
            structures=structures,
            interactions=interactions,
            atom_bonds=atom_bonds,
            modifications=modifications,
            copies=copies,
            symmetry=symmetry,
            secondary_structure=secondary_structure,
            cyclic=cyclic,
            min_length=min_length,
            max_length=max_length,
            residue_bias=residue_bias,
            insertions=insertions,
            deletions=deletions,
        )

class Protein(_BiopolymerEntity):
    """
    Protein entity
    """
    _entity_type = "protein"


class DNA(_BiopolymerEntity):
    """
    DNA entity
    """
    _entity_type = "dna"


class RNA(_BiopolymerEntity):
    """
    RNA entity
    """
    _entity_type = "rna"


class Ligand(Entity):
    """
    Create ligand entity. Syntactic sugar for direct instantiation
    of class Entity
    """
    def __init__(
        self,
        rep: str | RepSequence | None = None,
        id: str | None = None,  # noqa
        structures: StructureChainMap | None = None,
        ligand_rep_type: LigandRepType | None = None,
        interactions: Sequence[Interaction] | None = None,
        atom_bonds: Sequence[AtomBond] | None = None,
        copies: int | None = None,
        symmetry: SymmetryType | None = None,
    ):
        """
        Create new ligand entity

        Cf. Entity class documentation for parameters
        """
        super().__init__(
            type="ligand",
            rep=rep,
            id=id,
            structures=structures,
            ligand_rep_type=ligand_rep_type,
            interactions=interactions,
            atom_bonds=atom_bonds,
            copies=copies,
            symmetry=symmetry
        )

