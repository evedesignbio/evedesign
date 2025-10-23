"""
Specification of components of molecular design system (proteins, nucleic acids, ligands, etc.)
"""
from collections import UserList
from collections.abc import Sequence
from copy import deepcopy
from io import StringIO
from typing import Mapping, NamedTuple, Self, Any
import numpy as np

from protdesign.sequence import valid_sequence, Sequences
from protdesign.structure import Model, Structure
from protdesign.types import EntityType, Metadata, BioPolymers, RepSequence
from protdesign.constants import (
    VALID_AA_OR_GAP_SORTED, VALID_AA_SORTED,
    VALID_DNA_OR_GAP_SORTED, VALID_DNA_SORTED,
    VALID_RNA_OR_GAP_SORTED, VALID_RNA_SORTED,
    GAP
)
from protdesign.utils import ensure_sequence, shorten


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
sequence *before* applying any of the mutations (e.g. before any numbering shifts due to insertions)
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
StructureChainMap = dict[str, Model | list[Model]]

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
            model_deserialized = Structure(
                StringIO(model), format="cif"
            ).get_model()

            # extract single chain
            chains = model_deserialized.chains()
            assert (len(chains) == 1), "Only can deserialize single-chain models"
            deserialized_map[key].append(
                model_deserialized.get_chain(chains[0])
            )

    return deserialized_map


class Entity:
    def __init__(
        self,
        type: EntityType,  # noqa
        rep: str | RepSequence | None = None,
        id: str | None = None,  # noqa
        copies: int | None = None,
        first_index: int | None = None,
        sequences: Sequences | None = None,
        structures: StructureChainMap | None = None,
    ):
        """
        Create new generic entity for molecular system.

        Note: For clarity, preferentially use subclasses for specific types
        of entities (e.g. Protein class)

        TODO: parameters to be added at later point
         * modifications
         * different states / conformations
         * hotspots, pair restraints / constraints

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
        copies
            Number of entity copies in molecular system. Set to None
            to leave variable.
                sequences
        sequences
            Sequence record (e.g. multiple sequence alignment of homologs) of the target
            sequence represented by this entity (only applies to proteins and nucleotides)
        structures
            Structure chains representing this entity. Use dict with structure identifiers
            as keys to supply multiple different structures; use list to supply multiple copies
            of the chain within the structure (homooligomer)
        """
        self.type_ = type
        self.rep = _rep_to_np_array(rep)
        self.id_ = id
        self.copies = copies

        if self.type_ not in BioPolymers and sequences is not None:
            raise ValueError(
                "Sequence record only supported for biopolymer entities"
            )

        self.sequences = sequences
        self.structures = structures

        if self.type_ in BioPolymers and (first_index is None or first_index < 1):
            raise ValueError(
                f"first_index must be specified for type {self.type_} and must be >= 1"
            )

        self.first_index = first_index

    def __eq__(self, other):
        # only ever accept other entities for equality
        if not isinstance(other, Entity):
            return False

        # do not compare sequences and structures are these are auxiliary resources
        # for modeling the entity
        return (
            self.type_ == other.type_ and
            np.all(self.rep == other.rep) and
            self.id_ == other.id_ and
            self.copies == other.copies and
            self.first_index == other.first_index
        )

    def serialize(self) -> dict[str, Any]:
        """
        Serialize entity to JSON-compatible representation

        Returns
        -------
        Serialized entity represented as dict
        """
        return {
            "id": self.id_,
            "type": self.type_,
            "rep": "".join(self.rep) if self.rep is not None else None,
            "copies": self.copies,
            "first_index": self.first_index,
            "sequences": self.sequences.serialize() if self.sequences is not None else None,
            "structures": _serialize_chain_map(self.structures),
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
            self.type_ in BioPolymers and
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
        if self.type_ == "protein":
            a = VALID_AA_OR_GAP_SORTED if include_gap else VALID_AA_SORTED
            if include_inserts:
                a = a + [symbol.lower() for symbol in VALID_AA_SORTED]
        elif self.type_ == "dna":
            a = VALID_DNA_OR_GAP_SORTED if include_gap else VALID_DNA_SORTED
            if include_inserts:
                a = a + [symbol.lower() for symbol in VALID_DNA_SORTED]
        elif self.type_ == "rna":
            a = VALID_RNA_OR_GAP_SORTED if include_gap else VALID_RNA_SORTED
            if include_inserts:
                a = a + [symbol.lower() for symbol in VALID_RNA_SORTED]
        else:
            raise NotImplementedError(
                f"Alphabet for type {self.type_} not implemented"
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

Embedding = np.ndarray[
    tuple[int, int], np.dtype[float]
] | np.ndarray[
    tuple[int], np.dtype[float]
]

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
        return np.char.upper(self.rep[self.rep != GAP])

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
        id: str | None = None
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
        self.id_ = id

    def __repr__(self):
        return f"SystemInstance({self.data} id={self.id_} score={self.score})"

    def copy(self) -> Self:
        """
        Create a shallow copy of the system instance

        Returns
        -------
        Shallow copy
        """
        return type(self)(
            entity_instances=self.data.copy(),
            score=self.score,
            confidence=self.confidence,
            metadata=self.metadata.copy() if self.metadata is not None else None,
            id=self.id_
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
            "id": self.id_
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
        super().__init__(entities)

    def __eq__(self, other):
        # only ever accept other systems for equality
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

    def serialize(self) -> list[dict[str, Any]]:
        """
        Serialize system into JSON-compatible representation

        Returns
        -------
        List of serialized Entity objects
        """
        return [
            entity.serialize() for entity in self.data
        ]

    @classmethod
    def deserialize(cls, serialized_system: list[dict[str, Any]]) -> Self:
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
        return cls([
            Entity.deserialize(entity) for entity in serialized_system
        ])

    def copy(self) -> Self:
        """
        Create deep copy (for simplicity, usually system parts will not use too many resources) of system

        Returns
        -------
        Deep copy of system
        """
        return deepcopy(self)

    def valid_instance(
        self,
        instance: SystemInstance,
        validate_reps: bool = True,
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
            If True, verify if sequence representations are comprised of valid amino acids/nucleotides
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
            if entity.type_ in BioPolymers:
                if fixed_length:
                    valid = valid and (
                        entity.rep is None or (
                            entity_instance.rep is not None and len(entity.rep) == len(entity_instance.rep)
                         )
                    )

                if validate_reps and entity_instance.rep is not None:
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
                        for models in entity_instance.models.values():
                            models = ensure_sequence(models)
                            for model in models:
                                valid = valid and model.represents(
                                    positions, entity_instance.rep, allow_missing=True
                                )

                                # do not continue with comparison if we have at least one invalid structure
                                if not valid:
                                    break

        if not valid and raise_invalid:
            raise ValueError("Provided instance is not valid for biomolecular system")

        return valid

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
        # create mapping of valid position and reference symbol in each biopolymer entity instance with defined
        # sequence and first_index
        entity_to_pos = {
            entity_idx: {
                pos: ref_symbol for (pos, ref_symbol) in enumerate(
                    instance[entity_idx].rep, start=entity.first_index
                )
            } for entity_idx, entity in enumerate(self.data)
            # note: defined_sequence() is too strict of a check here as it required rep to be defined
            if entity.type_ in BioPolymers and entity.first_index is not None
        }

        # also record possible positions for insertion including N-terminal of first_index
        if insertions:
            entity_to_ins_pos = {
                entity_idx: (set(pos) | {min(pos) - 1}) for entity_idx, pos in entity_to_pos.items()
            }
        else:
            entity_to_ins_pos = {
                entity_idx: {} for entity_idx, pos in entity_to_pos.items()
            }

        entity_to_valid_subs = {
            entity_idx: set(entity.alphabet(include_gap=deletions, include_inserts=insertions))
            for entity_idx, entity in enumerate(self.data)
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
                    (subs.to.lower() == subs.to)
                )
            )
        ]

        valid = len(invalid_subs) == 0

        if not valid and raise_invalid:
            raise ValueError(f"Invalid mutants: {invalid_subs}")

        return valid, invalid_subs

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
                type=entity.type_,
                rep=entity_instance.normalized_rep(),
                id=entity.id_,
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
            EntityInstance(rep=entity.rep) for entity in self.data
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
        # TODO: implement this
        raise NotImplementedError()


class Protein(Entity):
    """
    Single protein chain entity
    """
    def __init__(
        self,
        id: str | None,  # noqa
        rep: str | None = None,
        first_index: int = 1,
        copies: int | None = None,
        sequences: Sequences | None = None,
        structures: StructureChainMap | None = None,
    ):
        """
        Create new protein entity

        Parameters
        ----------
        id
            Unique identifier of protein
        rep
            Sequence of protein (if None, auto-infer or leave open as needed for model).
            May contain any valid amino acid or the mask symbol.
        first_index
            Sequence index of first residue (1-based numbering)
        copies
            Number of copies of protein chain in system (None to leave unspecified/variable)
        sequences
            Sequence record (e.g. multiple sequence alignment of homologs) of the target
            sequence represented by this entity
        structures
            Structure chains representing this entity. Use dict with structure identifiers
            as keys to supply multiple different structures; use list to supply multiple copies
            of the chain within the structure (homooligomer)
        """
        # verify that protein sequence is valid if specified (including mask)
        if rep is not None:
            # allow representative to contain gaps, may want to mutate this to AA
            valid_seq, invalid_aa = valid_sequence(
                rep, VALID_AA_OR_GAP_SORTED, allow_mask=True
            )

            if not valid_seq:
                raise ValueError(f"Invalid protein sequence: {invalid_aa}")

        super().__init__(
            type="protein",
            id=id,
            rep=rep,
            first_index=first_index,
            copies=copies,
            sequences=sequences,
            structures=structures,
        )

# mapping from entity index to positions in entity (e.g. for fixing positions)
EntityPosList = Mapping[int, Sequence[int]]
