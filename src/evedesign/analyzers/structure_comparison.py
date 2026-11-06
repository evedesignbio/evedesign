"""
Compare predicted 3D models of system instances against reference (e.g. experimental)
structures, computing RMSD and TM-score on the full molecule and RMSD on named residue
subsets (e.g. active site residues).

Conventions (see evedesign.system.StructureChainMap):
- Reference: data["reference"] passed to analyze(), or else system[entity].structures[reference_id];
  one single-chain Structure per entity copy, numbered like the entity rep
- Models: instance[entity].models[model_id]; one single-chain Structure per entity copy, numbered
  along the (A3M-coded) entity instance rep and renumbered here into entity numbering
  (inserted residues are dropped)
- Atoms are matched by (entity index, copy index, residue number, atom name), never by chain
  identifier or atom order
"""
from inspect import signature
from itertools import permutations, product
from typing import Any, Literal, Mapping, Sequence, TypedDict
import numpy as np
import biotite.structure as struc
from loguru import logger

from evedesign.analysis import Analyzer
from evedesign.constants import GAP
from evedesign.structure import Structure
from evedesign.system import System, SystemInstance, Entity, EntityInstance
from evedesign.tools.structure_mapping import map_structure_chain_to_entity_pairwise
from evedesign.types import StructureComparison, STRUCTURE_COMPARISON_COMPONENT_KEY
from evedesign.utils import ensure_sequence

# (entity index, copy index, residue number in entity numbering)
ResidueKey = tuple[int, int, int]
# name of residue subset -> residues in subset (copy index refers to reference copies)
ResidueSubsets = Mapping[str, Sequence[ResidueKey]]
# entity index -> one single-chain Structure per copy, in entity numbering
Chains = Mapping[int, Structure | Sequence[Structure]]
# (entity index, copy index, residue number, atom name) -> (residue name, coordinates)
_AtomTable = dict[tuple[int, int, int, str], tuple[str, np.ndarray]]

FULL_SUBSET = "all"
MAX_COPIES_FOR_PERMUTATION = 4
# warn if reference sequence identity to entity rep is below this (e.g. wrong numbering offset)
MIN_REFERENCE_IDENTITY = 0.5
# atoms paired regardless of residue type; all other atoms only between identical residue types
BACKBONE_ATOMS = {"N", "CA", "C", "O", "P", "C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'"}


class StructureComparisonData(TypedDict, total=False):
    """
    data argument of ReferenceStructureComparison.analyze()
    """
    reference: Chains | None  # if None/missing, use system[entity].structures[reference_id]
    subsets: ResidueSubsets | None


def _atom_table(chains: Chains, atom_names: Sequence[str]) -> _AtomTable:
    table = {}
    for entity_idx, copies in chains.items():
        for copy_idx, chain in enumerate(ensure_sequence(copies)):
            arr = chain.atom_array
            # polymer residues only (keeps e.g. MSE, excludes ions such as calcium named "CA")
            sel = np.isin(arr.atom_name, atom_names) & (
                struc.filter_amino_acids(arr) | struc.filter_nucleotides(arr)
            )
            for res_id, res_name, atom_name, coord in zip(
                arr.res_id[sel], arr.res_name[sel], arr.atom_name[sel], arr.coord[sel]
            ):
                table[(entity_idx, copy_idx, int(res_id), str(atom_name))] = (str(res_name), coord)
    return table


def _paired_coords(
    ref: _AtomTable,
    model: _AtomTable,
    assignment: Mapping[int, Sequence[int]],
    residues: set[ResidueKey] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of atoms present in both tables (assignment[entity][ref_copy] = model_copy);
    non-backbone atoms are only paired if residue types agree (e.g. not at mutated positions)
    """
    pairs = []
    for (e, c, r, a), (ref_res, ref_coord) in ref.items():
        if residues is not None and (e, c, r) not in residues:
            continue
        model_res, model_coord = model.get((e, assignment[e][c], r, a), (None, None))
        if model_coord is not None and (a in BACKBONE_ATOMS or model_res == ref_res):
            pairs.append((ref_coord, model_coord))
    if not pairs:
        return np.empty((0, 3)), np.empty((0, 3))
    ref_xyz, model_xyz = zip(*pairs)
    return np.array(ref_xyz), np.array(model_xyz)


def _copy_assignments(reference: Chains, model: Chains, match_copies: bool) -> list[dict[int, tuple[int, ...]]]:
    """
    Candidate assignments of reference to model copies (chain identity in homo-oligomers is arbitrary)
    """
    per_entity = []
    for entity_idx in reference:
        n_ref, n_model = len(ensure_sequence(reference[entity_idx])), len(ensure_sequence(model[entity_idx]))
        if n_ref > n_model:
            raise ValueError(f"Entity {entity_idx}: reference has {n_ref} copies but model only {n_model}")
        if match_copies and n_model > MAX_COPIES_FOR_PERMUTATION:
            logger.warning(f"Entity {entity_idx}: {n_model} copies, skipping copy permutation search")
        if match_copies and n_model <= MAX_COPIES_FOR_PERMUTATION:
            options = permutations(range(n_model), n_ref)
        else:
            options = [tuple(range(n_ref))]
        per_entity.append([(entity_idx, opt) for opt in options])
    return [dict(combo) for combo in product(*per_entity)]


def _tm_score(ref_xyz: np.ndarray, model_xyz: np.ndarray, length: int) -> float:
    """
    TM-score (Zhang & Skolnick 2004, d0 as in biotite) normalized by reference length, maximized
    over the RMSD-optimal and an outlier-robust superposition (TM-score is defined as the maximum
    over superpositions; this is a lower bound of the TM-align/US-align optimum)
    """
    d0 = max(1.24 * max(length - 15, 0) ** (1 / 3) - 1.8, 0.5)
    transforms = [struc.superimpose(ref_xyz, model_xyz)]
    if len(ref_xyz) > 3:
        transforms.append(struc.superimpose_without_outliers(ref_xyz, model_xyz)[0])
    return max(
        float(np.sum(1 / (1 + (np.linalg.norm(ref_xyz - t.apply(model_xyz), axis=1) / d0) ** 2)) / length)
        for t in transforms
    )


def compare_structures(
    reference: Chains,
    model: Chains,
    subsets: ResidueSubsets | None = None,
    atom_names: Sequence[str] = ("CA",),
    subset_superposition: Literal["self", "global"] = "self",
    match_copies: bool = True,
) -> list[StructureComparison]:
    """
    Compare one model to one reference (both in entity numbering)

    Parameters
    ----------
    reference
        Map from entity index to reference chain(s), one per copy
    model
        Map from entity index to model chain(s), one per copy; must cover all entities in reference
    subsets
        Named residue subsets (entity, reference copy, pos) for which RMSD is computed in addition
        to the full set of shared atoms
    atom_names
        Atoms used for superposition and RMSD (TM-score always uses CA atoms). Atoms not in
        BACKBONE_ATOMS are only compared at positions with identical residue types
    subset_superposition
        "self": superimpose on subset itself (local geometry);
        "global": reuse full-molecule superposition (subset placement in global frame)
    match_copies
        Search assignments of reference to model copies (homo-oligomers) for lowest full RMSD

    Returns
    -------
    Comparison records (full molecule first, then one per subset with at least 3 shared atoms),
    each including the 4x4 transform applied to model coordinates
    """
    if missing := set(reference) - set(model):
        raise ValueError(f"Model lacks entities present in reference: {sorted(missing)}")
    if subsets and FULL_SUBSET in subsets:
        raise ValueError(f"Subset name '{FULL_SUBSET}' is reserved for the full comparison")
    ref_copies = {(e, c) for e, copies in reference.items() for c in range(len(ensure_sequence(copies)))}
    for name, residues in (subsets or {}).items():
        if invalid := {(int(e), int(c)) for e, c, _ in residues} - ref_copies:
            raise ValueError(f"Subset {name} refers to (entity, copy) not in reference: {sorted(invalid)}")

    ref_table, model_table = _atom_table(reference, atom_names), _atom_table(model, atom_names)

    best = None
    for assignment in _copy_assignments(reference, model, match_copies):
        ref_xyz, model_xyz = _paired_coords(ref_table, model_table, assignment)
        if len(ref_xyz) < 3:
            continue
        transform = struc.superimpose(ref_xyz, model_xyz)
        value = float(struc.rmsd(ref_xyz, transform.apply(model_xyz)))
        if best is None or value < best[0]:
            best = (value, assignment, transform, len(ref_xyz))
    if best is None:
        raise ValueError("Fewer than 3 shared atoms between reference and model")
    full_rmsd, assignment, full_transform, num_atoms = best

    ref_ca, model_ca = _atom_table(reference, ["CA"]), _atom_table(model, ["CA"])
    ref_ca_xyz, model_ca_xyz = _paired_coords(ref_ca, model_ca, assignment)
    tm = _tm_score(ref_ca_xyz, model_ca_xyz, len(ref_ca)) if len(ref_ca_xyz) >= 3 else None

    def record(subset, superposition, n, rmsd, tm_score, transform) -> StructureComparison:
        return StructureComparison(
            subset=subset, superposition=superposition, atoms=list(atom_names), num_atoms=n,
            rmsd=rmsd, tm_score=tm_score, copy_assignment={str(e): list(a) for e, a in assignment.items()},
            transform=transform.as_matrix().reshape(4, 4).tolist(),
        )

    results = [record(FULL_SUBSET, FULL_SUBSET, num_atoms, full_rmsd, tm, full_transform)]
    for name, residues in (subsets or {}).items():
        ref_xyz, model_xyz = _paired_coords(
            ref_table, model_table, assignment, {tuple(int(x) for x in r) for r in residues}
        )
        if len(ref_xyz) < 3:
            logger.warning(f"Subset {name}: only {len(ref_xyz)} shared atoms, skipping")
            continue
        if subset_superposition == "self":
            transform, superposition = struc.superimpose(ref_xyz, model_xyz), name
        else:
            transform, superposition = full_transform, FULL_SUBSET
        rmsd = float(struc.rmsd(ref_xyz, transform.apply(model_xyz)))
        results.append(record(name, superposition, len(ref_xyz), rmsd, None, transform))

    return results


def instance_to_entity_positions(entity: Entity, entity_instance: EntityInstance) -> dict[int, int]:
    """
    Map entity instance positions to entity positions. Following the A3M-style EntityInstance coding,
    uppercase symbols and gaps consume one entity position; lowercase symbols (insertions) and gaps
    (no residue) are not part of the mapping.
    """
    mapping, entity_pos = {}, entity.first_index
    for inst_pos, symbol in enumerate(map(str, entity_instance.rep), start=entity.first_index):
        if symbol.isupper():
            mapping[inst_pos] = entity_pos
        if symbol.isupper() or symbol == GAP:
            entity_pos += 1

    if entity_pos - entity.first_index != len(entity.rep):
        raise ValueError(
            f"Instance rep covers {entity_pos - entity.first_index} entity positions, entity has {len(entity.rep)}"
        )
    return mapping


def models_to_entity_numbering(
    entity: Entity, entity_instance: EntityInstance, chains: Structure | Sequence[Structure]
) -> list[Structure]:
    """
    Validate model chains against the entity instance (numbering and sequence) and renumber
    them into entity numbering, dropping inserted residues
    """
    if entity_instance.rep is None:
        raise ValueError("Entity instance rep must be defined to compare its models")
    positions = range(entity.first_index, entity.first_index + len(entity_instance.rep))
    mapping = instance_to_entity_positions(entity, entity_instance)
    renumbered = []
    for copy_idx, chain in enumerate(ensure_sequence(chains)):
        if not chain.represents(list(positions), entity_instance.rep, allow_missing=True):
            raise ValueError(
                f"Model copy {copy_idx} does not match numbering/sequence of the entity instance "
                f"(expected numbering from first_index along the instance rep)"
            )
        renumbered.append(chain.remap(mapping))
    return renumbered


def _map_to_entity(entity: Entity, chain: Structure, **kwargs) -> Structure:
    """
    Renumber reference chain into entity numbering by sequence alignment, keeping polymer residues only;
    kwargs are passed to map_structure_chain_to_entity_pairwise (e.g. gap_penalty, local)
    """
    arr = chain.atom_array
    polymer = Structure(arr[struc.filter_amino_acids(arr) | struc.filter_nucleotides(arr)])
    return map_structure_chain_to_entity_pairwise(entity, polymer, **kwargs)


class ReferenceStructureComparison(Analyzer):
    """
    Compare predicted models attached to instances (e.g. by BoltzFold) against a reference structure
    (e.g. experimental), storing results in instance.metadata["structure_comparison"]. Results from
    earlier runs with the same reference_id are replaced, results for other references are kept.
    """
    def __init__(
        self,
        reference_id: str = "reference",
        model_ids: Sequence[str] | None = None,
        atom_names: Sequence[str] = ("CA",),
        subset_superposition: Literal["self", "global"] = "self",
        match_copies: bool = True,
        map_reference: bool = False,
        map_reference_kwargs: Mapping[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        reference_id
            Key of reference in system[entity].structures (used if no reference is passed via data);
            also used as reference label in results
        model_ids
            Keys of models in instance[entity].models to compare (None: all models present for all
            compared entities)
        atom_names, subset_superposition, match_copies
            See compare_structures()
        map_reference
            If True, renumber reference chains passed via data into entity numbering by aligning their
            sequence to the entity rep (map_structure_chain_to_entity_pairwise), so chains can be passed
            with their original (e.g. PDB) numbering. Non-polymer residues (waters, ligands) are removed
            first; chains must not contain insertion codes (use use_author_fields=False). References
            attached to the system are never remapped since they are in entity numbering by convention
        map_reference_kwargs
            Keyword arguments passed to map_structure_chain_to_entity_pairwise if map_reference is True,
            e.g. {"local": False, "gap_penalty": (-12, -1)}
        """
        self.reference_id = reference_id
        self.model_ids = model_ids
        self.atom_names = list(atom_names)
        self.subset_superposition = subset_superposition
        self.match_copies = match_copies
        self.map_reference = map_reference
        self.map_reference_kwargs = dict(map_reference_kwargs or {})
        allowed = set(signature(map_structure_chain_to_entity_pairwise).parameters) - {"entity", "chain"}
        if invalid := set(self.map_reference_kwargs) - allowed:
            raise ValueError(f"Invalid map_reference_kwargs {sorted(invalid)}; allowed are {sorted(allowed)}")

    def _reference(self, system: System, reference: Chains | None, entity: int | None) -> dict[int, list[Structure]]:
        from_data = reference is not None
        if not from_data:
            reference = {
                idx: e.structures[self.reference_id] for idx, e in enumerate(system)
                if e.structures is not None and self.reference_id in e.structures
            }
        chains = {
            idx: list(ensure_sequence(c)) for idx, c in reference.items() if entity is None or idx == entity
        }
        if not chains:
            source = "passed via data" if from_data else f"'{self.reference_id}' attached to system"
            raise ValueError(f"No reference structure ({source}) for any selected entity")

        for idx, copies in chains.items():
            if not 0 <= idx < len(system):
                raise ValueError(f"Reference given for entity {idx} not in system")
            if self.map_reference and from_data:
                chains[idx] = copies = [
                    _map_to_entity(system[idx], chain, **self.map_reference_kwargs) for chain in copies
                ]
            # positions() is empty for entities without rep, so zip yields nothing then
            entity_seq = dict(zip(system[idx].positions(), map(str, system[idx].rep if system[idx].rep is not None else ())))
            for copy_idx, chain in enumerate(copies):
                if not chain.represents(list(entity_seq), allow_missing=True):
                    raise ValueError(
                        f"Reference for entity {idx}, copy {copy_idx} is not a single chain numbered "
                        f"consistently with the entity (remap numbering first)"
                    )
                # numbering within range can still be shifted; residue identity catches most offsets
                res = chain.res_df()
                identity = np.mean([entity_seq[r] == aa for r, aa in zip(res.res_id, res.res_name_oneletter)])
                if identity < MIN_REFERENCE_IDENTITY:
                    logger.warning(
                        f"Reference for entity {idx}, copy {copy_idx}: only {identity:.0%} sequence identity "
                        f"to entity rep, check numbering offset"
                    )
        return chains

    def analyze(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        data: StructureComparisonData | None = None,
        entity: int | None = None,
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances with predicted models attached
        data
            Optional dict with keys
            - "reference": {entity: Structure | [Structure per copy]} in entity numbering;
              if missing/None, system[entity].structures[reference_id] is used
            - "subsets": named residue subsets, e.g. {"active_site": [(0, 0, 57), (0, 1, 102)]}
              with tuples (entity, reference copy, position in entity numbering)
        entity
            Restrict comparison to this entity (None: all entities with a reference)

        Returns
        -------
        Unchanged system and shallow copies of instances with updated metadata
        """
        data = data or {}
        if unknown := set(data) - {"reference", "subsets"}:
            raise ValueError(f"Unknown keys in data: {sorted(unknown)}; allowed are 'reference', 'subsets'")

        reference = self._reference(system, data.get("reference"), entity)

        updated = []
        for inst in instances:
            inst = inst.copy()
            inst.metadata = inst.metadata or {}  # copy() already made a shallow copy
            records = [
                r for r in inst.metadata.get(STRUCTURE_COMPARISON_COMPONENT_KEY, [])
                if r.get("reference") != self.reference_id
            ]

            available = set.intersection(*[set(inst[idx].models or {}) for idx in reference])
            for model_id in sorted(available) if self.model_ids is None else self.model_ids:
                if model_id not in available:
                    logger.warning(f"Instance {inst.id}: model '{model_id}' missing, skipping")
                    continue
                model = {
                    idx: models_to_entity_numbering(system[idx], inst[idx], inst[idx].models[model_id])
                    for idx in reference
                }
                for r in compare_structures(
                    reference, model, data.get("subsets"), self.atom_names,
                    self.subset_superposition, self.match_copies,
                ):
                    records.append({"reference": self.reference_id, "model": model_id, **r})

            inst.metadata[STRUCTURE_COMPARISON_COMPONENT_KEY] = records
            updated.append(inst)

        return system, updated


def superposed_model(instance: SystemInstance, record: StructureComparison) -> Structure:
    """
    Superposed predicted model (all atoms, all entities and copies) for a comparison record,
    e.g. to write to file for visual inspection
    """
    matrix = np.array(record["transform"])
    chains = [
        chain
        for entity_instance in instance
        if entity_instance.models is not None and record["model"] in entity_instance.models
        for chain in ensure_sequence(entity_instance.models[record["model"]])
    ]
    merged = Structure.concat(chains).atom_array.copy()
    merged.coord = merged.coord @ matrix[:3, :3].T + matrix[:3, 3]
    return Structure(merged)
