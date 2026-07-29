"""
Translation layer between evedesign data structures and Boltz-2 inputs/outputs.

This module is the only place that knows about Boltz-2 internals.
boltzfold.py calls these functions without importing boltz directly.

NOTE: Template conditioning (Entity.structures → YAML templates)
is not yet implemented. Structures are ignored with a warning.
"""

from pathlib import Path
from typing import Literal

import yaml
import json
from loguru import logger

from evedesign.system import Entity, EntityInstance, System, SystemInstance, StructureChainMap, Structure
from evedesign.structure import StructureFile
from evedesign.types import Score, RepSequence

# 1. evedesign Entity --> to Boltz-2 YAML

# Chain ID generation
# Boltz-2 uses chain IDs to identify entities in the system. We need to generate unique chain IDs for each entity instance. 
# The convention is to use uppercase letters (A-Z) for the first 26 entities, then double letters (AA, AB, AC, ...) for additional entities. 
# This function generates the appropriate chain ID based on the entity's position in the system.
def _get_chain_id(chain_num: int) -> str:
    """Generate chain ID: A-Z, then AA, AB, AC, ..."""
    if chain_num < 26:
        return chr(65 + chain_num)
    else:
        first = chr(65 + (chain_num - 26) // 26)
        second = chr(65 + (chain_num - 26) % 26)
        return first + second


def _get_chain_ids(system: System) -> list[str]:
    """
    Generate one chain ID per chain in the system.

    IDs follow the sequence A-Z, then AA, AB, AC, ...
    Each entity consumes max(entity.copies, 1) IDs.
    """
    total = sum(
        max(e.copies, 1) if e.copies is not None else 1
        for e in system
    )
    return [_get_chain_id(i) for i in range(total)]


def _chain_to_entity_map(system: System) -> dict[str, int]:
    """Maps each Boltz-2 chain ID back to its evedesign entity index."""
    chain_ids = _get_chain_ids(system)
    result: dict[str, int] = {}
    pointer = 0
    for entity_idx, entity in enumerate(system):
        copies = entity.copies if entity.copies is not None else 1
        for chain_id in chain_ids[pointer:pointer + copies]:
            result[chain_id] = entity_idx
        pointer += copies
    return result


# A3M writer
# a3m format is a simple extension of FASTA that allows for insertions in the MSA.
def _write_a3m(
    entity: Entity,
    entity_instance: EntityInstance,
    output_path: Path,
    old_query: str | RepSequence | None = None,
) -> Path:
    """Write an A3M file with the query sequence followed by MSA hits.

    The MSA hits in ``entity.sequences`` were searched once against a base
    query (the system rep, ``entity.rep``). When folding a SystemInstance
    whose rep differs from that base query, the hit columns are remapped to
    the instance rep via :meth:`Sequences.remap_query`, so a single base MSA
    can be reused across many instances without re-running MMSeqs2.

    Parameters
    ----------
    old_query
        The query the current ``entity.sequences`` were aligned against
        (typically ``entity.rep``). If None, equal to the instance rep, or
        the remap fails, the hits are written verbatim (back-compatible
        behavior).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_query = "".join(entity_instance.rep)

    # Reuse the base MSA across instances by remapping its columns to the
    # current instance rep. Skip when there is nothing to remap against, the
    # query is unchanged, or the remap is not possible — folding with the raw
    # MSA is preferable to crashing.
    sequences = entity.sequences
    if old_query is not None:
        old_query_str = "".join(old_query)
        if old_query_str != new_query:
            try:
                sequences = entity.sequences.remap_query(
                    old_query_str, new_query, prepend_new_query=False
                )
            except (ValueError, NotImplementedError) as exc:
                logger.warning(
                    f"Entity '{entity.id}': could not remap MSA from base query "
                    f"to instance rep ({exc}); writing hits verbatim."
                )
                sequences = entity.sequences

    with open(output_path, "w") as f:
        # Query sequence first. The raw rep (with gaps/lowercase) is what
        # remap_query needs to identify column transitions, but the A3M query
        # line must be the actual designed sequence boltz folds: gaps stripped
        # (deleted positions are not residues) and insertions uppercased.
        header = entity.id or "query"
        query_line = EntityInstance.normalize_rep_str(new_query)
        f.write(f">{header}\n{query_line}\n")

        # MSA hits (remapped to the current instance when applicable)
        for seq in sequences.seqs:
            f.write(f">{seq.id_ or 'seq'}\n{seq.seq}\n")

    return output_path


def _write_csv(
    entity: Entity,
    entity_instance: EntityInstance,
    output_path: Path,
    old_query: str | RepSequence | None = None,
) -> Path:
    """
    Write a Boltz-2 compatible CSV MSA file preserving
    pairing keys from Sequence.key.

    Format matches what Boltz-2's own server path produces:
    - Header: key,sequence
    - Query sequence first with key=0
    - Paired sequences: each distinct key value maps to a
      stable integer taxonomy_id
    - Unpaired sequences with key=None (written as -1 in CSV)

    This format is required for paired MSAs in multi-chain
    complexes. Boltz-2's CSV parser uses the key column as
    taxonomy_id to match paired rows across chains.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_query = "".join(entity_instance.rep)

    sequences = entity.sequences
    if old_query is not None:
        old_query_str = "".join(old_query)
        if old_query_str != new_query:
            try:
                sequences = entity.sequences.remap_query(
                    old_query_str, new_query, prepend_new_query=False
                )
            except (ValueError, NotImplementedError) as exc:
                logger.warning(
                    f"Entity '{entity.id}': could not remap paired MSA from base "
                    f"query to instance rep ({exc}); writing hits verbatim."
                )
                sequences = entity.sequences

    rows = ["key,sequence"]
    rows.append(f"0,{EntityInstance.normalize_rep_str(new_query)}")
    key_to_taxid: dict = {}
    for seq in sequences.seqs:
        if seq.key is not None and seq.key not in key_to_taxid:
            key_to_taxid[seq.key] = len(key_to_taxid)
    for seq in sequences.seqs:
        taxid = key_to_taxid[seq.key] if seq.key is not None else -1
        rows.append(f"{taxid},{seq.seq}")
    output_path.write_text("\n".join(rows) + "\n")
    return output_path


def _resolve_msa_field(
    entity: Entity,
    entity_instance: EntityInstance,
    chain_id: str,
    yaml_path: Path,
    use_msa: bool,
) -> str | None:
    """
    Decide the msa field value for a single entity in the Boltz-2 YAML.

    Returns an absolute path string when a local MSA file was written
    (CSV when pairing keys are present, A3M otherwise), or "empty" when
    no MSA is available.
    """
    # Warn about unsupported template conditioning
    if entity.structures is not None and len(entity.structures) > 0:
        logger.warning(
            f"Entity '{entity.id}': structures are present but "
            f"template conditioning is not yet implemented — ignoring."
        )

    if (
        use_msa
        and entity.sequences is not None
        and len(entity.sequences.seqs) > 0
    ):
        # Paired sequences carry a non-None pairing key (see
        # add_sequences_mmseqs2). The key's format is not inspected —
        # its presence indicates a paired MSA requiring CSV output.
        has_pairing = any(
            s.key is not None for s in entity.sequences.seqs
        )
        if has_pairing:
            msa_path = _write_csv(
                entity, entity_instance,
                yaml_path.parent / "msa" / f"{chain_id}.csv",
                old_query=entity.rep,
            )
        else:
            # entity.rep is the base query MMSeqs2 searched on (see
            # add_sequences_mmseqs2); pass it so the hits can be remapped to
            # this instance's rep instead of triggering a fresh search.
            msa_path = _write_a3m(
                entity, entity_instance,
                yaml_path.parent / "msa" / f"{chain_id}.a3m",
                old_query=entity.rep,
            )
        return str(msa_path.resolve())

    return "empty"


def system_instance_to_yaml(
    system: System,
    instance: SystemInstance,
    output_path: Path,
    use_msa: bool = True,
) -> Path:
    """
    Convert an evedesign System + SystemInstance into a Boltz-2 input YAML.

    Parameters
    ----------
    system
        The evedesign System (defines entities, copies, MSA, structures).
    instance
        A specific SystemInstance whose sequences will be written.
    output_path
        Where to write the YAML file.
    use_msa
        If True and MSA data is available on the entity, write a local
        MSA file (CSV when pairing keys are present, A3M otherwise) and
        reference it. If False, set msa to "empty".

    Returns
    -------
    Path to the written YAML file.

    
    Homo-oligomer mapping:
        evedesign represents a homodimer as a single Entity with copies=2.
        There is one EntityInstance per entity regardless of copy count.
        Boltz-2 expects a list of chain IDs for homo-oligomers:
            copies=1  →  id: "A"        (scalar, monomer)
            copies=2  →  id: ["A","B"]  (list, homodimer)
        Boltz-2 then creates separate chains sharing the same entity_id
        and incrementing sym_id, so it knows they are symmetry-related.
    """
    chain_ids = _get_chain_ids(system)
    pointer = 0
    sequences = []

    for entity, entity_instance in zip(system, instance):
        copies = entity.copies if entity.copies is not None else 1
        first_chain = chain_ids[pointer]
        id_field = first_chain if copies == 1 else chain_ids[pointer:pointer + copies]

        seq = "".join(entity_instance.rep)

        msa = _resolve_msa_field(
            entity, entity_instance, first_chain, output_path,
            use_msa=use_msa,
        )

        entry: dict = {"id": id_field, "sequence": seq}
        if msa is not None:
            entry["msa"] = msa

        sequences.append({"protein": entry})
        pointer += copies

    data = {"version": 1, "sequences": sequences}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return output_path


def prediction_to_instance(
    record_id: str,
    predictions_dir: Path,
    system: System,
    instance: SystemInstance,
    chain_to_entity: dict[str, int],
    score_attribute: Literal[
        "iptm", "ptm", "confidence_score", "complex_plddt"
    ] = "confidence_score",
    confidence_attribute: Literal[
        "iptm", "ptm", "confidence_score", "complex_plddt"
    ] = "complex_plddt",
) -> SystemInstance:
    """
    Parse BoltzWriter output files for one record into a
    SystemInstance with populated structures and scores.

    Only the best-ranked model (model_0, highest confidence)
    is parsed and stored in EntityInstance.models.

    NOTE: Boltz-2 always numbers residues from 1 internally.
    Output residue numbering is remapped to match each
    entity's first_index before populating EntityInstance.models.

    NOTE: Support for returning all diffusion samples is not
    yet implemented. When added, the CIF parsing logic should
    be extracted into a separate _parse_cif_to_chain_map()
    helper and iterated over all model_{i}.cif files.
    """
    # Locate output files for this record
    record_dir = predictions_dir / record_id
    if not record_dir.exists():
        raise ValueError(
            f"No prediction output found for record "
            f"'{record_id}' in {predictions_dir}"
        )

    cif_files = sorted(record_dir.glob("*.cif"))
    json_files = sorted(record_dir.glob("confidence_*.json"))

    if not cif_files:
        raise ValueError(
            f"No .cif files found for record "
            f"'{record_id}' in {record_dir}"
        )

    # Load all per-rank confidence JSONs
    all_confidence: dict[str, dict] = {}
    for rank_idx, json_path in enumerate(json_files):
        rank_key = f"model_{rank_idx}"
        all_confidence[rank_key] = json.loads(
            json_path.read_text()
        )

    # The score and confidence on SystemInstance are
    # taken from model_0 (best ranked by Boltz-2)
    best_confidence = all_confidence.get("model_0", {})

    # SystemInstance.score holds the score_attribute
    # value (e.g. "confidence_score") of the best-ranked
    # diffusion sample (model_0). Per-sample scores for
    # all ranks are stored in metadata["scores"].
    score = best_confidence.get(score_attribute, None)
    if score is None and best_confidence:
        raise ValueError(
            f"'{score_attribute}' not found in Boltz-2 "
            f"confidence output. "
            f"Available keys: {list(best_confidence.keys())}"
        )

    # SystemInstance.confidence holds the
    # confidence_attribute value (e.g. "complex_plddt")
    # of the best-ranked diffusion sample (model_0).
    confidence_val = best_confidence.get(confidence_attribute, None)
    if confidence_val is None and best_confidence:
        raise ValueError(
            f"'{confidence_attribute}' not found in Boltz-2 "
            f"confidence output. "
            f"Available keys: {list(best_confidence.keys())}"
        )

    entity_models: dict[int, StructureChainMap] = {}

    for rank_idx, cif_path in enumerate(cif_files):
        rank_key = f"model_{rank_idx}"
        entity_chains: dict[int, list[Structure]] = {}

        sf = StructureFile(str(cif_path), format="cif")
        full_structure = sf.get_model()

        for chain_id in full_structure.chains():
            if chain_id not in chain_to_entity:
                logger.warning(
                    f"Chain '{chain_id}' not in "
                    f"chain_to_entity mapping — skipping"
                )
                continue
            entity_idx = chain_to_entity[chain_id]
            entity = system[entity_idx]

            chain_structure = full_structure.get_chain(chain_id)

            n = len(entity.rep)
            mapping = {
                i: i + entity.first_index - 1
                for i in range(1, n + 1)
            }
            remapped = chain_structure.remap(mapping)

            if entity_idx not in entity_chains:
                entity_chains[entity_idx] = []
            entity_chains[entity_idx].append(remapped)

        for entity_idx, chains in entity_chains.items():
            if entity_idx not in entity_models:
                entity_models[entity_idx] = {}
            if len(chains) == 1:
                entity_models[entity_idx][rank_key] = chains[0]
            else:
                entity_models[entity_idx][rank_key] = chains

    # Build output EntityInstance objects by copying inputs
    # and attaching the predicted structures
    new_entity_instances = []
    for i, entity_instance in enumerate(instance):
        new_ei = entity_instance.copy()
        new_ei.models = entity_models.get(i, None)
        new_entity_instances.append(new_ei)

    # Flatten per-rank confidence metrics into list[Score]
    # so metadata["scores"] matches the typed Metadata schema.
    # index = diffusion sample rank (0 = best).
    scores_list: list[Score] = []
    for rank_key, conf_dict in all_confidence.items():
        rank_idx = int(rank_key.removeprefix("model_"))
        for metric_name, value in conf_dict.items():
            if not isinstance(value, (int, float)):
                continue
            scores_list.append({
                "index": rank_idx,
                "name": metric_name,
                "weight": 1.0,
                "score": float(value),
                "ref_score": None,
            })

    new_instance = instance.copy()
    new_instance.data = new_entity_instances
    new_instance.score = score
    new_instance.confidence = confidence_val

    # Merge Boltz scores into existing metadata so caller-attached
    # keys (e.g. provenance) survive the transform.
    if new_instance.metadata is None:
        new_instance.metadata = {}
    new_instance.metadata["scores"] = scores_list

    return new_instance
