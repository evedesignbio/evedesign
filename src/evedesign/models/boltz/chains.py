"""
Chain ID utilities shared between BoltzFold and BoltzGen.

Boltz-2 and BoltzGen both use alphabetic chain IDs
(A-Z, then AA, AB, ...). This module centralizes the
chain ID generation and entity mapping logic.
"""
from evedesign.system import System


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
    """Map each chain ID back to its evedesign entity index."""
    chain_ids = _get_chain_ids(system)
    result: dict[str, int] = {}
    pointer = 0
    for entity_idx, entity in enumerate(system):
        copies = (
            entity.copies if entity.copies is not None
            else 1
        )
        for chain_id in chain_ids[pointer:pointer + copies]:
            result[chain_id] = entity_idx
        pointer += copies
    return result
