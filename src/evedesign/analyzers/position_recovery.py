"""
Sequence-level check of how many fixed positions (e.g. active site residues) are maintained in designs
relative to a wild-type (WT) sequence
"""
from typing import Mapping, Sequence, TypedDict

from evedesign.analysis import Analyzer
from evedesign.analyzers.structure_comparison import entity_sequence, instance_to_entity_positions
from evedesign.constants import GAP
from evedesign.system import System, SystemInstance
from evedesign.types import PositionRecovery, POSITION_RECOVERY_COMPONENT_KEY

# name of position set -> (entity index, position in entity numbering)
PositionSets = Mapping[str, Sequence[tuple[int, int]]]


class PositionRecoveryData(TypedDict, total=False):
    """
    data argument of FixedPositionRecovery.analyze()
    """
    positions: PositionSets  # required
    reference: Mapping[int, str] | None  # WT sequence per entity in entity numbering (default: entity rep)


class FixedPositionRecovery(Analyzer):
    """
    Fraction of fixed positions at which designs keep the WT residue, stored per position set in
    instance.metadata["position_recovery"]. Deleted positions count as not maintained, insertions are
    ignored. Results for position sets with the same name from earlier runs are replaced.
    """
    def analyze(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        data: PositionRecoveryData | None = None,
        entity: int | None = None,
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Designs to analyze (rep must be defined for all entities referenced by positions)
        data
            Dict with keys
            - "positions": named position sets, e.g. {"active_site": [(0, 34), (0, 47)]} with tuples
              (entity, position in entity numbering)
            - "reference" (optional): WT sequence per entity {entity: "MKV..."} in entity numbering
              (same length as entity rep); if missing/None, the entity rep is used
        entity
            Restrict analysis to positions of this entity (None: all entities)

        Returns
        -------
        Unchanged system and shallow copies of instances with updated metadata
        """
        if data is None:
            raise ValueError("data['positions'] must define at least one position set")
        if unknown := set(data) - {"positions", "reference"}:
            raise ValueError(f"Unknown keys in data: {sorted(unknown)}; allowed are 'positions', 'reference'")
        position_data = data.get("positions")
        if not position_data:
            raise ValueError("data['positions'] must define at least one position set")

        # duplicates removed (order kept) so each position counts once
        position_sets = {
            name: list(dict.fromkeys((int(e), int(p)) for e, p in positions if entity is None or int(e) == entity))
            for name, positions in position_data.items()
        }
        entities = sorted({e for positions in position_sets.values() for e, _ in positions})

        # WT residue per (entity, position): entity rep, or reference sequence if given
        reference = data.get("reference") or {}
        polymers, wt = {}, {}
        for e in entities:
            if not 0 <= e < len(system):
                raise ValueError(f"Entity {e} not in system")
            polymers[e] = system.biopolymer(e)
            wt[e] = entity_sequence(polymers[e])
            seq = reference.get(e)
            if seq is not None:
                if len(seq) != len(wt[e]):
                    raise ValueError(f"Reference for entity {e} has length {len(seq)}, entity has {len(wt[e])}")
                # uppercase to match design residues (only uppercase instance symbols map to positions)
                wt[e] = dict(zip(wt[e], (str(symbol).upper() for symbol in seq)))

        for name, positions in position_sets.items():
            if invalid := [(e, p) for e, p in positions if p not in wt[e]]:
                raise ValueError(f"Position set {name} contains positions outside entity: {invalid}")

        updated = []
        for inst in instances:
            # design residue per (entity, entity position); deleted positions are absent
            design = {}
            for e, polymer in polymers.items():
                instance_rep = inst[e].rep
                if instance_rep is None:
                    raise ValueError(f"Instance {inst.id}: rep of entity {e} not defined")
                for inst_pos, entity_pos in instance_to_entity_positions(polymer, inst[e]).items():
                    design[(e, entity_pos)] = str(instance_rep[inst_pos - polymer.first_index])

            inst = inst.copy()
            inst.metadata = inst.metadata or {}
            records = [
                r for r in inst.metadata.get(POSITION_RECOVERY_COMPONENT_KEY, [])
                if r["subset"] not in position_sets
            ]
            for name, positions in position_sets.items():
                mutated: list[list[int | str]] = [
                    [e, p, wt[e][p], design.get((e, p), GAP)]
                    for e, p in positions if design.get((e, p), GAP) != wt[e][p]
                ]
                records.append(PositionRecovery(
                    subset=name,
                    num_positions=len(positions),
                    num_maintained=len(positions) - len(mutated),
                    fraction=(len(positions) - len(mutated)) / len(positions) if positions else None,
                    mutated=mutated,
                ))
            inst.metadata[POSITION_RECOVERY_COMPONENT_KEY] = records
            updated.append(inst)

        return system, updated
