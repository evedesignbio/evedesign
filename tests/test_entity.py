from evedesign.system import System, Entity, SystemInstance, EntityInstance

def test_system_validation():
    system = System([
        Entity(type="dna", rep="AATT", first_index=1)
    ])

    # check length verification
    assert system.valid_instance(
        SystemInstance([EntityInstance(rep="AAAA")]),
        fixed_length=True
    )

    assert not system.valid_instance(
        SystemInstance([EntityInstance(rep="AAAAA")]),
        fixed_length=True
    )

    # check gap verification
    assert system.valid_instance(
        SystemInstance([EntityInstance(rep="----")]),
        allow_deletions=True,
        fixed_length=True
    )

    # check insert verification
    assert system.valid_instance(
        SystemInstance([EntityInstance(rep="AAtAA")]),
        fixed_length=False
    )


def test_apply_instance_takes_rep_from_instance():
    system = System([Entity(type="protein", rep="ACDE", first_index=5)])
    updated = system.apply_instance(
        SystemInstance([EntityInstance(rep="WYFG")])
    )

    assert "".join(updated[0].rep) == "WYFG"
    # entity-level attributes carry over
    assert updated[0].first_index == 5


def test_apply_instance_normalizes_inserts_and_deletions():
    system = System([Entity(type="protein", rep="ACDE", first_index=1)])
    updated = system.apply_instance(
        SystemInstance([EntityInstance(rep="AaC-E")])
    )

    # lowercase inserts become regular symbols, gaps are dropped
    assert "".join(updated[0].rep) == "AACE"


def test_apply_instance_keeps_ligand_rep_type():
    # ligand_rep_type is mandatory for ligands, so dropping it raised
    system = System([
        Entity(type="protein", rep="ACDE", first_index=1),
        Entity(type="ligand", rep="CCO", ligand_rep_type="smiles"),
    ])
    updated = system.apply_instance(SystemInstance([
        EntityInstance(rep="WYFG"),
        EntityInstance(rep="CCO"),
    ]))

    assert updated[1].ligand_rep_type == "smiles"


def test_apply_instance_falls_back_to_entity_rep():
    # EntityInstance.rep is None when only a backbone is available
    system = System([Entity(type="protein", rep="ACDE", first_index=1)])
    updated = system.apply_instance(
        SystemInstance([EntityInstance(rep=None)])
    )

    assert "".join(updated[0].rep) == "ACDE"
