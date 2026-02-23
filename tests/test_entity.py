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
