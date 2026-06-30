import math

import pytest

from evedesign.models.evcouplings import EVcouplingsMeanField
from evedesign.sequence import Sequence, Sequences
from evedesign.system import DNA, RNA, Entity, EntityInstance, Protein, System, SystemInstance

pytestmark = pytest.mark.skipif(
    not EVcouplingsMeanField.available,
    reason="evcouplings optional dependency is not importable",
)


def _msa(
    seqs: list[str],
    *,
    type_: str = "protein",
    keys: list[str] | None = None,
) -> Sequences:
    if keys is None:
        keys = [None] * len(seqs)

    return Sequences(
        [
            Sequence(seq=seq, id=f"seq{i}", key=key, type=type_)
            for i, (seq, key) in enumerate(zip(seqs, keys))
        ],
        aligned=True,
        type=type_,
        format="a3m",
    )


@pytest.mark.parametrize(
    ("entity_cls", "seq_type", "target", "variant"),
    [
        (DNA, "dna", "ACGT", "ACGA"),
        (RNA, "rna", "ACGU", "ACGA"),
    ],
)
def test_evcouplings_meanfield_models_single_nucleotide_entity(
    entity_cls,
    seq_type,
    target,
    variant,
):
    system = System([
        entity_cls(
            rep=target,
            first_index=1,
            sequences=_msa(
                [target, variant, target[:1] + "C" + target[2:], target[:2] + "G" + target[3:]],
                type_=seq_type,
            ),
        )
    ])

    can_model, reason = EVcouplingsMeanField.can_model(system)
    assert can_model, reason

    model = EVcouplingsMeanField(theta=0.8).build(system)
    instance = SystemInstance([EntityInstance(rep=variant)])

    assert model.positions() == [(0, 1), (0, 2), (0, 3), (0, 4)]

    scored = model.score([instance])
    assert len(scored) == 1
    assert math.isfinite(scored[0].score)

    scan = model.single_mutation_scan(instance, entity=0, positions=[1])
    assert list(scan.index.names) == ["entity", "pos", "ref"]
    assert set(scan.index.get_level_values("entity")) == {0}
    assert set(scan.index.get_level_values("pos")) == {1}
    assert list(scan.columns) == system[0].alphabet(include_gap=True, include_inserts=False)

    conditional = model.score_conditional([instance], entities=[0], positions=[1])
    assert list(conditional.columns) == system[0].alphabet(include_gap=True, include_inserts=False)


def test_evcouplings_rejects_unsupported_entity_type():
    can_model, reason = EVcouplingsMeanField.can_model(
        System([Entity(type="small_molecule", rep="ATP")])
    )
    assert not can_model
    assert "protein, DNA, or RNA" in reason


def test_evcouplings_meanfield_models_homogeneous_multimer():
    keys = ["focus", "pair1", "pair2", "pair3"]
    system = System([
        Protein(
            id="chain_a",
            rep="ACDE",
            first_index=10,
            sequences=_msa(["ACDE", "ACDF", "TCDE", "AC-E"], keys=keys),
        ),
        Protein(
            id="chain_b",
            rep="FGHI",
            first_index=50,
            sequences=_msa(["FGHI", "FGHI", "YGHI", "FG-I"], keys=keys),
        ),
    ])

    can_model, reason = EVcouplingsMeanField.can_model(system)
    assert can_model, reason

    model = EVcouplingsMeanField(theta=0.8).build(system)
    assert model.positions() == [
        (0, 10), (0, 11), (0, 12), (0, 13),
        (1, 50), (1, 51), (1, 52), (1, 53),
    ]

    instances = [
        SystemInstance([EntityInstance(rep="ACDF"), EntityInstance(rep="FGHI")]),
        SystemInstance([EntityInstance(rep="TCDE"), EntityInstance(rep="YGHI")]),
    ]
    scored = model.score(instances)
    assert len(scored) == 2
    assert all(math.isfinite(instance.score) for instance in scored)

    scan = model.single_mutation_scan(instances[0], entity=1, positions=[50])
    assert set(scan.index.get_level_values("entity")) == {1}
    assert set(scan.index.get_level_values("pos")) == {50}

    conditional = model.score_conditional([instances[0]], entities=[1], positions=[50])
    assert list(conditional.index[0]) == [0, 1, 50]


def test_evcouplings_rejects_mixed_type_multimer():
    system = System([
        Protein(rep="ACDE", first_index=1, sequences=_msa(["ACDE", "ACDF"])),
        DNA(rep="ACGT", first_index=1, sequences=_msa(["ACGT", "ACGA"], type_="dna")),
    ])

    can_model, reason = EVcouplingsMeanField.can_model(system)
    assert not can_model
    assert "same biopolymer type" in reason


def test_evcouplings_rejects_unpaired_msa_keys():
    system = System([
        Protein(
            rep="ACDE",
            first_index=1,
            sequences=_msa(["ACDE", "ACDF"], keys=["focus", "pair-a"]),
        ),
        Protein(
            rep="FGHI",
            first_index=1,
            sequences=_msa(["FGHI", "FGHI"], keys=["focus", "pair-b"]),
        ),
    ])

    can_model, reason = EVcouplingsMeanField.can_model(system)
    assert not can_model
    assert "sequence keys must match" in reason

    with pytest.raises(ValueError, match="sequence keys must match"):
        EVcouplingsMeanField(theta=0.8).build(system)
