import math

import pytest

from evedesign.models import oasis_humanness
from evedesign.models.oasis_humanness import OASisHumanness
from evedesign.system import (
    Entity,
    EntityInstance,
    Mutation,
    Protein,
    System,
    SystemInstance,
)


class FakePrombDatabase:
    def compute_peptide_content(self, sequence: str) -> float:
        return sequence.count("W") / len(sequence)


@pytest.fixture
def mock_promb(monkeypatch):
    monkeypatch.setattr(
        oasis_humanness, "init_db", lambda name, verbose=True: FakePrombDatabase()
    )
    monkeypatch.setattr(OASisHumanness, "available", True)


def test_raises_clear_error_without_optional_dependency():
    if oasis_humanness.IMPORT_AVAILABLE:
        pytest.skip("promb is installed")

    with pytest.raises(ImportError, match="promb"):
        OASisHumanness()


def test_can_model_accepts_protein_only_systems_without_sequences():
    valid, reason = OASisHumanness.can_model(System([Protein()]))
    assert valid, reason

    valid, reason = OASisHumanness.can_model(
        System([Entity(type="dna", rep="AAAAAAAAA", first_index=1)])
    )
    assert not valid
    assert "protein" in reason


def test_scores_all_chains_with_weighted_mean_aggregation(mock_promb):
    system = System(
        [
            Protein(id="heavy", rep="AAAAAAAAA", first_index=1),
            Protein(id="light", rep="WWWWWWWWWW", first_index=1),
        ]
    )
    instance = SystemInstance(
        [
            EntityInstance(rep="WWAAAAAAA"),
            EntityInstance(rep="WWWWWWWWWW"),
        ]
    )

    scored = OASisHumanness().build(system).score([instance])
    assert scored[0] is not instance
    assert instance.score is None
    assert scored[0].score == pytest.approx(((2 / 9) + (2 * 1.0)) / 3)


def test_supports_other_aggregation_methods(mock_promb):
    system = System([Protein(), Protein()])
    instance = SystemInstance(
        [
            EntityInstance(rep="WAAAAAAAA"),
            EntityInstance(rep="WWAAAAAAA"),
        ]
    )

    aggregations = ["mean", "min", "max", sum]
    scores = [
        OASisHumanness(aggregation).build(system).score([instance])[0].score
        for aggregation in aggregations
    ]
    assert scores == pytest.approx([1 / 6, 1 / 9, 2 / 9, 1 / 3])


def test_default_mutation_scorer_returns_relative_scores(mock_promb):
    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    instance = SystemInstance([EntityInstance(rep="AAAAAAAAA")])
    model = OASisHumanness().build(system)

    scored = model.score_mutants(
        instance,
        [[Mutation(entity=0, pos=1, ref="A", to="W")]],
    )

    assert math.isclose(scored[0].score, 1 / 9)


def test_database_is_loaded_only_while_scoring(monkeypatch):
    calls = []

    def spy_init_db(name, verbose=True):
        calls.append(name)
        return FakePrombDatabase()

    monkeypatch.setattr(oasis_humanness, "init_db", spy_init_db)
    monkeypatch.setattr(OASisHumanness, "available", True)

    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    model = OASisHumanness().build(system)
    assert calls == []
    assert model._db is None

    model.score([SystemInstance([EntityInstance(rep="AAAAAAAAA")])])
    assert calls == ["human-oas"]
    assert model._db is None

    model.score([SystemInstance([EntityInstance(rep="AAAAAAAAA")])])
    assert calls == ["human-oas", "human-oas"]
    assert model._db is None


def test_scores_short_instance_sequences_as_zero(mock_promb):
    system = System([Protein()])
    model = OASisHumanness().build(system)

    scored = model.score([SystemInstance([EntityInstance(rep="AAAAAAAA")])])
    assert scored[0].score == 0.0


def test_normalizes_insertions_and_deletions(mock_promb):
    system = System([Protein()])
    instance = SystemInstance([EntityInstance(rep="WWww-AAAAAAA")])

    scored = OASisHumanness().build(system).score([instance])
    assert scored[0].score == pytest.approx(4 / 11)


def test_score_of_empty_instance_list_is_empty(mock_promb):
    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    assert OASisHumanness().build(system).score([]) == []


@pytest.mark.skipif(
    not oasis_humanness.IMPORT_AVAILABLE, reason="promb is not installed"
)
def test_real_promb_ranks_humanized_above_murine():
    humanized_vh = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
        "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    )
    murine_vh = (
        "QVQLQQSGPELVKPGASVKISCKASGYTFTDYNMDWVKQSHGKSLEWIGDINPNNGGTIYNQKFKG"
        "KATLTVDKSSSTAYMELRSLTSEDTAVYYCARNYYGSSLSMDYWGQGTSVTVSS"
    )

    system = System([Protein(rep=humanized_vh, first_index=1)])
    model = OASisHumanness().build(system)
    scored = model.score(
        [
            SystemInstance([EntityInstance(rep=humanized_vh)]),
            SystemInstance([EntityInstance(rep=murine_vh)]),
        ]
    )

    assert 0.0 <= scored[1].score < scored[0].score <= 1.0
