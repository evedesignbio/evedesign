import json
import pickle

import pandas as pd
import pytest

from evedesign.models import sapiens_humanizer
from evedesign.models.sapiens_humanizer import SapiensHumanizer
from evedesign.system import DNA, Protein, System

HEAVY = "HAAAAAAAAAAA"
LIGHT = "LCCCCCCCCCCC"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class FakePosition:
    def __init__(self, in_cdr):
        self.in_cdr = in_cdr

    def is_in_cdr(self):
        return self.in_cdr


class FakeChain:
    def __init__(self, sequence, chain_type=None, **kwargs):
        self.seq = sequence
        self.chain_type = chain_type or ("H" if sequence.startswith("H") else "K")
        self.positions = [
            FakePosition(index in {1, 2, 3}) for index in range(len(sequence))
        ]

    def clone(self, replace_seq=None):
        return FakeChain(
            self.seq if replace_seq is None else replace_seq,
            chain_type=self.chain_type,
        )

    def graft_cdrs_onto(self, other, backmutate_vernier=False):
        sequence = list(other.seq)
        sequence[1:4] = self.seq[1:4]
        if backmutate_vernier:
            sequence[4] = self.seq[4]
        return self.clone("".join(sequence))


def fake_predict_scores(seq, chain_type):
    return pd.DataFrame(
        [
            {amino_acid: float(amino_acid == "W") for amino_acid in AMINO_ACIDS}
            for _ in seq
        ]
    )


@pytest.fixture
def mock_sapiens(monkeypatch):
    monkeypatch.setattr(sapiens_humanizer, "Chain", FakeChain)
    monkeypatch.setattr(sapiens_humanizer, "predict_scores", fake_predict_scores)
    monkeypatch.setattr(SapiensHumanizer, "available", True)


def antibody_system(heavy=HEAVY, light=LIGHT):
    return System(
        [
            Protein(id="VH", rep=heavy, first_index=1),
            Protein(id="VL", rep=light, first_index=1),
        ]
    )


def test_raises_clear_error_without_optional_dependency():
    if sapiens_humanizer.IMPORT_AVAILABLE:
        pytest.skip("Sapiens dependencies are installed")

    with pytest.raises(ImportError, match="sapiens.*optional dependency"):
        SapiensHumanizer()


@pytest.mark.parametrize(
    ("iterations", "error", "message"),
    [
        (0, ValueError, "positive integer"),
        (-1, ValueError, "positive integer"),
        (1.5, TypeError, "integer"),
        (True, TypeError, "integer"),
    ],
)
def test_rejects_invalid_iterations(mock_sapiens, iterations, error, message):
    with pytest.raises(error, match=message):
        SapiensHumanizer(iterations=iterations)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"scheme": ""}, ValueError, "scheme"),
        ({"scheme": None}, TypeError, "scheme"),
        ({"cdr_definition": ""}, ValueError, "cdr_definition"),
        ({"humanize_cdrs": 1}, TypeError, "humanize_cdrs"),
        ({"backmutate_vernier": 1}, TypeError, "backmutate_vernier"),
        (
            {"humanize_cdrs": True, "backmutate_vernier": True},
            ValueError,
            "Vernier",
        ),
    ],
)
def test_rejects_invalid_configuration(mock_sapiens, kwargs, error, message):
    with pytest.raises(error, match=message):
        SapiensHumanizer(**kwargs)


def test_can_model_requires_paired_antibody_variable_regions(mock_sapiens):
    valid, reason = SapiensHumanizer.can_model(antibody_system())
    assert valid, reason

    invalid_systems = [
        (System([Protein(rep=HEAVY)]), "exactly two"),
        (System([Protein(rep=HEAVY), DNA(rep="ATGC")]), "protein"),
        (System([Protein(rep=HEAVY), Protein()]), "defined rep"),
        (
            System(
                [
                    Protein(rep="HA-", deletions=True),
                    Protein(rep=LIGHT),
                ]
            ),
            "canonical",
        ),
        (antibody_system(light=HEAVY), "heavy and one light"),
    ]
    for system, expected in invalid_systems:
        valid, reason = SapiensHumanizer.can_model(system)
        assert not valid
        assert expected in reason

    valid, reason = SapiensHumanizer.can_model(antibody_system(), data={})
    assert not valid
    assert "data parameter" in reason


def test_build_keeps_only_serializable_framework_state(mock_sapiens):
    model = SapiensHumanizer().build(antibody_system())

    assert model.system is not None
    assert not any(isinstance(value, FakeChain) for value in vars(model).values())
    restored = pickle.loads(pickle.dumps(model))
    assert restored.ready
    assert restored.system == model.system


def test_generate_preserves_cdrs_fixed_positions_and_unselected_entities(
    mock_sapiens,
):
    system = antibody_system()
    designs = (
        SapiensHumanizer()
        .build(system)
        .generate(
            num_designs=2,
            entities=[0],
            fixed_pos={0: [6]},
        )
    )

    assert len(designs) == 2
    assert "".join(designs[0][0].rep[1:4]) == HEAVY[1:4]
    assert designs[0][0].rep[5] == HEAVY[5]
    assert "".join(designs[0][1].rep) == LIGHT
    assert "".join(system[0].rep) == HEAVY
    assert designs[0] is not designs[1]
    assert designs[0][0] is not designs[1][0]
    assert designs[0].metadata is not designs[1].metadata
    assert designs[0].metadata["sapiens"] is not designs[1].metadata["sapiens"]

    metadata = designs[0].metadata["sapiens"]
    assert metadata["entities"] == [0]
    assert metadata["fixed_positions"] == {0: [6]}
    assert metadata["mutations"]
    assert all(mutation["entity"] == 0 for mutation in metadata["mutations"])
    json.dumps(designs[0].serialize())


def test_can_humanize_cdrs_when_requested(mock_sapiens):
    design = (
        SapiensHumanizer(humanize_cdrs=True)
        .build(antibody_system())
        .generate(num_designs=1)[0]
    )

    assert "".join(design[0].rep[1:4]) == "WWW"
    assert "".join(design[1].rep[1:4]) == "WWW"


def test_can_backmutate_vernier_positions(mock_sapiens):
    design = (
        SapiensHumanizer(backmutate_vernier=True)
        .build(antibody_system())
        .generate(num_designs=1)[0]
    )

    assert design[0].rep[4] == HEAVY[4]
    assert design[1].rep[4] == LIGHT[4]


def test_runs_each_iteration_from_the_previous_design(mock_sapiens, monkeypatch):
    inputs = []

    def record_predictions(seq, chain_type):
        inputs.append(seq)
        return fake_predict_scores(seq, chain_type)

    monkeypatch.setattr(sapiens_humanizer, "predict_scores", record_predictions)
    SapiensHumanizer(iterations=2, humanize_cdrs=True).build(
        antibody_system()
    ).generate(num_designs=1, entities=[0])

    assert inputs == [HEAVY, "W" * len(HEAVY)]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"num_designs": 0}, ValueError, "positive integer"),
        ({"num_designs": True}, TypeError, "integer"),
        (
            {"num_designs": 1, "temperature": 0.5},
            ValueError,
            "temperature=1.0",
        ),
        ({"num_designs": 1, "temperature": True}, TypeError, "real number"),
        ({"num_designs": 1, "entities": []}, ValueError, "at least one"),
        ({"num_designs": 1, "entities": [0, 0]}, ValueError, "duplicate"),
        ({"num_designs": 1, "entities": [True]}, TypeError, "integer"),
        ({"num_designs": 1, "entities": [2]}, ValueError, "valid system"),
        ({"num_designs": 1, "entities": 0}, TypeError, "sequence"),
        (
            {"num_designs": 1, "entities": [0], "fixed_pos": {1: [2]}},
            ValueError,
            "selected entities",
        ),
        ({"num_designs": 1, "fixed_pos": []}, TypeError, "must map"),
        ({"num_designs": 1, "fixed_pos": {True: [2]}}, TypeError, "keys"),
        ({"num_designs": 1, "fixed_pos": {0: "2"}}, TypeError, "sequences"),
        (
            {"num_designs": 1, "fixed_pos": {0: [2, 2]}},
            ValueError,
            "duplicate",
        ),
        ({"num_designs": 1, "fixed_pos": {0: [False]}}, TypeError, "integers"),
        (
            {"num_designs": 1, "fixed_pos": {0: [100]}},
            ValueError,
            "Invalid positions",
        ),
    ],
)
def test_rejects_invalid_generation_options(mock_sapiens, kwargs, error, message):
    model = SapiensHumanizer().build(antibody_system())

    with pytest.raises(error, match=message):
        model.generate(**kwargs)


def test_requires_build_before_generation(mock_sapiens):
    with pytest.raises(ValueError, match=r"build\(\)"):
        SapiensHumanizer().generate(num_designs=1)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (pd.DataFrame([{"W": 1.0}]), "unexpected length"),
        (pd.DataFrame([{"?": 1.0}] * len(HEAVY)), "invalid predicted"),
        ([{"W": 1.0}] * len(HEAVY), "unexpected format"),
    ],
)
def test_rejects_malformed_sapiens_output(mock_sapiens, monkeypatch, scores, message):
    monkeypatch.setattr(
        sapiens_humanizer,
        "predict_scores",
        lambda seq, chain_type: scores,
    )
    model = SapiensHumanizer().build(antibody_system())

    with pytest.raises(RuntimeError, match=message):
        model.generate(num_designs=1, entities=[0])


def test_reports_status_after_validation(mock_sapiens):
    events = []
    SapiensHumanizer().build(antibody_system()).generate(
        num_designs=1,
        status_callback=lambda *event: events.append(event),
    )

    assert events == [
        ("running", None, "Humanizing antibody sequences"),
        ("done", None, "Humanization complete"),
    ]


@pytest.mark.sapiens
@pytest.mark.skipif(
    not sapiens_humanizer.IMPORT_AVAILABLE,
    reason="Sapiens dependencies are not installed",
)
def test_real_sapiens_preserves_cdrs_and_serializes_metadata():
    from abnumber import Chain

    heavy = (
        "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFK"
        "NRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
    )
    light = (
        "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGS"
        "RSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    )
    design = (
        SapiensHumanizer()
        .build(antibody_system(heavy=heavy, light=light))
        .generate(num_designs=1)[0]
    )

    humanized_heavy = "".join(design[0].rep)
    humanized_light = "".join(design[1].rep)
    assert humanized_heavy != heavy
    assert humanized_light != light

    for parental, humanized in [(heavy, humanized_heavy), (light, humanized_light)]:
        parental_chain = Chain(
            parental, scheme="kabat", cdr_definition="kabat", use_anarcii=True
        )
        humanized_chain = Chain(
            humanized, scheme="kabat", cdr_definition="kabat", use_anarcii=True
        )
        assert humanized_chain.cdr1_seq == parental_chain.cdr1_seq
        assert humanized_chain.cdr2_seq == parental_chain.cdr2_seq
        assert humanized_chain.cdr3_seq == parental_chain.cdr3_seq

    assert design.metadata["sapiens"]["mutations"]
    json.dumps(design.serialize())
