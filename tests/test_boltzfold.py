import pytest
import numpy as np
from evedesign.system import System, Protein, DNA
from evedesign.models.boltzfold import BoltzFoldTransformer
from evedesign.sequence import Sequences, Sequence
from evedesign.utils import ensure_sequence
from evedesign.tools.mmseqs2 import add_sequences_mmseqs2

# every test here constructs BoltzFoldTransformer / imports boltz internals
pytestmark = pytest.mark.boltz2fold

SEQ = "TSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLLSHRPVRDIDRERDLLERLITLGKAHHLDAHYITRLFQLIIEDSVLTQQALLQQH"

# unit tests no forward pass needed

def test_can_model_protein():
    """can_model accepts a single protein entity."""
    s = System([Protein(rep='MAST', id='test')])
    ok, msg = BoltzFoldTransformer.can_model(s)
    assert ok
    assert msg == ""


def test_can_model_rejects_non_protein():
    """can_model rejects DNA entities."""
    s = System([DNA(rep='ATCG', id='test')])
    ok, msg = BoltzFoldTransformer.can_model(s)
    assert not ok
    assert "protein" in msg.lower()


def test_build_sets_ready():
    """build() registers the system and sets ready=True."""
    s = System([Protein(rep='MAST', id='test')])
    m = BoltzFoldTransformer()
    assert not m.ready
    m.build(s)
    assert m.ready
    assert m.system is s


def test_build_rejects_non_protein():
    """build() raises ValueError for non-protein entities."""
    s = System([DNA(rep='ATCG', id='test')])
    m = BoltzFoldTransformer()
    with pytest.raises(ValueError, match="protein"):
        m.build(s)


def test_transform_entity_raises_not_implemented():
    """transform(entity=N) raises NotImplementedError."""
    s = System([Protein(rep='MAST', id='test')])
    m = BoltzFoldTransformer().build(s)
    with pytest.raises(NotImplementedError):
        m.transform([s.rep_to_instance()], entity=1)


# mini integration tests
# specific inputs and expected behaviours


@pytest.mark.slow
def test_transform_single_protein():
    """Single protein folds and returns structure and scores."""
    s = System([Protein(rep=SEQ, id='EcCM', first_index=2)])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    assert len(results) == 1
    assert results[0].score is not None
    assert results[0].confidence is not None
    assert results[0][0].models is not None
    assert "model_0" in results[0][0].models


@pytest.mark.slow
def test_transform_residue_numbering():
    """Output residue numbering matches entity.first_index."""
    s = System([Protein(rep=SEQ, id='EcCM', first_index=2)])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    structure = ensure_sequence(
        results[0][0].models["model_0"]
    )[0]
    assert structure.atom_array.res_id.min() == 2
    assert structure.atom_array.res_id.max() == 2 + len(SEQ) - 1


@pytest.mark.slow
def test_transform_multiple_instances():
    """Two instances of same system fold independently."""
    s = System([Protein(rep=SEQ, id='test')])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    results = m.transform([
        s.rep_to_instance(),
        s.rep_to_instance(),
    ])
    assert len(results) == 2
    assert results[0][0].models is not None
    assert results[1][0].models is not None


@pytest.mark.slow
def test_transform_diffusion_samples():
    """diffusion_samples=3 returns 3 ranked models."""
    s = System([Protein(rep=SEQ, id='test')])
    m = BoltzFoldTransformer(
        device='cpu', diffusion_samples=3, sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    models = results[0][0].models
    assert models is not None
    # diffusion_samples=3 should produce 3 ranked models
    assert "model_0" in models
    assert "model_1" in models
    assert "model_2" in models
    assert results[0].score is not None
    # Per-sample confidences in metadata: a flat
    # list[Score], each entry tagged with the
    # diffusion sample rank in "index".
    scores = results[0].metadata["scores"]
    assert isinstance(scores, list)
    # All three diffusion samples (indices 0,1,2)
    # should appear across the score entries.
    sample_indices = {s["index"] for s in scores}
    assert sample_indices == {0, 1, 2}
    # Each Score entry has the expected shape.
    for sc in scores:
        assert set(sc.keys()) >= {"index", "name", "score"}
        assert isinstance(sc["score"], float)


@pytest.mark.slow
def test_transform_homo_oligomer():
    """Homo-oligomer copies=2 returns list of 2 chain structures."""
    s = System([Protein(rep='MAST', id='homo', copies=2)])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    assert len(results[0]) == 1
    ei = results[0][0]
    assert ei.models is not None
    assert "model_0" in ei.models
    chains_list = ensure_sequence(ei.models["model_0"])
    assert len(chains_list) == 2
    chain_letters = sorted(s.chains()[0] for s in chains_list)
    assert chain_letters == ["A", "B"]


@pytest.mark.slow
def test_transform_with_msa_sequences():
    """Entity with precomputed MSA sequences folds correctly."""
    seqs = Sequences([
        Sequence(seq='MAST', id='hom1'),
        Sequence(seq='VAST', id='hom2'),
    ])
    s = System([Protein(rep='MAST', id='test', sequences=seqs)])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    assert results[0][0].models is not None


@pytest.mark.slow
def test_transform_score_returns_array():
    """score() runs transform internally and returns an ndarray of per-instance scores."""
    s = System([Protein(rep=SEQ, id='test')])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    scores = m.score([s.rep_to_instance()])
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
    assert scores[0] is not None


@pytest.mark.slow
def test_transform_a2b():
    """A2B: 2 copies of entity A + 1 copy of entity B."""
    s = System([
        Protein(rep='MAST', id='A_chain', copies=2),
        Protein(rep='GKLT', id='B_chain'),
    ])
    s = add_sequences_mmseqs2(
        s,
        use_pairing=True,
        user_agent="evedesign-test/test@example.com",
    )
    m = BoltzFoldTransformer(
        device='cpu', use_msa=True, sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    assert len(results) == 1
    assert len(results[0]) == 2
    ei0 = results[0][0]
    ei1 = results[0][1]
    assert ei0.models is not None
    assert ei1.models is not None
    assert "model_0" in ei0.models
    assert "model_0" in ei1.models
    ei0_chains = ensure_sequence(ei0.models["model_0"])
    ei1_chains = ensure_sequence(ei1.models["model_0"])
    assert len(ei0_chains) == 2, \
        f"entity 0: expected 2 chains got {len(ei0_chains)}"
    assert len(ei1_chains) == 1, \
        f"entity 1: expected 1 chain got {len(ei1_chains)}"
    ei0_letters = sorted(s.chains()[0] for s in ei0_chains)
    ei1_letters = [s.chains()[0] for s in ei1_chains]
    assert ei0_letters == ["A", "B"], \
        f"entity 0 letters: {ei0_letters}"
    assert ei1_letters == ["C"], \
        f"entity 1 letters: {ei1_letters}"


@pytest.mark.slow
def test_transform_a2b2():
    """A2B2: 2 copies each of entity A and entity B."""
    s = System([
        Protein(rep='MAST', id='A_chain', copies=2),
        Protein(rep='GKLT', id='B_chain', copies=2),
    ])
    s = add_sequences_mmseqs2(
        s,
        use_pairing=True,
        user_agent="evedesign-test/test@example.com",
    )
    m = BoltzFoldTransformer(
        device='cpu', use_msa=True, sampling_steps=100,
    ).build(s)
    results = m.transform([s.rep_to_instance()])
    assert len(results) == 1
    assert len(results[0]) == 2
    ei0 = results[0][0]
    ei1 = results[0][1]
    assert "model_0" in ei0.models
    assert "model_0" in ei1.models
    ei0_chains = ensure_sequence(ei0.models["model_0"])
    ei1_chains = ensure_sequence(ei1.models["model_0"])
    assert len(ei0_chains) == 2
    assert len(ei1_chains) == 2
    ei0_letters = sorted(s.chains()[0] for s in ei0_chains)
    ei1_letters = sorted(s.chains()[0] for s in ei1_chains)
    assert ei0_letters == ["A", "B"], \
        f"entity 0 letters: {ei0_letters}"
    assert ei1_letters == ["C", "D"], \
        f"entity 1 letters: {ei1_letters}"


def test_score_requires_build():
    """score() raises if build() was not called first."""
    m = BoltzFoldTransformer()
    s = System([Protein(rep='MAST', id='test')])
    with pytest.raises(Exception):
        m.score([s.rep_to_instance()])


@pytest.mark.slow
def test_score_returns_array():
    """score() returns a numpy array of floats."""
    s = System([Protein(rep="MAST", id='test')])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    scores = m.score([s.rep_to_instance()])
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
    assert scores.dtype == float or np.issubdtype(
        scores.dtype, np.floating
    )


@pytest.mark.slow
def test_score_multiple_instances():
    """score() returns one value per instance in input order."""
    s = System([Protein(rep="MAST", id='test')])
    m = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
    ).build(s)
    instances = [s.rep_to_instance(), s.rep_to_instance()]
    scores = m.score(instances)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert all(s is not None for s in scores)
    assert all(np.isfinite(s) for s in scores)


@pytest.mark.slow
def test_score_attribute_confidence_score():
    """score() uses score_attribute to select the metric."""
    s = System([Protein(rep="MAST", id='test')])

    m_ptm = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
        score_attribute='ptm',
    ).build(s)
    m_conf = BoltzFoldTransformer(
        device='cpu', sampling_steps=100,
        score_attribute='confidence_score',
    ).build(s)

    scores_ptm = m_ptm.score([s.rep_to_instance()])
    scores_conf = m_conf.score([s.rep_to_instance()])

    # Both should be valid floats but likely different values
    assert np.isfinite(scores_ptm[0])
    assert np.isfinite(scores_conf[0])
    assert scores_ptm[0] != scores_conf[0], \
        "ptm and confidence_score should differ"
