"""evedesign.remote.wrapping and wrapped models through the remote worker."""

import dill
import pytest

from evedesign.restraints.physicochemical import MolecularWeightRestraint
from evedesign.remote.wrapping import apply_wraps

SUBSET = "evedesign.model:system_subset_model"
TAGGED = f"{__name__}:tagged"
MW = {"class": "evedesign.restraints.physicochemical:MolecularWeightRestraint",
      "args": {"lower_bound": 0, "upper_bound": 1e9}}


def tagged(cls, label: str = "x"):
    class Tagged(cls):
        pass

    Tagged.label = label
    return Tagged


@pytest.fixture
def complex_():
    from evedesign.system import Protein, System

    return System([Protein(rep="MKTAYIAKQR", id="A"), Protein(rep="GSHMLEDPVK", id="B")])


def test_apply_wraps(complex_):
    wrap = [SUBSET, {"factory": TAGGED, "args": {"label": "z"}}]
    cls = apply_wraps(MolecularWeightRestraint, wrap)
    assert cls.label == "z" and cls.__mro__[1].__name__ == "SubsetModel"      # first entry innermost
    assert apply_wraps(MolecularWeightRestraint, wrap) is cls                  # cached
    model = cls(lower_bound=0, upper_bound=1e9).build(complex_, None, entity_subset=[1])
    assert {e for e, _ in model.positions(None)} == {1}
    with pytest.raises(TypeError, match="tagged.*lable"):
        apply_wraps(MolecularWeightRestraint, [{"factory": TAGGED, "args": {"lable": "z"}}])


def test_remote_wrap_and_build_args(complex_, monkeypatch):
    import evedesign.remote.api as api
    from evedesign.remote import open_built, open_model

    from .test_local import LOCAL, InProcessTransport

    monkeypatch.setattr(api, "launch", lambda target, image: InProcessTransport())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with open_model(LOCAL, {**MW, "wrap": [SUBSET]}) as m:
        m.build(complex_, None, entity_subset=[1])                             # keywords reach build()
        assert m.system is complex_ and {e for e, _ in m.positions(None)} == {1}

    nested = {**MW, "wrap": [SUBSET], "build_args": {"entity_subset": [1]}}   # per nested spec
    spec = {"class": "evedesign.samplers.gibbs:GibbsSampler", "args": {"num_sweeps": 1},
            "models": {"scorers": [nested, nested]}}
    with open_built(LOCAL, spec, complex_) as sampler:
        sampler.generate(num_designs=1)
        assert [r.get("wrap") for r in sampler.provenance["models"]][:2] == [[{"factory": SUBSET, "args": {}}]] * 2
        assert dill.loads(sampler.fetch_serialized()[0]).entity_subset == [1]


def keep(cls):
    return cls


def pair(cls, a: int = 0, b: list | None = None):
    class Pair(cls):
        pass

    return Pair


def test_wrapping_is_safe_with_user_specs():
    from evedesign.remote.wrapping import apply_wraps

    # the allowlist is checked before any import
    with pytest.raises(PermissionError, match="no_such_module:f"):
        apply_wraps(MolecularWeightRestraint, ["no_such_module:f"], allowed=[SUBSET])
    # argument order does not make a different class
    pair_ = f"{__name__}:pair"
    assert apply_wraps(MolecularWeightRestraint, [{"factory": pair_, "args": {"a": 1, "b": [2]}}]) is \
        apply_wraps(MolecularWeightRestraint, [{"factory": pair_, "args": {"b": [2], "a": 1}}])
    # a factory returning its input does not relabel the plain class
    assert apply_wraps(MolecularWeightRestraint, [f"{__name__}:keep"]) is MolecularWeightRestraint
    # wrapped CPU models do not claim a GPU (wrappers expose these as properties)
    from evedesign.remote.worker import _uses_gpu
    assert not _uses_gpu(apply_wraps(MolecularWeightRestraint, [SUBSET]))
