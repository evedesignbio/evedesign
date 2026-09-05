import pytest
from evedesign.mcp import registry


def test_discovers_known_classes_across_every_category():
    keys = {entry.key for entry in registry.list_all()}
    # one representative class per category; every category should be non-empty
    assert "GibbsSampler" in keys
    assert "IsoelectricPointRestraint" in keys
    assert "DNAChiselCodonOptimizer" in keys
    assert "SequenceSpaceMDS" in keys
    assert "OneHotEmbedder" in keys


def test_base_install_classes_are_available_without_optional_extras():
    entry = registry.get("OneHotEmbedder")
    assert entry.available is True
    assert entry.extras == ()
    assert "Transformer" in entry.interfaces


def test_heavy_dependency_classes_are_discovered_but_marked_unavailable():
    entry = registry.get("ESM2")
    assert entry.available is False
    assert "esm2" in entry.extras
    assert entry.citations


def test_supervised_module_extras_do_not_leak_between_its_two_classes():
    sklearn_entry = registry.get("SklearnPredictorOnEmbeddingsScores")
    gpytorch_entry = registry.get("GpytorchModel")
    assert sklearn_entry.extras == ()
    assert sklearn_entry.available is True
    assert "gpytorch" in gpytorch_entry.extras


def test_get_by_full_class_path():
    entry = registry.get("evedesign.samplers.gibbs.GibbsSampler")
    assert entry.key == "GibbsSampler"


def test_get_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get("NotARealModel")


def test_search_finds_relevant_entries_by_keyword():
    scored = registry.search("codon optimization")
    assert any(entry.key == "DNAChiselCodonOptimizer" for _, entry in scored)


def test_list_utilities_returns_callable_functions():
    utilities = registry.list_utilities()
    names = {u.name for u in utilities}
    assert "assign_scores_to_instances" in names
