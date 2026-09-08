import pickle
from copy import deepcopy

import numpy as np
import pytest

from evedesign.models.eve import EVE
from evedesign.models.evemodel.params import DEFAULT_MODEL_PARAMETERS
from evedesign.sequence import Sequence, Sequences
from evedesign.system import EntityInstance, Protein, System, SystemInstance


pytestmark = pytest.mark.eve


def _synthetic_system() -> System:
    sequences = Sequences(
        seqs=[
            Sequence("ACDE", id="query"),
            Sequence("ACDE", id="homolog_1"),
            Sequence("ACDF", id="homolog_2"),
            Sequence("ACNE", id="homolog_3"),
            Sequence("ASDE", id="homolog_4"),
            Sequence("VCDE", id="homolog_5"),
        ],
        aligned=True,
        format="a2m",
    )
    return System([Protein(rep="ACDE", id="synthetic", sequences=sequences)])


def _small_hyperparameters() -> dict:
    hyperparameters = deepcopy(DEFAULT_MODEL_PARAMETERS)
    hyperparameters["encoder_parameters"].update(
        hidden_layers_sizes=[8],
        z_dim=2,
    )
    hyperparameters["decoder_parameters"].update(
        hidden_layers_sizes=[8],
        z_dim=2,
        convolve_output=False,
        include_temperature_scaler=False,
    )
    hyperparameters["training_parameters"].update(
        num_training_steps=2,
        batch_size=4,
        log_training_info=False,
        log_training_freq=100,
        save_model_params_freq=100,
    )
    return hyperparameters


@pytest.mark.slow
def test_build_train_serialize_and_score() -> None:
    """Train on a synthetic MSA, serialize the model, and score two sequences."""
    system = _synthetic_system()
    model = EVE(
        model_hyperparameters=_small_hyperparameters(),
        model_checkpoint_path=None,
        use_weights=False,
        preprocess_msa=False,
        num_samples=2,
        batch_size=2,
        device="cpu",
    )

    assert not model.ready
    model.build(system)
    assert model.ready
    assert model.model is not None
    assert model._checkpoint_path is None

    restored = pickle.loads(pickle.dumps(model))
    instances = [
        system.rep_to_instance(),
        SystemInstance([EntityInstance(rep="ACDF")]),
    ]
    scored = restored.score(instances)

    assert restored.ready
    assert restored.model is not None
    assert len(scored) == 2
    assert all(instance.score is not None for instance in scored)
    assert np.isfinite([instance.score for instance in scored]).all()


@pytest.mark.slow
def test_transform_embeds_in_latent_space() -> None:
    """Embed sequences in the VAE latent space via transform()."""
    system = _synthetic_system()
    model = EVE(
        model_hyperparameters=_small_hyperparameters(),
        model_checkpoint_path=None,
        use_weights=False,
        preprocess_msa=False,
        num_samples=2,
        batch_size=2,
        device="cpu",
    ).build(system)

    instances = [
        system.rep_to_instance(),
        SystemInstance([EntityInstance(rep="ACDF")]),
    ]
    transformed = model.transform(instances)

    assert len(transformed) == 2
    for instance in transformed:
        # z_dim = 2 in the small test hyperparameters
        assert instance[0].embedding.shape == (2,)
        assert np.isfinite(instance[0].embedding).all()
        # transform() deliberately does not compute the (expensive) ELBO
        assert instance.score is None

    # different sequences should not collapse onto the same latent mean
    assert not np.allclose(transformed[0][0].embedding, transformed[1][0].embedding)

    # inputs must not be mutated
    assert all(instance[0].embedding is None for instance in instances)

    with pytest.raises(ValueError):
        model.transform(instances, entity=1)


@pytest.mark.slow
def test_generate_samples_sequences() -> None:
    """Sample new sequences from the VAE prior via generate()."""
    system = _synthetic_system()
    target = system[0]
    model = EVE(
        model_hyperparameters=_small_hyperparameters(),
        model_checkpoint_path=None,
        use_weights=False,
        preprocess_msa=False,
        num_samples=2,
        batch_size=2,
        device="cpu",
    ).build(system)

    designs = model.generate(num_designs=3)

    assert len(designs) >= 3
    for design in designs:
        assert len(design[0].rep) == len(target.rep)
        assert set(design[0].rep) <= set("ACDEFGHIKLMNPQRSTVWY")
        assert design.score is not None and np.isfinite(design.score)

    with pytest.raises(ValueError):
        model.generate(num_designs=2, entities=[1])

    # EVE's decoder cannot condition on held-fixed residues, so fixed_pos is rejected
    # rather than silently applied as a post-hoc overwrite
    with pytest.raises(ValueError):
        model.generate(num_designs=2, fixed_pos={0: [target.first_index]})

    with pytest.raises(ValueError):
        model.generate(num_designs=2, fixed_pos={0: [pos for _, pos in model.positions()]})

    with pytest.raises(ValueError):
        model.generate(num_designs=2, fixed_pos={1: [target.first_index]})

    # an empty request fixes nothing, so it is accepted as a no-op
    assert len(model.generate(num_designs=2, fixed_pos={})) >= 2
    assert len(model.generate(num_designs=2, fixed_pos={0: []})) >= 2
