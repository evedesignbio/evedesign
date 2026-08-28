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
