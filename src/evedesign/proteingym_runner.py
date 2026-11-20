"""Generic ProteinGym2 runner for evedesign models.

Installed as the ``evedesign-proteingym`` console script (``evedesign[proteingym]``).
ProteinGym model images use it as their entrypoint; the model card's
``model_class`` and ``hyper_parameters`` select and configure the model.
"""

import importlib
import pickle
from pathlib import Path
from typing import Annotated, Any

import dill
import polars as pl
import typer
from evedesign.dataset import LabeledInstanceDataset
from evedesign.model import Scorer, Transformer
from evedesign.proteingym import dataset_to_evedesign
from evedesign.system import System
from proteingym.base import Subsets
from proteingym.base.model import ModelCard
from rich.console import Console

app = typer.Typer(
    help="ProteinGym2 - generic evedesign model runner",
    add_completion=True,
)

console = Console()


class ContainerTrainingJobPath:
    PREFIX = Path("/opt/program")
    MODEL_CARD_PATH = PREFIX / "README.md"
    OUTPUT_PATH = PREFIX / "output"


def import_class(import_string: str) -> Any:
    """Resolve an import string to the class it points at.

    Accepts either entry-point style ("evedesign.models.esm2:ESM2") or fully
    dotted style ("evedesign.models.esm2.ESM2"). The resolved class is expected
    to be an evedesign-style model exposing build and score methods and
    a constructor that accepts the model card's hyper_parameters as kwargs.

    Args:
        import_string: Import path to the model class.

    Returns:
        The model class object.
    """
    module_path, _, attr = import_string.partition(":")
    if not attr:
        module_path, _, attr = import_string.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"Could not parse import string {import_string!r}; expected "
            "'package.module:ClassName' or 'package.module.ClassName'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def resolve_model_class(model_card: ModelCard) -> str:
    """Read the evedesign model class import string off the model card.

    Args:
        model_card: Loaded model card.

    Returns:
        The import string for the evedesign model class.

    Raises:
        typer.BadParameter: If the card is missing the model_class field.
    """
    model_class = getattr(model_card, "model_class", None)
    if not model_class:
        raise typer.BadParameter(
            "Model card is missing the required 'model_class' field (import "
            "string for the evedesign model class, e.g. "
            "'evedesign.models.esm2:ESM2')."
        )
    return model_class


def load_or_build_model(
    model_class: str,
    model_card: ModelCard,
    system: System,
    data: LabeledInstanceDataset | None,
    model_file: Path | None,
) -> Any:
    """Load a dill-pickled model, or build one and optionally persist it.

    Behaviour is driven entirely by model_file arg:

    - None: build the model from system/data and do not save it
    - path that exists: deserialize the model with dill and skip building
      (system/data build inputs are ignored).
    - path that does not exist: build the model, then dump it to that path

    Args:
        model_class: Import string for the evedesign model class
        model_card: Loaded model card (supplies constructor hyper_parameters).
        system: The evedesign system
        data: Build data
        model_file: Optional path to load from / dump to

    Returns:
        The loaded or freshly built model.
    """
    if model_file is not None and model_file.exists():
        console.print(f"Loading prebuilt model from {model_file}; skipping build.")
        with open(model_file, "rb") as f:
            return dill.load(f)

    ModelClass = import_class(model_class)
    model = ModelClass(**model_card.hyper_parameters)
    model = model.build(system, data=data)

    if model_file is not None:
        with open(model_file, "wb") as f:
            dill.dump(model, f)
        console.print(f"Saved built model to {model_file}")

    return model


@app.command()
def train(
    dataset_file: Annotated[
        Path,
        typer.Option(
            help="Path to the dataset file",
        ),
    ],
    target: Annotated[
        str,
        typer.Option(
            help="Target name to use",
        ),
    ],
    split: Annotated[
        str | None,
        typer.Option(
            help="Split name to use. Omit for zero-shot / whole-dataset scoring.",
        ),
    ] = None,
    test_fold: Annotated[
        int | None,
        typer.Option(
            help="Test fold index. Omit for zero-shot / whole-dataset scoring.",
        ),
    ] = None,
    model_card_file: Annotated[
        Path,
        typer.Option(
            help="Path to the model card markdown file",
        ),
    ] = ContainerTrainingJobPath.MODEL_CARD_PATH,
    precomputed_instances_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to a precomputed embeddings file (as written by the "
            "'embed' command). When supplied, the enhanced instances are "
            "substituted into the train/test datasets before building.",
        ),
    ] = None,
    model_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to a dill-pickled model. If it exists, the model is "
            "loaded and building is skipped. If path is given but does not "
            "exist, the built model is dumped there for reuse. Omit to build "
            "without saving model.",
        ),
    ] = None,
):
    subsets = Subsets.from_path(dataset_file)
    model_card = ModelCard.from_path(model_card_file)

    # Import string for the model class is carried on the model card itself.
    model_class = resolve_model_class(model_card)

    zero_shot = "zero-shot" in model_card.tags

    if not zero_shot and (split is None or test_fold is None):
        raise typer.BadParameter(
            "--split and --test-fold are required for non-zero-shot models"
            "they may only be omitted when the model card is tagged zero-shot."
        )

    system, training_dataset, test_dataset = dataset_to_evedesign(
        subsets,
        split=None if zero_shot else split,
        target=target,
        test_fold=None if zero_shot else test_fold,
    )

    # Optionally swap in instances enhanced with precomputed embeddings/scores
    # (step 1 of an evedesign pipeline). The embeddings are produced upstream by
    # an embeddable model, so this is independent of the current card's tags.
    if precomputed_instances_file is not None:
        console.print(f"Loading precomputed embeddings from {precomputed_instances_file}")
        with open(precomputed_instances_file, "rb") as f:
            embedded_instances = pickle.load(f)

        # Map each enhanced instance back to its sequence so it can be
        # substituted into the train/test datasets. Let it fail with KeyError
        # if the precomputed set does not cover every dataset instance.
        instance_map = {
            "".join(instance[0].rep): instance for instance in embedded_instances
        }

        if training_dataset is not None:
            training_dataset.instances = [
                instance_map["".join(instance[0].rep)]
                for instance in training_dataset.instances
            ]

        if test_dataset is not None:
            test_dataset.instances = [
                instance_map["".join(instance[0].rep)]
                for instance in test_dataset.instances
            ]

    # When a prebuilt model is loaded, build is skipped. Warn about any
    # build-time inputs that were supplied but won't be used.
    if model_file is not None and model_file.exists():
        if training_dataset is not None:
            console.print(
                "Warning: a prebuilt model will be loaded from "
                f"{model_file}, the provided training data will not be used to "
                "build it."
            )
        if precomputed_instances_file is not None:
            console.print(
                "Warning: a prebuilt model will be loaded, "
                "--embeddings-file was not used to build it (test-set "
                "embeddings are still applied for scoring)."
            )

    # load -> build -> score, using the imported model API. training data
    # is None in the zero-shot/unsupervised case
    model = load_or_build_model(
        model_class, model_card, system, training_dataset, model_file
    )

    # Score the test dataset (the whole dataset in the zero-shot case).
    instances, values, _, _ = test_dataset.select(name=target, drop_missing=False)

    # In the supervised case also score the training folds: the benchmark's
    # train_available/per_fold score modes need a prediction for every sequence
    if training_dataset is not None:
        train_instances, _, _, _ = training_dataset.select(
            name=target, drop_missing=False
        )
        instances = list(instances) + list(train_instances)

    # score() raises on the whole batch if any instance is invalid for the model
    # (e.g. indels or non-standard residues for a fixed-length model). Score only
    # the valid ones, using the same checks as score(); the rest are left without
    # a prediction (null) by predictions_delta below.
    scorable_instances = [
        instance
        for instance in instances
        if model.system.valid_instance(
            instance,
            validate_reps=True,
            require_reps=True,
            fixed_length=model.requires_fixed_length,
            allow_deletions=model.handles_deletions,
            raise_invalid=False,
        )
    ]
    if len(scorable_instances) < len(instances):
        console.print(
            f"Warning: {len(instances) - len(scorable_instances)} of "
            f"{len(instances)} instances cannot be scored by this model (e.g. "
            "indels or non-standard residues) and are left without a prediction."
        )

    # score() returns scored instance copies (order preserved)
    scored_instances = model.score(scorable_instances)

    sequences = ["".join(instance[0].rep) for instance in scored_instances]

    predictions_df = pl.DataFrame(
        {
            "sequence": sequences,
            target: [float(instance.score) for instance in scored_instances],
        }
    )

    predictions_dataset = subsets.dataset.predictions_delta(
        predictions_df, target=target, allow_extra_predictions=True
    )

    output_file = Path(ContainerTrainingJobPath.OUTPUT_PATH) / (
        f"{subsets.dataset.name}_predictions.pgdata"
    )
    predictions_dataset.dump(path=ContainerTrainingJobPath.OUTPUT_PATH)
    console.print(f"Saved predictions to {output_file}")


@app.command()
def embed(
    dataset_file: Annotated[
        Path,
        typer.Option(
            help="Path to the dataset file",
        ),
    ],
    target: Annotated[
        str,
        typer.Option(
            help="Target name to use",
        ),
    ],
    model_card_file: Annotated[
        Path,
        typer.Option(
            help="Path to the model card markdown file",
        ),
    ] = ContainerTrainingJobPath.MODEL_CARD_PATH,
    model_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to a dill-pickled model. If it exists, the model is "
            "loaded and building is skipped. If path is given but does not "
            "exist, the built model is dumped there for reuse. Omit to build "
            "without saving model.",
        ),
    ] = None,
):
    """Precompute embeddings/scores for the whole dataset and serialize them.

    Applies the model's Transformer (or, failing that, Scorer) interface to
    every instance to enhance it with embeddings/scores, then pickles the
    enhanced instances so a downstream model can load them via
    'train --embeddings-file'.
    """
    subsets = Subsets.from_path(dataset_file)
    model_card = ModelCard.from_path(model_card_file)

    # import string for the model class is carried on the model card itself
    model_class = resolve_model_class(model_card)

    # embedding availability is advertised by a capability tag on the card
    if "embeddable" not in model_card.tags:
        raise typer.BadParameter(
            "This model does not support embedding; its model card is not "
            "tagged 'embeddable'."
        )

    # retrieve the entire dataset without splitting
    system, _, test_dataset = dataset_to_evedesign(
        subsets,
        split=None,
        target=target,
        test_fold=None,
    )

    # load -> build, using the imported model wrapper (data=None for embedding)
    model = load_or_build_model(model_class, model_card, system, None, model_file)

    # only the instances are needed here (the full dataset in this case)
    instances, _, _, _ = test_dataset.select(name=target, drop_missing=False)

    # Prefer the Transformer interface, which computes embeddings Both transform()
    # and score() return scored/transformed instance copies, so swap them in directly.
    if isinstance(model, Transformer):
        instances = model.transform(instances)
    elif isinstance(model, Scorer):
        instances = model.score(instances)
    else:
        raise typer.BadParameter(
            "Model must implement the Transformer or Scorer interface to embed."
        )

    # Embeddings are large--pickle the enhanced instances rather than
    # serializing them to text
    output_file = f"{ContainerTrainingJobPath.OUTPUT_PATH}/embeddings.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(instances, f)

    console.print(f"Saved embeddings to {output_file}")


@app.command()
def ping():
    console.print("pong")


if __name__ == "__main__":
    app()
