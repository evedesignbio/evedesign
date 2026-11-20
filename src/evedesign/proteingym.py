"""ProteinGym2 dataset conversion and the generic evedesign model runner.

The runner is installed as the ``evedesign-proteingym`` console script
(``evedesign[proteingym]``). ProteinGym model images use it as their entrypoint;
the model card's ``model_class`` and ``hyper_parameters`` select and configure
the model.
"""

import importlib
import pickle
from pathlib import Path
from typing import Annotated, Any, Sequence

import dill
import polars as pl
import typer
from biotite.structure import AtomArray
from rich.console import Console

from evedesign import sequence
from evedesign.dataset import LabeledInstanceDataset
from evedesign.model import Scorer, Transformer
from evedesign.structure import Structure
from evedesign.system import EntityInstance, Protein, System, SystemInstance

from proteingym.base import Dataset, Subsets
from proteingym.base.model import ModelCard
from proteingym.base.sequence import SequenceType



def wildtype_or_none(dataset: Dataset) -> str | None:
    """Extract the wildtype sequence from a dataset to set rep.

    Parameters
    ----------
    dataset : Dataset
        ProteinGym dataset to extract the wildtype sequence from.

    Returns
    -------
    str or None
        The wildtype sequence if available, otherwise None.
    """
    wt = [
        seq for seq in dataset.sequences if seq.type == SequenceType.WILD_TYPE
    ]

    if len(wt) > 0:
        return str(wt[0].value)
    else:
        return None


def msa_to_sequences(dataset: Dataset) -> sequence.Sequences | None:
    """Convert the first MSA in a dataset into an evedesign Sequences object.

    Parameters
    ----------
    dataset : Dataset
        ProteinGym dataset to extract the MSA from.

    Returns
    -------
    sequence.Sequences or None
        The first MSA as an evedesign Sequences object, or None if the
        dataset has no MSAs.

    Raises
    ------
    NotImplementedError
        If the MSA region start is different from 1.
    """
    if len(dataset.msas) == 0:
        return None

    # take first MSA by default for now
    first_msa = dataset.msas[0].value

    if len(dataset.msa_weights) == 0:
        weights = None
    else:
        weights = dataset.msa_weights[0].value
        assert len(weights) == len(first_msa), "MSA and weights length does not match"

    return sequence.Sequences(
        seqs=[sequence.Sequence(seq=str(seq), id=None) for seq in first_msa],
        aligned=True,
        type="protein",
        weights=weights,
        format="a3m", # everything will be a3m
    )


def update_structure(atom_array: AtomArray) -> Structure:
    """Wrap a ProteinGym structure into the evedesign Structure model.

    Assumes monomers.

    Parameters
    ----------
    atom_array : AtomArray
        ProteinGym structure as a biotite AtomArray.

    Returns
    -------
    Structure
        The wrapped evedesign Structure.
    """
    s = Structure(atom_array)
    assert len(s.chains()) == 1
    return s


def seqs_to_instances(sequences: Sequence[str]) -> list[SystemInstance]:
    """Transform standard string-based sequences into evedesign instances.

    Parameters
    ----------
    sequences : Sequence[str]
        String-based sequences to transform.

    Returns
    -------
    list[SystemInstance]
        The sequences wrapped as evedesign instances.
    """
    return [
        SystemInstance([
            EntityInstance(rep=seq)
        ])
        for seq in sequences
    ]


def system_from_dataset(dataset: Dataset) -> System:
    """Build an evedesign System from a ProteinGym Dataset.

    We default to a single-component protein system as there are no other cases
    in ProteinGym (yet...) and assume structure numbering matches the rep
    sequence and first_index by convention.

    Parameters
    ----------
    dataset : Dataset
        ProteinGym dataset to build the System from.

    Returns
    -------
    System
        An evedesign System with a single Protein.
    """
    return System([
        Protein(

            first_index=1, # this is the case for 90% of the database, would
            # be good to have as an inferred param eventually - we aren't using
            # mutation string labels for anything, so this won't be an issue

            # extract WT sequence, otherwise set to None - but we shouldn't ever
            # have that case with ProteinGym? - agreed
            rep=wildtype_or_none(dataset),

            # extract first MSA; we need to discuss the case where MSA target
            # region != full sequence
            sequences=msa_to_sequences(dataset),

            # Wrap each dataset structure (a biotite AtomArray) directly into the
            # evedesign Structure model
            structures={
                struc.name: update_structure(struc.value)
                for struc in dataset.structures
            },
        )
    ])


def labeled_dataset_from_df(df: pl.DataFrame, target: str) -> LabeledInstanceDataset:
    """Build a LabeledInstanceDataset from a ProteinGym dataframe slice.

    Comment:
    these instances have lowercase/gaps, so I guess it will be on each
    individual model to featurize accordingly.

    Parameters
    ----------
    df : pl.DataFrame
        ProteinGym dataframe slice containing a sequence column and the
        target column.
    target : str
        Name of the target column to extract labels from.

    Returns
    -------
    LabeledInstanceDataset
        Dataset mapping each sequence instance to its target value.
    """
    sequences = df["sequence"].to_list()
    # missing target values come back as None
    values = df[target].to_list()

    return LabeledInstanceDataset(
        instances=seqs_to_instances(sequences),
        labels={target: values},
    )


def dataset_to_evedesign(
    subsets: Subsets,
    split: str | None,
    target: str,
    test_fold: int | None,
) -> tuple[System, LabeledInstanceDataset | None, LabeledInstanceDataset | None]:
    """Map a ProteinGym dataset (subset) to evedesign representations.

    Follows the train/test conventions of the benchmark training entrypoint.

    Parameters
    ----------
    subsets : Subsets
        Loaded ProteinGym Subsets object (ex. Subsets.from_path(path)).
    split : str or None
        Name of the split column in the dms csv. Unused when test_fold is
        None (the unsupervised case), so may be passed as None there.
    target : str
        Name of the assay target to extract, ex. 'DMS Score'.
    test_fold : int or None
        Index of the slice to use as the test set. The remaining folds
        form the training set. If None (unsupervised), the whole dataset 
        is used as the training set and the test set is None.

    Returns
    -------
    system : System
        evedesign System with a single Protein. We default to a
        single-component protein system as there are no other cases in ProteinGym
        (yet), and assume structure numbering matches the rep sequence and
        first_index by convention.
    training_data : LabeledInstanceDataset
        Training dataset mapping each assay sequence (as a SystemInstance) to its target
        value (split into train/test sets according to test_fold)
    test_data : LabeledInstanceDataset
        Test dataset mapping each assay sequence (as a SystemInstance) to its target
        value (split into train/test sets according to test_fold)

    Raises
    ------
    ValueError
        If target is not present in the dataset assay targets, or if
        test_fold is out of range for the given split.
    """

    dataset = subsets.dataset

    system = system_from_dataset(dataset)

    valid_targets = [t.name for t in dataset.assay_targets]
    if target not in valid_targets:
        raise ValueError(
            f"Target '{target}' is not present in dataset assay targets, "
            f"valid options are: {', '.join(valid_targets)}"
        )

    # zero-shot case, assign everything to test set
    if test_fold is None:
        test_set = labeled_dataset_from_df(dataset.to_df(), target)
        return system, None, test_set

    subset_split = subsets[split]

    n_folds = len(subset_split.slices)
    if not 0 <= test_fold < n_folds:
        raise ValueError(
            f"test_fold {test_fold} is out of range for split '{split}' "
            f"with {n_folds} folds"
        )

    # test set is the requested fold; training set is everything else
    test_df = dataset[subset_split.slices[test_fold]].to_df()
    test_set = labeled_dataset_from_df(test_df, target)

    train_dfs = [
        dataset[subset_split.slices[i]].to_df()
        for i in range(n_folds)
        if i != test_fold
    ]

    train_df = pl.concat(train_dfs) if train_dfs else test_df
    training_set = labeled_dataset_from_df(train_df, target)

    return system, training_set, test_set


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
