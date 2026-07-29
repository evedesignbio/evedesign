from itertools import combinations
from typing import Sequence, Literal
import numpy as np
from sklearn.model_selection import KFold
from evedesign.system import SystemInstance
from evedesign.types import DatasetSplit, DatasetSplitMap


class LabeledInstanceDataset:
    """
    Basic mapping of instances to one or multiple labels in "long" format (map from string key to values).
    Can be used for regression and classification tasks.

    Missing labels must be encoded with None (do not use NaN as this will break JSON serialization). Datasets
    also may not contain -inf or inf values to allow JSON serialization.
    """
    def __init__(
        self,
        instances: Sequence[SystemInstance],
        labels: dict[str, Sequence[float | None]],
        splits: DatasetSplitMap | tuple[[Literal["cv"]], int] | None = None,
        final_train_val_split: DatasetSplit | None = None,
    ):
        """
        Create new dataset of instances ("X") and corresponding labels ("y")

        Parameters
        ----------
        instances
            Instance list
        labels
            Map from value name to a sequence of values. Each sequence must have exactly
            the same length as instances. Missing values must be encoded with None.
        splits
            Sequence of train/validation/test splits (e.g. single train/test split,
            cross-validation, ...) for model evaluation. Can also supply a tuple
            ("cv", fold_number) to perform a random, non-stratified cross-validation
            split. Data will be shuffled with a fixed random seed before fold assignment.
            Note that random cross-validation splits are usually not meaningful
            in the context of biomolecular design, so using this convenience feature is not
            encouraged.
            When using select() drop_missing=True, affected datapoints will be removed from
            splits. If requiring exactly balanced splits, it is best to exclude those
            datapoints before instantiating the LabeledInstanceDataset.
        final_train_val_split:
            Training/validation set split for final model training on full dataset (not to be
            used for evaluation). Whether this split will be used is dependent on the implementation
            of the model that is fitted to the dataset.
        """
        if len(labels) == 0:
            raise ValueError(
                "Must specify at least one series in labels dictionary"
            )

        for name, series in labels.items():
            if len(instances) != len(series):
                raise ValueError(
                    f"Length of instances and values for series {name} does not agree for training set"
                )

            if not np.isfinite(series).all():
                raise ValueError(
                    f"Series {name} contains non-finite values (nan, inf or -inf)"
                )

        def _validate_split_or_raise(split_to_check, required_keys, split_name):
            for k in required_keys:
                if k not in split_to_check:
                    raise ValueError("Missing required split key '{}'".format(k))

            # check all train/val/test subsets
            all_indices = set()
            for subset_name, subset_indices in split_to_check.items():
                invalid_idx = [
                    index for index in subset_indices if index < 0 or index >= len(instances)
                ]
                if len(invalid_idx) > 0:
                    raise ValueError(
                        f"Subset {subset_name} contains out-of-bound indices: {invalid_idx}"
                    )
                all_indices.update(subset_indices)

            if len(all_indices) != len(instances):
                raise ValueError(f"Split {split_name} does not contain full dataset")

            # check there is no overlap between train/val/test within each split
            for (s1_name, s1), (s2_name, s2) in combinations(split_to_check.items(), r=2):
                overlap = set(s1).intersection(set(s2))
                if len(overlap) > 0:
                    raise ValueError(
                        f"Elements overlap between {s1_name} and {s2_name}: {overlap}"
                    )

        # validate splits if defined
        if splits is not None:
            if isinstance(splits, tuple) and len(splits) == 2 and splits[0] == "cv":
                # fix random seed for reproducibility; in any more advanced cases user can supply own splits
                kfold = KFold(
                    n_splits=splits[1], shuffle=True, random_state=42
                )

                splits = {
                    f"cv{fold_index}": {
                        "train": train_indices,
                        "test": test_indices,
                    } for fold_index, (train_indices, test_indices) in enumerate(
                        kfold.split(list(range(len(instances))))
                    )
                }
            elif isinstance(splits, dict):
                for split_name, split in splits.items():
                    # validation split must always contain train and test
                    _validate_split_or_raise(
                        split, required_keys=["train", "test"], split_name=split_name
                    )
            else:
                raise ValueError(
                    "Invalid split specification"
                )

        if final_train_val_split is not None:
            _validate_split_or_raise(
                final_train_val_split, required_keys=["train", "val"], split_name="final_train_val_split"
            )

        self.instances = instances
        self.labels = labels
        self._splits = splits
        self.final_train_val_split = final_train_val_split

    @property
    def names(self) -> list[str]:
        """
        Get dataset series names

        Returns
        -------
        List of dataset series names
        """
        return list(self.labels.keys())

    def select(
        self,
        name: str | None,
        drop_missing: bool = True
    ) -> tuple[list[SystemInstance], list[float | None], DatasetSplitMap | None, DatasetSplit | None]:
        """
        Select a single series from dataset

        Parameters
        ---------
        name
             The name of the series to select from dataset. If only one series is present,
             can pass None and it will be selected by default, otherwise a ValueError
             will be raised
        drop_missing
            Remove instance/label value pairs where the label value is missing (None)

        Returns
        -------
        Dataset sliced to selected series
        """
        if name is None:
            if len(self.labels) > 1:
                raise ValueError(
                    "Dataset has multiple label types, need to specify name to select"
                )

            name = list(self.labels)[0]
        else:
            if name not in self.labels:
                raise ValueError(
                    f"Series {name} is not present in dataset, valid options are: {', '.join(self.labels.keys())}"
                )

        series = self.labels[name]

        instances_filt = [
            inst for i, inst in enumerate(self.instances) if series[i] is not None or not drop_missing
        ]

        series_filt = [
            value for value in series if value is not None or not drop_missing
        ]

        # adjust splits to filtered list, as indices might change to due element removal
        # index mapping for retained elements
        index_map = {
            old_index: new_index
            for (new_index, old_index) in
            enumerate(index for index, value in enumerate(series) if value is not None or not drop_missing)
        }

        # map evaluation splits
        if self._splits is not None:
            splits_mapped = {
                split_name: {
                    subset_name: [
                        index_map[index] for index in subset if index in index_map
                    ] for subset_name, subset in split.items()
                }
                for split_name, split in self._splits.items()
            }
        else:
            splits_mapped = None

        # map final train/val split
        if self.final_train_val_split is not None:
            final_split_mapped = {
                subset_name: [
                    index_map[index] for index in subset if index in index_map
                ] for subset_name, subset in self.final_train_val_split.items()
            }
        else:
            final_split_mapped = None

        return instances_filt, series_filt, splits_mapped, final_split_mapped
