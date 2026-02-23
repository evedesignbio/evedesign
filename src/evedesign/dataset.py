from typing import Sequence
from evedesign.system import SystemInstance


class LabeledInstanceDataset:
    """
    Basic mapping of instances to one or multiple labels in "long" format (map from string key to values).
    Can be used for regression and classification tasks.

    Missing labels must be encoded with None (do not use NaN as this will break JSON serialization)

    # TODO: add utility method to create from dataframe
    # TODO: add utility method for creating train/test split
    """
    def __init__(
        self,
        instances: Sequence[SystemInstance],
        labels: dict[str, Sequence[float | None]],
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

        self.instances = instances
        self.labels = labels

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
    ) -> tuple[list[SystemInstance], list[float | None]]:
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

        return instances_filt, series_filt


class LabeledInstanceTrainTestDataset:
    def __init__(
        self,
        training_set: LabeledInstanceDataset,
        test_set: LabeledInstanceDataset | None = None
    ):
        """
        Training/test dataset split

        Parameters
        ----------
        training_set
            Labeled instances belonging to training data
        test_set
            Labeled instances belonging to test data. Can be None if no explicit
            test set is specified.
        """
        if test_set is not None:
            if set(training_set.names) != set(test_set.names):
                raise ValueError(
                    "Training and test data must contain the same series names"
                )

        # make sure datasets agree
        self.training_set = training_set
        self.test_set = test_set
