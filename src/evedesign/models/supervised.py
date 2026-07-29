"""
Supervised regression models trained on top of embeddings and/or scores from zero-shot models
"""
import copy
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Sequence, Literal
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score, average_precision_score, roc_auc_score
from sklearn.utils import all_estimators
from sklearn.utils.validation import check_is_fitted
from scipy.stats import pearsonr, spearmanr
from evedesign.dataset import LabeledInstanceDataset
from evedesign.system import System, SystemInstance
from evedesign.model import Transformer, Scorer, SupervisedBaseModel, MutationScorer, \
    ConditionalMutationScorer, assign_scores_to_instances
from evedesign.types import StatusCallback, ModelStats, BioPolymers, BatchSize, DeviceType

try:
    import torch
    import gpytorch
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False

spearman_score = lambda y_true, y_pred: spearmanr(y_true, y_pred).correlation
pearson_score = lambda y_true, y_pred: pearsonr(y_true, y_pred).correlation


class SupervisedPredictorOnEmbeddingsScores(SupervisedBaseModel, Scorer, MutationScorer, ConditionalMutationScorer, ABC):
    """
    Supervised property prediction from pooled molecular embeddings and scores. Abstract base class to inherit from
    for specific supervised models.

    Currently only uses embeddings for biopolymers. Multi-entity systems are supported, but all embeddings
    must have same feature dimensionality.

    Note that passed in data must be pre-transformed to fit with framework conventions (higher values for more
    functional/fit sequences and lower values for less functional/fit sequences), ideally on a log-like scale;
    e.g. log-transformed read ratios vs WT

    Note: Possible extensions/updates in future
    - Multi-output learning (but need to think about how to integrate with score() with expects scalar per instance,
      but main purpose would be able to regress out other variables anyways, e.g. stability from activity)
    """
    available = True
    name: str = "Supervised predictor on sequence embeddings/scores"
    citations: list[str] = ["doi:10.1038/s41587-021-01146-5"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = True

    # property handling is all done by predictor and embedder, so return None to indicate that attributes
    # are irrelevant for model
    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        embedder: Transformer | None = None,
        scorer: Scorer | None = None,
        use_embeddings: bool = True,
        use_scores: bool = True,
        override_models_for_training: bool = False,
        target_name: str | None = None,
        pooling: Literal["mean", "max"] | None = "mean",
        batch_size: BatchSize = 128,
        is_classifier: bool = False
    ):
        """
        Train supervised regression model on top of molecular model embeddings/scores. Positional embeddings
        can be pooled to one feature vector along the position dimension.

        Can be used in either of two modes with pre-computed embeddings/scores, or through on-the-fly computation
        (cf. embedder param). The latter mode is needed to use mutation scoring methods, e.g. for Gibbs sampling
        or calculation of single mutation matrices.

        Parameters
        ----------
        embedder
            Molecular model to use for computing embeddings on the fly. If None, will use values available on supplied
            instances for build() and score(); in this mode, mutation scoring methods cannot be used).
            If this model is able to compute scores and no explicit scorer is specified, this model will also be used
            for scoring if use_scores = True.
            Also note override_models_for_training for multi-system training of models.
        scorer
            Separate molecular model to use for computing scores on the fly (overrides scoring with embedder).
            This e.g. allows to combine one-hot encoding embeddings with scores from sequence/structure models.
            Also note override_models_for_training for multi-system training of models.
        use_embeddings
            If True, include embeddings as a model feature (will raise an error if embeddings are absent and
            cannot be computed with embedder).
        use_scores
            If True, include instance score as a model feature (will raise an error if scores are absent and
            cannot be computed with embedder or scorer).
        override_models_for_training
            If True, use embeddings/scores already on instances, even if embedder/scorer is specified. This allows to
            train a model on a dataset with instances from multiple systems (e.g. stability measurements for many
            different proteins). The embedder/scorer will still be used at prediction time to allow mutation prediction
            methods to be used. Note this assumes all systems have the same number of entities and entity types,
            which will not be verified (trivially true for single-component protein systems).
        target_name
            Name of target series in LabeledInstanceDataset to retrieve. If the dataset only contains a single series,
            it can be extracted as a default by setting this parameter to None (an exception will be raised otherwise)
        pooling
            Aggregation to apply to positional embeddings across position dimension (to obtain one feature vector
            per entity). If None, do not apply any pooling and flatten embedding array instead; this requires
            that embeddings have the same length/number of positions across all instances.
        batch_size
            Assemble X features in batches of this size. Helps to address out of memory errors as embedding memory
            usage can become very large if predicting many instances at the same time
        is_classifier
            Set to True if trained model is a classifier, to False if a regression model (determines default
            evaluation metrics)
        """
        # make sure we are left with some features
        if not use_scores and not use_embeddings:
            raise ValueError(
                "At least one of use_scores or use_embeddings must be True"
            )

        # modelled system
        self._system = None

        # note: embedder needs to be built already built outside by convention if a BaseModel
        self.embedder = embedder
        self.scorer = scorer
        self.override_models_for_training = override_models_for_training
        self.target_name = target_name
        self.use_scores = use_scores
        self.use_embeddings = use_embeddings
        self.pooling_strategy = pooling

        if batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")
        self.batch_size = batch_size

        # update class variable defaults on instance as these will be used by mixin scoring function defaults
        if self.embedder is not None:
            self.handles_insertions = embedder.handles_insertions
            self.handles_deletions = embedder.handles_deletions
            self.requires_fixed_length = embedder.requires_fixed_length
            self.requires_target = embedder.requires_target

        if self.scorer is not None:
            # all methods must be handling insertions and deletions for composition to be able to handle them
            self.handles_insertions = self.handles_insertions and scorer.handles_insertions
            self.handles_deletions = self.handles_deletions and scorer.handles_deletions

            # require fixed length and target if at least one method needs it
            self.requires_fixed_length = self.requires_fixed_length or scorer.requires_fixed_length
            self.requires_target = self.requires_target or scorer.requires_target

        # set evaluation scores depending if we have a classifier or regressor
        self._is_classifier = is_classifier
        if self._is_classifier:
            # for now, we only allow to use sklearn metrics that receive scores/probabilities, not class labels
            self._eval_scores = {
                "rocauc": roc_auc_score,
                "average_precision": average_precision_score,
            }
        else:
            # default to regression
            self._eval_scores = {
                "spearman": spearman_score,
                "pearson": pearson_score,
                "r2": r2_score
            }

        # performance statistics
        self._y_true = None
        self._y_pred = None
        self._scores = None

    @property
    def system(self) -> System | None:
        return self._system

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        self.ready_or_raise()

        if self.embedder is not None and self.scorer is None:
            return self.embedder.positions(instance)
        elif self.embedder is None and self.scorer is not None:
            return self.scorer.positions(instance)
        elif self.embedder is not None and self.scorer is not None:
            return sorted(
                set(self.embedder.positions(instance)) & set(self.scorer.positions(instance))
            )
        else:
            raise ValueError(
                "No explicit embedder specified, cannot use positions()"
            )

    @classmethod
    def can_model(cls, system: System, data: LabeledInstanceDataset) -> tuple[bool, str]:
        biopolymer_entities = [
            entity for entity in system if entity.type in BioPolymers
        ]

        if len(biopolymer_entities) == 0:
            return False, "Can only handle systems with at least one biopolymer entity"

        if data is None:
            return False, "Labelled instance must be supplied for building model"

        return True, ""

    def _transform_and_validate_instances_batch(
        self,
        instances: Sequence[SystemInstance],
        override_models: bool,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        # start with instances as they are as transformed instances, will add to these as needed further down
        instances_t = instances

        embedder_added_scores = False
        if self.use_embeddings:
            # compute embeddings on the fly and replace instances if we have the model explicitly specified;
            # pass status_callback through as this is mostly heavy part of the computation
            if self.embedder is not None and not override_models:
                # in this case, we leave instance validation to the embedder; we verify that embeddings
                # are complete directly after this if/else clause;
                # depending on model capabilities, this call may also set the score attribute of the instance
                instances_t = self.embedder.transform(
                    instances, entity=None, status_callback=status_callback
                )
            else:
                # perform instance validation; this does not imply all instances actually have an embedding
                # so must check this as well
                [
                    self.system.valid_instance(
                        instance,
                        validate_reps=True,
                        require_reps=False,
                        validate_embeddings=True,
                        fixed_length=self.requires_fixed_length,
                        allow_deletions=self.handles_deletions,
                        raise_invalid=True,
                    ) for instance in instances_t
                ]

            # extract embeddings and verify they are complete; implementation right now
            # assumes that multi-entity embeddings for biopolymers have all the same feature dimensionality;
            # note: not creating a numpy array on outer dimension as length of embeddings in position
            # dimension may vary before pooling
            embeddings_in_parts = [
                [
                    inst[entity_idx].embedding
                    for entity_idx, entity in enumerate(self.system)
                    if entity.type in BioPolymers
                ] for inst in instances_t
            ]


            # make sure embeddings are defined for all entity instances in each system instance,
            # and that they all have the same feature dimensionality
            embeddings = [
                np.concatenate(instance_parts, axis=0)  # noqa
                for instance_parts in embeddings_in_parts
                if not any([part is None for part in instance_parts])  # all embeddings specified
                and len(set([part.shape[-1] for part in instance_parts])) == 1  # all have same feature dimensionality
            ]

            # check embedding completeness
            if len(embeddings) != len(instances_t):
                raise ValueError(
                    "All instances must have valid embeddings if use_embeddings is True. "
                    "Precompute or specify a model to compute on the fly. "
                    "If embeddings are specified, check all biopolymer entities have an embedding, and all have "
                    "same feature dimensionality."
                )

            # check embeddings all have same dimensionality (vector or matrix) across instances
            embedding_shapes = {
                len(emb.shape) for emb in embeddings
            }

            if len(embedding_shapes) != 1:
                raise ValueError(
                    f"Embeddings must all have same shape (vector or matrix) but found {embedding_shapes}"
                )

            embedding_dims = {
                emb.shape[-1] for emb in embeddings
            }

            if len(embedding_dims) != 1:
                raise ValueError(
                    f"Embeddings must all have same feature dimensionality but found {embedding_dims}"
                )

            # if embedding matrix, apply pooling across sequence dimension;
            # use nan versions of functions to allow blanking out other positions
            if list(embedding_shapes)[0] == 2:
                if self.pooling_strategy == "mean":
                    pooling_func = lambda emb: np.nanmean(emb, axis=0)
                elif self.pooling_strategy == "max":
                    pooling_func = lambda emb: np.nanmax(emb, axis=0)
                elif self.pooling_strategy is None:
                    pooling_func = lambda emb: emb.flatten()
                else:
                    raise ValueError("Invalid pooling strategy")

                embeddings = np.array(
                    [pooling_func(emb) for emb in embeddings]
                )
        else:
            embeddings = np.zeros((len(instances_t), 0))

        if self.use_scores:
            # extract scores from transformed instances (if using transform() above, these may have been
            # computed already so need to pay special attention to this case) or be precomputed from outside
            scores = np.array([
                inst.score for inst in instances_t if inst.score is not None
            ])

            if override_models or self.scorer is None:
                if len(scores) != len(instances_t):
                    raise ValueError(
                        "Missing scores on instances but must be all defined when using " 
                        "override_models = True and use_scores = True"
                    )
            else:
                # we always compute scores on the fly if an explicit scorer is defined
                scores = np.array([
                    inst.score for inst in self.scorer.score(instances)
                ])

            # expand axes for concatenation with feature matrix
            scores = scores[:, np.newaxis]
        else:
            scores = np.zeros((len(instances_t), 0))

        # concatenate along feature dimension and return
        x = np.concatenate(
            (embeddings, scores), axis=1
        )

        return x

    def _transform_and_validate_instances(
        self,
        instances: Sequence[SystemInstance],
        override_models: bool,
        status_callback: StatusCallback | None = None  # noqa
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        if self.batch_size is None:
            batch_size = len(instances)
        else:
            batch_size = self.batch_size

        all_x = []
        for batch_start in range(0, len(instances), batch_size):
            batch_instances = instances[batch_start:batch_start + batch_size]

            x_batch = self._transform_and_validate_instances_batch(
                batch_instances,
                override_models=override_models,
                # TODO: implement sensible way to handle status updates
            )

            all_x.append(x_batch)

        return np.concatenate(all_x, axis=0)  # noqa

    @abstractmethod
    def _fit(
        self,
        instances_transformed: np.ndarray[tuple[int, int], np.dtype[float]],
        values
    ) -> None:
        """
        Abstract method to overwrite in concrete implementations that fits models on embeddings/scores.

        Parameters
        ----------
        instances_transformed
            Pretransformed instances with embeddings and scores (X)
        values
            Labels for all instances (y)
        """
        pass

    @abstractmethod
    def _score(
        self,
        instances_transformed: np.ndarray[tuple[int, int], np.dtype[float]],
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        """
        Abstract method to overwrite in concrete implementations that scores pre-transformed
        instances with embeddings and scores using a previously fitted model

        Parameters
        ----------
        instances_transformed

        Returns
        -------
        Predicted y values
        """
        pass

    def build(
        self,
        system: System,
        data: LabeledInstanceDataset,
        status_callback: StatusCallback | None = None
    ):
        # verify if we can model the system
        self.can_model_or_raise(system, data)

        # make record of modelled system
        self._system = system

        if ((self.embedder is not None and self.system != self.embedder.system) or
                (self.scorer is not None and self.system != self.scorer.system)):
            raise ValueError(
                "system does not agree to embedder or scorer"
            )

        # retrieve target series, do not use missing values; we drop the test/val split as not used
        # by this model implementation
        instances, values, splits, _ = data.select(
            self.target_name, drop_missing=True
        )

        # training set
        x = self._transform_and_validate_instances(
            instances, self.override_models_for_training, status_callback
        )

        y = np.array(values)

        # iterate arbitrary splits to estimate model performance
        if splits is not None:
            self._scores = {}
            self._y_true = []
            self._y_pred = []

            for split_name, split in splits.items():
                x_split_train = x[split["train"]]
                y_split_train = y[split["train"]]

                x_split_test = x[split["test"]]
                y_split_test = y[split["test"]]

                # fit model on current training split and predict test split
                self._fit(x_split_train, y_split_train)
                y_pred = self._score(x_split_test)

                self._y_true += [y_split_test]
                self._y_pred += [y_pred]

                # remove -inf values from log probabilities for classifiers
                # or otherwise scoring will crash
                if self._is_classifier:
                    # copy so we don't modify list above
                    y_pred = y_pred.copy()
                    y_pred[y_pred == -np.inf] = np.finfo(np.float64).tiny

                # compute evaluation scores
                for name, eval_score in self._eval_scores.items():
                    self._scores[name] = self._scores.get(name, []) + [
                        eval_score(y_split_test, y_pred)
                    ]

        # fit final predictor on full training set (this could also implicitly be GridSearchCV/RandomSearchCV)
        self._fit(x, y)

        return self

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> list[SystemInstance]:
        self.ready_or_raise()

        x_pred = self._transform_and_validate_instances(
            instances, override_models=False, status_callback=status_callback
        )

        scores = self._score(x_pred)

        return assign_scores_to_instances(instances, scores)

    def stats(self) -> ModelStats | None:
        """
        Return summary statistics about built model (e.g. cross validation statistics) after
        a model has been prepared with build()

        Returns
        -------
        Model statistics
        """
        # only able to provide statistics once model has been built
        self.ready_or_raise()

        res: ModelStats = {
            "y_true": self._y_true,
            "y_pred": self._y_pred,
            "scores": self._scores,
        }

        return res


class SklearnPredictorOnEmbeddingsScores(SupervisedPredictorOnEmbeddingsScores):
    """
    Supervised property prediction from pooled molecular embeddings/scores. Can stack any
    scikit-learn-compatible predictors that implement fit() and predict()
    methods, including pipelines.
    """
    available = True
    name: str = "Supervised scikit-learn predictor on sequence embeddings/scores"
    citations: list[str] = ["doi:10.1038/s41587-021-01146-5", "doi:10.48550/arXiv.1201.0490"]

    def __init__(
        self,
        predictor: Any | str,
        predictor_kwargs: dict[str, Any] | None = None,
        embedder: Transformer | None = None,
        scorer: Scorer | None = None,
        use_embeddings: bool = True,
        use_scores: bool = True,
        override_models_for_training: bool = False,
        target_name: str | None = None,
        pooling: Literal["mean", "max"] | None = "mean",
        batch_size: BatchSize = 128,
    ):
        """
        Train supervised scikit-learn regression/classification model on top of molecular model embeddings/scores.
        Positional embeddings can be pooled to one feature vector along the position dimension.

        Note that classifiers will return scores based on class probabilities rather than class labels to facilitate
        use for quantative variant scoring.

        Can be used in either of two modes with pre-computed embeddings/scores, or through on-the-fly computation
        (cf. embedder param). The latter mode is needed to use mutation scoring methods, e.g. for Gibbs sampling
        or calculation of single mutation matrices.

        Parameters
        ----------
        predictor
            Scikit-learn regressor instance or model name string as available through
            sklearn.utils.all_estimators(type_filter=["regressor", "classifier"])
        predictor_kwargs
            Constructor parameters to use if predictor is a string (will be ignored if predictor is a model instance)
        embedder
            Molecular model to use for computing embeddings on the fly. If None, will use values available on supplied
            instances for build() and score(); in this mode, mutation scoring methods cannot be used).
            If this model is able to compute scores and no explicit scorer is specified, this model will also be used
            for scoring if use_scores = True.
            Also note override_models_for_training for multi-system training of models.
        scorer
            Separate molecular model to use for computing scores on the fly (overrides scoring with embedder).
            This e.g. allows to combine one-hot encoding embeddings with scores from sequence/structure models.
            Also note override_models_for_training for multi-system training of models.
        use_embeddings
            If True, include embeddings as a model feature (will raise an error if embeddings are absent and
            cannot be computed with embedder).
        use_scores
            If True, include instance score as a model feature (will raise an error if scores are absent and
            cannot be computed with embedder or scorer).
        override_models_for_training
            If True, use embeddings/scores already on instances, even if embedder/scorer is specified. This allows to
            train a model on a dataset with instances from multiple systems (e.g. stability measurements for many
            different proteins). The embedder/scorer will still be used at prediction time to allow mutation prediction
            methods to be used. Note this assumes all systems have the same number of entities and entity types,
            which will not be verified (trivially true for single-component protein systems).
        target_name
            Name of target series in LabeledInstanceDataset to retrieve. If the dataset only contains a single series,
            it can be extracted as a default by setting this parameter to None (an exception will be raised otherwise)
        pooling
            Aggregation to apply to positional embeddings across position dimension (to obtain one feature vector
            per entity). If None, do not apply any pooling and flatten embedding array instead; this requires
            that embeddings have the same length/number of positions across all instances.
        batch_size
            Assemble X features in batches of this size. Helps to address out of memory errors as embedding memory
            usage can become very large if predicting many instances at the same time
        """
        # instantiate predictor from model name string or store provided instance
        if isinstance(predictor, str):
            all_predictors = dict(all_estimators(type_filter=["regressor", "classifier"]))
            if predictor in all_predictors:
                if predictor_kwargs is None:
                    predictor_kwargs = {}
                self.predictor: Any = all_predictors[predictor](
                    **predictor_kwargs
                )
            else:
                raise ValueError(
                    f"Invalid regressor, valid options are {', ' .join(list(all_predictors))}"
                )
        else:
            # unfortunately no good typing options available, so verify attributes like scikit-learn does
            if not hasattr(predictor, "fit") or not hasattr(predictor, "predict"):
                raise ValueError(
                    "Predictor must have scikit-learn fit() and predict methods()"
                )

            self.predictor = predictor

        is_classifier = isinstance(self.predictor, ClassifierMixin)

        super().__init__(
            embedder=embedder,
            scorer=scorer,
            use_embeddings=use_embeddings,
            use_scores=use_scores,
            override_models_for_training=override_models_for_training,
            target_name=target_name,
            pooling=pooling,
            batch_size=batch_size,
            is_classifier=is_classifier,
        )

        self.predictor_kwargs = predictor_kwargs if predictor_kwargs is not None else {}

    @property
    def ready(self):
        # model only required if embeddings are non pre-specified
        fitted = True
        try:
            check_is_fitted(self.predictor)
        except NotFittedError:
            fitted = False

        return self.system is not None and fitted

    def _fit(
        self,
        instances_transformed,
        values
    ) -> None:
        # Fit model with scikit-learn
        self.predictor.fit(instances_transformed, values)

    def _score(
        self,
        instances_transformed,
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        # Predict with sckit-learn model; if a classifier
        # we use log probabilities instead of class labels
        if self._is_classifier:
            return self.predictor.predict_log_proba(
                instances_transformed
            )[:, 1].astype(float)
        else:
            return self.predictor.predict(
                instances_transformed
            ).astype(float)


if GPYTORCH_AVAILABLE:
    class _ExactGPModel(gpytorch.models.ExactGP):
        """
        Generic exact GP that wires a supplied mean and covariance module into the standard
        forward, MultivariateNormal(mean(x), covar(x)), paired with the supplied likelihood.

        This covers standard exact GP regression for any combination of gpytorch mean/kernel/likelihood
        modules, which can all be instantiated upfront and passed to GpytorchModel. Models requiring a
        custom forward (e.g. deep kernels with neural feature extractors, multitask GPs) are not handled.
        """
        def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = mean_module
            self.covar_module = covar_module

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GpytorchModel(SupervisedPredictorOnEmbeddingsScores):
    """
    Supervised Gaussian process regression on pooled molecular embeddings/scores, backed by gpytorch.

    The fitted model and likelihood are kept on CPU outside of inference (see _model_on_device) so the model
    stays serializable
    """
    available = GPYTORCH_AVAILABLE
    name: str = "Gaussian process predictor on sequence embeddings/scores"
    citations: list[str] = ["doi:10.1038/s41587-021-01146-5", "doi:10.48550/arXiv.1809.11165"]

    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = True

    def __init__(
        self,
        mean_module: "gpytorch.means.Mean | None" = None,
        covar_module: "gpytorch.kernels.Kernel | None" = None,
        likelihood: "gpytorch.likelihoods.Likelihood | None" = None,
        num_iters: int = 100,
        learning_rate: float = 0.1,
        optimizer: "type[torch.optim.Optimizer] | None" = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        mll: "type[gpytorch.mlls.MarginalLogLikelihood] | None" = None,
        standardize_targets: bool = True,
        device: DeviceType = "cpu",
        embedder: Transformer | None = None,
        scorer: Scorer | None = None,
        use_embeddings: bool = True,
        use_scores: bool = True,
        override_models_for_training: bool = False,
        target_name: str | None = None,
        pooling: Literal["mean", "max"] | None = "mean",
        batch_size: BatchSize = 128,
    ):
        """
        Train an exact Gaussian process regression model on top of molecular model embeddings/scores.

        The GP is specified by supplying instantiated gpytorch mean, covariance, and likelihood
        modules. These are fed to a standard exact-GP forward (MultivariateNormal(mean(x), covar(x)))
        and copied for each fit so repeated fits start from the same initial hyperparameters

        Parameters
        ----------
        mean_module
            A fully-instantiated gpytorch mean module (gpytorch.means.Mean). If None, a ConstantMean is used
        covar_module
            A fully-instantiated gpytorch kernel module (gpytorch.kernels.Kernel). If None, a scaled RBF kernel
            ScaleKernel(RBFKernel()) is used
        likelihood
            A fully-instantiated gpytorch Likelihood instance to use. If None, a GaussianLikelihood is used
        num_iters
            Number of optimizer steps for the training loop
        learning_rate
            Learning rate for the optimizer
        optimizer
            A torch.optim optimizer class to use, constructed per fit as optimizer(model.parameters(), lr=learning_rate, 
            **optimizer_kwargs). If None, torch.optim.Adam is used
        optimizer_kwargs
            Extra keyword arguments forwarded to the optimizer constructor
        mll
            A gpytorch.mlls marginal log-likelihood class constructed per fit as mll(likelihood, model) and maximized 
            during training. If None, ExactMarginalLogLikelihood is used.
        standardize_targets
            If True, standardize y to zero mean / unit variance before fitting and invert the transform on
            predictions
        device
            Device on which to run inference ("cpu", "cuda" or "mps")
        embedder, scorer, use_embeddings, use_scores, override_models_for_training, target_name, pooling, batch_size
            See SupervisedPredictorOnEmbeddingsScores
        """
        if not self.available:
            raise ImportError(
                "gpytorch package could not be imported"
                "Install with the optional dependency pip install evedesign[gpytorch]"
            )

        # validate module specs (safe to reference gpytorch after availability check above)
        if mean_module is not None and not isinstance(mean_module, gpytorch.means.Mean):
            raise ValueError(
                "mean_module must be a gpytorch Mean instance, or None for a ConstantMean"
            )

        if covar_module is not None and not isinstance(covar_module, gpytorch.kernels.Kernel):
            raise ValueError(
                "covar_module must be a gpytorch Kernel instance, or None for a scaled RBF kernel"
            )

        if likelihood is not None and not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            raise ValueError(
                "likelihood must be a gpytorch Likelihood instance, or None for a GaussianLikelihood"
            )

        if optimizer is not None and not (isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer)):
            raise ValueError(
                "optimizer must be a torch.optim.Optimizer subclass (e.g. torch.optim.Adam), "
                "or None for torch.optim.Adam"
            )

        if mll is not None and not (isinstance(mll, type) and issubclass(mll, gpytorch.mlls.MarginalLogLikelihood)):
            raise ValueError(
                "mll must be a gpytorch.mlls.MarginalLogLikelihood subclass "
                "(e.g. gpytorch.mlls.ExactMarginalLogLikelihood), or None for ExactMarginalLogLikelihood"
            )

        super().__init__(
            embedder=embedder,
            scorer=scorer,
            use_embeddings=use_embeddings,
            use_scores=use_scores,
            override_models_for_training=override_models_for_training,
            target_name=target_name,
            pooling=pooling,
            batch_size=batch_size,
            is_classifier=False,
        )

        self.mean_spec = mean_module
        self.covar_spec = covar_module
        self.likelihood_spec = likelihood
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.mll = mll if mll is not None else gpytorch.mlls.ExactMarginalLogLikelihood
        self.standardize_targets = standardize_targets
        self.device = device

        # fitted state (set by _fit)
        self._model = None
        self._likelihood = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    @property
    def ready(self) -> bool:
        return self.system is not None and self._model is not None and self._likelihood is not None

    def _make_mean(self):
        # copy the supplied module (or build a default) so each fit starts from the same initial hyperparameters
        if self.mean_spec is None:
            return gpytorch.means.ConstantMean()
        return copy.deepcopy(self.mean_spec)

    def _make_covar(self):
        if self.covar_spec is None:
            return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        return copy.deepcopy(self.covar_spec)

    def _make_likelihood(self):
        if self.likelihood_spec is None:
            return gpytorch.likelihoods.GaussianLikelihood()
        return copy.deepcopy(self.likelihood_spec)

    def _build_gp(self, train_x, train_y, likelihood):
        # construct a GP model from the training tensors, wiring in fresh mean/covariance modules
        return _ExactGPModel(
            train_x, train_y, likelihood, self._make_mean(), self._make_covar()
        )

    def _release_cache(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    @contextmanager
    def _model_on_device(self):
        """
        Yield the fitted GP model and likelihood on self.device for the duration of the context.

        """
        if self.device == "cpu" or self._model is None or self._likelihood is None:
            # nothing to move
            yield self._model, self._likelihood
            return

        device = torch.device(self.device)
        # deepcopy before moving so the canonical CPU state on the instance is never mutated
        model = copy.deepcopy(self._model).to(device)
        likelihood = copy.deepcopy(self._likelihood).to(device)
        try:
            yield model, likelihood
        finally:
            # release the device copies and clear the device cache
            del model, likelihood
            self._release_cache()

    def _to_tensor(self, array) -> "torch.Tensor":
        return torch.as_tensor(
            np.asarray(array), dtype=torch.get_default_dtype(), device=torch.device(self.device)
        )

    def _fit(
        self,
        instances_transformed,
        values
    ) -> None:
        train_x = self._to_tensor(instances_transformed)

        # standardize targets for GP numerical stability; invert on prediction
        y = np.asarray(values, dtype=float)
        if self.standardize_targets:
            self._y_mean = float(y.mean())
            # guard against constant targets (zero variance)
            self._y_std = float(y.std()) or 1.0
        else:
            self._y_mean, self._y_std = 0.0, 1.0

        train_y = self._to_tensor((y - self._y_mean) / self._y_std)

        # build fresh likelihood and model on the target device (train_x/train_y already on device)
        device = torch.device(self.device)
        likelihood = self._make_likelihood().to(device)
        model = self._build_gp(train_x, train_y, likelihood).to(device)

        # marginal log-likelihood to maximize during training
        mll = self.mll(likelihood, model)

        model.train()
        likelihood.train()

        optimizer = self.optimizer(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        for _ in range(self.num_iters):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        # keep fitted state on CPU so the model is serializable
        self._model = model.to("cpu")
        self._likelihood = likelihood.to("cpu")
        self._release_cache()

    def _posterior(
        self,
        instances_transformed,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior predictive mean and standard deviation (on the original target scale)
        for pre-transformed instances using the fitted GP.
        """
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), self._model_on_device() as (model, likelihood):
            # model/likelihood are on self.device inside this context
            test_x = self._to_tensor(instances_transformed)

            model.eval()
            likelihood.eval()

            # pass through likelihood to obtain distribution
            posterior = likelihood(model(test_x))
            mean = posterior.mean.detach().cpu().numpy().astype(float)
            std = posterior.stddev.detach().cpu().numpy().astype(float)

        # invert target standardization
        mean = mean * self._y_std + self._y_mean
        std = std * self._y_std

        return mean, std

    # this doesn't actually get used, but just leaving this here for now
    def _score(
        self,
        instances_transformed,
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        mean, _ = self._posterior(instances_transformed)
        return mean

    # override base score() to attach the posterior predictive standard deviation
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> list[SystemInstance]:
        self.ready_or_raise()

        x_pred = self._transform_and_validate_instances(
            instances, override_models=False, status_callback=status_callback
        )

        mean, std = self._posterior(x_pred)

        # store the posterior predictive stddev under the confidence attribute
        return assign_scores_to_instances(instances, mean, confidences=std)
