"""
Wrapper class around EVE (Evolutionary model of Variant Effects)

EVE is a family-specific Bayesian VAE trained on an MSA of homologs (Frazer et al. 2021).
"""
import os
import copy
import json
import random
import tempfile
from os import PathLike
from types import SimpleNamespace
from typing import Any, Self, Sequence

import numpy as np

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance, EntityInstance
from evedesign.constants import GAP
from evedesign.utils import status_start, status_done
from evedesign.types import DeviceType, StatusCallback, BatchSize
from .evemodel.params import DEFAULT_MODEL_PARAMETERS

torch = None
DataLoader = None
VAE_model = None
data_utils = None


def _import_eve() -> bool:
    """
    Import torch + the vendored EVE codebase, populating the module-level references.
    Returns True on success, False otherwise.
    """
    global torch, DataLoader, VAE_model, data_utils

    try:
        import torch
        from torch.utils.data import DataLoader
        from .evemodel import VAE_model, data_utils  # noqa

        return True
    except ImportError:
        return False


IMPORT_AVAILABLE = _import_eve()


class EVE(BaseModel, Scorer, MutationScorer):
    """
    Wrapper class around the EVE model (build + score only, for VEP)
    """
    available = IMPORT_AVAILABLE
    name: str = "EVE"
    citations: list[str] = ["doi:10.1038/s41586-021-04043-8"]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    # need a multiple sequence alignment of homologs to train the family-specific VAE
    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        model_hyperparameters: str | PathLike | dict | None = None,
        model_checkpoint_path: str | PathLike | None = None,
        theta: float = 0.2,
        use_weights: bool = True,
        msa_weights_path: str | PathLike | None = None,
        preprocess_msa: bool = True,
        num_samples: int = 20000,
        batch_size: BatchSize = 2048,
        random_seed: int = 42,
        device: DeviceType = "cpu",
        max_msa_depth: int | None = 200_000,
        threshold_focus_cols_frac_gaps: float = 1.0,
    ):
        """
        Instantiate new EVE model wrapper

        Parameters
        ----------
        model_hyperparameters
            VAE hyperparameters (encoder/decoder/training): a path to a JSON file in the
            format of EVE's default parameters, or a parsed dict. If None, uses EVE's
            defaults.
        model_checkpoint_path
            Path to a trained EVE VAE checkpoint. If it exists, it is loaded in build()
            (no training); otherwise the VAE is trained on the MSA and saved to this path.
            If None, the trained model is retained in this object without being saved.
        theta
            MSA sequence re-weighting threshold (EVE default 0.2; viruses ~0.01).
        use_weights
            If True, use weights attached to the System's Sequences object where
            available, otherwise compute EVE weights.
        msa_weights_path
            Location to load/save EVE-computed MSA sequence weights (.npy); only used
            when the System's Sequences object has no weights.
        preprocess_msa
            Run EVE's MSA pre-processing (drop fragment sequences and low-occupancy
            columns). Focus columns may be a subset of the target positions, but sequence
            representations remain the full target length; excluded positions are ignored
            during scoring.
        num_samples
            Number of samples used to approximate the ELBO when scoring (EVE uses 20000).
            Higher is more accurate but slower.
        batch_size
            Maximum number of sequences to score concurrently.
        random_seed
            Random seed for the VAE.
        device
            Device to use for computations.
        max_msa_depth
            Maximum number of non-focus MSA sequences to train on. If the MSA has more,
            it is randomly subsampled down to this many (the focus/reference sequence is
            always kept) before re-weighting. EVE's sequence re-weighting is O(depth^2),
            and the original EVE paper's own alignment-construction methodology caps depth
            at 200,000 for this reason (README: "N such that 100,000 >= N >= 10L... down
            to N <= 200,000"); unlike the paper's pipeline, ProteinGym MSAs are not
            necessarily pre-capped, so this is enforced here instead. Set to None to
            disable (train on the full MSA, however deep).
        threshold_focus_cols_frac_gaps
            EVE's threshold for excluding a wild-type-covered column from training when
            too many family members have a gap there (default in the original EVE code
            is 0.3). Real, evolutionarily diverse alignments almost always have some
            natural indel variation. Defaulting to 1.0 keeps every wild-type-covered
            column regardless of family gap coverage; pass 0.3 to restore EVE's default
            (stricter) column-dropping behavior. Excluded columns are omitted from
            positions() and ignored when scoring.
        """
        if not _import_eve():
            raise ImportError(
                "EVE codebase could not be imported. Ensure torch is installed "
                "(the 'eve' extra)."
            )
        self.available = True

        # resolve model hyperparameters
        if model_hyperparameters is None:
            model_hyperparameters = DEFAULT_MODEL_PARAMETERS
        if isinstance(model_hyperparameters, dict):
            self._model_hyperparameters = copy.deepcopy(model_hyperparameters)
        else:
            with open(model_hyperparameters) as f:
                self._model_hyperparameters = json.load(f)

        for key in ("encoder_parameters", "decoder_parameters", "training_parameters"):
            if key not in self._model_hyperparameters:
                raise ValueError(
                    f"model_hyperparameters is missing required section '{key}'"
                )

        self.model_checkpoint_path = model_checkpoint_path
        self.theta = theta
        self.use_weights = use_weights
        self.msa_weights_path = msa_weights_path
        self.preprocess_msa = preprocess_msa
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.max_msa_depth = max_msa_depth
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.device = device

        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        if self.batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")

        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # modelled system
        self._system: System | None = None

        # Trained VAE, retained in memory unless an existing checkpoint is loaded lazily.
        self.model: Any | None = None

        # model name (per-family); derived from entity id at build time
        self.model_name: str | None = None

        # Path to a final checkpoint supplied by the caller.
        self._checkpoint_path: str | None = None

        # information required to reconstruct the VAE and to encode sequences,
        # populated by build() (kept lightweight so the MSA need not be retained)
        self._seq_len: int | None = None
        self._alphabet_size: int | None = None
        self._neff: float | None = None
        self._aa_dict: dict | None = None
        self._focus_cols: list[int] | None = None

        # working directory for training logs / intermediate checkpoints
        self._work_dir: str | None = None

    @property
    def ready(self):
        return (
            self._system is not None and
            (self.model is not None or self._checkpoint_path is not None) and
            self._seq_len is not None and
            self._focus_cols is not None
        )

    @property
    def system(self) -> System | None:
        return self._system

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        """
        Return the target positions retained as EVE focus columns.
        """
        self.ready_or_raise()
        first_index = self.system[0].first_index
        return [(0, first_index + col) for col in self._focus_cols]

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have defined rep sequence"

        # focus (wild-type) sequence must be a clean amino acid sequence (no gaps/mask)
        valid_aa = set(target.alphabet(include_gap=False))
        if any(symbol not in valid_aa for symbol in target.rep):
            return False, "Target rep must be a gap-free, unmasked amino acid sequence"

        if target.sequences is None or len(target.sequences.seqs) == 0:
            return False, "Must provide an MSA (sequences) to train/score the EVE model"

        if not target.sequences.aligned:
            return False, "Provided sequences must be aligned"

        return True, ""

    def _write_msa(self, target, msa_path: str) -> list[float] | None:
        """
        Write the entity MSA to disk in the format expected by EVE's MSA_processing:
        a fixed-width a2m alignment whose focus (first) sequence header carries the
        "/start-stop" position range. EVE ignores A3M inserts, so they are removed
        explicitly here rather than through a generic, implicit format conversion.
        """
        if target.sequences.format_ == "a3m":
            seqs = target.sequences.remove_inserts().seqs
        else:
            seqs = target.sequences.to_a2m().seqs

        weights = target.sequences.weights if self.use_weights else None
        if weights is not None and len(weights) != len(seqs):
            raise ValueError("Number of MSA weights must match number of sequences")

        # re-weighting is O(depth^2); cap non-focus depth to keep it tractable (see
        # max_msa_depth docstring). Sampling (not truncating) avoids any bias from
        # the MSA's original ordering.
        selected_indices = list(range(1, len(seqs)))
        if self.max_msa_depth is not None and len(selected_indices) > self.max_msa_depth:
            rng = random.Random(self.random_seed)
            selected_indices = rng.sample(selected_indices, self.max_msa_depth)
        selected_indices = [0] + selected_indices
        seqs = [seqs[i] for i in selected_indices]

        # number of match-state columns in the query (uppercase, non-gap)
        focus_len = sum(
            1 for symbol in seqs[0].seq if symbol.isupper() and symbol != GAP
        )
        start = target.first_index
        stop = start + focus_len - 1
        focus_name = target.id if target.id is not None else "query"

        # per-row headers must be unique: EVE's MSA_processing keys sequences by
        # header name and merges rows that share one, so a missing/duplicate id_
        # (e.g. ProteinGym MSAs, which carry no per-sequence id) would silently
        # collapse most of the alignment into a single row.
        with open(msa_path, "w") as msa_file:
            msa_file.write(f">{focus_name}/{start}-{stop}\n{seqs[0].seq}\n")
            for i, seq in enumerate(seqs[1:], start=1):
                seq_id = seq.id_ if seq.id_ is not None else f"seq_{i}"
                msa_file.write(f">{seq_id}_{i}\n{seq.seq}\n")

        if weights is None:
            return None
        return [float(weights[i]) for i in selected_indices]

    def _load_model(self):
        # avoid reloading if already loaded
        if self.model is not None:
            return

        # reconstruct the VAE from stored shape info (avoids retaining the full MSA);
        # encoder/decoder params are mutated in place by VAE_model, so pass copies
        shim_data = SimpleNamespace(
            seq_len=self._seq_len,
            alphabet_size=self._alphabet_size,
            Neff=self._neff,
        )

        model = VAE_model.VAE_model(
            model_name=self.model_name,
            data=shim_data,
            encoder_parameters=copy.deepcopy(
                self._model_hyperparameters["encoder_parameters"]
            ),
            decoder_parameters=copy.deepcopy(
                self._model_hyperparameters["decoder_parameters"]
            ),
            random_seed=self.random_seed,
        )

        # EVE forces cuda-if-available internally; honour the requested device instead
        self._set_device(model)

        checkpoint = torch.load(
            self._checkpoint_path, map_location=torch.device(self.device)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model

    def _set_device(self, model) -> None:
        """
        Move the VAE to the requested device. EVE sets device = cuda-if-available
        independently on the VAE_model and on its encoder/decoder submodules, so all
        three must be overridden, otherwise sampling tensors are created on a different
        device than the parameters (device-mismatch RuntimeError).
        """
        device = torch.device(self.device)
        model.device = device
        model.encoder.device = device
        model.decoder.device = device
        model.to(device)

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self._release_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None
    ) -> Self:
        # verify if we can model the system
        self.can_model_or_raise(system, data)

        # store system with this instance
        self._system = system
        target = self.system[0]

        self.model_name = (target.id if target.id is not None else "protein") + "_EVE"

        # working directory for MSA file, sequence weights, and training artefacts
        self._work_dir = tempfile.mkdtemp(prefix="eve_")
        msa_path = os.path.join(self._work_dir, "msa.a2m")
        msa_weights = self._write_msa(target, msa_path)

        weights_location = (
            str(self.msa_weights_path) if self.msa_weights_path is not None
            else os.path.join(self._work_dir, "msa_weights.npy")
        )

        # process the MSA (re-weighting + focus column extraction)
        status_start(status_callback, "Processing MSA")
        msa_data = data_utils.MSA_processing(
            MSA_location=msa_path,
            theta=self.theta,
            use_weights=self.use_weights,
            weights_location=weights_location,
            preprocess_MSA=self.preprocess_msa,
            threshold_focus_cols_frac_gaps=self.threshold_focus_cols_frac_gaps,
            sequence_weights=msa_weights,
        )

        focus_cols = [int(col) for col in msa_data.focus_cols]
        if not focus_cols:
            raise ValueError(
                "All positions exceed threshold_focus_cols_frac_gaps"
            )
        if any(col < 0 or col >= len(target.rep) for col in focus_cols):
            raise ValueError(
                "EVE focus columns do not map onto the target sequence positions"
            )

        # store lightweight info needed for reconstruction / encoding at scoring time
        self._seq_len = msa_data.seq_len
        self._alphabet_size = msa_data.alphabet_size
        self._neff = msa_data.Neff
        self._aa_dict = dict(msa_data.aa_dict)
        self._focus_cols = focus_cols

        # load an existing checkpoint if available, otherwise train and save
        if self.model_checkpoint_path is not None and os.path.exists(self.model_checkpoint_path):
            status_start(status_callback, "Loading EVE checkpoint")
            self._checkpoint_path = str(self.model_checkpoint_path)
        else:
            status_start(status_callback, "Training EVE VAE")
            self._train(msa_data)

        status_done(status_callback, "EVE model ready")

        # return self to allow method chaining
        return self

    def _train(self, msa_data) -> None:
        """
        Train the EVE VAE and optionally save it to the caller-supplied checkpoint.
        """
        encoder_parameters = copy.deepcopy(
            self._model_hyperparameters["encoder_parameters"]
        )
        decoder_parameters = copy.deepcopy(
            self._model_hyperparameters["decoder_parameters"]
        )
        training_parameters = copy.deepcopy(
            self._model_hyperparameters["training_parameters"]
        )

        # redirect logs / intermediate checkpoints to the working directory
        training_parameters["training_logs_location"] = self._work_dir
        training_parameters["model_checkpoint_location"] = self._work_dir

        model = VAE_model.VAE_model(
            model_name=self.model_name,
            data=msa_data,
            encoder_parameters=encoder_parameters,
            decoder_parameters=decoder_parameters,
            random_seed=self.random_seed,
        )
        self._set_device(model)

        model.train_model(data=msa_data, training_parameters=training_parameters)

        # Persist only when the caller requested a checkpoint. Otherwise the trained
        # model remains part of this object and is serialized with it.
        if self.model_checkpoint_path is not None:
            checkpoint_path = str(self.model_checkpoint_path)
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            model.save(
                model_checkpoint=checkpoint_path,
                encoder_parameters=self._model_hyperparameters["encoder_parameters"],
                decoder_parameters=self._model_hyperparameters["decoder_parameters"],
                training_parameters=training_parameters,
            )
            self._checkpoint_path = checkpoint_path

        model.eval()
        self.model = model

    def _one_hot_encode(self, seqs: Sequence[str]) -> np.ndarray:
        """
        One-hot encode fixed-length sequences over the EVE amino acid alphabet.
        Gaps (and any symbol outside the alphabet) yield an all-zero row, matching
        EVE's own encoding of mutated sequences.
        """
        encoding = np.zeros(
            (len(seqs), self._seq_len, self._alphabet_size)
        )
        for i, seq in enumerate(seqs):
            for j, col in enumerate(self._focus_cols):
                letter = seq[col]
                k = self._aa_dict.get(letter)
                if k is not None:
                    encoding[i, j, k] = 1.0
        return encoding

    def _compute_elbo(
        self,
        seqs: Sequence[str],
        status_callback: StatusCallback | None = None,
    ) -> np.ndarray:
        """
        Compute the sample-averaged ELBO for a batch of fixed-length sequences.
        """
        one_hot = self._one_hot_encode(seqs)
        self._load_model()

        tensor = torch.tensor(one_hot)
        dataloader = DataLoader(
            tensor, batch_size=self.batch_size, shuffle=False
        )

        prediction_matrix = torch.zeros((len(seqs), self.num_samples))

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x = batch.type(self.model.dtype).to(self.device)
                offset = batch_idx * self.batch_size
                for sample_idx in range(self.num_samples):
                    elbo, _, _ = self.model.all_likelihood_components(x)
                    prediction_matrix[offset:offset + len(x), sample_idx] = elbo

                if status_callback is not None:
                    progress = ((batch_idx + 1) * self.batch_size / len(seqs)) * 100
                    status_callback(
                        "running", min(progress, 100.0),
                        f"Scored {min(offset + len(x), len(seqs))}/{len(seqs)} sequences"
                    )

        mean_elbo = prediction_matrix.mean(dim=1).detach().cpu().numpy()

        assert len(mean_elbo) == len(seqs), "Length of scores does not match number of sequences"

        return mean_elbo

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        self.ready_or_raise()

        # ProteinGym instances mark mutated positions in lowercase; case carries no
        # meaning for EVE's fixed-length one-hot encoding, so validate/score against
        # uppercased copies while returning scores against the original instances.
        normalized = [
            SystemInstance([EntityInstance(rep="".join(instance[0].rep).upper())])
            for instance in instances
        ]

        # validate sequences against the modelled system (fixed length, no deletions)
        self._validate_instances(normalized)

        seqs = [
            "".join(instance[0].rep) for instance in normalized
        ]

        scores = self._compute_elbo(seqs, status_callback)

        # confidence will be set to None in call
        return assign_scores_to_instances(instances, scores)
