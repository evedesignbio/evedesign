"""
BoltzFold: wraps Boltz-2 structure prediction into the
evedesign Transformer interface.

NOTE: Template conditioning via Entity.structures is not
yet implemented. Structures present on entities will be
ignored with a warning.
"""

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Self, Sequence, cast

import numpy as np
from loguru import logger

try:
    import torch
    from boltz.main import (
        download_boltz2,
        process_inputs,
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
        BoltzSteeringParams,
    )
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzWriter
    from evedesign.models.boltz.convert import (
        _chain_to_entity_map,
        system_instance_to_yaml,
        prediction_to_instance,
    )
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/boltz")

import tempfile

from evedesign.model import BaseModel, Transformer, Scorer
from evedesign.utils import model_param_context
from evedesign.system import System, SystemInstance
from evedesign.types import DeviceType, StatusCallback, BatchSize


class BoltzFoldTransformer(BaseModel, Transformer, Scorer):
    """
    Wraps Boltz-2 into the evedesign Transformer interface.
    Folds protein sequences into 3D structures.
    Confidence scores are returned as a side effect of transform().
    """
    available = IMPORT_AVAILABLE
    name: str = "Boltz2"
    citations: list[str] = ["doi:10.1101/2025.06.14.659707"]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = ["copies"]

    def __init__(
        self,
        model_dir_path: str | None = DEFAULT_CACHE_DIR,
        batch_size: BatchSize = 1,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
        sampling_steps: int = 200,
        diffusion_samples: int = 5,
        recycling_steps: int = 3,
        use_msa: bool = True,
        score_attribute: Literal[
            "iptm", "ptm", "confidence_score", "complex_plddt"
        ] = "confidence_score",
        confidence_attribute: Literal[
            "iptm", "ptm", "confidence_score", "complex_plddt"
        ] = "complex_plddt",
    ):
        if not self.available:
            raise ValueError(
                "boltz package could not be imported. Is it installed already?"
            )

        self.model_dir_path = model_dir_path
        self.batch_size = batch_size
        self.keep_model_after_build = keep_model_after_build

        # by default, keep parameters loaded once loaded for prediction purposes to avoid reloading over and over
        self.keep_model_after_pred = True
        self.device = device
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.recycling_steps = recycling_steps
        self.use_msa = use_msa
        self.score_attribute = score_attribute
        self.confidence_attribute = confidence_attribute

        self._system: System | None = None
        self.model: Any | None = None

    @property
    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) == 0:
            return False, "System must have at least one entity"

        for i, entity in enumerate(system):
            if entity.type != "protein":
                return False, (
                    f"Entity {i} has type '{entity.type}'. "
                    "Only protein entities are supported. "
                    "DNA/RNA/ligand coming soon."
                )

            if not entity.defined_sequence():
                return False, f"Entity {i} must have a defined sequence"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """Validate system and register for folding."""
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self._release_cache()

    def _ensure_weights(self) -> Path:
        """
        Download Boltz-2 weights and supporting data on first use.
        Returns the cache directory path.

        The cache directory is set via the model_dir_path
        constructor parameter (default: ~/.cache/boltz).
        """
        cache = Path(self.model_dir_path or DEFAULT_CACHE_DIR).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)
        return cache

    def _load_model(self):
        """Download weights if needed and load Boltz-2 into memory."""
        if self.model is not None:
            return

        cache = self._ensure_weights()
        checkpoint = cache / "boltz2_conf.ckpt"

        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5

        predict_args = {
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "diffusion_samples": self.diffusion_samples,
            "max_parallel_samples": None, 
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        self.model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels="cuda" in str(self.device),  # different from default
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(MSAModuleArgs(use_paired_feature=True)), # different from default
            steering_args=asdict(BoltzSteeringParams()),
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Boltz-2 loaded from {checkpoint}")

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        use_msa: bool | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Fold each instance's sequences into 3D structures via Boltz-2.

        Currently folds all entities in the system together as a complex.

        Not yet implemented:
        - Per-entity folding: when entity is not None, only that entity
          should be folded in isolation and the result merged back into
          the full SystemInstance.
        - Multiple diffusion samples: when diffusion_samples > 1, only
          the best-ranked sample (model_0) is returned. Support for
          returning all samples as a list[SystemInstance] is planned.
        - Template conditioning: Entity.structures is ignored. Passing
          known structures as Boltz-2 templates is not yet supported.
        """
        self.ready_or_raise()

        if use_msa is None:
            use_msa = self.use_msa

        # Validate all instances against the system
        self._validate_instances(instances)

        if entity is not None:
            raise NotImplementedError(
                "Per-entity folding is not yet implemented. "
                "Pass entity=None to fold all entities as a complex."
            )

        fold_system = self._system
        fold_instances = list(instances)

        # Map chain IDs back to entity indices for output reconstruction
        chain_to_entity = _chain_to_entity_map(fold_system)

        # Write one YAML per instance into a shared temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="boltzfold_"))
        (tmp_dir / "inputs").mkdir()
        yaml_paths: list[Path] = []
        record_ids: list[str] = []
        for inst_idx, instance in enumerate(fold_instances):
            record_id = f"instance_{inst_idx}"
            yaml_path = tmp_dir / "inputs" / f"{record_id}.yaml"
            system_instance_to_yaml(
                fold_system, instance, yaml_path,
                use_msa=use_msa,
            )
            yaml_paths.append(yaml_path)
            record_ids.append(record_id)

        try:
            processed_dir = tmp_dir / "processed"
            cache = self._ensure_weights()
            mol_dir = cache / "mols"

            process_inputs(
                data=yaml_paths,
                out_dir=tmp_dir,
                ccd_path=cache / "ccd.pkl",  # unused in boltz2 mode but required by signature
                mol_dir=mol_dir,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
                use_msa_server=False,
                boltz2=True,
            )

            manifest = cast(
                Manifest,
                Manifest.load(processed_dir / "manifest.json")
            )
            if not manifest.records:
                raise ValueError(
                    "Boltz-2 processed no records from the input YAMLs. "
                    "Check that input sequences are valid proteins and "
                    "that Boltz-2 dependencies are correctly installed."
                )

            data_module = Boltz2InferenceDataModule(
                manifest=manifest,
                target_dir=processed_dir / "structures",
                msa_dir=processed_dir / "msa",
                mol_dir=mol_dir,
                num_workers=0,
                constraints_dir=(
                    processed_dir / "constraints"
                    if (processed_dir / "constraints").exists()
                    else None
                ),
                template_dir=(
                    processed_dir / "templates"
                    if (processed_dir / "templates").exists()
                    else None
                ),
                extra_mols_dir=(
                    processed_dir / "mols"
                    if (processed_dir / "mols").exists()
                    else None
                ),
            )
            data_module.setup(stage="predict")
            dataloader = data_module.predict_dataloader()

            record_id_to_idx = {
                rid: i for i, rid in enumerate(record_ids)
            }
            structures_dir = processed_dir / "structures"

            predictions_dir = tmp_dir / "predictions"
            predictions_dir.mkdir(exist_ok=True)

            writer = BoltzWriter(
                data_dir=str(structures_dir),
                output_dir=str(predictions_dir),
                output_format="mmcif",
                boltz2=True,
            )

            with model_param_context(
                self._load_model, self._delete_model,
                self.keep_model_after_pred,
            ):
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        batch_device = {
                            k: v.to(self.device)
                            if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()  # noqa
                        }
                        pred_dict = self.model.predict_step(
                            batch_device, batch_idx=batch_idx
                        )
                        if not pred_dict.get("exception", False):
                            writer.write_on_batch_end(
                                trainer=None,  # type: ignore
                                pl_module=None,  # type: ignore
                                prediction=pred_dict,
                                batch_indices=[],
                                batch=batch,  # noqa
                                batch_idx=batch_idx,
                                dataloader_idx=0,
                            )
                        else:
                            record = batch["record"][0]  # noqa
                            logger.warning(
                                f"Prediction failed for record {record.id}"
                            )

            all_files = list(predictions_dir.rglob("*"))
            logger.info(f"Boltz-2 output written to: {predictions_dir}")
            logger.info(f"Files written ({len(all_files)}):")
            for f in sorted(all_files):
                if f.is_file():
                    logger.info(
                        f"  {f.relative_to(predictions_dir)} "
                        f"({f.stat().st_size} bytes)"
                    )

            results_by_idx = {
                inst_idx: prediction_to_instance(
                    record_id=record_id,
                    predictions_dir=predictions_dir,
                    system=fold_system,
                    instance=fold_instances[inst_idx],
                    chain_to_entity=chain_to_entity,
                    score_attribute=self.score_attribute,
                    confidence_attribute=self.confidence_attribute,
                )
                for record_id, inst_idx in record_id_to_idx.items()
            }

            return [results_by_idx[i] for i in range(len(fold_instances))]

        finally:
            shutil.rmtree(tmp_dir)

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[float]]:  # type: ignore
        """
        Run Boltz-2 prediction for each instance and
        return an array of confidence scores.

        Uses self.score_attribute (set at construction)
        to select which metric to return. To use a
        different metric, construct a new instance with
        the desired score_attribute.
        """
        transformed = self.transform(
            instances,
            status_callback=status_callback,
        )
        return np.array(
            [inst.score for inst in transformed]
        )

