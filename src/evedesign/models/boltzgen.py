"""
BoltzGen Generator: wraps BoltzGen diffusion-based de novo
protein structure design into the evedesign Generator interface.

NOTE: Requires the boltzgen package (pip install evedesign[boltzgen]).
A CUDA GPU is mandatory: the boltzgen CLI calls torch.cuda.get_device_capability()
unconditionally, so there is no CPU path.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Self, Sequence

from loguru import logger

from evedesign.model import BaseModel, Generator
from evedesign.models.boltz.convert_design import (
    parse_design_output,
    system_to_boltzgen_yaml,
)
from evedesign.system import System, SystemInstance
from evedesign.types import DeviceType, EntityPosList, StatusCallback
from evedesign.utils import ensure_sequence

# boltzgen is CLI-only, so availability is a PATH check, not an import
IMPORT_AVAILABLE = shutil.which("boltzgen") is not None


# Default checkpoint references (HuggingFace)
DEFAULT_DESIGN_CHECKPOINTS = [
    "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
    "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
]
DEFAULT_INVERSE_FOLD_CHECKPOINT = (
    "huggingface:boltzgen/boltzgen-1:boltzgen1_ifold.ckpt"
)
DEFAULT_FOLDING_CHECKPOINT = (
    "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt"
)

PROTOCOLS = [
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
]


class BoltzGenGenerator(BaseModel, Generator):
    """
    Wraps BoltzGen diffusion-based de novo structure
    design into the evedesign Generator interface.

    Generates de novo protein backbones, optionally
    conditioned on a target. Returned SystemInstance
    objects have both structures and designed sequences
    populated.

    BoltzGen runs as a subprocess, so binary may point at
    an executable in another environment
    """
    available = IMPORT_AVAILABLE
    name: str = "BoltzGen"
    citations: list[str] = ["doi.org/10.1101/2025.11.20.689494"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = False
    handles_insertions: bool = True
    requires_gpu: bool = True
    supports_gpu: bool = True
    supports_gpu_parallel: bool = True
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = [
        "structures",
        "secondary_structure",
        "interactions",
        "atom_bonds",
        "copies",
        "cyclic",
        "insertions",
        "ligand_rep_type",
        "min_length",
        "max_length",
    ]

    def __init__(
        self,
        protocol: str = "protein-anything",
        device: DeviceType = "cuda",
        num_devices: int | None = None,
        num_workers: int = 1,
        diffusion_batch_size: int | None = None,
        design_checkpoints: list[str] | None = None,
        inverse_fold_checkpoint: str | None = None,
        folding_checkpoint: str | None = None,
        skip_inverse_folding: bool = False,
        inverse_fold_num_sequences: int = 1,
        step_scale: str | None = None,
        noise_scale: str | None = None,
        budget: int = 30,
        alpha: float | None = None,
        keep_tmp_dir: bool = False,
        binary: str = "boltzgen",
    ):
        # Checked per instance, not via the class-level available
        # flag, so a binary in another environment is resolved
        # instead of whatever is on PATH
        if shutil.which(binary) is None:
            logger.warning(
                f"boltzgen CLI not found at '{binary}'. Install with "
                "the optional dependency pip install "
                "evedesign[boltzgen], or pass binary=<path> to run it "
            )

        if protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol '{protocol}', "
                f"valid options: {PROTOCOLS}"
            )

        if device != "cuda":
            raise ValueError(
                f"BoltzGen requires a CUDA GPU (got device='{device}'). "
                "The boltzgen CLI calls "
                "torch.cuda.get_device_capability() unconditionally "
                "(cli/boltzgen.py:921), so there is no CPU or MPS path."
            )

        self.protocol = protocol
        self.device = device
        self.num_devices = num_devices
        self.num_workers = num_workers
        self.diffusion_batch_size = diffusion_batch_size
        self.design_checkpoints = (
            design_checkpoints
            or DEFAULT_DESIGN_CHECKPOINTS
        )
        self.inverse_fold_checkpoint = (
            inverse_fold_checkpoint
            or DEFAULT_INVERSE_FOLD_CHECKPOINT
        )
        self.folding_checkpoint = (
            folding_checkpoint
            or DEFAULT_FOLDING_CHECKPOINT
        )
        self.skip_inverse_folding = skip_inverse_folding
        self.inverse_fold_num_sequences = (
            inverse_fold_num_sequences
        )
        self.step_scale = step_scale
        self.noise_scale = noise_scale
        self.budget = budget
        self.alpha = alpha
        self.keep_tmp_dir = keep_tmp_dir
        self.binary = binary

        self._system = None

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(
        cls,
        system: System,
        data: Any = None,
    ) -> tuple[bool, str]:
        """
        Check if the system is suitable for BoltzGen
        de novo design.

        Which entities are designed is a per-call choice
        (generate(entities=...)), so it is not checked here.
        """
        if data is not None:
            return False, (
               "Model does not support data parameter (must be None)"
            )

        if len(system) == 0:
            return False, "System has no entities"

        for i, entity in enumerate(system):
            if entity.type not in ("protein", "ligand"):
                return False, (
                    f"Entity {i} has type "
                    f"'{entity.type}'. Only protein "
                    "and ligand entities are currently "
                    "supported."
                )

            if entity.deletions:
                return False, (
                    f"Entity {i} sets deletions, which BoltzGen cannot "
                    "express."
                )

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """Validate system and register for generation."""
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    def _build_cli_command(
        self,
        yaml_path: Path,
        output_dir: Path,
        num_designs: int,
    ) -> list[str]:
        """
        Build the boltzgen CLI command from the
        generator's configuration and per-call args.

        Returns the command as a list of strings
        suitable for subprocess.run.
        """
        cmd = [
            self.binary, "run",
            str(yaml_path),
            "--output", str(output_dir),
            "--protocol", self.protocol,
            "--num_designs", str(num_designs),
            "--num_workers", str(self.num_workers),
            "--inverse_fold_num_sequences",
            str(self.inverse_fold_num_sequences),
            "--budget", str(self.budget),
        ]
        cmd += [
            "--design_checkpoints"
        ] + self.design_checkpoints
        cmd += [
            "--inverse_fold_checkpoint",
            self.inverse_fold_checkpoint,
        ]
        cmd += [
            "--folding_checkpoint",
            self.folding_checkpoint,
        ]

        if self.num_devices is not None:
            cmd += ["--devices", str(self.num_devices)]
        if self.diffusion_batch_size is not None:
            cmd += [
                "--diffusion_batch_size",
                str(self.diffusion_batch_size),
            ]
        if self.step_scale is not None:
            cmd += ["--step_scale", self.step_scale]
        if self.noise_scale is not None:
            cmd += ["--noise_scale", self.noise_scale]
        if self.alpha is not None:
            cmd += ["--alpha", str(self.alpha)]
        if self.skip_inverse_folding:
            cmd += ["--skip_inverse_folding"]

        return cmd

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Generate N de novo designs via BoltzGen.

        Writes a BoltzGen design YAML from self._system,
        shells out to the boltzgen CLI, then parses
        the output CIFs into SystemInstance objects.

        entities selects which entities are designed, the rest are
        held fixed. Defaults to all of them.

        fixed_pos holds the given 1-based positions of an
        entity fixed while the rest of that chain is
        designed (motif scaffolding). The fixed residues
        are read from the entity's rep, which must be set.
        """
        self.ready_or_raise()

        if entities is not None:
            entities = ensure_sequence(entities)
            invalid = set(entities).difference(range(len(self._system)))
            if invalid:
                raise ValueError(f"Invalid entities: {sorted(invalid)}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="boltzgen_"))
        try:
            # 1. Write the YAML design spec
            yaml_path = tmp_dir / "design_spec.yaml"
            system_to_boltzgen_yaml(
                self._system, yaml_path,
                fixed_pos=fixed_pos, entities=entities,
            )
            logger.info(
                f"BoltzGen YAML written to {yaml_path}"
            )

            # 2. Build the CLI command and invoke
            output_dir = tmp_dir / "output"
            output_dir.mkdir()

            cmd = self._build_cli_command(
                yaml_path, output_dir, num_designs
            )

            logger.info(f"Running BoltzGen: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"BoltzGen failed (exit code {result.returncode})\n"
                    f"{result.stderr[-2000:]}"
                )

            # 3. Parse outputs into list[SystemInstance]
            instances = parse_design_output(
                output_dir=output_dir,
                system=self._system,
            )

            return instances

        finally:
            if not self.keep_tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                logger.info(
                    f"keep_tmp_dir=True; outputs preserved "
                    f"at {tmp_dir}"
                )
