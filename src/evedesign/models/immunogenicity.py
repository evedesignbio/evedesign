import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from math import log10
from os import PathLike, path, environ
from tempfile import TemporaryDirectory
from typing import Sequence, Any, Self
import numpy as np
import pandas as pd

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback, Site
from evedesign.utils import status_start, status_done, available_cpus

MHCII_EPITOPE_CORE_LENGTH = 9


class LRU(OrderedDict):
    """
    Helper class for maintaining our own LRU dictionary,
    which allows to add computations in a batched fashion
    """
    def __init__(self, maxsize=128, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)          # mark as recently used
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class MixMHC2Pred(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around MixMHC2pred MHCII display predictor
    """
    available = True
    name: str = "MixMHC2pred"
    citations: list[str] = ["doi.org/10.1038/s41587-019-0289-6", "10.1016/j.immuni.2023.03.009"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = False  # TODO: can be set to True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        alleles: dict[str, float],
        binary: PathLike,
        peptide_lengths: Sequence[int] = (15,),
        floor_rank: float = 1e-05,
        truncate_rank: float | None = None,
        prediction_cache_size: int = 10 ** 6,
        cpu: int | None = None
    ):
        """
        Create new MixMHC2pred predictor

        Parameters
        ----------
        alleles
            Mapping from allele (e.g. HLA-DRB1*03:01 or DRB1*03:01) to weight/population frequency.
            Alleles will be remapped to MixMHC2pred naming convention with underscores internally.
            Values of dictionary will be used to weight binding core scores in epitope burden
            calculations.
        binary
            Path to MixMHC2pred binary
        peptide_lengths
            Length of peptides that sequence will be chunked into for scanning
        floor_rank
            Floor the rank at this value based on number of random peptides that method
            was calibrated on (e.g. 1e-05 when calibrated on 100k peptides)
        truncate_rank
            Exclude epitopes below this rank from scoring and inclusion in output
        prediction_cache_size
            Size of LRU cache to reuse calculations for peptides (larger cache = quicker
            calculations on large mutation sets, but higher memory footprint)
        cpu
            Number of CPU cores to use. If None, use all available cores.
        """
        self.alleles = {
            allele.replace(
                "HLA-", ""
            ).replace(
                "*", "_"
            ).replace(
                ":", "_"
            ).replace(
                "/", "__"
            ): weight
            for allele, weight in alleles.items()
        }
        self.peptide_lengths = peptide_lengths
        self.binary = binary
        self.truncate_rank = truncate_rank
        self.floor_rank = floor_rank
        self.cpu = cpu
        self._lru_cache = LRU(
            maxsize=prediction_cache_size
        )
        self._system = None

    @staticmethod
    def extract_peptides(
        seq: str,
        lengths: Sequence[int],
        n_flank_length: int | None = None,
        c_flank_length: int | None = None,
    ):
        peps = []
        for length in lengths:
            for i in range(len(seq) - length + 1):
                pep = seq[i:i + length]

                if n_flank_length is not None:
                    n_flank = seq[max(0, i - n_flank_length):i]
                else:
                    n_flank = None

                if c_flank_length is not None:
                    c_flank = seq[i + length:i + length + c_flank_length]
                else:
                    c_flank = None

                peps.append(
                    (pep, i, n_flank, c_flank)
                )

        return peps

    def _run_mixmhc2pred_chunk(self, chunk: list[tuple[str, str]]) -> pd.DataFrame:
        with TemporaryDirectory() as tmpdir:
            input_file = path.join(tmpdir, "input.txt")
            output_file = path.join(tmpdir, "output.txt")

            with open(input_file, "w") as f:
                f.writelines(f"{pep}\t{pep_ctx}\n" for pep, pep_ctx in chunk)

            cmd = [
                self.binary, "--input", input_file, "--output", output_file, "-a",
                *self.alleles,
            ]
            subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                # keep the binary from spawning its own threads on top of ours
                env={**environ, "OMP_NUM_THREADS": "1"},
            )
            return pd.read_csv(output_file, sep="\t", comment="#")


    def _run_mixmhc2pred(
        self,
        peptides_with_context: set[tuple[str, str]],
        min_chunk_size: int = 500,
    ) -> pd.DataFrame:
        items = list(peptides_with_context)
        if not items:
            return pd.DataFrame()

        n_jobs = self.cpu or available_cpus()
        # don't spawn more workers than there is meaningful work for
        n_chunks = max(1, min(n_jobs, len(items) // min_chunk_size or 1))

        if n_chunks == 1:
            return self._run_mixmhc2pred_chunk(items)

        chunks = [items[i::n_chunks] for i in range(n_chunks)]  # round-robin split

        with ThreadPoolExecutor(max_workers=n_chunks) as pool:
            dfs = list(pool.map(self._run_mixmhc2pred_chunk, chunks))

        return pd.concat(dfs, ignore_index=True)

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        # for now only handle single-protein systems but can trivially extend to multiple proteins
        for entity_idx, entity in enumerate(system):
            if entity.type != "protein":
                return False, "Can only handle protein entities"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score immunogenicity for full sequence (higher = more immunogenic)
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        status_start(status_callback, "Starting scoring")

        # first collect all peptides for this run
        unique_peps = {}

        # also collect which peptides are not yet in cache
        new_peps = set()
        reusing = 0

        for instance_idx, instance in enumerate(instances):
            for entity_idx, entity in enumerate(instance):
                # remove deletions and insertions from sequence
                seq = "".join(entity.normalized_rep())

                # extract all peptides for current instance
                peps = self.extract_peptides(
                    seq=seq,
                    lengths=self.peptide_lengths,
                    n_flank_length=3,
                    c_flank_length=3,
                )

                # iterate peptides, compute context and store in global map
                for pep_seq, seq_idx, ctx_n, ctx_c in peps:
                    ctx = "".join([
                        ctx_n.ljust(3, "-"), pep_seq[:3], pep_seq[-3:], ctx_c.rjust(3, "-")
                    ])

                    key = (pep_seq, ctx)

                    if key not in unique_peps:
                        unique_peps[key] = []

                    unique_peps[key].append(
                        (instance_idx, entity_idx, seq_idx)
                    )

                    if key not in self._lru_cache:
                        new_peps.add(key)
                    else:
                        reusing += 1

        # predict peptides we haven't seen before and stored in cache
        if len(new_peps) > 0:
            new_preds = self._run_mixmhc2pred(
                new_peps
            ).set_index(
                ["Peptide", "Context"]
            ).to_dict("index")
        else:
            new_preds = {}

        # deduplicate binding cores
        core_map = {}

        for key, hit_list in unique_peps.items():
            try:
                stats = new_preds[key]
            except KeyError:
                stats = self._lru_cache[key]

            for allele, allele_weight in self.alleles.items():
                rank = stats["%Rank_" + allele]
                core_hit = stats["CoreP1_" + allele]

                # skip cores if truncation is enabled and rank not high enough
                if self.truncate_rank is not None and rank > self.truncate_rank:
                    continue

                # assign cores to instances/entities
                for instance_idx, entity_idx, pos_idx in hit_list:
                    if instance_idx not in core_map:
                        core_map[instance_idx] = {}

                    # 1-based index for returned binding cores, add pos from hit list and
                    # first_index of respective entity
                    pos_remapped = self._system[entity_idx].first_index + pos_idx + core_hit - 1
                    instance_pos = (entity_idx, pos_remapped)

                    if instance_pos not in core_map[instance_idx]:
                        core_map[instance_idx][instance_pos] = {}

                    if rank < core_map[instance_idx][instance_pos].get(allele, 999):
                        core_map[instance_idx][instance_pos][allele] = rank


        # update cache with latest predictions (only at end to avoid losing entries)
        for k, v in new_preds.items():
            self._lru_cache[k] = v

        # assign scores to instances, this creates shallow copy after which we
        # can also attach binding cores as metadata
        scores = np.zeros((len(instances)))
        metadata = [None] * len(instances)

        if self.truncate_rank is not None:
            t = self.truncate_rank
        else:
            t = 100.0

        for instance_idx, pos_to_cores in core_map.items():
            # compute weighted sum of allele frequency * log-transformed rank for all cores,
            # apply floor to bound rank range based on # random peptides used for calibration,
            # and sum py core starting position
            pos_to_burden = {
                pos: sum(
                    self.alleles[core] * max(0.0, log10(t / max(rank, self.floor_rank))) for core, rank in cores.items()
                )
                for pos, cores in pos_to_cores.items()
            }

            # also accumulate flat list of epitopes to store in metadata
            metadata[instance_idx]: Site = [  # noqa
                {
                    "entity": entity_idx,
                    "pos": pos,
                    "length": MHCII_EPITOPE_CORE_LENGTH,
                    "type": "t_cell_epitope",
                    "subtype": core,
                    "score": rank,
                    # "score2": self.alleles[core]  * -log10(max(rank, self.floor_rank)),
                    "weight": self.alleles[core],
                }
                for (entity_idx, pos), cores in pos_to_cores.items()
                for core, rank in cores.items()
            ]


            # sum over entire entity for aggregated score
            scores[instance_idx] = sum(pos_to_burden.values())

        # first assign scores, this creates a shallow copy of instances
        scored_instances = assign_scores_to_instances(
            instances, np.asarray(scores, dtype=float)
        )

        # also attach epitope hits to instances
        for instance_idx, pos_to_cores in core_map.items():
            if scored_instances[instance_idx].metadata is None:
                scored_instances[instance_idx].metadata = {}

            if "sites" not in scored_instances[instance_idx].metadata:
                scored_instances[instance_idx].metadata["sites"] = []

            scored_instances[instance_idx].metadata["sites"] += metadata[instance_idx]

        status_done(status_callback, "Finished scoring")

        return scored_instances

