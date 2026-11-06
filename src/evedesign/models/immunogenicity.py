import subprocess
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import log10, inf
from os import PathLike, path, environ
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Any, Self, Literal, Iterable
import pandas as pd

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback, Site, Metadata
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


@dataclass(frozen=True, slots=True)
class AlleleHit:
    """
    Predictor-agnostic result for a single peptide / allele pair
    """
    rank: float
    core_offset: int
    score: float | None = None
    affinity: float | None = None


# canonical prediction result: peptide key -> allele -> hit
PredictionBatch = dict[tuple[str, str], dict[str, AlleleHit]]


class MHCIIPresentationPredictor(
    BaseModel, Scorer, MutationScorer, ConditionalMutationScorer
):
    """
    Shared implementation for external MHC class II presentation predictors
    """
    # core properties shared by all peptide-scanning MHCII predictors
    available = True
    requires_target: bool = False

    # Limit scoring to fixed length sequences for now; handling deletions and
    # insertions will require to adjust the index of the returned sites in the instance reps
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    # Floor is smallest resolvable rank percentage,
    # both predictors calibrate on 100k random peptides.
    floor_rank: float = 1e-03

    # whether the predictor reports the binding core position counting from 0 or from 1
    # (MixMHC2pred: one-based, NetMHCIIpan: zero-based). Applied by _core_offset() so that
    # AlleleHit.core_offset is always zero-based.
    core_index_base: int = 0

    # peptide context layout: residues taken from the protein either side of the peptide, and
    # residues taken from each end of the peptide itself
    context_flank_length: int = 3
    context_peptide_terminus_length: int = 3

    # symbol padding the context where the peptide sits at a protein terminus
    context_pad_char: str = "-"

    # minimum number of peptides per worker chunk; set higher for predictors with a large per-invocation cost
    min_chunk_size: int = 500

    def __init__(
        self,
        alleles: dict[str, float],
        binary: PathLike,
        peptide_lengths: Sequence[int] = (15,),
        truncate_rank: float | None = None,
        prediction_cache_size: int = 10 ** 6,
        cpu: int | None = None,
    ):
        """
        Parameters
        ----------
        alleles
            Mapping from allele (e.g. HLA-DRB1*03:01 or DRB1*03:01) to weight/population frequency.
            Alleles will be remapped to internal naming convention of the predictor.
            Values of dictionary will be used to weight binding core scores in epitope burden
            calculations.
        binary
            Path to the predictor binary
        peptide_lengths
            Length of peptides that sequence will be chunked into for scanning
        truncate_rank
            Exclude epitopes below this rank from scoring and inclusion in output
        prediction_cache_size
            Size of LRU cache to reuse calculations for peptides (larger cache = quicker
            calculations on large mutation sets, but higher memory footprint)
        cpu
            Number of CPU cores to use. If None, use all available cores.
        """
        # Alleles keyed by normalized name; reverse map takes reported sites back
        self.alleles = {
            self._normalize_allele(allele): weight
            for allele, weight in alleles.items()
        }

        self._allele_input_names = {
            self._normalize_allele(allele): allele
            for allele in alleles
        }

        if len(self.alleles) != len(alleles):
            raise ValueError(
                "Allele names became non-unique after normalization"
            )

        self.peptide_lengths = peptide_lengths
        self.binary = binary
        self.truncate_rank = truncate_rank
        self.cpu = cpu
        self._lru_cache = LRU(
            maxsize=prediction_cache_size
        )
        self._system: System | None = None

    # predictor-specific methods
    @staticmethod
    @abstractmethod
    def _normalize_allele(allele: str) -> str:
        """
        Translate a standard allele name (e.g. HLA-DRB1*03:01) into the naming convention
        expected by the predictor binary.
        """

    @classmethod
    def _core_offset(cls, reported: int | str) -> int:
        """
        Convert a core position as reported by the predictor into the zero-based offset for AlleleHit
        """
        return int(reported) - cls.core_index_base

    def allele_display_name(self, allele: str) -> str:
        """
        Map a normalized allele name back to the name the caller supplied
        """
        return self._allele_input_names[allele]

    @abstractmethod
    def _predict_chunk(
        self, chunk: Sequence[tuple[str, str]]
    ) -> PredictionBatch:
        """
        Run the predictor on one chunk of (peptide, context) pairs and return canonical results.
        Implementations must return an entry for every input pair and every allele in
        self.alleles; missing entries will surface as a KeyError during scoring.
        """

    # shared peptide / context handling

    @staticmethod
    def extract_peptides(
        seq: str,
        lengths: Sequence[int],
        n_flank_length: int = 0,
        c_flank_length: int = 0,
    ) -> list[tuple[str, int, str, str]]:
        """
        Slide a window of each requested length over seq, returning one
        (peptide, start index, N-flank, C-flank) tuple per window. A flank length of 0
        yields an empty flank; flanks are truncated at the sequence termini.
        """
        peps = []
        for length in lengths:
            for i in range(len(seq) - length + 1):
                pep = seq[i:i + length]
                peps.append((
                    pep,
                    i,
                    seq[max(0, i - n_flank_length):i],
                    seq[i + length:i + length + c_flank_length],
                ))

        return peps

    @classmethod
    def build_context(cls, pep_seq: str, n_flank: str, c_flank: str) -> str:
        """
        Build the 12-mer context string expected by both MixMHC2pred and NetMHCIIpan:
        3 residues upstream of the peptide, first 3 residues of the peptide, last 3 residues
        of the peptide, 3 residues downstream of the peptide
        """
        k = cls.context_flank_length
        t = cls.context_peptide_terminus_length
        pad = cls.context_pad_char

        return "".join([
            n_flank.rjust(k, pad),
            pep_seq[:t],
            pep_seq[-t:],
            c_flank.ljust(k, pad),
        ])

    # shared execution / chunking

    def _predict(
        self,
        peptides_with_context: Iterable[tuple[str, str]],
    ) -> PredictionBatch:
        """
        Run the predictor over a set of (peptide, context) pairs, split across worker threads
        """
        items = list(peptides_with_context)
        if not items:
            return {}

        n_jobs = self.cpu or available_cpus()
        # don't spawn more workers than there is meaningful work for
        n_chunks = max(1, min(n_jobs, len(items) // self.min_chunk_size or 1))

        if n_chunks == 1:
            return self._predict_chunk(items)

        chunks = [items[i::n_chunks] for i in range(n_chunks)]  # round-robin split

        with ThreadPoolExecutor(max_workers=n_chunks) as pool:
            results = list(pool.map(self._predict_chunk, chunks))

        merged: PredictionBatch = {}
        for result in results:
            merged.update(result)

        return merged

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

    # scoring methods

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

        assert self._system is not None  # guaranteed by ready_or_raise above
        first_indices: list[int] = []
        for idx, system_entity in self._system.biopolymers():
            first_indices.append(system_entity.first_index)

        status_start(status_callback, "Starting scoring")

        # first collect all peptides for this run
        unique_peps: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)

        # also collect which peptides are not yet in cache
        new_peps = set()

        for instance_idx, instance in enumerate(instances):
            for entity_idx, entity in enumerate(instance):
                # identical to rep given the flags above, but normalize defensively
                seq = "".join(entity.normalized_rep())

                # extract all peptides for current instance
                peps = self.extract_peptides(
                    seq=seq,
                    lengths=self.peptide_lengths,
                    n_flank_length=self.context_flank_length,
                    c_flank_length=self.context_flank_length,
                )

                # iterate peptides, compute context and store in global map
                for pep_seq, seq_idx, ctx_n, ctx_c in peps:
                    ctx = self.build_context(pep_seq, ctx_n, ctx_c)

                    key = (pep_seq, ctx)
                    unique_peps[key].append(
                        (instance_idx, entity_idx, seq_idx)
                    )

                    if key not in self._lru_cache:
                        new_peps.add(key)

        # predict peptides we haven't seen before and stored in cache
        new_preds = self._predict(new_peps)

        # deduplicate binding cores: instance -> (entity, position) -> allele -> best rank
        core_map: dict[int, dict[tuple[int, int], dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        for key, hit_list in unique_peps.items():
            try:
                hits = new_preds[key]
            except KeyError:
                hits = self._lru_cache[key]

            for allele in self.alleles:
                hit = hits[allele]

                # skip cores if truncation is enabled and rank not high enough
                if self.truncate_rank is not None and hit.rank > self.truncate_rank:
                    continue

                # assign cores to instances/entities
                for instance_idx, entity_idx, pos_idx in hit_list:
                    # 1-based index for returned binding cores, add pos from hit list and
                    # first_index of respective entity (hit.core_offset is zero-based for
                    # every predictor, see AlleleHit)
                    pos_remapped = first_indices[entity_idx] + pos_idx + hit.core_offset

                    # keep the best rank per core, which several windows may report
                    cores = core_map[instance_idx][entity_idx, pos_remapped]
                    if hit.rank < cores.get(allele, inf):
                        cores[allele] = hit.rank

        # update cache with latest predictions (only at end to avoid losing entries)
        for k, v in new_preds.items():
            self._lru_cache[k] = v

        # assign scores to instances, this creates shallow copy after which we
        # can also attach binding cores as metadata
        scores = [0.0] * len(instances)
        metadata: list[list[Site] | None] = [None] * len(instances)

        # rank ceiling the burden is measured against; ranks are percentages
        t = self.truncate_rank if self.truncate_rank is not None else 100.0

        for instance_idx, pos_to_cores in core_map.items():
            # compute weighted sum of allele frequency * log-transformed rank for all cores,
            # apply floor to bound rank range based on # random peptides used for calibration,
            # and sum py core starting position
            pos_to_burden = {
                pos: sum(
                    self.alleles[core] * max(0.0, log10(t / max(rank, self.floor_rank)))
                    for core, rank in cores.items()
                )
                for pos, cores in pos_to_cores.items()
            }

            # also accumulate flat list of epitopes to store in metadata
            metadata[instance_idx] = [
                Site(
                    entity=entity_idx,
                    pos=pos,
                    length=MHCII_EPITOPE_CORE_LENGTH,
                    type="t_cell_epitope",
                    # report in the caller's allele convention, not the predictor's
                    subtype=self.allele_display_name(core),
                    score=rank,
                    weight=self.alleles[core],
                    # attribution, so that re-scoring can replace this model's own sites and
                    # sites from two MHCII predictors stay distinguishable downstream
                    model=self.name,
                )
                for (entity_idx, pos), cores in pos_to_cores.items()
                for core, rank in cores.items()
            ]

            # sum over entire entity for aggregated score
            scores[instance_idx] = sum(pos_to_burden.values())

        # first assign scores, this creates a shallow copy of instances
        scored_instances = assign_scores_to_instances(instances, scores)

        # copy() only shallow-copies metadata, so "sites" is still the caller's list object
        for inst, new_sites in zip(scored_instances, metadata):
            existing: list[Site] = (inst.metadata or {}).get("sites") or []
            sites = [
                site for site in existing if site.get("model") != self.name
            ] + (new_sites or [])

            if not sites:
                continue

            meta: Metadata = inst.metadata if inst.metadata is not None else {}
            meta["sites"] = sites
            inst.metadata = meta

        status_done(status_callback, "Finished scoring")

        return scored_instances


class MixMHC2Pred(MHCIIPresentationPredictor):
    """
    Wrapper around MixMHC2pred MHCII display predictor
    """
    name: str = "MixMHC2pred"
    citations: list[str] = ["doi.org/10.1038/s41587-019-0289-6", "10.1016/j.immuni.2023.03.009"]

    # MixMHC2pred reports the 1-based position of the core's P1 residue
    core_index_base: int = 1

    # MixMHC2pred documents '-' for context positions outside the protein
    context_pad_char: str = "-"

    # default_floor_rank and min_chunk_size inherited: %Rank is a percentage calibrated on
    # 100k random peptides, and the binary is cheap to invoke

    @staticmethod
    def _normalize_allele(allele: str) -> str:
        """
        MixMHC2pred naming, e.g. HLA-DRB1*03:01 -> DRB1_03_01,
        HLA-DPA1*01:03/DPB1*04:01 -> DPA1_01_03__DPB1_04_01
        """
        return allele.replace(
            "HLA-", ""
        ).replace(
            "*", "_"
        ).replace(
            ":", "_"
        ).replace(
            "/", "__"
        )

    def _predict_chunk(
        self, chunk: Sequence[tuple[str, str]]
    ) -> PredictionBatch:
        with TemporaryDirectory() as tmpdir:
            input_file = path.join(tmpdir, "input.txt")
            output_file = path.join(tmpdir, "output.txt")

            with open(input_file, "w") as f:
                f.writelines(f"{pep}\t{pep_ctx}\n" for pep, pep_ctx in chunk)

            cmd = [
                str(self.binary), "--input", input_file, "--output", output_file, "-a",
                *self.alleles,
            ]
            subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                # keep the binary from spawning its own threads on top of ours
                env={**environ, "OMP_NUM_THREADS": "1"},
            )
            table = pd.read_csv(output_file, sep="\t", comment="#")

        # MixMHC2pred returns one row per peptide with one column block per allele,
        # translate into the canonical long representation
        result: PredictionBatch = {}

        for row in table.to_dict("records"):
            key = (row["Peptide"], row["Context"])
            result[key] = {
                allele: AlleleHit(
                    rank=float(row[f"%Rank_{allele}"]),
                    core_offset=self._core_offset(row[f"CoreP1_{allele}"]),
                    score=_opt_float(row.get(f"Score_{allele}")),
                )
                for allele in self.alleles
            }

        return result


PredictionMode = Literal["EL", "BA"]


class NetMHCIIpan(MHCIIPresentationPredictor):
    """
    Wrapper around the NetMHCIIpan 4.3 MHCII presentation predictor.

    NetMHCIIpan provides two prediction heads which are selected via ``prediction_mode``:

    * ``"EL"`` (default) -- eluted ligand likelihood, trained on MS-eluted ligand data. This is
      the head to use for display/presentation-driven immunogenicity, and is the direct
      counterpart to MixMHC2pred.
    * ``"BA"`` -- binding affinity, trained on in-vitro affinity measurements. Useful when the
      downstream question is affinity per se, or for comparability with older BA-based pipelines.

    In both cases scoring consumes the corresponding percentile rank (``%Rank_EL`` / ``%Rank_BA``),
    so the shared epitope burden logic is unchanged between modes. Passing ``-BA`` also makes the
    binary emit the EL columns, so both ranks are parsed and cached regardless of mode; the mode
    only decides which of them is handed to the scoring logic.
    """
    name: str = "NetMHCIIpan"
    citations: list[str] = [
        "10.1126/sciadv.adj6367",            # NetMHCIIpan-4.3
        "10.1038/s42003-023-04749-7",        # NetMHCIIpan-4.2
        "10.1021/acs.jproteome.9b00874",     # NetMHCIIpan-4.0
    ]

    # NetMHCIIpan's "Of" column is the 0-based offset of the core within the peptide
    core_index_base: int = 0

    # netMHCIIpan encodes context extending beyond the protein as 'X' by default
    # (per its -termAcon flag description, which switches the encoding to 'A')
    context_pad_char: str = "X"

    # default_floor_rank inherited: %Rank_EL and %Rank_BA are both percentages calibrated
    # against 100k random natural peptides, which is what the base class assumes

    # NetMHCIIpan pays a substantial per-invocation cost loading its network ensemble,
    # so use much larger chunks than MixMHC2pred
    min_chunk_size: int = 5000

    # columns parsed out of the tabular stdout, per prediction mode
    _rank_column: dict[PredictionMode, str] = {"EL": "%Rank_EL", "BA": "%Rank_BA"}
    _score_column: dict[PredictionMode, str] = {"EL": "Score_EL", "BA": "Score_BA"}

    def __init__(
        self,
        alleles: dict[str, float],
        binary: PathLike,
        prediction_mode: PredictionMode = "EL",
        peptide_lengths: Sequence[int] = (15,),
        truncate_rank: float | None = None,
        prediction_cache_size: int = 10 ** 6,
        cpu: int | None = None,
        max_alleles_per_run: int = 15,
    ):
        """
        Create new NetMHCIIpan 4.3 predictor

        Parameters
        ----------
        alleles
            Mapping from allele to weight/population frequency. Accepts standard names
            (HLA-DRB1*03:01, DRB1*03:01, HLA-DQA1*01:01/DQB1*05:01) and remaps them to
            NetMHCIIpan naming (DRB1_0301, HLA-DQA10101-DQB10501) internally.
        binary
            Path to the netMHCIIpan executable
        prediction_mode
            Which prediction head to score with, "EL" (eluted ligand likelihood, default)
            or "BA" (binding affinity). See class docstring.
        peptide_lengths
            Length of peptides that sequence will be chunked into for scanning
        truncate_rank
            Exclude epitopes below this rank from scoring and inclusion in output
        prediction_cache_size
            Size of LRU cache to reuse calculations for peptides
        cpu
            Number of CPU cores to use. If None, use all available cores.
        max_alleles_per_run
            Maximum number of alleles passed to the binary in a single invocation; larger
            allele sets are split across several invocations.

        Notes
        -----
        Each invocation is given its own scratch directory via -tdir, so concurrent chunks do
        not collide. The netMHCIIpan script's own TMPDIR must still be set at install time.
        """
        if prediction_mode not in ("EL", "BA"):
            raise ValueError(
                f"Invalid prediction_mode: {prediction_mode} (must be 'EL' or 'BA')"
            )

        self.prediction_mode: PredictionMode = prediction_mode
        self.max_alleles_per_run = max_alleles_per_run

        super().__init__(
            alleles=alleles,
            binary=binary,
            peptide_lengths=peptide_lengths,
            truncate_rank=truncate_rank,
            prediction_cache_size=prediction_cache_size,
            cpu=cpu,
        )

        # make the active head visible in model provenance (shadows the class attribute)
        # self.name = f"{type(self).name}-{prediction_mode}"

    @staticmethod
    def _normalize_allele(allele: str) -> str:
        """
        Map standard allele names onto NetMHCIIpan naming:

        * DR (single chain):   HLA-DRB1*03:01            -> DRB1_0301
        * DQ/DP (paired):      HLA-DQA1*01:01/DQB1*05:01 -> HLA-DQA10101-DQB10501
        * non-human / already normalized names (H-2-IAb, DRB1_0301, ...) pass through unchanged
        """
        raw = allele.strip()

        # already in NetMHCIIpan form, or a mouse/other species allele we do not rewrite
        if "*" not in raw and ":" not in raw:
            return raw

        body = raw[4:] if raw.upper().startswith("HLA-") else raw

        # split paired alpha/beta chain alleles; both "/" and "-" are accepted as separators
        chains = [c for c in body.replace("-", "/").split("/") if c]

        def _split_chain(chain: str) -> tuple[str, str]:
            gene, _, digits = chain.partition("*")
            if not digits:
                raise ValueError(
                    f"Cannot parse allele chain '{chain}' from '{allele}'"
                )
            return gene, digits.replace(":", "")

        if len(chains) == 1:
            gene, digits = _split_chain(chains[0])
            return f"{gene}_{digits}"

        if len(chains) == 2:
            return "HLA-" + "-".join(
                f"{gene}{digits}" for gene, digits in map(_split_chain, chains)
            )

        raise ValueError(
            f"Cannot parse allele '{allele}' (expected one or two chains)"
        )

    def _build_command(
        self, input_file: str, alleles: Sequence[str], tdir_template: str
    ) -> list[str]:
        cmd = [
            str(self.binary),
            "-f", input_file,
            # peptide list with a context column; we do our own scanning rather than feeding
            # FASTA so that peptide-level caching works
            "-inptype", "2",
            # -inptype 2 only says the input carries a context column; -context selects the
            # network that uses it (the default synapse file is "synlist_nocontext.bin")
            "-context",
            "-a", ",".join(alleles),
        ]

        # -BA adds the binding affinity columns on top of the eluted ligand columns
        if self.prediction_mode == "BA":
            cmd.append("-BA")

        # -tdir takes an mkdtemp TEMPLATE that the binary creates itself -- passing an
        # existing path fails with "Cannot make tmpdir". One per invocation isolates chunks.
        cmd += ["-tdir", tdir_template]

        return cmd

    def _predict_chunk(
        self, chunk: Sequence[tuple[str, str]]
    ) -> PredictionBatch:
        chunk = list(chunk)
        if not chunk:
            return {}

        result: PredictionBatch = {key: {} for key in chunk}

        allele_list = list(self.alleles)
        allele_batches = [
            allele_list[i:i + self.max_alleles_per_run]
            for i in range(0, len(allele_list), self.max_alleles_per_run)
        ]

        with TemporaryDirectory() as tmpdir:
            input_file = path.join(tmpdir, "input.txt")

            with open(input_file, "w") as f:
                # netMHCIIpan splits the peptide/context columns on whitespace, so tab
                # and space are both accepted (verified against the binary)
                f.writelines(f"{pep}\t{ctx}\n" for pep, ctx in chunk)

            for batch_idx, alleles in enumerate(allele_batches):
                # template only -- the binary creates this directory, so we must not
                tdir_template = path.join(tmpdir, f"nm_{batch_idx}_XXXXXX")

                proc = subprocess.run(
                    self._build_command(input_file, alleles, tdir_template),
                    capture_output=True, text=True, check=False,
                    # keep the binary from spawning its own threads on top of ours
                    env={**environ, "OMP_NUM_THREADS": "1"},
                )

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"netMHCIIpan failed with exit code {proc.returncode}:\n"
                        f"{proc.stderr.strip()}"
                    )

                n_parsed = self._parse_output(proc.stdout, chunk, result)

                if n_parsed == 0:
                    raise RuntimeError(
                        "netMHCIIpan exited successfully but produced no prediction rows "
                        f"for alleles {list(alleles)}.\n"
                        f"Command: "
                        f"{' '.join(self._build_command(input_file, alleles, tdir_template))}\n"
                        f"First input lines:\n{_head(Path(input_file).read_text())}\n"
                        f"stdout:\n{_head(proc.stdout)}\n"
                        f"stderr:\n{_head(proc.stderr)}"
                    )

        # fail loudly rather than silently dropping peptides or alleles
        for key, hits in result.items():
            missing = set(self.alleles) - set(hits)
            if missing:
                raise RuntimeError(
                    f"netMHCIIpan returned no prediction for peptide {key[0]} "
                    f"and allele(s) {sorted(missing)}"
                )

        return result

    def _parse_output(
        self,
        stdout: str,
        chunk: Sequence[tuple[str, str]],
        result: PredictionBatch,
    ) -> int:
        """
        Parse the tabular stdout of netMHCIIpan into ``result``.

        One block per allele, each with a header line, then one row per input peptide in input
        order. A peptide alone is not a unique key (it can recur with different contexts), so
        rows are matched positionally and the peptide column is checked on every row, turning
        any drift into an error rather than silently mismatched results.

        Returns the number of prediction rows parsed.
        """
        rank_col = self._rank_column[self.prediction_mode]
        score_col = self._score_column[self.prediction_mode]

        # empty until the first header line is seen (rather than None, so that the mapping
        # never needs narrowing at the many lookups below)
        columns: dict[str, int] = {}
        row_idx = 0
        n_parsed = 0

        for line in stdout.splitlines():
            stripped = line.strip()

            if not stripped or stripped.startswith(("#", "-")):
                continue

            fields = stripped.split()

            # header line: rebuild the column map and restart the per-allele row counter
            if fields[0] == "Pos":
                columns = {col: idx for idx, col in enumerate(fields)}
                row_idx = 0
                continue

            # trailing summary lines ("Number of strong binders ...", "Protein ...")
            if not columns or not fields[0].isdigit():
                continue

            if row_idx >= len(chunk):
                raise RuntimeError(
                    "netMHCIIpan returned more rows than peptides submitted"
                )

            key = chunk[row_idx]
            peptide = fields[columns["Peptide"]]

            if peptide != key[0]:
                raise RuntimeError(
                    f"netMHCIIpan output out of sync at row {row_idx}: "
                    f"expected peptide {key[0]}, got {peptide}"
                )

            allele = fields[columns["MHC"]]
            if allele not in self.alleles:
                raise RuntimeError(
                    f"netMHCIIpan reported unexpected allele '{allele}'"
                )

            if rank_col not in columns:
                raise RuntimeError(
                    f"netMHCIIpan output does not contain column '{rank_col}'; "
                    f"check that prediction_mode='{self.prediction_mode}' is supported "
                    f"by this binary version"
                )

            affinity_col = columns.get("Affinity(nM)")

            result[key][allele] = AlleleHit(
                rank=float(fields[columns[rank_col]]),
                core_offset=self._core_offset(fields[columns["Of"]]),
                score=_opt_float(
                    fields[columns[score_col]] if score_col in columns else None
                ),
                affinity=_opt_float(
                    fields[affinity_col] if affinity_col is not None else None
                ),
            )

            row_idx += 1
            n_parsed += 1

        return n_parsed


def _head(text: str, n_lines: int = 25) -> str:
    """
    First few lines of a captured stream, for inclusion in diagnostics.
    """
    if not text or not text.strip():
        return "    (empty)"

    lines = text.splitlines()
    out = "\n".join("    " + line for line in lines[:n_lines])

    if len(lines) > n_lines:
        out += f"\n    ... ({len(lines) - n_lines} more lines)"

    return out


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
