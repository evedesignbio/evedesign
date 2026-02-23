"""
Codon optimization with the DNA Chisel package

TODO: if implementing any other codon optimizers in the future, extract useful shared functionality
 into abstract base class
"""
import multiprocessing as mp
from typing import Sequence, Literal
import pandas as pd
from dnachisel import NoSolutionError
from loguru import logger

from evedesign.nucleotides import ProteinToDnaOptimizer, CodonUsageTable

try:
    import dnachisel as dc  # noqa
    from dnachisel.builtin_specifications.codon_optimization.BaseCodonOptimizationClass import (
        BaseCodonOptimizationClass  # noqa
    )
    from Bio.Seq import Seq  # noqa
    from Bio.Restriction.Restriction_Dictionary import rest_dict  # noqa
    from Bio.Data.CodonTable import unambiguous_dna_by_name  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

from evedesign.constants import GAP, VALID_DNA_SORTED
from evedesign.system import System, SystemInstance, EntityInstance
from evedesign.sequence import valid_sequence

OPTIMIZATION_METHODS = [
    "use_best_codon",
    "match_codon_usage"
]


class DNAChiselCodonOptimizer(ProteinToDnaOptimizer):
    available = IMPORT_AVAILABLE

    def __init__(
        self,
        method: Literal["use_best_codon", "match_codon_usage"],
        codon_usage_table: str | CodonUsageTable,
        avoid_sites: list[str] | None,
        gc_min: float | None = 0.4,
        gc_max: float | None = 0.6,
        gc_window: int | None = 30,
        max_homopolymer_length: int | None = 5,
        max_repeat_length: int | None = 9,
        avoid_hairpins: bool = True,
        genetic_code: str = "Standard",
        extra_constraints: Sequence[dc.Specification] | None = None,
        cpu: int = 1,
    ):
        """
        Create new codon optimizer based on DNA Chisel

        Parameters
        ----------
        method
            Optimize codon usage by maximizing codon adaption index (use_best_codon), or by matching
             match codon frequencies in target species (match_codon_usage)
        codon_usage_table
            Codon usage table for optimization. Can be any species valid for the python_codon_tables package
             (str; e.g. h_sapiens or e_coli), a taxonomy identifier (str of numeric code, will be downloaded
            from web), or an explicit codon usage table dictionary (CodonUsageTable)
        avoid_sites
            List of restriction enzyme sites to avoid during optimization (e.g. "BsaI"). For all valid options,
            see Bio.Restriction.Restriction_Dictionary.rest_dict
        gc_min
            Minimum GC content to enforce in optimized nucleotide sequence
        gc_max
            Maximum GC content to enforce in optimized nucleotide sequence
        gc_window
            If specified, compute GC content in a local window; otherwise will compute over entire nucleotide sequence
        max_homopolymer_length
            Maximum acceptable length of homopolymers (e.g. AAAAA), will remove any homopolymers with
            length > max_homopolymer_length. If None, will not constrain homopolymer occurrence.
        max_repeat_length
            Maximum acceptable length of arbitrary repeats; enforced during quantitative optimization
            (i.e. not as strict constraint).
        genetic_code
            Genetic code to ensure nucleotide sequence translates into specified amino acid sequences
            (note this is redundant to codon_usage_table but internally needed by dnachisel)
        extra_constraints
            Extra dnachisel specifications to use during optimization
        cpu
            If cpu > 1, parallelize optimization over different instances with specified number of processes.
        """
        if not self.available:
            raise ValueError(
                "dnachisel or biopython package could not be imported. Are they already installed?"
            )

        if method not in OPTIMIZATION_METHODS:
            raise ValueError(
                f"Invalid optimization method, valid options are {OPTIMIZATION_METHODS} "
            )

        self.method = method

        # verify we have a valid genetic code specified
        if genetic_code not in dc.biotools.CODON_TABLE_NAMES:
            raise ValueError(
                f"Invalid codon table, valid options are {dc.biotools.CODON_TABLE_NAMES}"
            )

        self.genetic_code = genetic_code
        self.start_codons = unambiguous_dna_by_name[self.genetic_code].start_codons
        self.stop_codons =  unambiguous_dna_by_name[self.genetic_code].stop_codons

        # retrieve explicit codon table as dictionary right away so we can verify against genetic code
        if isinstance(codon_usage_table, str):
            self.codon_table = BaseCodonOptimizationClass.get_codons_table(
                species=codon_usage_table, codon_usage_table=None
            )
        elif isinstance(codon_usage_table, dict):
            self.codon_table = codon_usage_table
        else:
            raise ValueError("Invalid codon_table argument")

        # verify that genetic code matches codon table
        for codon, aa in unambiguous_dna_by_name[self.genetic_code].forward_table.items():
            if codon not in self.codon_table[aa]:
                raise ValueError(
                    f"Mismatch between codon_usage_table and genetic_code:" +
                    f"aa: {aa} codon: {codon} options: {self.codon_table[aa]}"
                )

        # extra specifications to be added to optimization problem
        if extra_constraints is not None:
            self.specifications = list(extra_constraints)
        else:
            self.specifications = []

        self.max_homopolymer_length = max_homopolymer_length
        self.max_repeat_length = max_repeat_length
        self.avoid_hairpins = avoid_hairpins

        if (gc_min is None and gc_max is not None) or (gc_min is not None and gc_max is None):
            raise ValueError(
                "gc_min and gc_max need to be both specified or None"
            )

        if gc_min is not None and gc_max is not None:
            if not 0 <= gc_min < gc_max <= 1:
                raise ValueError(
                    "GC content specification must be 0 <= gc_min < gc_max <= 1"
                )

        self.gc_min = gc_min
        self.gc_max = gc_max
        self.gc_window = gc_window

        if avoid_sites is not None:
            for site in avoid_sites:
                if site not in rest_dict:
                    raise ValueError(
                        f"Restriction site {site} not available through biopython rest_dict"
                    )

        self.avoid_sites = avoid_sites

        # Number of CPUs to use for parallelization
        if not cpu >= 1:
            raise ValueError(
                "cpu must be >= 1 (1 for serial execution, > 1 for parallel execution)"
            )

        self.cpu = cpu

    def _optimize_seq(
        self,
        seq: str,
        upstream_dna: str,
        downstream_dna: str,
        reference_seq: str | None = None,
        reference_dna: str | None = None,
    ) -> tuple[str, float]:
        """
        Codon-optimize a single sequence

        Parameters
        ----------
        seq
            Protein sequence for which to create codon-optimized coding DNA sequence
        upstream_dna
            Upstream nucleotides before coding sequence (e.g. assembly/cloning overhangs)
        downstream_dna
            Downstream nucleotides after coding sequence (e.g. assembly/cloning overhangs)
        reference_seq
            If specified, only optimize positions in seq that differ from reference_seq
            (reference_dna must be specified as well)
        reference_dna
            Use this DNA sequence as template for optimization if reference_seq is specified,
            keeping any positions that do not differ compared to seq constant on the DNA level

        Returns
        -------
        Tuple with
         (i) optimized DNA sequence for seq (*excluding* upstream and downstream DNA)
         (ii) final optimization score
        """
        if (
            (reference_seq is None and reference_dna is not None) or
            (reference_seq is not None and reference_dna is None)
        ):
            raise ValueError(
                "Both reference_seq and reference_dna must be specified together or both None"
            )

        # normalized version of sequence (no gaps, insertions to uppercase)
        seq_norm = EntityInstance.normalize_rep_str(seq)
        upstream_dna = upstream_dna.upper()
        downstream_dna = downstream_dna.upper()

        # first, simply initialize the sequence (if we have a reference, we will backfill codons in next step)
        seq_dna = dc.reverse_translate(seq_norm)

        # region in full_dna to optimize (corresponds to seq_dna, i.e. keep upstream/downstream sequence fixed)
        seq_dna_start = len(upstream_dna)
        seq_dna_end = len(upstream_dna) + len(seq_dna)
        seq_dna_loc = (seq_dna_start, seq_dna_end)
        seq_dna_loc_both_strands = (*seq_dna_loc, 0)

        # fix codons based on reference sequence if specified
        fixed_codon_constraints = []

        if reference_dna is not None and reference_seq is not None:
            ref_codon_idx = 0
            # iterate through (potentially non-normal reference seq), we are only interested in "match states";
            # note that reference_dna is based on normalized reference sequence so may have different length
            ref_aa_codons = []
            for ref_idx, ref_aa in enumerate(reference_seq):
                if ref_aa == GAP:
                    # if we have a gap, this is a match state but no corresponding codon;
                    # keep alignment information by appending None, but do not increase codon index as
                    # this position will not be present in reference_dna sequence
                    ref_aa_codons.append(None)
                else:
                    # if match state, we keep the codon
                    if ref_aa.upper() == ref_aa:
                        ref_aa_codons.append(
                            (ref_idx, ref_aa, reference_dna[ref_codon_idx : ref_codon_idx + 3])
                        )

                    # increase codon index in any case (match and insertion), to skip over insertion codon
                    ref_codon_idx += 3

            seq_dna = list(seq_dna)

            # iterate through optimized sequence and update codons where needed; jointly iterate through
            # extracted reference sequence codons with ref_idx
            ref_idx = 0
            for i, aa in enumerate(seq):
                # only fix codons if we are in a match state (i.e., not an insert/lowercase symbol)
                if aa == GAP or aa == aa.upper():
                    cur_ref = ref_aa_codons[ref_idx]

                    # we can only use a reference codon if there is no deletion in reference (coded by None)
                    if cur_ref is not None:
                        _, ref_aa, ref_codon = cur_ref

                        # we can only keep the codon if the current position has the same aa as reference
                        if aa == ref_aa:
                            # replace codon with reference codon
                            seq_dna[i * 3 : (i + 1) * 3] = ref_codon

                            # fix codon during optimization
                            fixed_codon_constraints.append(
                                dc.AvoidChanges(
                                    location=(seq_dna_start + i * 3, seq_dna_start + (i + 1) * 3)
                                )
                            )

                    ref_idx += 1

            # verify that number of match states agrees between reference and optimized sequence
            if ref_idx != len(ref_aa_codons):
                raise ValueError(
                    "Number of aligned positions between reference and optimized sequence do not match: " +
                    f"ref: {reference_seq}, seq: {seq}"
                )

            seq_dna = "".join(seq_dna)

        # full sequence context for optimization problem
        full_dna = upstream_dna + seq_dna + downstream_dna

        # enforce correct translation of sequence and do not change upstream/downstream sequences
        seq_constraints = [
            dc.EnforceTranslation(
                location=seq_dna_loc, genetic_table=self.genetic_code, translation=seq_norm
            ),
            dc.AvoidChanges(
                location=(0, seq_dna_start),
            ),
            dc.AvoidChanges(
                location=(seq_dna_end, len(full_dna)),
            )
        ]

        # apply restriction enzyme site constraints only to optimized sequence, as these sites
        # may by design occur in the upstream/downstream sequences
        if self.avoid_sites is not None:
            for site in self.avoid_sites:
                # match pattern on both strands (0) in optimized region, "localized" function in code indicates
                # this takes partially overlapping matches into account as well
                seq_constraints.append(
                    dc.AvoidPattern(dc.EnzymeSitePattern(site), location=seq_dna_loc_both_strands)
                )

        if self.max_homopolymer_length is not None:
            for nuc in ["A", "C", "G", "T"]:
                # add 1 to length as max_homopolymer_length is maximum *acceptable* length
                seq_constraints.append(
                    dc.AvoidPattern(
                        dc.HomopolymerPattern(nuc, self.max_homopolymer_length + 1),
                        location=seq_dna_loc_both_strands
                    )
                )

        if self.max_repeat_length is not None:
            seq_constraints.append(
                dc.AvoidPattern(
                    dc.RepeatedKmerPattern(2, self.max_repeat_length + 1),
                    location=seq_dna_loc_both_strands
                )
            )

        if self.avoid_hairpins:
            seq_constraints.append(
                dc.AvoidHairpins(location=seq_dna_loc_both_strands)
            )

        # restraints
        quant_specifications = []

        # both min/max or neither specified
        if self.gc_min is not None and self.gc_max is not None:
            quant_specifications.append(
                dc.EnforceGCContent(
                    mini=self.gc_min, maxi=self.gc_max, window=self.gc_window, location=seq_dna_loc_both_strands
                )
            )

        problem = dc.DnaOptimizationProblem(
            sequence=full_dna,
            constraints=self.specifications + seq_constraints + fixed_codon_constraints,
            objectives=[dc.CodonOptimize(
                codon_usage_table=self.codon_table,
                method=self.method,
                location=seq_dna_loc,
            )] + quant_specifications,
            logger=None
        )

        # raw_score = problem.objective_scores_sum()
        try:
            problem.resolve_constraints()
            problem.optimize()
            opt_score = problem.objective_scores_sum()
        except NoSolutionError as e:
            # wrap around NoSolutionError as there is an error passing it through
            # to parent process in multiprocessing setting
            raise ValueError(
                f"Unable to optimize sequence '{seq}', relax constraints?"
            ) from e

        # extract full optimized sequence with upstream/downstream DNA
        dna_opt = problem.sequence
        assert len(dna_opt) == len(upstream_dna) + len(seq_dna) + len(downstream_dna)
        assert dna_opt[:seq_dna_start] == upstream_dna, "Upstream DNA sequence does not match input"
        assert dna_opt[seq_dna_end:] == downstream_dna, "Downstream DNA sequence does not match input"

        # extract optimized protein-coding DNA sequence and verify it translates correctly
        dna_seq_opt = dna_opt[seq_dna_start:seq_dna_end]
        dna_seq_transl = Seq(dna_seq_opt).translate(table=self.genetic_code)
        assert dna_seq_transl == seq_norm, "Translation of optimized sequence does not match input"

        # print(problem.mutation_space.string_representation())
        # print(problem.objectives_text_summary())

        return dna_seq_opt, opt_score

    def optimize(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int,
        upstream_dna: str,
        downstream_dna: str,
        reference: SystemInstance | None = None,
        reference_dna: str | None = None,
    ) -> pd.DataFrame:
        """
        Create codon-optimize DNA sequences for protein entity instances
        (needs to be called once per protein entity in multi-entity systems)

        Notes:
        i) Returned sequences will *not* include upstream_dna and downstream_dna,
         this is solely used as context for parametrizing the codon optimization
         algorithm

        ii) The method will return duplicate DNA sequences for duplicate protein
         sequences (output is not deduplicated, this is responsibility of user)

        iii) User is entirely responsible for ensuring that generated
         DNA sequences are correctly inserted into an ORF with upstream_dna
         and downstream_dna; verifying this without the full plasmid
         context is not possible in general way in this function as we cannot
         make the assumption that start/stop codons are necessarily part of
         upstream_dna and downstream_dna, respectively (e.g. when cloning
         a domain into a longer protein)

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which codon-optimized DNA sequences should be created
        entity
            Index of protein entity for which DNA sequence should be created
        upstream_dna
            Fixed DNA sequence directly upstream of DNA that will be generated here
            (e.g. assembly handle)
        downstream_dna
            Fixed DNA sequence directly downstream of DNA that will be generated here
            (e.g. assembly handle)
        reference
            Instance that should be used as reference to generate new DNA sequences.
            If specified, will reuse codons from the reference in any non-indel positions
            (to keep DNA background as constant as possible where needed)
        reference_dna
            DNA sequence for reference sequence (must translate into reference). If not specified,
            the reference will be first codon-optimized on its own to create the DNA reference
            from which codons will be reused.

        Returns
        -------
        Dataframe (guaranteed to be of same length and in same order as supplied instances list)
        with columns :
         i. "rep" containing protein sequence as in instance
         ii. "dna" containing the optimized DNA sequence guaranteed to translate into "rep",
         iii. "score" with optimization score (should be set to NaN if not available)
        """
        # verify that valid entity is selected
        if not 0 <= entity <= len(system):
            raise ValueError("Invalid entity index")

        if system[entity].type != "protein":
            raise ValueError("Can only optimize protein entities")

        # make sure all sequences are uppercase to simplify later handling
        upstream_dna = upstream_dna.upper()
        downstream_dna = downstream_dna.upper()

        # check if upstream/downstream DNA includes start and stop codons, respectively;
        # if not, warn the user (this may however be a valid input if trying to insert
        # into a larger ORF)
        if not any([upstream_dna.endswith(codon) for codon in self.start_codons]):
            logger.warning("upstream_dna does not end with a start codon. Is this intentional?")

        if not any([downstream_dna.startswith(codon) for codon in self.stop_codons]):
            logger.warning("downstream_dna does not start with a stop codon. Is this intentional?")

        # validate provided instances
        [
            system.valid_instance(
                instance, validate_reps=True, fixed_length=False, allow_deletions=True, raise_invalid=True,
            ) for instance in instances
        ]

        # validate upstream/downstream DNA sequences
        for s in [upstream_dna, downstream_dna]:
            s_valid, invalid_s_pos = valid_sequence(
                s, VALID_DNA_SORTED, allow_mask=False
            )

            if not s_valid:
                raise ValueError(
                    f"upstream_dna or downstream_dna is not a valid DNA sequence, invalid symbols: {invalid_s_pos}"
                )

        # check if we optimize a given reference sequence
        if reference is not None:
            # validate reference first
            system.valid_instance(
                reference, validate_reps=True, fixed_length=False, allow_deletions=True, raise_invalid=True,
            )

            # create normalized and raw versions of instance sequence
            # (the latter to keep potential alignment information)
            reference_seq_norm = "".join(reference[entity].normalized_rep())
            reference_seq = "".join(reference[entity].rep)

            # if we don't have a reference sequence, optimize it
            if reference_dna is None:
                # optimize reference sequence first (as this is reference, do this without being constrained
                # by any other sequence)
                reference_dna, reference_dna_score = self._optimize_seq(
                    seq=reference_seq_norm, upstream_dna=upstream_dna, downstream_dna=downstream_dna
                )
            else:
                reference_dna = reference_dna.upper()
                valid_ref, invalid_ref_pos = valid_sequence(
                    reference_dna, VALID_DNA_SORTED, allow_mask=False
                )

                if not valid_ref:
                    raise ValueError(
                        f"reference_dna is not a valid DNA sequence, invalid symbols: {invalid_ref_pos}"
                    )

                # verify that reference_dna has valid length and translation matches (with specified genetic code)
                if len(reference_dna) != len(reference_seq_norm) * 3:
                    raise ValueError(
                        "reference_dna length must be length of instance sequence * 3"
                    )

                if Seq(reference_dna).translate(table=self.genetic_code) != reference_seq_norm:
                    raise ValueError(
                        "reference_dna does not translate into reference instance sequence"
                    )
        else:
            reference_seq = None

        # extract and deduplicate protein sequences (do not perform unnecessary codon optimizations);
        # do not normalize to keep potential alignment information
        all_seqs = pd.Series(
            "".join(inst[entity].rep) for inst in instances
        )
        unique_seqs = all_seqs.drop_duplicates()

        # prepare list of individual optimization jobs
        jobs = [
            (seq, upstream_dna, downstream_dna, reference_seq, reference_dna)
            for seq in unique_seqs.tolist()
        ]

        # run jobs serially or in parallel
        if self.cpu == 1:
            res = [
                self._optimize_seq(*job) for job in jobs
            ]
        elif self.cpu > 1:
            with mp.Pool(processes=self.cpu) as pool:
                res = pool.starmap(
                    self._optimize_seq, jobs
                )

        unique_res_df = unique_seqs.to_frame("rep").assign(
            dna=[dna_seq for (dna_seq, score) in res],
            score=[score for (dna_seq, score) in res],
        )

        res_df_all = all_seqs.to_frame("rep").merge(
            unique_res_df, on="rep", how="left",
        )

        # verify we kept the original order of designs after merging
        assert (res_df_all.rep == all_seqs.values).all()  # noqa

        return res_df_all
