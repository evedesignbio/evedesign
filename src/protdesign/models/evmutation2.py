"""
Wrapper class around EVmutation2/picasso model
"""
from os import PathLike
from typing import Self, Tuple, Sequence, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
from loguru import logger
import torch

from protdesign.model import (
    BaseModel, Scorer, Generator, RequiredResources, MutationScorer, ConditionalMutationScorer
)
from protdesign.entity import System, SystemInstance, EntityInstance, EntityPosList, Mutant
from protdesign.constants import MASK
from protdesign.utils import ensure_sequence, model_param_context
from protdesign.types import DeviceType, StatusCallback, BatchSize

try:
    from picasso_model import model, features, parsers  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class EVmutation2(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, Generator):
    """
    Wrapper class around EVmutation2/picasso model
    """
    available = IMPORT_AVAILABLE
    name: str = "EVmutation2"

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    # molecular model properties
    requires_heavy_build: bool = False
    requires_seqs: bool = True
    requires_msa: bool = True
    requires_3d: bool = False

    def __init__(
        self,
        model_file_path: str | PathLike,
        encoder_num_samples: int = 1,
        encoder_num_recycling_steps: int = 4,
        encoder_max_num_msa: int | None = 2048,
        decoder_batch_size: BatchSize = 64,
        decoder_num_full_samples: int = 16,
        decoder_num_mutant_samples: int = 16,
        decoder_share_order_across_encodings: bool = True,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate new EVcouplings2 model

        TODO: support min_p sampling and with extra parameters on generate() method

        Parameters
        ----------
        model_file_path
            Path to Lightning checkpoint
        encoder_num_samples
            Number of encoder samples to draw (at least 1), can improve model performance
        encoder_num_recycling_steps
            Recycling steps to run when computing encoding
        encoder_max_num_msa
            Number of sequences to sample from MSA when computing encoding
        decoder_batch_size
            Maximum number of sequences to decode concurrently
        decoder_num_full_samples
            Number of sampled decoding orders when computing full sequence scores
        decoder_num_mutant_samples
            Number of sampled decoding orders when computing mutant scores
        decoder_share_order_across_encodings
            Reuse decoding order across multiple encodings (if more than 1 used)
        keep_model_after_build
            If True, keep model parameters asssociated to instance after build step
            to avoid reloading when scoring/generating. If serializing model, set to
            False to avoid storing model parameters repeatedly.
        device
            Device to use for computations
        """
        if not self.available:
            raise ValueError("EVmutation2 package could not be imported. Is it installed already?")

        self.model_file_path = model_file_path
        self.keep_model_after_build = keep_model_after_build

        # by default, keep parameters loaded once loaded for prediction purposes to avoid reloading over and over
        self.keep_model_after_pred = True
        self.device = device

        # modelled system
        self._system = None

        # lazy-load model when needed
        self.model = None

        # model parameters for encoding and decoding during inference
        self.encoder_num_samples = encoder_num_samples
        self.encoder_num_recycling_steps = encoder_num_recycling_steps
        self.encoder_max_num_msa = encoder_max_num_msa
        self.decoder_batch_size = decoder_batch_size
        self.decoder_num_full_samples = decoder_num_full_samples
        self.decoder_num_mutant_samples = decoder_num_mutant_samples
        self.decoder_share_order_across_encodings = decoder_share_order_across_encodings

        if self.encoder_num_samples < 1 or self.decoder_num_full_samples < 1 or self.decoder_num_mutant_samples < 1:
            raise ValueError(
                "encoder_num_samples, decoder_num_single_samples and decoder_num_single_samples must all be > 0"
            )

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        if self.decoder_batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")

        # encodings created when calling build() method;
        # first for permanent association with object
        self.encoding = None
        self.pos_mask = None

        self._single_rep_on_device = None
        self._pair_rep_on_device = None
        self._pos_mask_on_device = None

    @property
    def ready(self):
        return self.system is not None and self.encoding is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None=None) -> Tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) != 1 or system[0].type_ != "protein":
            return False, "Can only handle single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have defined rep sequence"

        if target.sequences is None or len(target.sequences.seqs) == 0:
            return False, "Must provide sequences for model inference"

        if not target.sequences.aligned:
            return False, "Provided sequences must be aligned"

        # this should be ensured by construction of system but check again to be safe
        # if not valid_protein_sequence(
        #     target.rep, allow_mask=True, allow_gap=True, allow_ambiguous=True
        # ):
        #     return False, "Input sequence may only contain AA symbols or mask"

        # TODO: more checks on alignment: does length match target rep;
        #  and is alignment compatible with a3m format

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: System,
        data: None = None,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        raise NotImplementedError(
            "Resource estimation not yet implemented"
        )
        # TODO: implement meaningful requirements depending on target size instead of made up values
        # return RequiredResources(
        #     min_gpu_cores=1,
        #     min_gpu_memory_per_core=16000,
        #     min_cpu_cores=1,
        #     min_cpu_memory_per_core=16000,
        #     max_batch_size=512,
        #     time=1,
        # )

    def _load_model(self):
        # avoid reloading if already loaded
        if self.model is not None:
            return

        self.model = model.Model.load_from_checkpoint(
            self.model_file_path, map_location=torch.device(self.device)
        )

        # switch to evaluation mode
        self.model.eval()

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

        # load MSA
        # TODO: clunky hack - reassemble sequences back into one string and pass into parser;
        #  should really update parser to receive sequences and headers
        msa_a3m = target.sequences.to_a3m()
        a3m_lines = "".join(
            f">{seq.id_}\n{seq.seq}\n" for seq in msa_a3m.seqs
        )

        msa = parsers.parse_a3m(a3m_lines)

        # ideally would move this check over to can_model() but checking can
        # then become more resource-intensive
        if len(msa.sequences[0]) != len(target.rep):
            raise ValueError(
                "Length of MSA does not map to length of target representation"
            )

        # featurize and batch; add structure features here eventually as well when
        # that part of EVmutation2 model is finished
        d = features.extract_msa_feature_data(msa)
        f = features.prepare_msa_features(*d)
        input_features = features.batch_features(
            [f], device=self.device
        )

        # also store position mask for prediction time
        self.pos_mask = input_features.pos_mask.cpu()

        # context for loading (and possibly destroying model parameters)
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            with torch.no_grad():
                s, p = [], []
                # create requested number of encoder samples (single and pair representation)
                for i in range(self.encoder_num_samples):
                    cur_s, cur_p = self.model.encoder(
                        input_features,
                        num_recycling_steps=self.encoder_num_recycling_steps,
                        max_num_msa=self.encoder_max_num_msa,
                    )
                    s.append(cur_s)
                    p.append(cur_p)

                # concatenate into one tensor each for single and pair representation
                s = torch.cat(s, dim=0)
                p = torch.cat(p, dim=0)

                # attach encodings to instance if we keep model parameters
                if self.keep_model_after_build:
                    self._single_rep_on_device = s
                    self._pair_rep_on_device = p
                    self._pos_mask_on_device = input_features.pos_mask

                # store encodings, make sure these are moved to CPU for good serialization behaviour
                self.encoding = (
                    s.cpu(), p.cpu()
                )

        # TODO: automatically estimate robust maximum possible batch size for decoder if set to "auto"
        #  depending on available resources, and update self.decoder_batch_size

        # return self to allow method chaining
        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> List[Tuple[int, int]]:
        self.ready_or_raise()

        # implementation here is very simple: we model all positions of exactly one target
        # protein sequence of fixed length (i.e. can ignore the passed instance);
        # none of the positions along the sequence are excluded so we can simply enumerate starting from first_index
        target = self.system[0]
        return [
            (0, pos) for pos, _ in enumerate(target.rep, start=target.first_index)
        ]

    def _validate_instances(
        self,
        instances: Sequence[SystemInstance],
    ) -> None:
        # validate instance sequences; must all have the same length
        [
            self.system.valid_instance(
                instance,
                validate_reps=True,
                fixed_length=True,
                allow_deletions=True,
                raise_invalid=True,
            ) for instance in instances
        ]

    @contextmanager
    def _reps_on_device(self, keep: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper to move all necessary information to target device
        when calling inference methods

        Parameters
        ----------
        keep
            If True, keep single/pair representations on device after exiting the manager;
            if False, remove them and clear cache where applicable

        Returns
        -------
        Tuple of single representations, pair representations, and position_mask on
        target device
        """
        # move representations and position mask to device
        try:
            # reload representations if anything is missing
            if (
                self._pos_mask_on_device is None or
                self._single_rep_on_device is None or
                self._pair_rep_on_device is None
            ):
                (s, p) = self.encoding
                assert s.shape[0] == p.shape[0], "Number of single and pair representations does not agree"
                self._single_rep_on_device = s.to(self.device)
                self._pair_rep_on_device = p.to(self.device)
                self._pos_mask_on_device = self.pos_mask.to(self.device)

            yield
        finally:
            # if not keeping representations, release them again and clear cache
            if not keep:
                self._single_rep_on_device = None
                self._pair_rep_on_device = None
                self._pos_mask_on_device = None
                self._release_cache()
                assert False, "Should not come here with current implementation"

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        """
        TODO: support min_p sampling parameter eventually
        """
        self.ready_or_raise()

        # verify validity of entity selection, even if not used since
        # at this point method can only handle single entity
        if entities is not None:
            entities = ensure_sequence(entities)
            if len(entities) != 1 or entities[0] != 0:
                raise ValueError("Can only design single entity (entities = [0] | None)")
        else:
            # not used for now
            entities = [0]  # noqa

        target = self.system[0]

        # extract fixed pos for single chain
        if fixed_pos is not None:
            if len(fixed_pos) != 1 or list(fixed_pos)[0] != 0:
                raise ValueError(
                    "Only accepting position mapping for entity 0"
                )

            # verify if all positions are valid
            self.valid_positions(fixed_pos[0], entities=0, raise_invalid=True)
            fixed_pos = set(fixed_pos[0])
        else:
            fixed_pos = set()

        if len(fixed_pos) == len(target.rep):
            raise ValueError("All positions fixed, need to sample at least one position")

        # mark which positions to design (with mask symbol)
        base_seq = [
            symbol if pos in fixed_pos else MASK
            for pos, symbol in enumerate(
                target.rep, start=target.first_index
            )
        ]

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred),
            self._reps_on_device(self.keep_model_after_pred)
        ):
            # sampling function expects number of designs to be a multiple of batch_size,
            # so adjust accordingly
            if rem := num_designs % self.decoder_batch_size != 0:
                num_designs_adj = num_designs + (self.decoder_batch_size - rem)
            else:
                num_designs_adj = num_designs

            logger.warning(
                "Sampling using a preliminary inefficient O(N^3) implementation which needs to be "
                "improved to O(N^2) for production use"
            )

            # note: method has @torch.inference_mode() so no_grad not necessary here
            # TODO: update sampling method to update generation status dynamically with callback
            designs, _ = self.model.decoder.sample_inefficient(
                single=self._single_rep_on_device,
                pairwise=self._pair_rep_on_device,
                pos_mask=self._pos_mask_on_device,
                seq=base_seq,
                batch_size=self.decoder_batch_size,
                num_samples=num_designs_adj,
                temperature=temperature,
                sample_gaps=deletions,
                # min_p=None,  # TODO: implement
            )

        # score the designs relative to entity sequence (ideally, user supplied WT sequence, but user can
        # always rescore the designs later if needed)

        # prepend reference sequence, and create instances;
        # note ref_and_designs is list[str] that is transformed to np array by EntityInstance constructor
        ref_and_designs = ["".join(target.rep)] + list(designs.seq)
        instances = [
            SystemInstance(
                EntityInstance(rep=rep)
            ) for rep in ref_and_designs
        ]

        # score and attach to instances (normalize by reference score)
        scores = self.score(instances)
        ref_score = scores[0]

        # remove reference in first position again
        instances_with_score = [
            SystemInstance(
                EntityInstance(rep=seq),
                score=score - ref_score
            ) for seq, score in zip(ref_and_designs, scores)
        ][1:]

        assert len(instances_with_score) >= num_designs, "Not returning minimum guaranteed number of designs"

        return instances_with_score

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # validate sequences
        self._validate_instances(instances)

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred),
            self._reps_on_device(self.keep_model_after_pred)
        ):
            # note that score_full_probability could also handle numpy arrays but not yet documented,
            # so turn into list[str] for due diligence
            str_instances = [
                "".join(instance[0].rep) for instance in instances
            ]

            scores = self.model.decoder.score_full_probability(
                str_instances,
                single=self._single_rep_on_device,
                pairwise=self._pair_rep_on_device,
                pos_mask=self._pos_mask_on_device,
                batch_size=self.decoder_batch_size,
                num_samples=self.decoder_num_full_samples,
                share_decoding_order_across_encodings=self.decoder_share_order_across_encodings,
            )

        # average the logits across encoder and decoder samples,
        # and make sure aggregated dataframe it is sorted by sequence index
        scores_agg = scores.groupby(
            level="seq_idx"
        ).mean().sort_index()

        assert len(scores_agg) == len(instances), "Length of scores does not length of instances"

        # return as numpy vector
        return scores_agg["score"].values

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Note: could express this function through newer score_conditionals to simplify codebase
        and avoid redundancy (either here, or inside picasso_model package, tbd)
        """
        self.ready_or_raise()

        # check instance against molecular system, requiring fixed length of sequence
        # as was used for entity specification as we have a fixed-length model
        self._validate_instances([instance])

        if positions is not None and entity is None:
            raise ValueError(
                "Parameter entity must be explicitly specified if using parameter positions"
            )

        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # extract single target entity from system, nd get sequence from instance
        # (we safely can access this as we have verified instance against system)
        target = self.system[0]
        instance_seq = instance[0].rep

        # validate positions
        if positions is not None:
            self.valid_positions(positions, entities=0, raise_invalid=True)

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred),
            self._reps_on_device()
        ):
            # get number of encoder samples (single/pair representations)
            num_encodings = self._single_rep_on_device.shape[0]

            # iterate through encoder samples; we can average these as these are log-odds scores, i.e.
            # different decoding orders will already have cancelled out. ultimately, this functionality should
            # probably go inside the score_single_mutants() method in picasso...
            effects = {}
            for idx_enc in range(num_encodings):
                # note: method has @torch.inference_mode() so no_grad not necessary here
                effects[idx_enc] = self.model.decoder.score_single_mutants(
                    seq=instance_seq,
                    first_index=target.first_index,
                    single=self._single_rep_on_device[[idx_enc]],
                    pairwise=self._pair_rep_on_device[[idx_enc]],
                    pos_mask=self._pos_mask_on_device,
                    position_subset=positions,
                    num_samples=self.decoder_num_mutant_samples,
                    batch_size=self.decoder_batch_size,
                )

        # assemble multiple scores
        effects = pd.concat(
            effects, axis=0, names=["encoder_sample"]
        )

        # subtract WT score, average and limit to relevant output symbols
        effects = effects.sub(
            effects["wt"], axis=0,
        ).groupby(
            level=["pos", "wt_aa"]
        ).mean().reindex(
            target.alphabet(include_gap=True), axis=1
        )

        effects.index.names = ["pos", "ref"]

        # add entity 0 to index
        effects = pd.concat(
            {entity: effects}, names=["entity"]
        )

        assert (
            (positions is None and len(effects) == len(target.rep)) or
            (positions is not None and len(effects) == len(positions))
        ), "Invalid number of positions in output dataframe"

        return effects

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # check instance against molecular system, requiring fixed length of sequence
        # as was used for entity specification as we have a fixed-length model
        self._validate_instances([instance])

        # verify if mutants are valid relative to system and instance
        self.system.valid_mutants(
            instance, mutants, deletions=True, insertions=False, raise_invalid=True
        )

        # extract single target entity from system, and get sequence from instance
        # (we safely can access this as we have verified instance against system)
        target = self.system[0]
        instance_seq = instance[0].rep

        # transform mutants into format expected by EVmutation2
        mutants_transformed = [
            [
                (subs.pos, subs.ref, subs.to) for subs in mutant
            ] for mutant in mutants
        ]

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred),
            self._reps_on_device()
        ):
            # get number of encoder samples (single/pair representations)
            num_encodings = self._single_rep_on_device.shape[0]

            # iterate through encoder samples; we can average these as these are log-odds scores, i.e.
            # different decoding orders will already have cancelled out. ultimately, this functionality should
            # probably go inside the score_single_mutants() method in picasso...
            effects = {}
            for idx_enc in range(num_encodings):
                # note: method has @torch.inference_mode() so no_grad not necessary here
                effects[idx_enc] = self.model.decoder.score_mutants(
                    seq=instance_seq,
                    mutants=mutants_transformed,
                    first_index=target.first_index,
                    single=self._single_rep_on_device[[idx_enc]],
                    pairwise=self._pair_rep_on_device[[idx_enc]],
                    pos_mask=self._pos_mask_on_device,
                    num_samples=self.decoder_num_mutant_samples,
                    batch_size=self.decoder_batch_size,
                )

            # assemble multiple scores and average logits; careful not to resort dataframe so
            # we keep the original mutant order
            effects = pd.concat(
                effects, axis=0, names=["encoder_sample"]
            ).groupby(
                level="mutant",
                sort=False,
            ).mean()

        # return as simple 1D numpy array
        return effects.score.values

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        self.ready_or_raise()

        # validate instance sequences
        self._validate_instances(instances)

        # validate entity specification (only handle single entity for now)
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        if not len(instances) == len(entities) == len(positions):
            raise ValueError("Sequences for instances, entities and positions must all have same length")

        target = self.system[0]

        # validate positions
        self.valid_positions(positions, entities=0, raise_invalid=True)

        # extract sequences;
        # note: could also pass numpy rep directly but use proper signature for due diligence
        seqs = [
            "".join(instance[0].rep) for instance in instances
        ]

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred),
            self._reps_on_device()
        ):
            scores = self.model.decoder.score_conditional(
                seqs=seqs,
                positions=positions,
                first_index=target.first_index,
                single=self._single_rep_on_device,
                pairwise=self._pair_rep_on_device,
                pos_mask=self._pos_mask_on_device,
                batch_size=self.decoder_batch_size,
                num_samples=self.decoder_num_mutant_samples,
                share_decoding_order_across_encodings=self.decoder_share_order_across_encodings,
            )

        # average encoder and decoder samples
        conditionals = scores.groupby(
            level=["seq_idx", "pos"], sort=False
        ).mean().reindex(
            target.alphabet(include_gap=True), axis=1
        )
        conditionals.index.names =["instance", "pos"]

        # add entity 0 to index, then move instance index to outermost level
        conditionals = pd.concat(
            {0: conditionals}, names=["entity"]
        ).swaplevel(
            i=0, j=1, axis=0
        )

        assert len(conditionals) == len(entities), "Length mismatch between output and input"

        return conditionals

