from os import PathLike
from typing import Literal, Self, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import torch

from evedesign.model import (
    BaseModel, Scorer, Generator, MutationScorer, ConditionalMutationScorer, Transformer
)
from evedesign.system import System, SystemInstance, EntityInstance, Mutant
from evedesign.utils import model_param_context
from evedesign.types import DeviceType, StatusCallback, BatchSize, EntityPosList
from evedesign.samplers.gibbs import GibbsSampler, ScanOrder, InitStrategy, TemperatureSchedule

try:
    from transformers import EsmForMaskedLM, AutoTokenizer  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class ESM2(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, Generator, Transformer):
    """
    Wrapper class around ESM2 model

    Note: warnings upon loading can be ignored (https://github.com/huggingface/transformers/issues/39405)
    """
    available = IMPORT_AVAILABLE
    name: str = "ESM2"
    citations: list[str] = ["doi:10.1126/science.ade2574"]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = False
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        model_name: Literal[
            "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"
        ] = "esm2_t33_650M_UR50D",
        model_dir_path: str | PathLike | None = None,
        batch_size: BatchSize = 64,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
        # GibbsSampler hyperparameters
        num_sweeps: int = 1000,
        init_strategy: InitStrategy = "system",
        scan_order: ScanOrder = "random",
        temperature_schedule: TemperatureSchedule | None = None
    ):
        if not self.available:
            raise ValueError(
                "transformers package could not be imported. Is it installed already?"
            )

        self.model_name = model_name
        self.model_dir_path = Path(
            model_dir_path
        ) if model_dir_path is not None else None
        self.keep_model_after_build = keep_model_after_build
        self.keep_model_after_pred = True
        self.device = device

        # Define maximum sequence length for ESM2 models (1024 tokens - 2 for special tokens)
        self.max_seq_length = 1022

        self._system = None
        self.model = None
        self.tokenizer = None  # Changed from alphabet to tokenizer

        self.batch_size = batch_size

        # Store GibbsSampler hyperparameters
        self.num_sweeps = num_sweeps
        self.init_strategy = init_strategy
        self.scan_order = scan_order
        self.temperature_schedule = temperature_schedule

        if self.batch_size != "auto" and self.batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        if self.batch_size == "auto":
            raise NotImplementedError(
                "Automatic batch_size not yet implemented"
            )

        self.token_ids = None
        self.encoding = None

    @property
    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have defined rep sequence"

        # Add check for sequence length
        max_seq_length = 1022  # 1024 - 2 for special tokens
        if len(target.rep) > max_seq_length:
            return False, f"Sequence length ({len(target.rep)}) exceeds maximum allowed ({max_seq_length})"

        return True, ""

    def _load_model(self):
        if self.model is not None:
            return

        if self.model_dir_path is None:
            # Load from HuggingFace hub
            try:
                # For remote loading from HuggingFace
                self.model = EsmForMaskedLM.from_pretrained(
                    f"facebook/{self.model_name}"
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"facebook/{self.model_name}"
                )
            except Exception as e:
                logger.error(f"Error loading model from HuggingFace: {e}")
                raise ValueError(
                    f"Failed to load model {self.model_name} from HuggingFace: {e}"
                )
        else:
            # Load from local file path
            try:
                # For local loading from a directory
                self.model = EsmForMaskedLM.from_pretrained(
                    self.model_dir_path
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir_path
                )
            except Exception as e:
                logger.error(f"Error loading model from local path: {e}")

                raise ValueError(
                    f"Failed to load model from {self.model_dir_path}: {e}"
                )

        self.model.eval()

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self.tokenizer = None  # Changed from alphabet to tokenizer

        self._release_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system

        # Additional check for sequence length
        target = system[0]
        if len(target.rep) > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({len(target.rep)}) exceeds maximum allowed by ESM2 ({self.max_seq_length})"
            )

        self.encoding = None
        self.token_ids = None

        return self

    def _validate_instances_and_max_length(
        self,
        instances: Sequence[SystemInstance],
    ) -> None:
        # Validate all instances in a single loop
        for instance in instances:
            # First validate the instance with system validation
            self.system.valid_instance(
                instance,
                validate_reps=True,
                require_reps=True,
                fixed_length=False,
                allow_deletions=False,
                raise_invalid=True,
            )

            # Now that we know the instance is valid
            seq = instance[0].rep
            seq_len = len(seq)

            # Check sequence length
            if seq_len > self.max_seq_length:
                raise ValueError(
                    f"Sequence length ({seq_len}) exceeds maximum allowed by ESM2 ({self.max_seq_length})"
                )

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Generate protein sequences using the ESM2 model with the GibbsSampler

        Parameters
        ----------
        num_designs
            Number of protein sequences to generate
        entities
            Indices of entities to redesign (default: [0])
        fixed_pos
            Positions to keep fixed during design
        temperature
            Initial temperature for sampling
        status_callback
            Optional callback function for progress updates

        Returns
        -------
        List[SystemInstance]
            Generated protein sequence instances
        """
        self.ready_or_raise()

        entities = entities if entities is not None else [0]
        if len(entities) != 1 or entities[0] != 0:
            raise ValueError(
                "Can only design single entity (entities = [0] | None)"
            )

        # Adjust num_designs to be a multiple of batch_size
        if rem := num_designs % self.batch_size:
            num_designs_adj = num_designs + (self.batch_size - rem)
            num_designs = num_designs_adj

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            logger.info(
                f"Generating {num_designs} designs with ESM2 using GibbsSampler"
            )

            # Create a GibbsSampler using the configured hyperparameters
            sampler = GibbsSampler(
                scorers=[self],
                weights=None,
                num_sweeps=self.num_sweeps,
                init_strategy=self.init_strategy,
                scan_order=self.scan_order,
                temperature_schedule=self.temperature_schedule,
                require_strict_pos=True,
                record_full_chain=False
            )

            # Generate designs
            instances = sampler.generate(
                num_designs=num_designs,
                entities=entities,
                fixed_pos=fixed_pos,
                temperature=temperature,
                status_callback=status_callback
            )

        # Score designs relative to reference
        target = self.system[0]
        ref_instance = SystemInstance(EntityInstance(rep="".join(target.rep)))
        all_instances = [ref_instance] + instances

        logger.info(f"Scoring {len(instances)} generated designs")
        scores = self.score(all_instances)
        ref_score = scores[0]

        # Attach normalized scores to instances
        for i, instance in enumerate(instances):
            instance.score = (scores[i+1] - ref_score)

        return instances

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()
        self._validate_instances_and_max_length(instances)

        # Convert any sequence arrays to strings
        sequences = []
        for instance in instances:
            seq = instance[0].rep
            seq = "".join(seq)
            sequences.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            scores = []

            # Process in batches
            for batch_start in range(0, len(sequences), self.batch_size):
                batch_end = min(
                    batch_start + self.batch_size, len(sequences)
                )
                batch_seqs = sequences[batch_start:batch_end]

                # Prepare batch data with tokenizer
                inputs = self.tokenizer(
                    batch_seqs, return_tensors="pt", padding=True
                ).to(self.device)

                # Compute log-likelihoods
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Calculate log-likelihood for each sequence
                    for i, seq in enumerate(batch_seqs):
                        # Get sequence length (excluding padding)
                        # -2 for special tokens
                        seq_len = len(self.tokenizer.encode(seq)) - 2

                        # Extract logits for the actual sequence (excluding padding and the last token)
                        # Skip the first special token
                        seq_logits = logits[i, 1:seq_len+1]

                        # Get target tokens (shifted by one position)
                        # +2 to include one more token as target
                        target_tokens = inputs.input_ids[i, 1:seq_len+1]

                        # Calculate log probabilities
                        token_probs = torch.log_softmax(seq_logits, dim=-1)

                        # Gather log probs for the target tokens
                        seq_log_probs = torch.gather(
                            token_probs,
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

                        # Sum log probs to get sequence log likelihood
                        seq_log_likelihood = seq_log_probs.sum().item()

                        scores.append(seq_log_likelihood)

        return np.array(scores)

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Perform a single mutation scan for the given instance using the Masked marginal probability approach
        """
        self.ready_or_raise()
        self._validate_instances_and_max_length([instance])

        if positions is not None and entity is None:
            raise ValueError(
                "Parameter entity must be explicitly specified if using parameter positions"
            )

        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # Get sequence and convert to string if needed
        target = self.system[0]
        instance_seq = instance[0].rep
        instance_seq = "".join(instance_seq)

        # Validate positions
        if positions is not None:
            self.valid_positions(
                positions, instance=instance, entities=0, raise_invalid=True
            )
        else:
            positions = list(
                range(target.first_index, target.first_index + len(target.rep))
            )

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            mutation_effects = []

            # For each position to scan
            for pos in positions:
                pos_idx = pos - target.first_index
                wt_aa = instance_seq[pos_idx]

                # Adjust for tokenizer offsets (assuming 1-to-1 mapping + 1 for start token)
                token_idx = pos_idx + 1

                # Tokenize and create masked input
                inputs = self.tokenizer(
                    instance_seq, return_tensors="pt"
                ).to(self.device)
                masked_inputs = inputs.copy()

                # Mask the position we want to predict
                # Get mask token id from tokenizer
                mask_token_id = self.tokenizer.mask_token_id
                masked_inputs['input_ids'][0, token_idx] = mask_token_id

                # Forward pass with masked input
                with torch.no_grad():
                    outputs = self.model(**masked_inputs)
                    masked_logits = outputs.logits[0]

                    # Convert logits to log probabilities for the masked position
                    pos_log_probs = torch.log_softmax(
                        masked_logits[token_idx], dim=-1
                    )

                    # Score each possible substitution
                    mut_scores = {}
                    for aa in target.alphabet(include_gap=False):
                        if aa == '-':  # Skip gap character
                            continue

                        # If same as wildtype, effect is 0
                        if aa == wt_aa:
                            mut_scores[aa] = 0.0
                            continue

                        # Get the token index for this amino acid
                        aa_token = self.tokenizer.convert_tokens_to_ids(aa)
                        wt_token = self.tokenizer.convert_tokens_to_ids(wt_aa)

                        # For masked marginal probability, calculate:
                        # log(p(mut_aa | masked_context)) - log(p(wt_aa | masked_context))
                        score_diff = (pos_log_probs[aa_token].item() -
                                      pos_log_probs[wt_token].item())

                        mut_scores[aa] = score_diff

                    # Store results for this position
                    mutation_effects.append({
                        'pos': pos,
                        'ref': wt_aa,
                        **mut_scores
                    })

                # Update status callback if provided
                if status_callback:
                    progress = (len(mutation_effects) / len(positions)) * 100
                    status_callback(
                        "running", progress, f"Processing position {pos}: {progress:.1f}% complete"
                    )

        # Convert to dataframe with proper index format
        df = pd.DataFrame(mutation_effects)
        df = df.set_index(['pos', 'ref'])
        df = pd.concat({entity: df}, names=["entity"])

        return df

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()
        self._validate_instances_and_max_length([instance])
        self.system.valid_mutants(
            instance, mutants, deletions=False, insertions=False, raise_invalid=True
        )

        # Get instance sequence
        target = self.system[0]
        instance_seq = instance[0].rep
        instance_seq = "".join(instance_seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            # Get all unique positions that will be mutated
            mutated_positions = set()
            for mutant in mutants:
                for sub in mutant:
                    pos_idx = sub.pos - target.first_index
                    mutated_positions.add(pos_idx)

            # Pre-compute masked probabilities for each position that will be mutated
            position_log_probs = {}
            position_list = list(mutated_positions)

            with torch.no_grad():
                # Process positions in batches
                for batch_start in range(0, len(position_list), self.batch_size):
                    batch_end = min(
                        batch_start + self.batch_size, len(position_list)
                    )
                    batch_positions = position_list[batch_start:batch_end]

                    if status_callback:
                        # First 50% for position processing
                        progress = (batch_start / len(position_list)) * 50
                        status_callback(
                            "running", progress, f"Computing masked probabilities batch {batch_start//self.batch_size + 1}"
                        )

                    # Create masked sequences for this batch
                    masked_seqs = []
                    for pos_idx in batch_positions:
                        masked_seq = list(instance_seq)
                        masked_seq[pos_idx] = self.tokenizer.mask_token
                        masked_seqs.append("".join(masked_seq))

                    # Tokenize batch
                    batch_inputs = self.tokenizer(
                        masked_seqs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)

                    # Single forward pass for batch
                    outputs = self.model(**batch_inputs)

                    # Extract probabilities for each position in the batch
                    for batch_idx, pos_idx in enumerate(batch_positions):
                        # Find mask token position in this sequence
                        mask_token_id = self.tokenizer.mask_token_id
                        mask_positions = (batch_inputs.input_ids[batch_idx] == mask_token_id).nonzero(  # noqa
                            as_tuple=True
                        )[0]

                        if len(mask_positions) > 0:
                            mask_pos = mask_positions[0]
                            logits = outputs.logits[batch_idx, mask_pos]
                            log_probs = torch.log_softmax(logits, dim=-1)
                            position_log_probs[pos_idx] = log_probs
                        else:
                            raise ValueError(
                                f"Mask token not found for position {pos_idx}")

            # Calculate scores for all mutants using the pre-computed masked probabilities
            mutant_scores = []
            for i, mutant in enumerate(mutants):
                if status_callback:
                    # Second 50% for mutant scoring
                    progress = 50 + ((i + 1) / len(mutants)) * 50
                    status_callback(
                        "running", progress, f"Scoring mutant {i + 1}/{len(mutants)}"
                    )

                total_score = 0.0

                for sub in mutant:
                    pos_idx = sub.pos - target.first_index
                    wt_aa = instance_seq[pos_idx]
                    mut_aa = sub.to

                    if wt_aa == mut_aa:
                        continue  # No change in score for unchanged positions

                    # Get token IDs
                    wt_token = self.tokenizer.convert_tokens_to_ids(wt_aa)
                    mut_token = self.tokenizer.convert_tokens_to_ids(mut_aa)

                    # Get the pre-computed log probabilities for this position
                    log_probs = position_log_probs[pos_idx]

                    # Calculate score difference for this mutation
                    wt_log_prob = log_probs[wt_token].item()
                    mut_log_prob = log_probs[mut_token].item()

                    score_diff = (mut_log_prob - wt_log_prob)
                    total_score += score_diff

                mutant_scores.append(total_score)

        return np.array(mutant_scores)

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Score conditional probabilities for specified positions in the sequences
        using masked-marginals approach with batching for efficiency
        """
        self.ready_or_raise()
        self._validate_instances_and_max_length(instances)

        # Validate input parameters
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length"
            )

        # Validate positions
        target = self.system[0]
        for instance, pos in zip(instances, positions):
            self.valid_positions(
                [pos], instance=instance, entities=0, raise_invalid=True
            )

        # Convert sequences to strings if needed
        seqs = []
        for instance in instances:
            seq = instance[0].rep
            seq = "".join(seq)
            seqs.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            conditionals_list = []

            # Process in batches for efficiency
            for batch_start in range(0, len(seqs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(seqs))
                batch_seqs = seqs[batch_start:batch_end]
                batch_positions = positions[batch_start:batch_end]
                batch_entities = entities[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))

                if status_callback:
                    progress = (batch_start / len(seqs)) * 100
                    status_callback(
                        "running", progress, f"Processing batch {batch_start//self.batch_size + 1}"
                    )

                # Create masked sequences for this batch
                masked_seqs = []
                for seq, pos in zip(batch_seqs, batch_positions):
                    pos_idx = pos - target.first_index
                    seq_list = list(seq)
                    seq_list[pos_idx] = self.tokenizer.mask_token
                    masked_seqs.append("".join(seq_list))

                # Tokenize all masked sequences in the batch
                inputs = self.tokenizer(
                    masked_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                with torch.no_grad():
                    # Single forward pass for the entire batch
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Process each sequence in the batch
                    for batch_idx, (orig_idx, pos, entity) in enumerate(
                        zip(batch_indices, batch_positions, batch_entities)
                    ):
                        # Find the position of the mask token in this sequence
                        mask_token_id = self.tokenizer.mask_token_id
                        mask_positions = (inputs.input_ids[batch_idx] == mask_token_id).nonzero(  # noqa
                            as_tuple=True
                        )[0]

                        if len(mask_positions) == 0:
                            raise ValueError(
                                f"Mask token not found in sequence {orig_idx}")

                        # Take first mask position
                        mask_pos = mask_positions[0]

                        # Get logits for the masked position
                        pos_logits = logits[batch_idx, mask_pos]

                        # Apply log softmax to get log probabilities
                        log_probs = torch.log_softmax(pos_logits, dim=-1)

                        # Convert to amino acid probabilities
                        aa_probs = {}
                        for aa in target.alphabet(include_gap=False):
                            if aa == '-':  # Skip gap character
                                aa_probs[aa] = 0.0
                            else:
                                aa_token_id = self.tokenizer.convert_tokens_to_ids(
                                    aa)
                                aa_probs[aa] = log_probs[aa_token_id].item()

                        # Store results
                        conditionals_list.append({
                            'instance': orig_idx,
                            'entity': entity,
                            'pos': pos,
                            **aa_probs
                        })

        # Create dataframe with proper index format
        conditionals = pd.DataFrame(conditionals_list)
        conditionals = conditionals.set_index(['instance', 'entity', 'pos'])

        return conditionals

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None   # noqa
    ) -> list[SystemInstance]:
        """
        Transform system instances by adding embeddings from the ESM2 model
        """
        self.ready_or_raise()
        self._validate_instances_and_max_length(instances)

        # Default to entity 0 if not specified
        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            transformed_instances = []

            # Process in batches
            for batch_start in range(0, len(instances), self.batch_size):
                batch_end = min(
                    batch_start + self.batch_size, len(instances)
                )
                batch_instances = instances[batch_start:batch_end]

                # Prepare batch sequences
                sequences = []
                for instance in batch_instances:
                    seq = instance[0].rep
                    seq = "".join(seq)
                    sequences.append(seq)

                # Tokenize sequences
                inputs = self.tokenizer(
                    sequences, return_tensors="pt", padding=True
                ).to(self.device)

                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Get the hidden states from the last layer
                    # Note: For EsmForMaskedLM, the hidden states are typically accessed as:
                    # hidden_states = outputs.hidden_states[-1]
                    # Last layer hidden states
                    hidden_states = outputs.hidden_states[-1]

                    # Process each instance in the batch
                    for i, instance in enumerate(batch_instances):
                        # Create new entity instance
                        new_entity = instance[0].copy()

                        # Create a new SystemInstance with this entity
                        new_instance = instance.copy()
                        new_instance.entity_instances = [new_entity]

                        # Get sequence length (excluding padding)
                        # -2 for special tokens
                        seq_len = len(self.tokenizer.encode(sequences[i])) - 2

                        # Store the embedding (excluding the first token which is the start token)
                        new_entity.embedding = hidden_states[
                            i, 1:seq_len+1
                        ].cpu().numpy()
                        # Replace the entity instance in copied system instance
                        new_instance.data = [new_entity]

                        # Calculate and store score
                        logits = outputs.logits[i, 1:seq_len+1]  # exclude last token
                        token_probs = torch.log_softmax(logits, dim=-1)

                        # Get the target tokens (shifted by one)
                        target_tokens = inputs.input_ids[i, 1:seq_len+1]

                        # Calculate log probabilities for target tokens
                        seq_log_probs = torch.gather(
                            token_probs,
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

                        new_instance.score = seq_log_probs.sum().item()
                        transformed_instances.append(new_instance)

        return transformed_instances
