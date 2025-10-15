import numpy as np
import torch
import tempfile
import os
from typing import List, Dict, Optional, Sequence, Callable
from protdesign.entity import System, SystemInstance, EntityInstance
from protdesign.entity import EntityPosList
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Tuple, Sequence, List

# Import the LigandMPNN modules
from .data_utils import (
    featurize,
    parse_PDB,
    restype_str_to_int,
    restype_int_to_str,
    get_score,
    get_seq_rec
)
from .model_utils import ProteinMPNN


def ensure_sequence(value):
    """Convert single values to sequences"""
    if isinstance(value, (list, tuple)):
        return value
    return [value]


class LigandMPNNWrapper:
    """
    Wrapper for LigandMPNN that works with System objects.

    Usage:
        wrapper = LigandMPNNWrapper()
        wrapper.build(system)
        instances = wrapper.generate(num_designs=10, temperature=0.2)
        scores = wrapper.score(instances)
    """

    def __init__(self,
                 model_type: str = "ligand_mpnn",
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the LigandMPNN wrapper.

        Args:
            model_type: Type of model ("ligand_mpnn", "protein_mpnn", "soluble_mpnn", etc.)
            checkpoint_path: Path to model checkpoint
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Set default checkpoint paths
        if checkpoint_path is None:
            default_paths = {
                "ligand_mpnn": "./model_params/ligandmpnn_v_32_010_25.pt",
                "protein_mpnn": "./model_params/proteinmpnn_v_48_020.pt",
                "soluble_mpnn": "./model_params/solublempnn_v_48_020.pt",
            }
            checkpoint_path = default_paths.get(
                model_type, default_paths["ligand_mpnn"])

        self.checkpoint_path = checkpoint_path
        self.model = None

        # State that gets set during build()
        self.system = None
        self.feature_dict = None
        self.entity_lengths = None
        self.chain_mapping = None
        self.symmetry_residues = None
        self.symmetry_weights = None
        self.native_seq = None
        self.pdb_path = None
        self._is_built = False

        self._load_model()

    def _load_model(self):
        """Load the model from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract model parameters
        if self.model_type == "ligand_mpnn":
            atom_context_num = checkpoint.get("atom_context_num", 25)
            k_neighbors = checkpoint.get("num_edges", 32)
        else:
            atom_context_num = 1
            k_neighbors = checkpoint.get("num_edges", 48)

        # Initialize model
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=self.model_type,
            ligand_mpnn_use_side_chain_context=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def ready_or_raise(self):
        """Check if the model is ready (built) or raise an error"""
        if not self._is_built:
            raise RuntimeError(
                "Model not built. Call build(system) before generate() or score()"
            )

    def build(self,
              system: System,
              ligand_cutoff: float = 8.0,
              use_ligand_context: bool = True) -> 'LigandMPNNWrapper':
        """
        Build/prepare the system for sequence generation.

        Args:
            system: System object containing protein entities and structures
            ligand_cutoff: Distance cutoff for ligand context
            use_ligand_context: Whether to use ligand context

        Returns:
            self for method chaining
        """
        self.system = system

        # Get entity sequence lengths
        self.entity_lengths = [(idx, len(entity.rep) if entity.rep is not None else 0)
                               for idx, entity in enumerate(system)]

        # Convert system to PDB file
        self.pdb_path = self._system_to_pdb_file(system)

        # Parse PDB with LigandMPNN
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            self.pdb_path,
            device=self.device,
            chains=[],
            parse_all_atoms=True,
            parse_atoms_with_zero_occupancy=False,
        )

        # Create mappings
        self.chain_mapping = self._create_chain_mapping(system)

        # Determine homo-oligomer symmetry if applicable
        self.symmetry_residues, self.symmetry_weights = self._determine_homooligomer_symmetry(
            system)

        # Set up chain mask (which residues to design)
        chain_mask = torch.ones_like(protein_dict["mask"], dtype=torch.float32)
        protein_dict["chain_mask"] = chain_mask

        # Featurize the protein
        self.feature_dict = featurize(
            protein_dict,
            cutoff_for_score=ligand_cutoff,
            use_atom_context=use_ligand_context,
            number_of_ligand_atoms=getattr(self.model, 'atom_context_num', 25),
            model_type=self.model_type,
        )

        # Store native sequence
        self.native_seq = "".join([
            restype_int_to_str[aa] for aa in self.feature_dict["S"][0].cpu().numpy()
        ])

        self._is_built = True
        return self

    def _system_to_pdb_file(self, system: System) -> str:
        """Convert a System object to a temporary PDB file."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(temp_fd)

        structure_keys = set()
        for entity in system:
            structure_keys.update(entity.structures.keys()
                                  ) if entity.structures else None

        if len(structure_keys) > 1:
            raise NotImplementedError(
                "Multi-state design not currently supported")

        structure_key = list(structure_keys)[0] if structure_keys else None

        # Collect all chains from all entities for this structure
        all_chains = []
        for entity in system:
            if entity.structures and structure_key in entity.structures:
                entity_chains = entity.structures[structure_key]
                if not isinstance(entity_chains, list):
                    entity_chains = [entity_chains]
                all_chains.extend(entity_chains)
                entity_id = entity.id_ or f'entity_{len(all_chains)}'
                print(
                    f"Adding chains from entity {entity_id}: {len(entity_chains)} chains")

        print(f"Total chains to write: {len(all_chains)}")

        # Write combined structure to PDB file
        if all_chains:
            if hasattr(all_chains[0], 'to_file') and len(all_chains) == 1:
                all_chains[0].to_file(temp_path, format="pdb")
            elif hasattr(all_chains[0], 'to_file'):
                try:
                    combined_structure = all_chains[0]
                    for chain in all_chains[1:]:
                        if hasattr(combined_structure, 'add_chain'):
                            combined_structure.add_chain(chain)
                        else:
                            raise NotImplementedError(
                                "Need to implement chain combination")
                    combined_structure.to_file(temp_path, format="pdb")
                except (AttributeError, NotImplementedError):
                    temp_files = []
                    for i, chain in enumerate(all_chains):
                        temp_fd, temp_chain_path = tempfile.mkstemp(
                            suffix=f'_chain{i}.pdb')
                        os.close(temp_fd)
                        chain.to_file(temp_chain_path, format="pdb")
                        temp_files.append(temp_chain_path)

                    with open(temp_path, 'w') as outfile:
                        for temp_file in temp_files:
                            with open(temp_file, 'r') as infile:
                                for line in infile:
                                    if line.startswith(('ATOM', 'HETATM', 'TER', 'END')):
                                        outfile.write(line)

                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
            else:
                with open(temp_path, 'w') as f:
                    for chain in all_chains:
                        if hasattr(chain, 'to_pdb_string'):
                            f.write(chain.to_pdb_string())
                        else:
                            raise NotImplementedError(
                                "Cannot write PDB - Structure class missing to_file or to_pdb_string method")

        print(temp_path)
        return temp_path

    def _split_concatenated_sequences(self, concatenated_sequences: List[str],
                                      entity_lengths: List[Tuple[int, int]]) -> Dict[int, List[str]]:
        """Split concatenated sequences back into per-entity sequences."""
        separated_sequences = {entity_idx: []
                               for entity_idx, _ in entity_lengths}

        for concat_seq in concatenated_sequences:
            start_pos = 0
            for entity_idx, length in entity_lengths:
                entity_seq = concat_seq[start_pos:start_pos + length]
                separated_sequences[entity_idx].append(entity_seq)
                start_pos += length

        return separated_sequences

    def _create_chain_mapping(self, system: System) -> Dict[str, str]:
        """Create mapping from PDB chain IDs to entity names/indices."""
        chain_to_entity = {}

        for idx, entity in enumerate(system):
            entity_id = entity.id_ or f'entity_{idx}'

            if entity.structures:
                for structure_key, chains in entity.structures.items():
                    if not isinstance(chains, list):
                        chains = [chains]
                    for chain in chains:
                        if hasattr(chain, 'chain_id'):
                            chain_id = chain.chain_id
                        elif hasattr(chain, 'get_id'):
                            chain_id = chain.get_id()
                        else:
                            chain_id = getattr(chain, 'id', f'chain_{idx}')

                        chain_to_entity[chain_id] = entity_id
                        print(f"Mapped chain {chain_id} to entity {entity_id}")

        return chain_to_entity

    def _determine_homooligomer_symmetry(self, system: System) -> Tuple[List[List[int]], List[List[float]]]:
        """Determine symmetry constraints for homo-oligomers."""
        symmetry_residues = []
        symmetry_weights = []

        for entity in system:
            if entity.structures:
                for structure_key, chains in entity.structures.items():
                    if not isinstance(chains, list):
                        chains = [chains]

                    if len(chains) > 1:  # Homo-oligomer
                        seq_length = len(
                            entity.rep) if entity.rep is not None else 0
                        for pos in range(seq_length):
                            residue_group = []
                            weight_group = []

                            for i, _ in enumerate(chains):
                                residue_group.append(pos + i * seq_length)
                                weight_group.append(1.0 / len(chains))

                            symmetry_residues.append(residue_group)
                            symmetry_weights.append(weight_group)

        return symmetry_residues, symmetry_weights

    def _create_chain_mask(self, fixed_pos: EntityPosList | None) -> torch.Tensor:
        """
        Create chain mask from fixed positions.

        Args:
            fixed_pos: Mapping of entity_idx -> list of fixed positions

        Returns:
            Chain mask tensor (1 = design, 0 = fixed)
        """
        chain_mask = torch.ones_like(
            self.feature_dict["mask"], dtype=torch.float32)

        if fixed_pos is not None:
            # Start position for each entity in the concatenated sequence
            entity_starts = {}
            current_pos = 0
            for idx, (entity_id, length) in enumerate(self.entity_lengths):
                entity_starts[idx] = current_pos
                current_pos += length

            # Set fixed positions to 0
            for entity_idx, positions in fixed_pos.items():
                start_pos = entity_starts[entity_idx]
                for pos in positions:
                    chain_mask[0, start_pos + pos] = 0.0

        return chain_mask

    def _create_bias_tensor(self, amino_acid_bias: Dict[str, float]) -> torch.Tensor:
        """Create bias tensor from amino acid bias dictionary."""
        bias_tensor = torch.zeros(
            [21], device=self.device, dtype=torch.float32)
        for aa, bias in amino_acid_bias.items():
            if aa in restype_str_to_int:
                bias_tensor[restype_str_to_int[aa]] = bias
        return bias_tensor

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 0.1,
        batch_size: int = 1,
        seed: Optional[int] = None,
        amino_acid_bias: Optional[Dict[str, float]] = None,
        omit_amino_acids: Optional[str] = None,
        use_ligand_context: bool = True,
        status_callback: Callable[[str], None] | None = None
    ) -> List[SystemInstance]:
        """
        Generate new sequences for the built structure and optionally score them.

        Args:
            num_designs: Number of designs to generate
            entities: Which entities to design (None = all)
            fixed_pos: Mapping of entity_idx -> list of fixed positions
            temperature: Sampling temperature
            batch_size: Batch size for generation
            seed: Random seed
            amino_acid_bias: Global amino acid biases
            omit_amino_acids: Amino acids to omit globally
            use_ligand_context: Whether to use ligand context
            status_callback: Optional callback for status updates

        Returns:
            List of SystemInstance objects with optional scores
        """
        # 1. Check model is ready
        self.ready_or_raise()

        # 2. Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 3. Validate entity selection
        if entities is not None:
            entities = ensure_sequence(entities)
            # Validate entities exist in system
            max_entity = len(self.system) - 1
            for entity_idx in entities:
                if entity_idx > max_entity:
                    raise ValueError(
                        f"Entity index {entity_idx} out of range (max: {max_entity})")
        else:
            entities = list(range(len(self.system)))

        # 4. Process fixed_pos into chain_mask
        chain_mask = self._create_chain_mask(fixed_pos)

        # 5. Update feature_dict with generation parameters
        feature_dict_copy = self.feature_dict.copy()
        feature_dict_copy["chain_mask"] = chain_mask
        feature_dict_copy["batch_size"] = batch_size
        feature_dict_copy["temperature"] = temperature
        feature_dict_copy["symmetry_residues"] = self.symmetry_residues or [[]]
        feature_dict_copy["symmetry_weights"] = self.symmetry_weights or [[]]

        # 6. Apply amino acid biases (always set bias tensor)
        B, L, _, _ = feature_dict_copy["X"].shape
        if amino_acid_bias:
            bias_tensor = self._create_bias_tensor(amino_acid_bias)
        else:
            bias_tensor = torch.zeros(
                [21], device=self.device, dtype=torch.float32)
        feature_dict_copy["bias"] = bias_tensor[None, None, :].repeat(1, L, 1)

        # 7. Generate sequences using the model
        L = feature_dict_copy["X"].shape[1]
        generated_sequences = []

        with torch.no_grad():
            num_batches = (num_designs + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                if status_callback:
                    status_callback(
                        f"Generating batch {batch_idx + 1}/{num_batches}")

                feature_dict_copy["randn"] = torch.randn(
                    [batch_size, L], device=self.device)
                output_dict = self.model.sample(feature_dict_copy)
                generated_sequences.append(output_dict["S"])

        S_stack = torch.cat(generated_sequences, 0)[:num_designs]

        # 8. Convert to sequences and split by entity
        concatenated_sequences = [
            "".join([restype_int_to_str[aa]
                    for aa in S_stack[i].cpu().numpy()])
            for i in range(S_stack.shape[0])
        ]

        separated_sequences = self._split_concatenated_sequences(
            concatenated_sequences, self.entity_lengths
        )

        # 9. Create SystemInstance objects
        system_instances = []
        for design_idx in range(num_designs):
            entity_instances = []

            # Calculate individual entity recoveries
            for entity_idx, (entity_id, length) in enumerate(self.entity_lengths):
                generated_seq = separated_sequences[entity_idx][design_idx]
                native_seq = self.native_seq[sum(length for (_, length) in self.entity_lengths[:entity_idx]):sum(
                    length for (_, length) in self.entity_lengths[:entity_idx+1])]

                # Calculate recovery
                matches = sum(1 for a, b in zip(
                    native_seq, generated_seq) if a == b)
                recovery = matches / len(native_seq)

                # Create EntityInstance
                # Ensure rep is a string, not a numpy array
                entity_instance = EntityInstance(
                    rep=''.join(generated_seq) if hasattr(
                        generated_seq, 'tolist') else generated_seq,
                    models=self.system[entity_idx].structures
                )

                entity_instances.append(entity_instance)

            # Calculate overall recovery
            overall_matches = sum(1 for a, b in zip(self.native_seq, "".join(
                str(inst.rep) for inst in entity_instances)) if a == b)
            overall_recovery = overall_matches / len(self.native_seq)

            # Create SystemInstance
            system_instance = SystemInstance(
                entity_instances=entity_instances,
                score=None,
                confidence=None,
                metadata={
                    'design_id': design_idx,
                    'overall_recovery': overall_recovery,
                    'log_probability': None,
                    'sampling_probabilities': None
                }
            )

            system_instances.append(system_instance)

        # 10. Score the generated instances
        scores = self.score(system_instances, status_callback=status_callback)

        # 11. Attach scores to instances
        for instance, raw_score in zip(system_instances, scores):
            instance.score = raw_score
            instance.metadata['log_probability'] = raw_score

        return system_instances

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: Callable[[str], None] | None = None
    ) -> np.ndarray:
        """
        Score sequences against the built structure.

        Args:
            instances: Sequence of SystemInstance objects to score
            status_callback: Optional callback for status updates

        Returns:
            Numpy array of scores (log probabilities)
        """
        # 1. Check model is ready
        self.ready_or_raise()

        # 2. Extract sequences from instances
        sequences = []
        for instance in instances:
            # Ensure each instance's rep is converted to a string
            concat_seq = "".join([
                ''.join(inst.rep) if hasattr(
                    inst.rep, 'tolist') else str(inst.rep)
                for inst in instance
            ])
            sequences.append(concat_seq)

        # 3. Score each sequence
        scores = []
        with torch.no_grad():
            for seq_idx, seq in enumerate(sequences):
                if status_callback:
                    status_callback(
                        f"Scoring sequence {seq_idx + 1}/{len(sequences)}")

                # Convert sequence to tensor
                S_tensor = torch.tensor(
                    [restype_str_to_int.get(aa, 20) for aa in seq],
                    device=self.device,
                    dtype=torch.int64
                )[None, :]

                # Create feature dict for this sequence
                feature_dict_copy = self.feature_dict.copy()
                feature_dict_copy["S"] = S_tensor
                feature_dict_copy["batch_size"] = 1
                feature_dict_copy["randn"] = torch.randn(
                    [1, len(seq)], device=self.device)
                feature_dict_copy["symmetry_residues"] = [[]]
                feature_dict_copy["symmetry_weights"] = [[]]

                # Score the sequence
                output_dict = self.model.score(
                    feature_dict_copy, use_sequence=True)

                # Calculate loss (negative log probability)
                loss, _ = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    self.feature_dict["mask"][:1]
                )

                # Convert to positive log likelihood
                scores.append(-loss.item())

        # 4. Return as numpy array
        return np.array(scores)

    def __del__(self):
        """Cleanup temporary files"""
        if hasattr(self, 'pdb_path') and self.pdb_path and os.path.exists(self.pdb_path):
            os.unlink(self.pdb_path)
