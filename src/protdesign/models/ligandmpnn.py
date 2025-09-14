import numpy as np
import torch
import tempfile
import os
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class GenerationConfig:
    """Configuration for sequence generation"""
    temperature: float = 0.1
    batch_size: int = 1
    number_of_batches: int = 1
    seed: Optional[int] = None
    # List of position IDs like "A10", "B25"
    fix_positions: Optional[List[str]] = None
    # Mutually exclusive with fix_positions
    redesign_positions: Optional[List[str]] = None
    # Global AA bias like {"A": 1.0, "P": -2.0}
    amino_acid_bias: Optional[Dict[str, float]] = None
    # Per-position bias
    bias_per_position: Optional[Dict[str, Dict[str, float]]] = None
    omit_amino_acids: Optional[str] = None  # String of AAs to omit globally
    omit_per_position: Optional[Dict[str, str]
                                ] = None  # Per-position AA omission
    use_ligand_context: bool = True
    ligand_cutoff: float = 8.0


class EntityInstance:
    """Represents a specific sequence instance of an entity"""

    def __init__(self, entity_id: str, sequence: str, native_sequence: str = None, recovery: float = None):
        self.entity_id = entity_id
        self.sequence = sequence
        self.native_sequence = native_sequence
        self.recovery = recovery

    def __repr__(self):
        return f"EntityInstance(entity_id='{self.entity_id}', sequence='{self.sequence[:20]}...', recovery={self.recovery:.3f})"


class SystemInstance:
    """Represents a complete system design with all entity instances"""

    def __init__(self, entity_instances: List[EntityInstance], design_id: int = None,
                 overall_recovery: float = None, log_probability: float = None,
                 sampling_probabilities: np.ndarray = None):
        self.entity_instances = entity_instances
        self.design_id = design_id
        self.overall_recovery = overall_recovery
        self.log_probability = log_probability
        self.sampling_probabilities = sampling_probabilities

    def __repr__(self):
        return f"SystemInstance(design_id={self.design_id}, entities={len(self.entity_instances)}, recovery={self.overall_recovery:.3f})"

    def get_entity(self, entity_id: str) -> Optional[EntityInstance]:
        """Get entity instance by ID"""
        for entity in self.entity_instances:
            if entity.entity_id == entity_id:
                return entity
        return None

    def get_concatenated_sequence(self) -> str:
        """Get the full concatenated sequence"""
        return "".join([entity.sequence for entity in self.entity_instances])

    def get_sequences_dict(self) -> Dict[str, str]:
        """Get sequences as a dictionary mapping entity_id -> sequence"""
        return {entity.entity_id: entity.sequence for entity in self.entity_instances}


class LigandMPNNWrapper:
    """
    Wrapper for LigandMPNN that works with System objects from the notebook.

    Handles conversion between the System representation and LigandMPNN's expected input format.
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

    def _system_to_pdb_file(self, system) -> str:
        """
        Convert a System object to a temporary PDB file.

        Args:
            system: System object with protein entities and structures

        Returns:
            Path to temporary PDB file
        """
        # Create temporary PDB file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(temp_fd)

        # For now, assume we're working with the first structure in the system
        # This could be extended to handle multi-state design
        structure_keys = set()
        for entity in system:
            structure_keys.update(entity.structures.keys())

        if len(structure_keys) > 1:
            raise NotImplementedError(
                "Multi-state design not currently supported")

        structure_key = list(structure_keys)[0]

        # Collect all chains from all entities for this structure
        all_chains = []
        for entity in system:
            if structure_key in entity.structures:
                all_chains.extend(entity.structures[structure_key])
                entity_id = getattr(entity, 'id', f'entity_{len(all_chains)}')
                print(
                    f"Adding chains from entity {entity_id}: {len(entity.structures[structure_key])} chains")

        print(f"Total chains to write: {len(all_chains)}")

        # Write combined structure to PDB file
        if all_chains:
            # Check if we can write all chains together
            if hasattr(all_chains[0], 'to_file') and len(all_chains) == 1:
                # Single chain case
                all_chains[0].to_file(temp_path, format="pdb")
            elif hasattr(all_chains[0], 'to_file'):
                # Multiple chains - need to combine them
                # Option 1: Try to create a combined structure
                try:
                    # If your Structure class supports combining chains
                    # Start with first chain
                    combined_structure = all_chains[0]
                    for chain in all_chains[1:]:
                        # This depends on your Structure class implementation
                        # You might need to use a different method to combine chains
                        if hasattr(combined_structure, 'add_chain'):
                            combined_structure.add_chain(chain)
                        else:
                            # Fallback: write each chain separately and combine files
                            raise NotImplementedError(
                                "Need to implement chain combination")

                    combined_structure.to_file(temp_path, format="pdb")

                except (AttributeError, NotImplementedError):
                    # Option 2: Write each chain to separate temp files and combine
                    temp_files = []
                    for i, chain in enumerate(all_chains):
                        temp_fd, temp_chain_path = tempfile.mkstemp(
                            suffix=f'_chain{i}.pdb')
                        os.close(temp_fd)
                        chain.to_file(temp_chain_path, format="pdb")
                        temp_files.append(temp_chain_path)

                    # Combine all temp files into final PDB
                    with open(temp_path, 'w') as outfile:
                        for temp_file in temp_files:
                            with open(temp_file, 'r') as infile:
                                # Skip header lines and only keep ATOM/HETATM records
                                for line in infile:
                                    if line.startswith(('ATOM', 'HETATM', 'TER', 'END')):
                                        outfile.write(line)

                    # Clean up temp files
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)

            else:
                # Fallback to string-based approach
                with open(temp_path, 'w') as f:
                    for chain in all_chains:
                        if hasattr(chain, 'to_pdb_string'):
                            f.write(chain.to_pdb_string())
                        else:
                            raise NotImplementedError(
                                "Cannot write PDB - Structure class missing to_file or to_pdb_string method")
        print(temp_path)
        return temp_path

    def _get_entity_sequence_lengths(self, system) -> List[Tuple[str, int]]:
        """
        Get the sequence length for each entity in order.

        Args:
            system: System object

        Returns:
            List of (entity_id, sequence_length) tuples in order
        """
        entity_lengths = []

        for idx, entity in enumerate(system):
            entity_id = getattr(entity, 'id', f'entity_{idx}')
            seq_length = len(entity.rep)
            entity_lengths.append((entity_id, seq_length))
            print(f"Entity {entity_id}: {seq_length} residues")

        return entity_lengths

    def _split_concatenated_sequences(self, concatenated_sequences: List[str], entity_lengths: List[Tuple[str, int]]) -> Dict[str, List[str]]:
        """
        Split concatenated sequences back into per-entity sequences.

        Args:
            concatenated_sequences: List of full-length sequences
            entity_lengths: List of (entity_id, length) tuples

        Returns:
            Dictionary mapping entity_id to list of sequences for that entity
        """
        separated_sequences = {entity_id: []
                               for entity_id, _ in entity_lengths}

        for concat_seq in concatenated_sequences:
            start_pos = 0
            for entity_id, length in entity_lengths:
                entity_seq = concat_seq[start_pos:start_pos + length]
                separated_sequences[entity_id].append(entity_seq)
                start_pos += length

        return separated_sequences

    def _create_chain_mapping(self, system) -> Dict[str, str]:
        """
        Create mapping from PDB chain IDs to entity names/indices.

        Args:
            system: System object

        Returns:
            Dictionary mapping chain_id -> entity_identifier
        """
        chain_to_entity = {}

        for idx, entity in enumerate(system):
            # Use index as entity identifier if no id attribute
            entity_id = getattr(entity, 'id', f'entity_{idx}')

            for structure_key, chains in entity.structures.items():
                for chain in chains:
                    # Extract chain ID from chain object
                    if hasattr(chain, 'chain_id'):
                        chain_id = chain.chain_id
                    elif hasattr(chain, 'get_id'):
                        chain_id = chain.get_id()
                    else:
                        # Try to get chain ID from the chain object's attributes
                        # This might vary depending on your Structure implementation
                        chain_id = getattr(chain, 'id', f'chain_{idx}')

                    chain_to_entity[chain_id] = entity_id
                    print(f"Mapped chain {chain_id} to entity {entity_id}")

        return chain_to_entity

    def _determine_homooligomer_symmetry(self, system) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Determine symmetry constraints for homo-oligomers.

        Args:
            system: System object

        Returns:
            Tuple of (symmetry_residues, symmetry_weights)
        """
        symmetry_residues = []
        symmetry_weights = []

        for entity in system:
            for structure_key, chains in entity.structures.items():
                if len(chains) > 1:  # Homo-oligomer
                    # Create symmetry groups for each position
                    seq_length = len(entity.rep)
                    for pos in range(seq_length):
                        residue_group = []
                        weight_group = []

                        for i, chain in enumerate(chains):
                            # Create position identifier
                            residue_group.append(pos + i * seq_length)
                            weight_group.append(1.0 / len(chains))

                        symmetry_residues.append(residue_group)
                        symmetry_weights.append(weight_group)

        return symmetry_residues, symmetry_weights

    def _prepare_constraints(self, system, config: GenerationConfig, chain_mapping: Dict[str, str]) -> Dict:
        """
        Prepare position constraints and biases.

        Args:
            system: System object
            config: Generation configuration
            chain_mapping: Mapping from chain IDs to entity IDs

        Returns:
            Dictionary with constraint information
        """
        constraints = {
            'fixed_residues': [],
            'redesigned_residues': [],
            'bias_per_residue': {},
            'omit_per_residue': {},
        }

        # Handle position fixing/redesigning
        if config.fix_positions:
            constraints['fixed_residues'] = config.fix_positions
        elif config.redesign_positions:
            constraints['redesigned_residues'] = config.redesign_positions

        # Handle per-position biases
        if config.bias_per_position:
            constraints['bias_per_residue'] = config.bias_per_position

        # Handle per-position omissions
        if config.omit_per_position:
            constraints['omit_per_residue'] = config.omit_per_position

        return constraints

    def _create_system_instances(self, separated_sequences: Dict[str, List[str]],
                                 native_separated: Dict[str, str],
                                 recoveries_by_entity: Dict[str, List[float]],
                                 overall_recoveries: List[float],
                                 probs_stack: torch.Tensor,
                                 log_probs_stack: torch.Tensor,
                                 entity_lengths: List[Tuple[str, int]]) -> List[SystemInstance]:
        """
        Create SystemInstance objects from the generated sequences.

        Args:
            separated_sequences: Dictionary mapping entity_id to list of sequences
            native_separated: Dictionary mapping entity_id to native sequence
            recoveries_by_entity: Dictionary mapping entity_id to list of recoveries
            overall_recoveries: List of overall sequence recoveries
            probs_stack: Tensor of sampling probabilities
            log_probs_stack: Tensor of log probabilities
            entity_lengths: List of (entity_id, length) tuples

        Returns:
            List of SystemInstance objects
        """
        system_instances = []

        # Determine number of designs
        first_entity_id = list(separated_sequences.keys())[0]
        num_designs = len(separated_sequences[first_entity_id])

        for design_idx in range(num_designs):
            entity_instances = []

            # Create EntityInstance for each entity in this design
            for entity_id, _ in entity_lengths:
                entity_sequence = separated_sequences[entity_id][design_idx]
                native_sequence = native_separated[entity_id]
                recovery = recoveries_by_entity[entity_id][design_idx]

                entity_instance = EntityInstance(
                    entity_id=entity_id,
                    sequence=entity_sequence,
                    native_sequence=native_sequence,
                    recovery=recovery
                )
                entity_instances.append(entity_instance)

            # Create SystemInstance
            system_instance = SystemInstance(
                entity_instances=entity_instances,
                design_id=design_idx,
                overall_recovery=overall_recoveries[design_idx],
                log_probability=log_probs_stack[design_idx].sum().item(),
                sampling_probabilities=probs_stack[design_idx].cpu().numpy()
            )

            system_instances.append(system_instance)

        return system_instances

    def generate(self, system, config: GenerationConfig = None) -> List[SystemInstance]:
        """
        Generate protein sequences for the given system.

        Args:
            system: System object containing protein entities and structures
            config: Generation configuration

        Returns:
            List of SystemInstance objects, each containing EntityInstance objects for all entities
        """
        if config is None:
            config = GenerationConfig()

        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Get entity sequence lengths for splitting later
        entity_lengths = self._get_entity_sequence_lengths(system)

        # Convert system to PDB file
        pdb_path = self._system_to_pdb_file(system)

        try:
            # Parse PDB with LigandMPNN
            protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                pdb_path,
                device=self.device,
                chains=[],  # Parse all chains
                parse_all_atoms=True,
                parse_atoms_with_zero_occupancy=False,
            )

            # Create mappings
            chain_mapping = self._create_chain_mapping(system)

            # Determine homo-oligomer symmetry if applicable
            symmetry_residues, symmetry_weights = self._determine_homooligomer_symmetry(
                system)

            # Prepare constraints
            constraints = self._prepare_constraints(
                system, config, chain_mapping)

            # Set up chain mask (which residues to design)
            chain_mask = torch.ones_like(
                protein_dict["mask"], dtype=torch.float32)

            # Apply position constraints
            if constraints['fixed_residues'] or constraints['redesigned_residues']:
                # This would need to be implemented based on the residue naming scheme
                # For now, assume all positions are designable
                pass

            protein_dict["chain_mask"] = chain_mask

            # Featurize the protein
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=config.ligand_cutoff,
                use_atom_context=config.use_ligand_context,
                number_of_ligand_atoms=getattr(
                    self.model, 'atom_context_num', 25),
                model_type=self.model_type,
            )

            # Set up generation parameters
            feature_dict["batch_size"] = config.batch_size
            feature_dict["temperature"] = config.temperature
            feature_dict["symmetry_residues"] = symmetry_residues or [[]]
            feature_dict["symmetry_weights"] = symmetry_weights or [[]]

            # Set up amino acid biases
            bias_tensor = torch.zeros(
                [21], device=self.device, dtype=torch.float32)
            if config.amino_acid_bias:
                for aa, bias in config.amino_acid_bias.items():
                    if aa in restype_str_to_int:
                        bias_tensor[restype_str_to_int[aa]] = bias

            B, L, _, _ = feature_dict["X"].shape
            feature_dict["bias"] = bias_tensor[None, None, :].repeat(1, L, 1)

            # Generate sequences
            generated_sequences = []
            sampling_probs = []
            log_probs = []

            with torch.no_grad():
                for batch_idx in range(config.number_of_batches):
                    # Add random noise for decoding order
                    feature_dict["randn"] = torch.randn(
                        [config.batch_size, L], device=self.device
                    )

                    # Sample sequences
                    output_dict = self.model.sample(feature_dict)

                    generated_sequences.append(output_dict["S"])
                    sampling_probs.append(output_dict["sampling_probs"])
                    log_probs.append(output_dict["log_probs"])

            # Combine results
            S_stack = torch.cat(generated_sequences, 0)
            probs_stack = torch.cat(sampling_probs, 0)
            log_probs_stack = torch.cat(log_probs, 0)

            # Convert to concatenated sequences
            concatenated_sequences = []
            for i in range(S_stack.shape[0]):
                seq = "".join([restype_int_to_str[aa]
                              for aa in S_stack[i].cpu().numpy()])
                concatenated_sequences.append(seq)

            # Split sequences by entity
            separated_sequences = self._split_concatenated_sequences(
                concatenated_sequences, entity_lengths)

            # Calculate sequence recovery for concatenated sequence
            native_seq = "".join([restype_int_to_str[aa]
                                 for aa in feature_dict["S"][0].cpu().numpy()])

            # Split native sequence by entity too
            native_separated = self._split_concatenated_sequences(
                [native_seq], entity_lengths)
            native_by_entity = {
                entity_id: seqs[0] for entity_id, seqs in native_separated.items()}

            # Calculate recovery per entity
            recoveries_by_entity = {}
            for entity_id in separated_sequences:
                entity_recoveries = []
                native_entity_seq = native_by_entity[entity_id]

                for generated_entity_seq in separated_sequences[entity_id]:
                    # Calculate simple sequence identity
                    matches = sum(1 for a, b in zip(
                        native_entity_seq, generated_entity_seq) if a == b)
                    recovery = matches / len(native_entity_seq)
                    entity_recoveries.append(recovery)

                recoveries_by_entity[entity_id] = entity_recoveries

            # Overall recoveries for concatenated sequences
            overall_recoveries = []
            for i in range(S_stack.shape[0]):
                recovery = get_seq_rec(
                    feature_dict["S"][:1],
                    S_stack[i:i+1],
                    feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
                )
                overall_recoveries.append(recovery.item())

            # Create SystemInstance objects
            system_instances = self._create_system_instances(
                separated_sequences=separated_sequences,
                native_separated=native_by_entity,
                recoveries_by_entity=recoveries_by_entity,
                overall_recoveries=overall_recoveries,
                probs_stack=probs_stack,
                log_probs_stack=log_probs_stack,
                entity_lengths=entity_lengths
            )

            return system_instances

        finally:
            # Clean up temporary file
            if os.path.exists(pdb_path):
                os.unlink(pdb_path)

    def score_sequences(self, system, sequences: List[str], config: GenerationConfig = None) -> Dict:
        """
        Score given sequences against the system structure.

        Args:
            system: System object containing protein entities and structures
            sequences: List of sequences to score
            config: Generation configuration

        Returns:
            Dictionary containing scores and probabilities
        """
        if config is None:
            config = GenerationConfig()

        # Convert system to PDB file
        pdb_path = self._system_to_pdb_file(system)

        try:
            # Parse PDB
            protein_dict, _, _, _, _ = parse_PDB(
                pdb_path,
                device=self.device,
                chains=[],
                parse_all_atoms=True,
                parse_atoms_with_zero_occupancy=False,
            )

            # Add chain mask before featurizing
            chain_mask = torch.ones_like(
                protein_dict["mask"], dtype=torch.float32)
            protein_dict["chain_mask"] = chain_mask

            # Featurize
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=config.ligand_cutoff,
                use_atom_context=config.use_ligand_context,
                number_of_ligand_atoms=getattr(
                    self.model, 'atom_context_num', 25),
                model_type=self.model_type,
            )

            scores = []
            log_probs_list = []

            with torch.no_data():
                for seq in sequences:
                    # Convert sequence to tensor
                    S_tensor = torch.tensor(
                        [restype_str_to_int.get(aa, 20) for aa in seq],
                        device=self.device,
                        dtype=torch.int64
                    )[None, :]  # Add batch dimension

                    # Update feature dict with sequence
                    feature_dict_copy = feature_dict.copy()
                    feature_dict_copy["S"] = S_tensor
                    feature_dict_copy["batch_size"] = 1
                    feature_dict_copy["randn"] = torch.randn(
                        [1, len(seq)], device=self.device)
                    feature_dict_copy["symmetry_residues"] = [[]]

                    # Score the sequence
                    output_dict = self.model.score(
                        feature_dict_copy, use_sequence=True)

                    # Calculate loss (negative log probability)
                    loss, _ = get_score(
                        output_dict["S"],
                        output_dict["log_probs"],
                        feature_dict["mask"][:1]
                    )

                    # Convert to positive log likelihood
                    scores.append(-loss.item())
                    log_probs_list.append(
                        output_dict["log_probs"].cpu().numpy())

            return {
                "sequences": sequences,
                "scores": scores,
                "log_probabilities": log_probs_list,
                "system": system,
            }

        finally:
            # Clean up temporary file
            if os.path.exists(pdb_path):
                os.unlink(pdb_path)
