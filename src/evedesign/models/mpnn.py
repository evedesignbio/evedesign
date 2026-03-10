from io import StringIO
import os
from tempfile import NamedTemporaryFile
from typing import Sequence, Self, Literal
import urllib.request

import numpy as np
import torch
from loguru import logger

from evedesign.model import (
    BaseModel, Scorer, Generator, MutationScorer, ConditionalMutationScorer
)
from evedesign.system import System, SystemInstance
from evedesign.structure import Structure
from evedesign.utils import ensure_sequence, model_param_context
from evedesign.types import DeviceType, StatusCallback, BatchSize, EntityPosList

# Import the LigandMPNN modules
from evedesign.models.ligandmpnn.data_utils import (
    featurize,
    parse_PDB,
    restype_str_to_int,
    restype_int_to_str,
    get_score,
)
from evedesign.models.ligandmpnn.model_utils import ProteinMPNN
try:
    import prody
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

MODEL_BASE_URL = "https://files.ipd.uw.edu/pub/ligandmpnn"
DEFAULT_CHECKPOINT_PATH = os.path.expanduser("~/.cache/mpnn")

# Model checkpoint URLs
MODEL_URLS = {
    # Original ProteinMPNN weights
    "proteinmpnn_v_48_002": f"{MODEL_BASE_URL}/proteinmpnn_v_48_002.pt",
    "proteinmpnn_v_48_010": f"{MODEL_BASE_URL}/proteinmpnn_v_48_010.pt",
    "proteinmpnn_v_48_020": f"{MODEL_BASE_URL}/proteinmpnn_v_48_020.pt",
    "proteinmpnn_v_48_030": f"{MODEL_BASE_URL}/proteinmpnn_v_48_030.pt",
    # LigandMPNN with num_edges=32; atom_context_num=25
    "ligandmpnn_v_32_005_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_030_25.pt",
    # Per residue label membrane ProteinMPNN
    "per_residue_label_membrane_mpnn_v_48_020": f"{MODEL_BASE_URL}/per_residue_label_membrane_mpnn_v_48_020.pt",
    # Global label membrane ProteinMPNN
    "global_label_membrane_mpnn_v_48_020": f"{MODEL_BASE_URL}/global_label_membrane_mpnn_v_48_020.pt",
    # SolubleMPNN
    "solublempnn_v_48_002": f"{MODEL_BASE_URL}/solublempnn_v_48_002.pt",
    "solublempnn_v_48_010": f"{MODEL_BASE_URL}/solublempnn_v_48_010.pt",
    "solublempnn_v_48_020": f"{MODEL_BASE_URL}/solublempnn_v_48_020.pt",
    "solublempnn_v_48_030": f"{MODEL_BASE_URL}/solublempnn_v_48_030.pt",
    # LigandMPNN for side-chain packing (multi-step denoising model)
    "ligandmpnn_sc_v_32_002_16": f"{MODEL_BASE_URL}/ligandmpnn_sc_v_32_002_16.pt",
}

def download_checkpoint(model_name: str, save_dir: str) -> str:
    """
    Download model checkpoint from URL if not already present.

    Args:
        model_name: Name of the model to download
        save_dir: Directory to save the checkpoint

    Returns:
        Path to the downloaded checkpoint

    Raises:
        ValueError: If model_name is not recognized
        RuntimeError: If download fails
    """
    if model_name not in MODEL_URLS:
        available_models = ", ".join(MODEL_URLS.keys())
        raise ValueError(
            f"Model '{model_name}' not recognized. Available models: {available_models}"
        )

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Construct file path
    checkpoint_path = os.path.join(save_dir, f"{model_name}.pt")

    # Download if not already present
    if not os.path.exists(checkpoint_path):
        url = MODEL_URLS[model_name]
        logger.info(f"Downloading {model_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"Successfully downloaded to {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name}: {str(e)}")
    else:
        logger.info(f"Using cached checkpoint at {checkpoint_path}")

    return checkpoint_path


class LigandMPNN(BaseModel, Scorer, Generator, MutationScorer, ConditionalMutationScorer):
    """
    evedesign wrapper for LigandMPNN/ProteinMPNN

    TODO: extend to also handle ligand entities
    TODO: implement specialized scoring methods that move known positions to front to score all substitutions at once
     (currently handled with standard mixins, which are not the most efficient solution for this particular model)
    """
    available = IMPORT_AVAILABLE
    name: str = "LigandMPNN"
    citations: list[str] = ["doi: 10.1038/s41592-025-02626-1"]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["structures"]
    optional_entity_attributes: list[str] | None = ["residue_bias"]

    def __init__(
        self,
        model_name: Literal[tuple(MODEL_URLS.keys())],  # noqa
        model_file_path: str | os.PathLike | None = None,
        batch_size: BatchSize = 1,
        use_ligand_context: bool = True,
        ligand_cutoff: float = 6.0,
        fix_full_decoding_order: bool = False,
        vary_decoding_order_per_instance: bool = False,
        keep_model_after_build: bool = False,
        cache_dir: str | None = DEFAULT_CHECKPOINT_PATH,
        device: DeviceType = "cpu"
    ):
        """
        Initialize the LigandMPNN wrapper

        Parameters
        ----------
        model_name
            Name of MPNN model. If checkpoint_path is specified, must match the loaded model.
        model_file_path
            Path to checkpoint file to load. If None, will attempt to download from web.
        batch_size
            Batch sized used for generation. Will not be used while scoring due to implementation limitations
            inside original MPNN code.
        use_ligand_context
            If True, ligand atoms will be included during calculations
        keep_model_after_build
            If True, keep model parameters asssociated to instance after build step
            to avoid reloading when scoring/generating. If serializing model, set to
            False to avoid storing model parameters repeatedly.
        fix_full_decoding_order
            If True, fix decoding order across calls to score()
        vary_decoding_order_per_instance:
            if True, will use different decoding order *per instance* when calling score()
            (will only have an effect if fix_full_decoding_order is False)
        ligand_cutoff
            Cutoff distance in angstroms to select residues that are considered to be close to ligand atoms
        cache_dir
            Directory to use for storing downloaded model parameters (only relevant if checkpoint_path is None)
        device
            Device to use for computations
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_ligand_context = use_ligand_context
        self.keep_model_after_build = keep_model_after_build
        self.ligand_cutoff = ligand_cutoff
        self.fix_full_decoding_order = fix_full_decoding_order
        self.vary_decoding_order_per_instance = vary_decoding_order_per_instance

        # Determine model type from model_name
        if "ligand" in model_name.lower():
            self.model_type = "ligand_mpnn"
        else:
            self.model_type = "protein_mpnn"

       # Handle checkpoint path
        if model_file_path is None:
            # Download from web using model_name
            self.checkpoint_path = download_checkpoint(model_name, cache_dir)
        else:
            self.checkpoint_path = model_file_path

        self.model = None

        # State that gets set during build()
        self._system = None
        self._feature_dict = None
        self._entity_lengths = None
        self._symmetry_residues = None
        self._symmetry_weights = None
        self._native_seq = None
        self._pdb = None
        self._pdb_to_entity_mapping = None  # Map PDB positions to entity positions
        self._entity_to_pdb_chains = None  # Map entity_idx to list of PDB chain IDs
        self._entity_pos_to_pdb_mapping = None
        self._randn = None

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

        # Check that all entities are proteins with structures
        for entity in system:
            if entity.type != "protein":
                return False, "Can only handle protein entities"
            if not entity.defined_sequence():
                return False, "Entity must have defined rep sequence"
            if not entity.structures or len(entity.structures) == 0:
                return False, "All entities must have 3D structures"

        return True, ""

    def _load_model(self):
        """
        Load the model from checkpoint
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

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
        self.can_model_or_raise(system, data)
        self._system = system

        # Get entity sequence lengths
        self._entity_lengths = [
            (idx, len(entity.rep) if entity.rep is not None else 0)
            for idx, entity in enumerate(system)
        ]

        # Convert system to PDB file and build mappings simultaneously
        self._pdb, self._pdb_to_entity_mapping, self._entity_to_pdb_chains, self._entity_pos_to_pdb_mapping = (
            self._system_to_pdb_file(system)
        )

        # Parse PDB with LigandMPNN from temporary file
        # TODO: cleaner solution would replace prody-based parsing and use our structure model directly
        with NamedTemporaryFile(mode="w", suffix=".pdb", delete=True) as f:
            f.write(self._pdb)
            f.flush()

            protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                f.name,
                device=self.device,
                chains=[],
                parse_all_atoms=True,
                parse_atoms_with_zero_occupancy=False,
            )

        # Tie positions on homomultimer chains with equal weights
        self._symmetry_residues = [
            pdb_pos_list for pdb_pos_list in self._entity_pos_to_pdb_mapping.values() if len(pdb_pos_list) > 1
        ]
        self._symmetry_weights = [
            [1.0 / len(pdb_pos_list)] * len(pdb_pos_list) for pdb_pos_list in self._symmetry_residues
        ]

        # Set up chain mask (which residues to design)
        chain_mask = torch.ones_like(protein_dict["mask"], dtype=torch.float32)
        protein_dict["chain_mask"] = chain_mask

        # Featurize the protein
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            self._feature_dict = featurize(
                protein_dict,
                cutoff_for_score=self.ligand_cutoff,
                use_atom_context=self.use_ligand_context,
                number_of_ligand_atoms=getattr(self.model, 'atom_context_num', 25),
                model_type=self.model_type,
            )

        # Store native sequence
        self._native_seq = "".join([
            restype_int_to_str[aa] for aa in self._feature_dict["S"][0].cpu().numpy()
        ])

        if self.fix_full_decoding_order:
            # keep on CPU for easy serialization
            self._randn = torch.randn(
                [1, len(self._native_seq)], device="cpu"
            ).numpy()
        else:
            self._randn = None

        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        self.ready_or_raise()

        # only enumerate positions that are covered by structures;
        # given that this is a fixed-length model we can ignore the optional instance supplied as parameter
        positions = sorted([
            (entity, pos)
            for entity, pos in set(self._pdb_to_entity_mapping.values())
            if self._system[entity].type == "protein"
        ])

        return positions

    @staticmethod
    def _system_to_pdb_file(system: System) -> tuple[str, dict, dict, dict]:
        """
        Convert a System object to a temporary PDB file.
        """
        # Track which entity each chain belongs to
        entity_to_pdb_chains = {i: [] for i in range(len(system))}
        pdb_to_entity_mapping = {}

        # Use single letter chain IDs: A, B, C, ..., Z, AA, AB, etc.
        def get_chain_id(chain_num: int) -> str:
            """Generate chain ID: A-Z, then AA, AB, AC, ..."""
            if chain_num < 26:
                return chr(65 + chain_num)  # A-Z
            else:
                # AA, AB, AC, ...
                first = chr(65 + (chain_num - 26) // 26)
                second = chr(65 + (chain_num - 26) % 26)
                return first + second

        chain_counter = 0
        current_pdb_pos = 0
        models_to_concat = []

        for entity_idx, entity in enumerate(system):
            # make sure check from can_model() holds
            assert entity.structures is not None and len(entity.structures) > 0
            structure_key = list(entity.structures.keys())[0]

            if len(entity.structures) > 1:
                logger.warning(
                    f"More than one structure key on entity {entity_idx}, defaulting to first: {structure_key}"
                )

            entity_chains = ensure_sequence(
                entity.structures[structure_key]
            )

            for chain_obj in entity_chains:
                # Perform deep copy
                model_copy = chain_obj.copy()

                # Assign new chain ID
                new_chain_id = get_chain_id(chain_counter)
                entity_to_pdb_chains[entity_idx].append(new_chain_id)

                # Modify chain_id directly in the Model's atom array
                model_copy.atom_array.chain_id[:] = new_chain_id

                # Build position mapping using residue table;
                # res_id on structure chains by definition is position in entity including first_index
                for entity_pos in list(model_copy.res_df().res_id):
                    pdb_to_entity_mapping[current_pdb_pos] = (
                        entity_idx, entity_pos
                    )
                    current_pdb_pos += 1

                models_to_concat.append(model_copy)
                chain_counter += 1

        # invert pdb_to_entity_mapping
        entity_pos_to_pdb_mapping = {}
        for pdb_pos, (entity_idx, entity_pos) in pdb_to_entity_mapping.items():
            entity_pos_to_pdb_mapping[(entity_idx, entity_pos)] = (
                entity_pos_to_pdb_mapping.get((entity_idx, entity_pos), []) + [pdb_pos]
            )

        # write concatenated model to PDB format (do not write to temporary file to allow model
        # to be serialized after build())
        pdb_string = StringIO()
        Structure.concat(models_to_concat).to_file(pdb_string, format="pdb")
        return pdb_string.getvalue(), pdb_to_entity_mapping, entity_to_pdb_chains, entity_pos_to_pdb_mapping

    def _create_chain_mask(self, fixed_pos: EntityPosList | None, entities: Sequence[int]) -> torch.Tensor:
        """
        Create chain mask from fixed positions.
        """
        chain_mask = torch.ones_like(
            self._feature_dict["mask"], dtype=torch.float32
        )

        # set fixed positions or entities that are not designed to 0 in pos_mask
        for pdb_pos, (entity_idx, entity_pos) in self._pdb_to_entity_mapping.items():
            if entity_idx not in entities or (
                    fixed_pos is not None and entity_idx in fixed_pos and entity_pos in fixed_pos.get(entity_idx, [])
            ):
                chain_mask[0, pdb_pos] = 0.0

        return chain_mask

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 0.1,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        self.ready_or_raise()

        # validate entity selection
        protein_entities = [
            entity_idx for entity_idx, _ in enumerate(self.system) if self.system[entity_idx].type == "protein"
        ]

        if entities is not None:
            entities = ensure_sequence(entities)
            invalid_entities = set(entities).difference(protein_entities)
            if len(invalid_entities) > 0:
                raise ValueError(
                    f"Invalid entities: {invalid_entities}"
                )
        else:
            entities = protein_entities

        if fixed_pos is not None:
            for entity_idx, pos_list in fixed_pos.items():
                self.valid_positions(pos_list, entities=entity_idx, raise_invalid=True)

        # process fixed_pos and designable entities into chain_mask
        chain_mask = self._create_chain_mask(fixed_pos, entities)
        if chain_mask.sum().item() <= 0:
            raise ValueError("No positions left to design after removing fixed positions")

        # update feature_dict with generation parameters
        feature_dict_copy = self._feature_dict.copy()
        feature_dict_copy["chain_mask"] = chain_mask
        feature_dict_copy["batch_size"] = self.batch_size
        feature_dict_copy["temperature"] = temperature
        feature_dict_copy["symmetry_residues"] = self._symmetry_residues or [[]]
        feature_dict_copy["symmetry_weights"] = self._symmetry_weights or [[]]

        # apply amino acid biases (always set bias tensor)
        B, L, _, _ = feature_dict_copy["X"].shape  # noqa

        bias_tensor = torch.zeros(
            [L, 21], device=self.device, dtype=torch.float32
        )
        for entity_idx, entity in enumerate(self.system):
            expanded_bias = entity.expand_residue_bias()
            for entity_pos, bias_map in expanded_bias.items():
                if (entity_idx, entity_pos) not in self._entity_pos_to_pdb_mapping:
                    continue

                # one-to-many mapping of positions
                all_pdb_pos = self._entity_pos_to_pdb_mapping[(entity_idx, entity_pos)]
                for pdb_pos in all_pdb_pos:
                    for symbol, bias_value in bias_map.items():
                        bias_tensor[pdb_pos, restype_str_to_int[symbol]] = bias_value

        feature_dict_copy["bias"] = bias_tensor[None, :, :]

        # generate sequences using the model
        L = feature_dict_copy["X"].shape[1]  # noqa
        generated_sequences = []

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            with torch.no_grad():
                num_batches = (num_designs + self.batch_size - 1) // self.batch_size
                for batch_idx in range(num_batches):
                    if status_callback:
                        progress = ((batch_idx + 1) / num_batches) * 100
                        status_callback(
                            "running", progress, f"Generating batch {batch_idx + 1}/{num_batches}"
                        )

                    feature_dict_copy["randn"] = torch.randn(
                        [self.batch_size, L], device=self.device
                    )
                    output_dict = self.model.sample(feature_dict_copy)
                    generated_sequences.append(output_dict["S"].cpu())

        # keep all sequences (even if more than num_designs, this is allowed by specification)
        S_stack = torch.cat(generated_sequences, 0).numpy()  #[:num_designs]  # noqa

        # create SystemInstance objects
        system_instances = []
        for design_idx in range(num_designs):
            # create entity instances based on rep of system entity first;
            # this will pass through any positions not designed by MPNN as-is
            system_instance = self.system.rep_to_instance()

            # update entity instances based on sequence from MPNN
            for pdb_pos, (entity_idx, entity_pos) in self._pdb_to_entity_mapping.items():
                system_instance[entity_idx].rep[
                    entity_pos - self._system[entity_idx].first_index
                ] = restype_int_to_str[
                    S_stack[design_idx, pdb_pos]
                ]

            system_instances.append(system_instance)

        # score the generated instances
        scores = self.score(system_instances, status_callback=status_callback)

        # try to score target sequence as well
        target_instance = self._system.rep_to_instance()
        if self._validate_instances([target_instance], raise_invalid=False):
            target_score = self.score([target_instance])[0]
        else:
            target_score = 0.0

        # attach scores to instances
        for instance, raw_score in zip(system_instances, scores):
             instance.score = raw_score - target_score

        return system_instances

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray:
        self.ready_or_raise()
        self._validate_instances(instances)

        # Extract sequences sliced to positions covered by PDB structure
        pdb_sequences = []
        for instance in instances:
            pdb_seq = ["_"] * len(self._native_seq)

            # Map symbol at corresponding instance positions to all PDB positions
            for pdb_pos, (entity_idx, entity_pos) in self._pdb_to_entity_mapping.items():
                pdb_seq[pdb_pos] = instance[entity_idx].rep[entity_pos - self.system[entity_idx].first_index]

            # we should have replaced all positions, if not, something is wrong
            assert "_" not in pdb_seq
            pdb_sequences.append("".join(pdb_seq))

        # Score sequences
        scores = []
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            with torch.no_grad():
                # fix one decoding order for all sequences
                if self._randn is None:
                    decoding_order = torch.randn(
                        [1, len(self._native_seq)], device=self.device
                    )
                else:
                    # use previously stored order
                    decoding_order = torch.from_numpy(self._randn).to(self.device)

                for seq_idx, seq in enumerate(pdb_sequences):
                    if status_callback:
                        progress = ((seq_idx + 1) / len(pdb_sequences)) * 100
                        status_callback(
                            "running", progress, f"Scoring sequence {seq_idx + 1}/{len(pdb_sequences)}"
                        )

                    # Convert sequence to tensor
                    S_tensor = torch.tensor(  # noqa
                        [restype_str_to_int.get(aa, 20) for aa in seq],
                        device=self.device,
                        dtype=torch.int64
                    )[None, :]

                    # Create feature dict for this sequence
                    feature_dict_copy = self._feature_dict.copy()
                    feature_dict_copy["S"] = S_tensor
                    feature_dict_copy["batch_size"] = 1
                    if self.vary_decoding_order_per_instance:
                        feature_dict_copy["randn"] = torch.randn(
                            [1, len(seq)], device=self.device
                        )
                    else:
                        feature_dict_copy["randn"] = decoding_order

                    feature_dict_copy["symmetry_residues"] = [[]]
                    feature_dict_copy["symmetry_weights"] = [[]]

                    # Score the sequence
                    output_dict = self.model.score(
                        feature_dict_copy, use_sequence=True
                    )

                    # Calculate loss (negative log probability)
                    loss, _ = get_score(
                        output_dict["S"],
                        output_dict["log_probs"],
                        self._feature_dict["mask"][:1]
                    )

                    # Convert to positive log likelihood
                    scores.append(-loss.item())

        # 5. Return as numpy array
        return np.array(scores)