from typing import Literal, Callable, Any, TypedDict, NotRequired, Mapping, Sequence
import numpy as np

BioPolymers = {"protein", "dna", "rna"}
BioPolymer = Literal["protein", "dna", "rna"]
EntityType = BioPolymer | Literal["ligand"]
LigandRepType = Literal["smiles", "ccd", "user_ccd"]
BondType = Literal["covalent", "hydrogen", "vdw", "ionic"]
SecondaryStructureType = Literal["H", "E", "C"]  # helix, sheet, coil
SymmetryType = Literal["C", "D", "T", "O", "I"]
DeviceType = Literal["cpu", "cuda", "mps"]
BatchSize = int | Literal["auto"] | None
SiteType = Literal["t_cell_epitope", "liability_motif"]


class DesignChain(TypedDict):
    # mapping from entity to
    init: dict[int, str]

    # tuple: entity, position, new symbol, score difference, temperature
    chain: list[tuple[int, int, str, float, float]]


class Score(TypedDict):
    index: int
    name: str
    weight: float
    score: float
    ref_score: float | None


class Site(TypedDict):
    entity: int
    pos: int
    length: int
    type: SiteType  # e.g. t_cell_epitope
    subtype: str   # e.g. HLA allele
    score: float | None  # e.g. HLA prediction rank
    weight: float | None  # e.g. population weight
    model: NotRequired[str]  # model that identified the site


class StructureComparison(TypedDict):
    reference: NotRequired[str]  # key in Entity.structures (e.g. PDB id)
    model: NotRequired[str]  # key in EntityInstance.models (e.g. "model_0")
    subset: str  # "all" or name of residue subset
    superposition: str  # subset used for superposition
    atoms: list[str]  # atom names used for superposition/RMSD
    num_atoms: int
    rmsd: float
    tm_score: float | None  # only computed for full comparison
    copy_assignment: dict[str, list[int]]  # entity -> model copy index per reference copy
    transform: list[list[float]]  # 4x4 matrix applied to model coordinates


class PositionRecovery(TypedDict):
    subset: str  # name of position set
    num_positions: int
    num_maintained: int
    fraction: float | None  # num_maintained / num_positions (None for empty set)
    mutated: list[list[int | str]]  # [entity, pos, WT residue, design residue ("-" if deleted)]


SCORE_COMPONENT_KEY = "scores"
CHAIN_COMPONENT_KEY = "design_chain"
SEQSPACE_PROJECTION_COMPONENT_KEY = "seqspace_projection"
STRUCTURE_COMPARISON_COMPONENT_KEY = "structure_comparison"
POSITION_RECOVERY_COMPONENT_KEY = "position_recovery"


class Metadata(TypedDict):
    scores: NotRequired[list[Score]]
    design_chain: NotRequired[DesignChain]
    seqspace_projection: NotRequired[list[float]]
    sites: NotRequired[list[Site]]


class SequenceMetadata(TypedDict):
    seqspace_projection: NotRequired[list[float]]
    taxonomy_id: NotRequired[int]
    taxonomy_lineage: NotRequired[str]

EvaluationScoreName = Literal[
    "r2", "pearson", "spearman", "rocauc", "mcc", "average_precision"
]

class ModelStats(TypedDict):
    y_true: NotRequired[Sequence[Sequence[float]]]
    y_pred: NotRequired[Sequence[Sequence[float]]]

    # different types of evaluation scores
    scores: NotRequired[dict[EvaluationScoreName, np.ndarray]]

class DatasetSplit(TypedDict):
    train: Sequence[int]
    test: NotRequired[Sequence[int]]
    val: NotRequired[Sequence[int]]

DatasetSplitMap = dict[str, DatasetSplit]

# status, progress (optional), message (optional)
Status = Literal["running", "done", "failed"]
StatusCallback = Callable[[Status, float | None, str | None], Any]
RepSequence = np.ndarray[tuple[int], np.dtype["U1"]]
Embedding = np.ndarray[
    tuple[int, int], np.dtype[float]
] | np.ndarray[
    tuple[int], np.dtype[float]
]
EntityPosList = Mapping[int, Sequence[int]]
