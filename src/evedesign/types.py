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

SCORE_COMPONENT_KEY = "scores"
CHAIN_COMPONENT_KEY = "design_chain"
SEQSPACE_PROJECTION_COMPONENT_KEY = "seqspace_projection"

class Metadata(TypedDict):
    scores: NotRequired[list[Score]]
    design_chain: NotRequired[DesignChain]
    seqspace_projection: NotRequired[list[float]]


class SequenceMetadata(TypedDict):
    seqspace_projection: NotRequired[list[float]]
    taxonomy_id: NotRequired[int]
    taxonomy_lineage: NotRequired[str]

EvaluationScoreName = Literal[
    "r2", "pearson", "spearman", "rocauc", "mcc", "average_precision"
]

class ModelStats(TypedDict):
    y_true: NotRequired[np.ndarray]
    y_pred: NotRequired[np.ndarray]

    # different types of evaluation scores
    scores: NotRequired[dict[EvaluationScoreName, np.ndarray]]

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
