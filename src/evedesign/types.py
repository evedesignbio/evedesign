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


class SapiensMutation(TypedDict):
    entity: int
    pos: int
    ref: str
    to: str


class SapiensGeneration(TypedDict):
    iterations: int
    scheme: str
    cdr_definition: str
    humanize_cdrs: bool
    backmutate_vernier: bool
    entities: list[int]
    fixed_positions: dict[int, list[int]]
    mutations: list[SapiensMutation]


SCORE_COMPONENT_KEY = "scores"
CHAIN_COMPONENT_KEY = "design_chain"
SEQSPACE_PROJECTION_COMPONENT_KEY = "seqspace_projection"

class Metadata(TypedDict):
    scores: NotRequired[list[Score]]
    design_chain: NotRequired[DesignChain]
    seqspace_projection: NotRequired[list[float]]
    sites: NotRequired[list[Site]]
    sapiens: NotRequired[SapiensGeneration]


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
