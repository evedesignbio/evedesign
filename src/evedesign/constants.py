# Mapping according to HHblits convention
# https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L975

MASK = "*"
GAP = "-"

AA_TO_INDEX = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
}

INDEX_TO_AA = {
    idx: symbol for symbol, idx in AA_TO_INDEX.items() if symbol not in {"U", "B", "Z", "J", "O"}
}

VALID_AA_TO_INDEX = {
    symbol: idx for symbol, idx in  AA_TO_INDEX.items() if symbol not in {"U", "B", "Z", "J", "O", "X"}
}

VALID_AA = set(VALID_AA_TO_INDEX)
VALID_AA_SORTED = sorted(VALID_AA)

VALID_AA_OR_GAP = VALID_AA | {GAP}
VALID_AA_OR_GAP_SORTED = sorted(VALID_AA) + [GAP]

VALID_DNA = {"A", "C", "G", "T"}
VALID_DNA_SORTED = sorted(VALID_DNA)
VALID_DNA_OR_GAP = VALID_DNA | {GAP}
VALID_DNA_OR_GAP_SORTED = sorted(VALID_DNA) + [GAP]

VALID_RNA = {"A", "C", "G", "U"}
VALID_RNA_SORTED = sorted(VALID_RNA)
VALID_RNA_OR_GAP = VALID_RNA | {GAP}
VALID_RNA_OR_GAP_SORTED = sorted(VALID_RNA) + [GAP]

# extracted from https://github.com/biopython/biopython/blob/186d3825c4b6b289c347c6ea95a89ffb2717e848/Bio/Data/PDBData.py#L295
# Wilke: Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635
RESIDUE_MAX_SASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "B": "ASX",
    "Z": "GLX",
    "X": "UNK",
}

AA3_to_AA1 = {
    **{
        v: k for k, v in AA1_TO_AA3.items()
    },
    "MSE": "M",
    "SEC": "C",
}
