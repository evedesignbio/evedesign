import random
import time
from io import StringIO
from typing import TypedDict, Sequence

from loguru import logger

from evedesign.tools.api_utils import _request_with_retries
from evedesign.system import System
from evedesign.structure import StructureFile, Structure
from evedesign.constants import GAP
from evedesign.__about__ import __version__

AFDB_DOWNLOAD_URL = "https://alphafold.ebi.ac.uk/files/{id_}.cif"

def _clean_sequence(seq):
    return "".join(seq.split()).upper()


def _predict_3di(sequence, host_url, headers):
    res = _request_with_retries(
        "GET",
        f"{host_url}/predict/{sequence}",
        headers=headers,
        context="Foldseek 3Di server",
    )
    text = res.text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    text = text.replace('"', "").replace("'", "")
    return "".join(text.split())


def _build_3di_query(sequence, header, host_url, headers):
    seq = _clean_sequence(sequence)
    if not seq:
        raise ValueError("Empty sequence for Foldseek 3Di prediction")
    three_di = _predict_3di(seq, host_url, headers)
    return f">{header}\n{seq}\n>3DI\n{three_di}\n"


def _foldseek_submit(query_text, databases, mode, host_url, headers):
    payload = [("q", query_text), ("mode", mode)]
    for db in databases:
        payload.append(("database[]", db))
    res = _request_with_retries(
        "POST",
        f"{host_url}/api/ticket",
        data=payload,
        headers=headers,
        context="Foldseek server",
    )
    try:
        return res.json()
    except ValueError:
        logger.error(f"Server didn't reply with json: {res.text}")
        return {"status": "ERROR"}


def _foldseek_status(ticket, host_url, headers):
    res = _request_with_retries(
        "GET",
        f"{host_url}/api/ticket/{ticket}",
        headers=headers,
        context="Foldseek server",
    )
    try:
        return res.json()
    except ValueError:
        logger.error(f"Server didn't reply with json: {res.text}")
        return {"status": "ERROR"}


def _foldseek_result(ticket, entry, host_url, headers, params=None):
    res = _request_with_retries(
        "GET",
        f"{host_url}/api/result/{ticket}/{entry}",
        params=params,
        headers=headers,
        context="Foldseek server",
    )
    try:
        return res.json()
    except ValueError:
        return res.text


def _extract_hits_brief(result_obj):
    results = []
    if isinstance(result_obj, list):
        if all(isinstance(item, dict) for item in result_obj):
            return result_obj
        return results

    if isinstance(result_obj, dict):
        results_list = result_obj.get("results", [])
        if not isinstance(results_list, list):
            return results
        for result in results_list:
            if not isinstance(result, dict):
                continue
            alignments = result.get("alignments", [])
            if not isinstance(alignments, list):
                continue
            for alignment_group in alignments:
                if not isinstance(alignment_group, list):
                    continue
                for hit in alignment_group:
                    if isinstance(hit, dict):
                        results.append(hit)
        return results

    return results


def foldseek_search_sequence(
    sequence,
    databases: list[str] = ("pdb100",),
    mode: str = "3diaa-print3di",
    host_url: str = "https://search.foldseek.com",
    predict_host_url: str = "https://3di.foldseek.com",
    user_agent: str | None = None,
):
    """
    Submit a single AA sequence to Foldseek and return brief hits without full C-alpha coords.
    Use foldseek_fetch_full_hit to retrieve full hit data including C-alpha coordinates.
    """
    if user_agent is None:
        user_agent = "evedesign/" + __version__

    headers = {"User-Agent": user_agent} if user_agent else None

    query_text = _build_3di_query(
        sequence,
        header="query",
        host_url=predict_host_url,
        headers=headers,
    )

    out = _foldseek_submit(query_text, databases, mode, host_url, headers)
    while out.get("status") in ["UNKNOWN", "RATELIMIT"]:
        sleep_time = 5 + random.randint(0, 5)
        logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
        time.sleep(sleep_time)
        out = _foldseek_submit(query_text, databases, mode, host_url, headers)

    if out.get("status") == "ERROR":
        raise Exception(
            "Foldseek API is giving errors. Please confirm your query is valid. "
            "If error persists, please try again an hour later."
        )

    if out.get("status") == "MAINTENANCE":
        raise Exception(
            "Foldseek API is undergoing maintenance. Please try again in a few minutes."
        )

    ticket = out.get("id")
    if not ticket:
        raise RuntimeError(f"Foldseek did not return a ticket id: {out}")

    out = {"status": "UNKNOWN"}
    while out.get("status") in ["UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"]:
        t = 5 + random.randint(0, 5)
        logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
        time.sleep(t)
        out = _foldseek_status(ticket, host_url, headers)

    if out.get("status") == "MAINTENANCE":
        raise Exception(
            "Foldseek API is undergoing maintenance. Please try again in a few minutes."
        )

    if out.get("status") == "ERROR":
        raise Exception(
            "Foldseek API is giving errors. Please confirm your query is valid. "
            "If error persists, please try again an hour later."
        )

    if out.get("status") != "COMPLETE":
        raise RuntimeError(f"Unexpected Foldseek status: {out.get('status')}")

    result_obj = _foldseek_result(
        ticket,
        entry=0,
        host_url=host_url,
        headers=headers,
        # params={"format": "brief"},  # need tSeq for our mapping
    )
    hits = _extract_hits_brief(result_obj)
    return hits, ticket


def foldseek_fetch_full_hit(
    ticket,
    index,
    database,
    entry: int = 0,
    host_url: str = "https://search.foldseek.com",
    user_agent: str | None = None,
):
    """
    Fetch full hit data using index+database (format=brief).
    """
    if user_agent is None:
        user_agent = "evedesign/" + __version__

    headers = {"User-Agent": user_agent} if user_agent else None

    return _foldseek_result(
        ticket,
        entry=entry,
        host_url=host_url,
        headers=headers,
        params={"format": "brief", "index": index, "database": database},
    )



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
    "J": "XLE",
    "U": "SEC",
    "O": "PYL",
    "X": "UNK",
    "-": "UNK",
}


def _parse_ca_coords(tca):
    if tca is None:
        return []
    if isinstance(tca, str):
        parts = [p for p in tca.split(",") if p.strip()]
        try:
            coords = [float(p) for p in parts]
        except ValueError:
            return []
    elif isinstance(tca, (list, tuple)):
        try:
            coords = [float(p) for p in tca]
        except (TypeError, ValueError):
            return []
    else:
        return []

    if len(coords) < 3:
        return []
    if len(coords) % 3 != 0:
        coords = coords[: len(coords) - (len(coords) % 3)]
    return [coords[i:i + 3] for i in range(0, len(coords), 3)]


def _mock_pdb_from_ca(tca, seq, chain_id="A"):
    coords = _parse_ca_coords(tca)
    if not coords:
        return ""
    chain_id = (chain_id or "A")[:1]
    seq = _clean_sequence(seq) if seq else ""
    use_seq = len(seq) == len(coords)
    lines = []
    for idx, (x, y, z) in enumerate(coords, start=1):
        aa = seq[idx - 1] if use_seq else "A"
        res = AA1_TO_AA3.get(aa, "UNK")
        lines.append(
            f"ATOM  {idx:5d}  CA  {res:>3} {chain_id:1}{idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    return "\n".join(lines)


def build_structure_from_ca(tca, seq, chain_id="A"):
    pdb_text = _mock_pdb_from_ca(tca, seq, chain_id=chain_id)
    if not pdb_text:
        return ""
    return StructureFile(StringIO(pdb_text), format="pdb")


def build_structure_from_hit(hit, chain_id="A"):
    if not isinstance(hit, dict):
        return ""
    return build_structure_from_ca(hit.get("tCa"), hit.get("tSeq"), chain_id=chain_id)


def flatten_foldseek_hits(hit_groups):
    flattened = []
    for group in hit_groups:
        if isinstance(group, list):
            flattened.extend([hit for hit in group if isinstance(hit, dict)])
        elif isinstance(group, dict):
            flattened.append(group)
    return flattened


def hits_to_structures(hits, chain_id="A"):
    paths = {}
    for idx, hit in enumerate(hits):
        mmcif = build_structure_from_hit(hit, chain_id=chain_id)
        if not mmcif:
            continue
        paths[idx] = mmcif
    return paths


class FoldSeekHit(TypedDict):
    query: str
    target: str
    seqId: float
    alnLength: int
    missmatches: int  # noqa
    gapsopened: int  # noqa
    qStartPos: int
    qEndPos: int
    dbStartPos: int
    dbEndPos: int
    prob: int
    eval: float
    score: int
    qLen: int
    dbLen: int
    qAln: str
    dbAln: str
    taxId: int
    taxName: str
    q3di: str
    t3di: str
    tCa: int | str  # 0 if format "brief"
    tSeq: int | str  # 0 if format "brief"
    scoreAdj: float

"""

Mapping extracted from https://github.com/steineggerlab/foldseek/blob/8979d230fb64c7089380b652758d8705493ed4a5/src/strucclustutils/GemmiWrapper.cpp#L110
with following Python code, after manually editing fall-through case:

for row in AA_CODES.split("\n"):
    row = row.strip()
    if "return" not in row or row.startswith("//"):
        continue

    symbol = row.split('"')[1]
    code = row.split("return ")[1].split(";")[0].replace("'", "")

    if code not in code_to_symbol:
        code_to_symbol[code] = []

    code_to_symbol[code].append(symbol)
    symbol_to_code[symbol] = code
"""
FOLDSEEK_THREE_TO_ONE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ABA': 'A',
    'ASP': 'D',
    'ASX': 'B',
    'CYS': 'C',
    'CSH': 'S',
    'GLN': 'Q',
    'GLU': 'E',
    'GLX': 'Z',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'MSE': 'M',
    'ORN': 'A',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRY': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'UNK': 'X',
    'VAL': 'V',
    'SEC': 'C',
    'PYL': 'O',
    'SEP': 'S',
    'TPO': 'T',
    'PCA': 'E',
    'CSO': 'C',
    'PTR': 'Y',
    'KCX': 'K',
    'CSD': 'C',
    'LLP': 'K',
    'CME': 'C',
    'MLY': 'K',
    'DAL': 'A',
    'TYS': 'Y',
    'OCS': 'C',
    'M3L': 'K',
    'FME': 'M',
    'ALY': 'K',
    'HYP': 'P',
    'CAS': 'C',
    'CRO': 'T',
    'CSX': 'C',
    'DPR': 'P',
    'DGL': 'E',
    'DVA': 'V',
    'CSS': 'C',
    'DPN': 'F',
    'DSN': 'S',
    'DLE': 'L',
    'HIC': 'H',
    'NLE': 'L',
    'MVA': 'V',
    'MLZ': 'K',
    'CR2': 'G',
    'SAR': 'G',
    'DAR': 'R',
    'DLY': 'K',
    'YCM': 'C',
    'NRQ': 'M',
    'CGU': 'E',
    '0TD': 'D',
    'MLE': 'L',
    'DAS': 'D',
    'DTR': 'W',
    'CXM': 'M',
    'TPQ': 'Y',
    'DCY': 'C',
    'DSG': 'N',
    'DTY': 'Y',
    'DHI': 'H',
    'MEN': 'N',
    'DTH': 'T',
    'SAC': 'S',
    'DGN': 'Q',
    'AIB': 'A',
    'SMC': 'C',
    'IAS': 'D',
    'CIR': 'R',
    'BMT': 'T',
    'DIL': 'I',
    'FGA': 'E',
    'PHI': 'F',
    'CRQ': 'Q',
    'SME': 'M',
    'GHP': 'G',
    'MHO': 'M',
    'NEP': 'H',
    'TRQ': 'W',
    'TOX': 'W',
    'ALC': 'A',
    'SCH': 'C',
    'MDO': 'A',
    'MAA': 'A',
    'GYS': 'S',
    'MK8': 'L',
    'CR8': 'H',
    'KPI': 'K',
    'SCY': 'C',
    'DHA': 'S',
    'OMY': 'Y',
    'CAF': 'C',
    '0AF': 'W',
    'SNN': 'N',
    'MHS': 'H',
    'SNC': 'C',
    'PHD': 'D',
    'B3E': 'E',
    'MEA': 'F',
    'MED': 'M',
    'OAS': 'S',
    'GL3': 'G',
    'FVA': 'V',
    'PHL': 'F',
    'CRF': 'T',
    'BFD': 'D',
    'MEQ': 'Q',
    'DAB': 'A',
    'AGM': 'R',
    '4BF': 'Y',
    'B3A': 'A',
    'B3D': 'D',
    'B3K': 'K',
    'B3Y': 'Y',
    'BAL': 'A',
    'DBZ': 'A',
    'GPL': 'K',
    'HSK': 'H',
    'HY3': 'P',
    'HZP': 'P',
    'KYN': 'W',
    'MGN': 'Q'
}

def remap_structure_from_hit(hit: FoldSeekHit, structure_model: Structure, first_index: int) -> list[Structure]:
    """
    Extract chain(s) from a PDB structure mapped by a FoldSeekHit independent of auth/label chain IDs,
    and remap residue indices so they match the target/query structure sequence indices.

    Parameters
    ----------
    hit
        Single FoldSeek hit (from list returned by foldseek_search_sequence)
    structure_model
        Loaded PDB structure with all chains present
    first_index
        Index of first position of target sequeence

    Returns
    -------
    All chains in structure_model that are covered by hit, remapped to target sequence indices
    """
    # store mapping of sequence to chain to match with FoldSeek sequences
    seq_to_chain_id = {}

    # also store mapping from chain_id to filtered position and sequence
    chain_id_to_residues = {}

    # iterate individual chains
    for chain_id, chain_df in structure_model.atom_df().groupby("chain_id"):
        # limit to residues which have CA atoms, and those that
        # are contained in FoldSeek/gemmi residue mapping
        s_ca = chain_df.query("atom_name == 'CA'").drop_duplicates(
            subset=["chain_id", "res_id", "ins_code"]
        ).query(
            "res_name in @FOLDSEEK_THREE_TO_ONE"
        ).assign(
            res_name_oneletter=lambda df: df.res_name.map(FOLDSEEK_THREE_TO_ONE)
        )

        # ignore anything that does not survive filtering (will be mostly non-protein chains)
        if len(s_ca) == 0:
            continue

        # assemble sequence as output by FoldSeek (i.e. no CA or not mappable missing)
        chain_seq = "".join(s_ca.res_name_oneletter)
        seq_to_chain_id[chain_seq] = seq_to_chain_id.get(chain_seq, []) + [chain_id]

        # also store filtered residue table for later position index mapping
        chain_id_to_residues[chain_id] = s_ca

    # identify what chains our target sequence maps to
    try:
        target_chains = seq_to_chain_id[hit["tSeq"]]
    except KeyError as e:
        raise ValueError(
            f"Could not map hit to structure, seq_to_chain_id={seq_to_chain_id}, tSeq={hit.get('tSeq')}"
        ) from e

    # perform remapping of chains one by one
    remapped_chains = []
    for chain_id in target_chains:
        chain_pos = chain_id_to_residues[chain_id]
        chain_map = {}

        # current positions in query and database sequence; local alignment
        # should not start with gaps in either pos;
        # 1-based index in target sequence, note that qStartPos does not incorporate first index shifts
        q_idx_seq = first_index + hit["qStartPos"] - 1
        db_idx = hit["dbStartPos"] - 1  # 0-based index in string

        # iterate through pairwise alignment;
        # paired symbols at current alignment position
        for q_symbol, db_symbol in zip(hit["qAln"], hit["dbAln"]):
            # establish residue mapping if two residues are aligned (no gap in either sequence)
            if q_symbol != GAP and db_symbol != GAP:
                # check we are tracking position in db sequence correctly
                assert hit["tSeq"][db_idx] == db_symbol, "Sequence mismatch that should never occur"

                # get corresponding residue information from structure, and store
                # mapping into position in target sequence
                cur_pos = chain_pos.iloc[db_idx]
                assert cur_pos.res_name_oneletter == db_symbol, "Sequence mismatch that should never occur"
                chain_map[int(cur_pos.res_id)] = q_idx_seq

            # increase index in either sequence if position was not a gap
            if q_symbol != GAP:
                q_idx_seq += 1
            if db_symbol != GAP:
                db_idx += 1

        # perform remapping and store chain
        remapped_chain = structure_model.get_chain(chain_id).remap(chain_map)
        remapped_chains.append(remapped_chain)

    return remapped_chains


def correct_score_for_spaghetti(hit: FoldSeekHit) -> FoldSeekHit:
    """
    Rescale hit score to reduce inflated contribution of low-complexity
    regions relative to experimental structures ("spaghetti" in AF structures
    that are typically not resolved in PDB structures)

    Corresponds to rankStructureHits function in frontend

    Parameters
    ----------
    hit
        Raw hit from FoldSeek API

    Returns
    -------
    Hit with corrected score field
    """
    t_seq = hit["tSeq"]
    if not isinstance(t_seq, str):
        raise ValueError("Must run FoldSeek with full instead of brief mode for this feature")

    # extract 3Di aligned region from full DB 3Di sequence
    target_region = t_seq[hit["dbStartPos"] - 1:hit["dbEndPos"]]
    target_region_from_aln = hit["dbAln"].replace(GAP, "")
    target_region_3di = hit["t3di"][hit["dbStartPos"] - 1:hit["dbEndPos"]]

    assert (
        target_region == target_region_from_aln and len(target_region_3di) == len(target_region)
    ), "Structure alignment inconsistency, this should never happen"

    # current positions in query and database sequence; local alignment
    # should not start with gaps in either pos
    q_idx = hit["qStartPos" ] - 1
    db_idx = hit["dbStartPos" ] - 1

    aligned_pairs = 0
    good_pairs = 0

    # sliding window matching D or P state
    overhang = "DD"
    t3di_with_overhang = overhang + hit["t3di"] + overhang

    assert len(hit["qAln"]) == len(hit["dbAln"]), "Alignment length mismatch"

    # iterate through aligned amino acid pairs
    for i, (q_symbol, db_symbol) in enumerate(zip(hit["qAln"], hit["dbAln"])):
        assert i != 0 or (q_symbol != GAP and db_symbol != GAP), "Alignment starts with gaps (against code assumptions"

        # check if pair is aligned
        if q_symbol != GAP and db_symbol != GAP:
            assert t_seq[db_idx] == db_symbol, "Sequence mismatch that should never occur"

            # increase aligned pair count, this will be used for normalization
            aligned_pairs += 1

            q_3di = hit["q3di"][q_idx]
            db_3di = hit["t3di"][db_idx]

            # extract sliding window
            db_3di_window = t3di_with_overhang[
                db_idx:(db_idx + 1 + 2 * len(overhang))
            ].replace("P", "D")

            # only consider non-D/P state as informative
            if not (q_3di in {"D", "P"} and db_3di in {"D", "P"} and db_3di_window == overhang + "D" + overhang):
                good_pairs += 1

        # increase index in either sequence if position was not a gap
        if q_symbol != GAP:
            q_idx += 1
        if db_symbol != GAP:
            db_idx += 1

    # rescale score
    score_adj = hit["score"] * (good_pairs / aligned_pairs)

    # linter does not like unpacking here so do it old-fashioned way
    hit = hit.copy()
    hit["scoreAdj"] = score_adj
    return hit


def find_structures_foldseek(
    system: System,
    databases: list[str] = ("pdb100",),
    entity_subset: Sequence[int] | None = None,
    correct_spaghetti: bool = True,
    mode: str = "3diaa-print3di",
    host_url: str = "https://search.foldseek.com",
    predict_host_url: str = "https://3di.foldseek.com",
    user_agent: str | None = None,
) -> dict[int, list[FoldSeekHit]]:
    """
    Find related 3D structures for all protein entities in system
    by predicting 3Di for target sequence and searching
    it against 3Di databases.

    Note that structure searches will be performed independently per
    entity and need to be intersected afterwards by identifier to
    find structures of interactions.

    Parameters
    ----------
    system
        System for which to perform related structure search
    databases
        Target databases
    entity_subset
        If None, search structures for all protein entities, otherwise limit to specified
        entities (by index in system)
    correct_spaghetti
        If True, rescale hit score to reduce inflated contribution of low-complexity
        regions relative to experimental structures ("spaghetti" in AF structures
        that are typically not resolved in PDB structures)
    mode
        FoldSeek server search mode
    host_url
        FoldSeek server URL
    predict_host_url
        3Di prediction server URL
    user_agent
        User agent to send to servers for diagnostic purposes

    Returns
    -------
    Mapping from entity index to all identified structure hits
    """
    entity_to_hits = {}

    # search entities one by one
    for idx, entity in enumerate(system):
        # only search for protein entity with defined sequence
        if entity.type != "protein" or not entity.defined_sequence():
            continue

        # skip entities if filter is defined
        if entity_subset is not None and idx not in entity_subset:
            continue

        logger.info(f"Running foldseek for entity {idx}")

        # run search
        hits, _ = foldseek_search_sequence(
            "".join(entity.rep),
            databases=databases,
            mode=mode,
            host_url=host_url,
            predict_host_url=predict_host_url,
            user_agent=user_agent,
        )

        if correct_spaghetti:
            hits = [
                correct_score_for_spaghetti(hit) for hit in hits
            ]
        else:
            for hit in hits:
                hit["scoreAdj"] = hit["score"]

        entity_to_hits[idx] = hits

    return entity_to_hits


def _extract_structure_id(id_description: str) -> tuple[str, str, str]:
    """
    Extract structure ID and database type from FoldSeek hit target

    Parameters
    ----------
    id_description
        FoldSeek hit target

    Returns
    -------
    Tuple of structure ID, assembly and database type
    """
    # AFDB, e.g. "AF-A0A378GLZ1-F1-model_v6 Molybdopterin synthase sulfur carrier subunit"
    if id_description.startswith("AF-"):
        structure_id = id_description.split()[0]
        assembly_id = ""
        db_type = "afdb"

    # PDB, e.g. "1jwb-assembly1.cif.gz_D-2 Structure of the Covalent Acyl-Adenylate Form of the MoeB-MoaD Protein Complex"
    elif "assembly" in id_description:
        structure_id = id_description.split("-")[0]
        # use specific assembly ID or we can run into issues that chain cannot be found
        assembly_id = id_description.split("-assembly")[1].split(".")[0]
        db_type = "pdb"
    else:
        raise NotImplementedError(
            f"Database type not yet supported {id_description}"
        )

    return structure_id, assembly_id, db_type


def filter_structures_foldseek(
    entity_to_hits: dict[int, list[FoldSeekHit]],
    use_pairing: bool = False,
    ids: Sequence[str] | None = None,
    top_n: int | None = None,
) -> dict[int, list[FoldSeekHit]]:
    """
    Filter FoldSeek hits, with filters applied in the following order (if specified):
    1. use_pairing
    2. ids
    3. top_n

    Parameters
    ----------
    entity_to_hits
        Input structure mapping from find_structures_foldseek
    use_pairing
        If True, only keep structures where all entities have a hit to the respective structure
    ids
        If not None, only keep hits with these identifiers
    top_n
        If not None, restrict to top n hits per entity. Unless intersect is True, hits are ranked independently
         *per entity*, i.e. hits may have different orders across entities

    Returns
    -------
    Filtered hits
    """
    if use_pairing:
        # note that we keep full tuple from _extract_structure_id to include assembly ID for intersection
        shared_ids = set.intersection(*[
            set(_extract_structure_id(hit["target"]) for hit in hits)
            for entity_idx, hits in entity_to_hits.items()
        ])

        entity_to_hits = {
            idx: [hit for hit in hits if _extract_structure_id(hit["target"]) in shared_ids]
            for idx, hits in entity_to_hits.items()
        }

    if ids is not None:
        ids_lower = [id_.lower() for id_ in ids]
        entity_to_hits = {
            idx: [hit for hit in hits if _extract_structure_id(hit["target"])[0].lower() in ids_lower]
            for idx, hits in entity_to_hits.items()
        }

    if top_n:
        entity_to_hits_new = {}
        if use_pairing:
            # aggregate adjusted scores per hit across entities
            id_to_scores = {}
            for entity_idx, hits in entity_to_hits.items():
                chain_id_already_seen = {}
                for hit in hits:
                    _id = _extract_structure_id(hit["target"])
                    # only count each unique structure per entity once in case of multiple occurrences
                    if _id not in chain_id_already_seen:
                        id_to_scores[_id] = id_to_scores.get(_id, 0) + hit["scoreAdj"]
                        chain_id_already_seen[_id] = True

            # then sort and extract top n per entity
            for entity_idx, hits in entity_to_hits.items():
                hits_sorted = sorted(
                    hits, key=lambda hit: id_to_scores[_extract_structure_id(hit["target"])], reverse=True
                )

                entity_to_hits_new[entity_idx] = hits_sorted[:top_n]
        else:
            # sort independently by adjusted score per entity
            for entity_idx, hits in entity_to_hits.items():
                entity_to_hits_new[entity_idx] = sorted(
                    hits, key=lambda hit: hit["scoreAdj"], reverse=True
                )[:top_n]
        return entity_to_hits_new
    else:
        return entity_to_hits


def add_structures_foldseek(
    system: System,
    entity_to_hits: dict[int, list[FoldSeekHit]]
) -> System:
    """
    Add structure chains to entities in system

    Parameters
    ----------
    system
        System to which structure chains should be added
    entity_to_hits
        Mapping from entity index in system to FoldSeek hits, all
        passed hits will be added (so filter before if needed)

    Returns
    -------
    Copy of system with structural information added
    """
    system = system.copy()

    for entity_idx, hits in entity_to_hits.items():
        if system[entity_idx].structures is None:
            system[entity_idx].structures = {}

        for hit in hits:
            structure_id, assembly_id, db_type = _extract_structure_id(hit["target"])

            # AFDB, e.g. "AF-A0A378GLZ1-F1-model_v6 Molybdopterin synthase sulfur carrier subunit"
            if db_type == "afdb":
                response = _request_with_retries(
                    "GET", AFDB_DOWNLOAD_URL.format(id_=structure_id)
                )

                s = StructureFile(
                    StringIO(response.text), format="cif"
                ).get_model(
                    use_author_fields=False
                )
            elif db_type == "pdb":
                s = StructureFile.from_id(
                    structure_id.lower()
                ).get_assembly_model(
                    assembly_id=assembly_id,
                    use_author_fields=False
                )
            else:
                # should already raise in _extract_structure_id
                assert False, "Invalid DB type"

            # remap structure to target sequence numbering
            s_mapped = remap_structure_from_hit(
                hit, s, first_index=system[entity_idx].first_index
            )

            # attach chain(s) to system
            system[entity_idx].structures[structure_id] = s_mapped

    return system