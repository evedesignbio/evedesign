import io
import os
import random
import subprocess
import tarfile
import tempfile
import time
from typing import Literal, Sequence as SequenceType

from pathlib import Path
from loguru import logger

from evedesign.__about__ import __version__
from evedesign.tools.api_utils import _request_with_retries
from evedesign.system import System, Entity
from evedesign.sequence import read_fasta, Sequence, Sequences


def filter_sequences_mmseqs(
    sequences: list[str],
    target_num_sequences: int,
    brackets: SequenceType[float]=(0.2, 0.4, 0.6, 0.8, 1.0),
    max_seq_id: float | None = None,
    filter_min_enable: int | None = None,
    mmseqs_path: str = "mmseqs"
):
    """
    Reduce sequences down to a specified number of clusters with MMseqs filtera3m command

    Parameters
    ----------
    sequences
        Input sequences, must be aligned in a3m format
    target_num_sequences
        Target number of most diverse sequences (note the exact number returned may differ)
        per bucket.
    brackets:
        Reduce diversity of output MSAs using buckets with these identity boundaries to query sequences (--qid parameter)
    max_seq_id:
        Maximum sequence identity between any pair of sequence (--max-seq-id parameter)
    filter_min_enable:
        Minimum number of sequences in bracket required to enable filtering (--filter-min-enable parameter)
    mmseqs_path
        Path to mmseqs binary (optional, defaults to assuming mmseqs is on $PATH)
    """
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Write sequences to a temporary FASTA file
        input_ali = tempdir / "input.a3m"

        with open(input_ali, "w") as f:
            for i, sequence in enumerate(sequences):
                f.write(f">{i}\n{sequence}\n")

        output_ali = tempdir / "output.a3m"

        cmd = [
            mmseqs_path,
            "filtera3m",
            str(input_ali),
            str(output_ali),
            "--diff", str(target_num_sequences),
            "--qsc", "0",
            "--qid", ",".join([str(thr) for thr in brackets])
        ]

        if filter_min_enable is not None:
            cmd += ["--filter-min-enable", str(filter_min_enable)]

        if max_seq_id is not None:
            cmd += ["--max-seq-id", str(max_seq_id)]

        ret = subprocess.run(
            cmd, capture_output=True
        )
        if ret.returncode != 0:
            raise ValueError(
                f"Error running MMseqs2, retcode={ret.returncode} stdout={ret.stdout} stderr={ret.stderr}"
            )

        # parse output
        with output_ali.open() as f:
            filtered_ids = [
                int(line[1:]) for line in f if line.startswith(">")
            ]

    return filtered_ids


def filter_entity_sequences_mmseqs(
    entity: Entity,
    target_num_sequences: int = 3000,
    brackets: SequenceType[float]=(0.2, 0.4, 0.6, 0.8, 1.0),
    max_seq_id: float | None = 0.95,
    filter_min_enable: int | None = 1000,
    mmseqs_path: str = "mmseqs"
) -> Sequences | None:
    """
    Filter sequences on entity by ColabFold bucket-based method,
    with same default values as in Mirdita et al. (Nature Methods, 2022)

    Function does not consider if sequences are paired to other sequences
    with key or not, which may lose sequence pairs if applied to an
    entity from a complex system

    Parameters
    ----------
    entity
        Entity for which sequences should be filtered
    target_num_sequences
        Target number of most diverse sequences (note the exact number returned may differ)
        per bucket.
    brackets:
        Reduce diversity of output MSAs using buckets with these identity boundaries to query sequences (--qid parameter)
    max_seq_id:
        Maximum sequence identity between any pair of sequence (--max-seq-id parameter)
    filter_min_enable:
        Minimum number of sequences in bracket required to enable filtering (--filter-min-enable parameter)
    mmseqs_path
        Path to mmseqs binary (optional, defaults to assuming mmseqs is on $PATH)

    Returns
    -------
    Sequences object with filtered set of sequences (can be assigned to entity)
    """
    if entity.sequences is None:
        return None

    # get indices of remaining sequences
    idx_filt = filter_sequences_mmseqs(
        [x.seq for x in entity.sequences.seqs],
        target_num_sequences=target_num_sequences,
        brackets=brackets,
        max_seq_id=max_seq_id,
        filter_min_enable=filter_min_enable,
        mmseqs_path=mmseqs_path
    )

    # create updated Sequences object
    return Sequences(
        seqs=[entity.sequences.seqs[i] for i in idx_filt],
        aligned=entity.sequences.aligned,
        type=entity.sequences.type_,
        weights=None,  # reset weights as not meaningful for subset
        format=entity.sequences.format_
    )


def run_mmseqs2(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,  # noqa
    use_pairing=False,
    pairing_strategy="greedy",
    host_url="https://api.colabfold.com",
    user_agent: str = "",
):
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') "
            "to help us debug in case of problems. This warning will become an error in the future."
        )

    if use_templates:
        logger.warning("Template fetching disabled; proceeding without templates.")

    def submit(seqs, mode, N=101):  # noqa
        n, query = N, ""
        for seq in seqs:  # noqa
            query += f">{n}\n{seq}\n"
            n += 1

        res = _request_with_retries(
            "POST",
            f"{host_url}/{submission_endpoint}",
            data={"q": query, "mode": mode},
            timeout=6.02,
            headers=headers,
            context="MSA server",
        )
        try:
            out = res.json()  # noqa
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}  # noqa
        return out

    def status(ID):  # noqa
        res = _request_with_retries(
            "GET",
            f"{host_url}/ticket/{ID}",
            timeout=6.02,
            headers=headers,
            context="MSA server",
        )
        try:
            out = res.json()  # noqa
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}  # noqa
        return out

    def download(ID, path):  # noqa
        res = _request_with_retries(
            "GET",
            f"{host_url}/result/download/{ID}",
            timeout=6.02,
            headers=headers,
            context="MSA server",
        )
        with open(path, "wb") as out:  # noqa
            out.write(res.content)

    seqs = [x] if isinstance(x, str) else x

    if filter is not None:
        use_filter = filter

    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = f"{mode}-env"

    path = f"{prefix}_{mode}"
    os.makedirs(path, exist_ok=True)

    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True  # noqa

    seqs_unique = []
    for seq in seqs:
        if seq not in seqs_unique:
            seqs_unique.append(seq)
    Ms = [N + seqs_unique.index(seq) for seq in seqs]   # noqa

    if not os.path.isfile(tar_gz_file):
        while REDO:
            out = submit(seqs_unique, mode, N)
            while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                sleep_time = 5 + random.randint(0, 5)
                logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                time.sleep(sleep_time)
                out = submit(seqs_unique, mode, N)

            if out["status"] == "ERROR":
                raise Exception(
                    "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. "
                    "If error persists, please try again an hour later."
                )

            if out["status"] == "MAINTENANCE":
                raise Exception(
                    "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                )

            ID = out["id"]  # noqa
            while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                t = 5 + random.randint(0, 5)
                logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                time.sleep(t)
                out = status(ID)

            if out["status"] == "COMPLETE":
                REDO = False  # noqa

            if out["status"] == "ERROR":
                REDO = False  # noqa
                raise Exception(
                    "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. "
                    "If error persists, please try again an hour later."
                )

        download(ID, tar_gz_file)  # noqa

    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path, filter="data")

    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None  # noqa
        with open(a3m_file, "r") as handle:
            for line in handle:
                if line:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True  # noqa
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())  # noqa
                        update_M = False  # noqa
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    return ["".join(a3m_lines[n]) for n in Ms]


def _parse_a3m(a3m_text):
    return list(read_fasta(io.StringIO(a3m_text)))


def _sequences_from_entries(entries, keys=None):
    seqs = []
    for i, (header, seq) in enumerate(entries):
        key = None
        if keys is not None and i < len(keys):
            key = keys[i]
        seqs.append(Sequence(seq=seq, id=header.split()[0], key=key))
    return Sequences(seqs, aligned=True, format="a3m")


def add_sequences_mmseqs2(
    system: System,
    use_env: bool = False,
    use_filter: bool = True,
    filter=None,  # noqa
    use_pairing: bool = False,
    pair_mode: Literal["paired", "unpaired", "unpaired_paired"] = "unpaired_paired",
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    user_agent: str | None = None,
    keep_tmp_dir: bool = False,
    tmpdir: str | Path | None = None,
) -> System:
    """
    Attach MSAs to all protein entities in system

    Parameters
    ----------
    system
        System where sequences should be added for all protein entities
    use_env
        If True, search metagenomic sequences (cf. ColabFold documentation)
    use_filter
        If True, filter output MSA (cf. ColabFold documentation)
    filter
         Cf. ColabFold documentation
    use_pairing
        If True, pair sequences across entities with key attribute
    pair_mode
        Cf. ColabFold documentation
    pairing_strategy
        Strategy for pairing sequences (cf. ColabFold documentation)
    host_url
        MMseqs2 server API url (defaults to public ColabFold server)
    user_agent
        User agent to send to MMseqs server (If None, will default to
        "evedesign/" + version)
    keep_tmp_dir
        If True, keep temporary directory with outputs
    tmpdir
        Optional path to local temporary directory

    Returns
    -------
    System with added sequences per entity
    """
    protein_entity_reps = [
        (idx, "".join(entity.rep))
        for idx, entity in enumerate(system)
        if entity.type == "protein" and entity.defined_sequence()
    ]
    if not protein_entity_reps:
        return system.copy()

    query_seqs_unique = []
    for _, seq in protein_entity_reps:
        if seq not in query_seqs_unique:
            query_seqs_unique.append(seq)

    if user_agent is None:
        user_agent = "evedesign/" + __version__

    if not use_pairing:
        pair_mode = "unpaired"
    else:
        pair_mode = pair_mode.lower()
        if pair_mode not in {"paired", "unpaired", "unpaired_paired"}:
            raise ValueError(f"Invalid pair_mode: {pair_mode}")

    need_unpaired = pair_mode in {"unpaired", "unpaired_paired"}
    need_paired = pair_mode in {"paired", "unpaired_paired"}

    tmpdir_ctx = None
    if keep_tmp_dir:
        tmpdir_path = Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="mmseqs_"))
        tmpdir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Keeping MMseqs2 output in {tmpdir_path}")
    else:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir_ctx.name)

    try:
        unpaired_a3m_lines = None
        if need_unpaired:
            unpaired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                tmpdir_path.joinpath("mmseqs_out"),
                use_env=use_env,
                use_filter=use_filter,
                use_templates=False,
                filter=filter,
                use_pairing=False,
                pairing_strategy=pairing_strategy,
                host_url=host_url,
                user_agent=user_agent,
            )

        paired_a3m_lines = None
        if need_paired and len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                tmpdir_path.joinpath("mmseqs_out"),
                use_env=use_env,
                use_filter=use_filter,
                use_templates=False,
                filter=filter,
                use_pairing=True,
                pairing_strategy=pairing_strategy,
                host_url=host_url,
                user_agent=user_agent,
            )
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    unpaired_entries_by_seq = {}
    if unpaired_a3m_lines is not None:
        unpaired_entries_by_seq = {
            seq: _parse_a3m(a3m_text)
            for seq, a3m_text in zip(query_seqs_unique, unpaired_a3m_lines)
        }

    paired_entries_by_seq = {}
    paired_keys = None
    if paired_a3m_lines is not None:
        paired_entries_by_seq = {
            seq: _parse_a3m(a3m_text)
            for seq, a3m_text in zip(query_seqs_unique, paired_a3m_lines)
        }
        paired_lengths = [len(entries) for entries in paired_entries_by_seq.values()]
        if paired_lengths:
            paired_len = min(paired_lengths)
            if any(length != paired_len for length in paired_lengths):
                logger.warning(
                    "Paired MSA lengths differ across chains; assigning keys to the first %d rows.",
                    paired_len,
                )
            paired_keys = [f"pair-{i}" for i in range(paired_len)]

    system = system.copy()
    for entity_idx, rep in protein_entity_reps:
        paired_entries = paired_entries_by_seq.get(rep, [])
        unpaired_entries = unpaired_entries_by_seq.get(rep, [])

        seqs = []
        if paired_entries:
            seqs.extend(
                _sequences_from_entries(paired_entries, keys=paired_keys).seqs
            )
        if unpaired_entries:
            seqs.extend(_sequences_from_entries(unpaired_entries).seqs)

        if seqs:
            system[entity_idx].sequences = Sequences(seqs, aligned=True, format="a3m")

    return system

