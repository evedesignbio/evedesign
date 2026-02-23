"""
Tools for clustering designs on sequence/structure/embedding level
"""
import subprocess
import tempfile
import math
from pathlib import Path

def cluster_sequences_mmseqs(
    sequences: list[str],
    target_num_clusters: int,
    num_iterations: int = 10,
    priorities: list[float] | None = None,
    mmseqs_path: str = "mmseqs"
) -> tuple[list[int], list[int]]:
    """
    Reduce sequences down to a specified number of clusters with MMseqs

    Parameters
    ----------
    sequences
        Corresponds to what is supplied to shell script as input.fasta
    target_num_clusters
        Target number of clusters
    num_iterations
        Maximum number of binary search splits to get as close to target_num_clusters
        as possible
    priorities
        Corresponds to priority.tsv; in same order as sequence list
    mmseqs_path
        Path to mmseqs binary (optional, defaults to assuming mmseqs is on $PATH)

    Returns
    -------
    Tuple containing
    (i) indices of picked sequences (list[int]) in input sequence list; each list element represents a cluster
    (ii) corresponding cluster indices (list[int]) for all input sequences, same length as input sequence list
    """
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Write sequences to a temporary FASTA file
        fasta = tempdir / "sequences.fasta"
        with fasta.open("w") as f:
            for i, sequence in enumerate(sequences):
                f.write(f">{i}\n{sequence}\n")

        # Write priorities to a temporary TSV file if provided
        if priorities:
            priority_file = tempdir / "priority.tsv"
            with priority_file.open("w") as f:
                for i, weight in enumerate(priorities):
                    f.write(f"{i}\t{weight}\n")
        else:
            priority_file = None

        # Create MMseqs DB
        db = tempdir / "sequences.db"
        base_res  = tempdir / "res"
        subprocess.run(
            [mmseqs_path, "createdb", str(fasta), str(db)],
            capture_output=True
        )
        
        # Binary search on --min-seq-id
        low = 0.00
        high = 0.99
        best_diff = math.inf
        best_thr = 0.0
        best_count = 0
        best_resdir = ""

        for i in range(num_iterations):
            thr = round((low + high)/2, 2)
            thr_str = f"{thr:.2f}"
            resdir  = f"{base_res}_{thr_str.replace('.', '')}"

            cmd = [
                mmseqs_path, 
                "cluster",
                str(db), 
                resdir, 
                str(tempdir),
                "--min-seq-id", str(thr),
                "--cluster-mode", "2", # chooses longest sequence in cluster as rep
                "--cov-mode", "1", # enforces larger coverage of member sequences
                "-c", "0.90",
            ]
            if priorities:
                cmd.extend(["--weights", str(priority_file)])
            
            subprocess.run(cmd, capture_output=True)

            # Number of clusters
            count = sum(1 for _ in open(f"{resdir}.index", "rb"))
            diff  = abs(count - target_num_clusters)

            # print(f"    → {count} clusters (target {target_num_clusters}; diff {diff} , best_diff {best_diff})")

            if diff < best_diff:
                best_thr = thr
                best_count = count
                best_diff = diff
                best_resdir = resdir

                if diff <= target_num_clusters * 0.01:
                    break

            # Adjust search window
            if count > target_num_clusters:
                high = thr
            else:
                low = thr

        # print(f"==> Best: thr={best_thr} → {best_count} clusters")

        # Save best result to final directory
        subprocess.run([mmseqs_path, "mvdb", best_resdir, base_res], capture_output=True)

        # Create cluster.tsv file mapping representatives to members
        tsv = tempdir / "cluster.tsv"
        subprocess.run(
            [mmseqs_path, "createtsv", str(db), str(db), base_res, tsv],
            capture_output=True
        )

        # Parse cluster.tsv to build return values
        rep_indices: list[int] = []
        cluster_assignments  = [-1] * len(sequences)
        rep2idx: dict[str, int] = {} # representative sequence ID -> cluster number

        with tsv.open() as f:
            for line in f:
                rep_id, mem_id = line.rstrip().split('\t')

                # Add representative to rep2idx and rep_indices 
                if rep_id not in rep2idx:
                    rep2idx[rep_id] = len(rep_indices)
                    rep_indices.append(int(rep_id))

                # Assign member to cluster
                cluster_id = rep2idx[rep_id]
                mem_idx = int(mem_id)
                cluster_assignments[mem_idx] = cluster_id

    # Check for any unassigned sequences (-1)
    unassigned = [i for i, cid in enumerate(cluster_assignments) if cid == -1]
    assert not unassigned, f"Unassigned sequences: {unassigned}"

    # Check representative indices are within bounds
    assert all(0 <= idx < len(cluster_assignments) for idx in rep_indices), "Representative index out of range"

    # Check that number of representatives matches cluster count
    assert len(rep_indices) == len(set(cluster_assignments)), (
        f"Cluster/rep mismatch: {len(set(cluster_assignments))} clusters vs {len(rep_indices)} reps"
    )

    return rep_indices, cluster_assignments
