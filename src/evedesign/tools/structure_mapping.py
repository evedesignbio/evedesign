from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.sequence import ProteinSequence, NucleotideSequence
from evedesign.structure import Structure
from evedesign.system import Entity

def map_structure_chain_to_entity_pairwise(
    entity: Entity,
    chain: Structure,
    gap_penalty: int | tuple[int, int] = (-10, -2),
    local: bool = True
):
    """
    Map a structure chain to an entity rep by sequence alignment
    so position numbering agrees to entity.

    Also consider using the functions in evedesign.tools.foldseek
    (find_structures_foldseek, filter_structures_foldseek, add_structures_foldseek)
    to search and map structures for your system (this function may however
    not return the exact PDB structure you want due to redundancy reduction
    in the database).

    Parameters
    ----------
    entity
        Entity to which the structure chain should be mapped
    chain
        Single-chain PDB structure (may not contain insertion codes, best extracted from model with
         use_author_fields=False
    gap_penalty
        Fixed gap penalty if single int, or affine penalty if tuple (gap open penalty, gap extension penalty).
        Values must be negative.
    local
        If True, run local alignment, otherwise use global alignment

    Returns
    -------
    Remapped structure chain with position indices consistent with entity
    """
    chain.single_chain_no_inscode_or_raise()

    if not entity.is_biopolymer() or not entity.defined_sequence():
        raise ValueError(
            "Entity must be a biopolymer and have defined sequence to use this function"
        )

    # extract structure residue table and one-letter sequence
    res_df = chain.res_df()
    if res_df.res_name_oneletter.isnull().any():
        raise ValueError(
            "Chain contains residues without one-letter code, this suggests non-biopolymer residues contained "
            "which cannot be mapped"
        )

    chain_seq = "".join(res_df.res_name_oneletter)
    entity_seq = "".join(entity.rep)

    if entity.type == "protein":
        chain_seq = ProteinSequence(chain_seq)
        entity_seq = ProteinSequence(entity_seq)
        matrix = SubstitutionMatrix.std_protein_matrix()
    else:
        chain_seq = NucleotideSequence(chain_seq)
        entity_seq = NucleotideSequence(entity_seq)
        matrix = SubstitutionMatrix.std_nucleotide_matrix()

    alignments = align_optimal(
        entity_seq, chain_seq, matrix, gap_penalty=gap_penalty, local=local, max_number=1
    )

    assert len(alignments) > 0

    # construct mapping from alignment
    mapping = {
        int(res_df.iloc[idx_chain]["res_id"]): int(idx_entity + entity.first_index)
        for idx_entity, idx_chain in zip(alignments[0].trace[:, 0], alignments[0].trace[:, 1])
        if idx_entity != -1 and idx_chain != -1
    }

    return chain.remap(mapping)