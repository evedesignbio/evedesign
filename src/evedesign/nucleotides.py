"""
Functionality related to physically making designed sequences (e.g. expressing proteins by computing DNA sequences)
"""

from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd
from evedesign.system import System, SystemInstance

CodonUsageTable = dict[str, dict[str, float]]

class ProteinToDnaOptimizer(ABC):
    """
    Abstract base class for methods that receive a set of designed protein sequences
    and create optimized DNA sequences that translate to the protein sequence

    Note: constructor parameters should follow DNAChiselCodonOptimizer implementation
    as much as possible
    """
    @abstractmethod
    def optimize(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int,
        upstream_dna: str,
        downstream_dna: str,
        reference: SystemInstance | None = None,
        reference_dna: str | None = None,
    ) -> pd.DataFrame:
        """
        Create optimized DNA sequences for protein entity instances
        (needs to be called once per protein entity in multi-entity systems)

        Notes:
        i) Returned sequences will *not* include upstream_dna and downstream_dna,
         this is solely used as context for parametrizing the codon optimization
         algorithm

        ii) The method will return duplicate DNA sequences for duplicate protein
         sequences (output is not deduplicated, this is responsibility of user)

        iii) User is entirely responsible for ensuring that generated
         DNA sequences are correctly inserted into an ORF with upstream_dna
         and downstream_dna; verifying this without the full plasmid
         context is not possible in general way in this function as we cannot
         make the assumption that start/stop codons are necessarily part of
         upstream_dna and downstream_dna, respectively (e.g. when cloning
         a domain into a longer protein)

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which codon-optimized DNA sequences should be created
        entity
            Index of protein entity for which DNA sequence should be created
        upstream_dna
            Fixed DNA sequence directly upstream of DNA that will be generated here
            (e.g. assembly handle)
        downstream_dna
            Fixed DNA sequence directly downstream of DNA that will be generated here
            (e.g. assembly handle)
        reference
            Instance that should be used as reference to generate new DNA sequences.
            If specified, will reuse codons from the reference in any non-indel positions
            (to keep DNA background as constant as possible where needed)
        reference_dna
            DNA sequence for reference sequence (must translate into reference). If not specified,
            the reference will be first codon-optimized on its own to create the DNA reference
            from which codons will be reused.

        Returns
        -------
        Dataframe (guaranteed to be of same length and in same order as supplied instances list)
        with columns :
         i. "rep" containing protein sequence as in instance
         ii. "dna" containing the optimized DNA sequence guaranteed to translate into "rep",
         iii. "score" with optimization score (should be set to NaN if not available)
        """
        pass
