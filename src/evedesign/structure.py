"""
Biomolecular structure-related functionality (PDB structures etc.)

Thin wrapper around biotite structures for more convenient, unified access to PDBx and PDB formats; also
decouples internal codebase from biotite API through abstractions that we know work well from EVcouplings
package development.
"""
from copy import deepcopy
from typing import Literal, TextIO, BinaryIO, Self, Sequence
import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
from biotite.structure import AtomArray
from evedesign.constants import RESIDUE_MAX_SASA, AA3_to_AA1
from evedesign.utils import map_array

# allow to receive single chain, or map from identifier to single chain or list of chains
StructureFormat = Literal["bcif", "cif", "pdb"]
_INVALID_FORMAT_MSG = "Invalid PDB file type, options are: 'bcif', 'cif', 'pdb'"


class Structure:
    def __init__(self, atom_array: struc.AtomArray):
        self.atom_array = atom_array

        # dataframe representation (atom- and residue-level)
        self._atom_df = None
        self._res_df = None

    def copy(self) -> Self:
        """
        Create deep copy of current instance
        (will reset cached atom and residue dataframes)

        Returns
        -------
        Deep copy of instance
        """
        return Structure(
            deepcopy(self.atom_array)
        )

    def atom_df(
        self,
        sasa: bool = False,
    ) -> pd.DataFrame:
        """
        Return dataframe representation of model on *atom* level

        Note: do not mutate returned dataframe in place

        Parameters
        ----------
        sasa
            If true, add column with residue solvent accessibility
            (will be retained on subsequent calls even if sasa = False)

        Returns
        -------
        Dataframe representation of model
        """
        # built dataframe if not already existing
        if self._atom_df is None:
            cols = self.atom_array.get_annotation_categories()
            _df_raw = {
                col: self.atom_array.get_annotation(col) for col in cols
            }

            # unpack 3D coordinates into separate columns
            for i, col in enumerate(["x", "y", "z"]):
                _df_raw[col] = self.atom_array.coord[:, i]

            # replace masked values in custom fields not handled by biotite
            self._atom_df = pd.DataFrame(
                _df_raw
            ).replace(
                {"?": pd.NA, ".": pd.NA}
            )

            # the following custom fields do not have mask values replaced by biotite, do this here
            for col in [
                "label_entity_id", "label_seq_id", "auth_seq_id"
            ]:
                if col in self._atom_df.columns:
                    self._atom_df.loc[:, col] = self._atom_df.loc[:, col].astype("Int64")

        # create residue-level dataframe; can only extract secondary structure annotation on this level in biotite
        if self._res_df is None:
            _res_ids, _res_names = struc.get_residues(self.atom_array)
            _res_starts = struc.get_residue_starts(self.atom_array)
            _sse = struc.annotate_sse(self.atom_array)
            _chain_ids = self.atom_array[_res_starts].chain_id

            # need to add _ins_code for cases where author numbering is used
            # (otherwise residue id alone is not unique);
            _ins_code = self.atom_array[_res_starts].ins_code

            # use 3-state DSSP nomenclature for compatibility with everyone else (even if different algorithm)
            _sse = struc.annotate_sse(self.atom_array)

            assert len(_res_ids) == len(_res_names) == len(_res_starts) == len(_sse)

            self._res_df = pd.DataFrame({
                "res_id": _res_ids,
                "res_name": _res_names,
                "ins_code": _ins_code,
                "chain_id": _chain_ids,
                "sse": _sse,
                "atom_df_start_idx": _res_starts,
            })

            self._res_df.loc[:, "sse"] = self._res_df.loc[:, "sse"].replace({
                "a": "H",
                "b": "E",
                "c": "C",
                "": pd.NA,
            })

            self._res_df.loc[:, "res_name_oneletter"] = self._res_df.res_name.map(AA3_to_AA1)

        # add solvent accessibility
        if sasa and "sasa" not in self._atom_df.columns:
            # compute on per-atom level first
            sasa_vec = struc.sasa(self.atom_array)
            assert len(sasa_vec) == len(self._atom_df)
            self._atom_df.loc[:, "sasa"] = sasa_vec

            # group on residue level (drop null values first or aggregation will create zero value)
            sasa_res_sum = self._atom_df.dropna(subset=["sasa"]).groupby(
                ["res_id", "ins_code", "chain_id", "res_name"], sort=False
            )["sasa"].sum().to_frame("sasa_residue").reset_index()

            # annotate maximum accessibility for residue (extended peptide) to compute relative accessibility
            max_sasa = sasa_res_sum.res_name.map(RESIDUE_MAX_SASA)
            sasa_res_sum.loc[:, "rel_sasa_residue"] = sasa_res_sum["sasa_residue"] / max_sasa

            # merge back to residue-level dataframe
            self._res_df = self._res_df.merge(
                sasa_res_sum.drop(["res_name"], axis=1),
                how="left",
                on=["res_id", "ins_code", "chain_id"]
            )

        return self._atom_df

    def res_df(
        self,
        sasa: bool = False,
    ) -> pd.DataFrame:
        """
        Return dataframe representation of model on *residue* level

        Note: do not mutate returned dataframe in place

        Parameters
        ----------
        sasa
            If true, add column solvent accessibility
            (will be retained on subsequent calls even if sasa = False)

        Returns
        -------
        Dataframe representation of model
        """
        # create/update all internal representations including atom dataframe

        self.atom_df(sasa=sasa)
        return self._res_df

    def chains(self) -> list[str]:
        """
        List available chains in model (whether these are author
        or label chain IDs is determined by use_author_fields parameter
        when calling get_model / get_assembly_model

        Returns
        -------
        Alphabetically sorted list of chain identifiers
        """
        # return all available chains
        return sorted(
            set(struc.get_chains(self.atom_array))
        )

    def get_chain(
        self,
        chain_id: str
    ) -> Self:
        """
        Extract single chain from model. Use chains() to list
        available chain identifiers in model.

        Parameters
        ----------
        chain_id
            Identifier of chain to extract

        Returns
        -------
        Model limited to extracted chain
        """
        valid_chains = self.chains()
        if chain_id not in valid_chains:
            raise ValueError(
                f"Invalid chain identifier, valid options are: {valid_chains}"
            )

        return type(self)(
            self.atom_array[self.atom_array.chain_id == chain_id]
        )

    def single_chain_no_inscode_or_raise(self):
        """
        Verify that structure only contains a single chain and
        no insertion codes
        """
        unique_ins_codes = np.unique(self.atom_array.ins_code)
        if len(unique_ins_codes) != 1 or unique_ins_codes[0] != "":
            raise ValueError(
                "Model contains insertion codes, cannot remap numbering"
            )

        unique_chains = np.unique(self.atom_array.chain_id)
        if len(unique_chains) != 1:
            raise ValueError(
                "Can only map a unique chain identifier"
            )

    def remap(
        self,
        mapping: dict[int, int],
        chain_id: str | None = None
    ) -> Self:
        """
        Remap numbering of a single-chain model. Will raise a ValueError if more than one chain
        or insertion codes are present to avoid ambiguity in mapping.

        Parameters
        ----------
        mapping
            Map from current numbering to new numbering. Any residues not contained in the
            mapping will be removed from the model
        chain_id
            If not None, update chain identifier to this value (otherwise will keep as is)

        Returns
        -------
        Single-chain model with updated numbering
        """
        self.single_chain_no_inscode_or_raise()

        # determine which positions will be mapped, discard others
        mapped_pos = np.isin(self.atom_array.res_id, list(mapping))

        # make a copy and update residue identifiers
        new_chain: AtomArray = self.atom_array[mapped_pos].copy()  # noqa
        new_chain.res_id[:] = map_array(new_chain.res_id, mapping)

        # update chain if given
        if chain_id is not None:
            new_chain.chain_id[:] = chain_id

        return type(self)(new_chain)

    def represents(
        self,
        positions: Sequence[int],
        sequence: Sequence[str] | None = None,
        allow_missing: bool = True,
        raise_invalid: bool = False,
    ) -> bool:
        """
        Verify if 3D model is a valid representative of a sequence

        Parameters
        ----------
        positions
            Sequence positions that need to match with structure (not checking symbols/residues themselves)
        sequence:
            If specified, structure residues needs to match this sequence
        allow_missing
            If true allow sequence positions to be undefined in structure model
            (but in no case may structure define positions that are not present in the sequence)
        raise_invalid
            If true and structure is not a valid representative of sequence, raise a ValueError

        Returns
        -------
        True if structure is a valid representation of sequence, false otherwise
        """
        if sequence is not None and len(positions) != len(sequence):
            raise ValueError(
                "Parameters positions and sequence must have same length"
            )

        df = self.res_df()

        # first ensure that model only contains a single chain or comparison is meaningless
        valid = df.chain_id.nunique() == 1

        # ensure there are no insertion codes
        unique_ins_code = df.ins_code.unique()
        valid = valid and len(unique_ins_code) == 1 and unique_ins_code[0] == ""

        # check that all positions in structure are part of reference sequence to check against
        valid = valid and df.res_id.isin(positions).all()

        # also check reverse unless we allow missing positions in structure
        valid = valid and (
            allow_missing or np.isin(np.array(positions), df.res_id.values).all()
        )

        # check sequence if specified
        if sequence is not None:
            mismatch = pd.DataFrame({
                "res_id": positions,
                "res_name_compare": sequence
            }).merge(
                df, on="res_id"
            ).query(
                "res_name_compare != res_name_oneletter"
            )

            valid = valid and len(mismatch) == 0

        if not valid and raise_invalid:
            raise ValueError(
                "Model is not a valid structure representative of sequence"
            )

        return valid

    @classmethod
    def concat(
        cls,
        models: list[Self]
    ) -> Self:
        """
        Create new model by concatenating given models

        Note: Caller is responsible for making sure there are no duplicated residues or chains

        Parameters
        ----------
        models
            Concatenate these models into new model

        Returns
        -------
        Concatenated model
        """
        return cls(
            struc.concatenate([
                model.atom_array for model in models
            ])
        )

    def to_file(
        self,
        file: TextIO | BinaryIO | str,
        format: StructureFormat="cif"  # noqa
    ) -> None:
        """
        Save model coordinates to a file.

        Note that the underlying library biotite does not preserve both author and
        label ids/numbering Users of the written file should always ensure that
        identifiers in structure are treated appropriately (i.e. not mix numbering types).

        Parameters
        ----------
        file
            File-like object or path to file
        format
            PDB format to write file as
        """
        if format == "cif":
            out_file = pdbx.CIFFile()
            pdbx.set_structure(out_file, self.atom_array)
        elif format == "bcif":
            out_file = pdbx.BinaryCIFFile()
            pdbx.set_structure(out_file, self.atom_array)
        elif format == "pdb":
            out_file = pdb.PDBFile()
            pdb.set_structure(out_file, self.atom_array)
        else:
            raise ValueError(_INVALID_FORMAT_MSG)

        out_file.write(file)


class StructureFile:
    """
    Biomolecular 3D structure
    """
    # extra fields that can be added for any structure type, we retrieve these by default
    _extra_fields = [
        "atom_id", "b_factor", "occupancy", "charge"
    ]

    # extra fields only available through CIF/PDBx formats to get access to full numbering information
    _extra_fields_pdbx = _extra_fields + [
        "label_entity_id", "label_asym_id", "auth_asym_id", "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code"
    ]

    def __init__(
        self,
        file: TextIO | BinaryIO | str,
        format: StructureFormat,  # noqa
    ):
        """
        Load existing PDB structure

        Parameters
        ----------
        file
            Path or file handle to read structure from
        format
            Indicates whether provided structure is mmCIF ('cif'), binaryCIF ('bcif'),
            or legacy PDB format ('pdb')
        """

        if format == "bcif":
            self.data = pdbx.BinaryCIFFile.read(file)
            self.pdbx = True
        elif format == "cif":
            self.data = pdbx.CIFFile.read(file)
            self.pdbx = True
        elif format == "pdb":
            self.data = pdb.PDBFile.read(file)
            self.pdbx = False
        else:
            raise ValueError(_INVALID_FORMAT_MSG)

    @classmethod
    def from_id(cls, pdb_id: str):
        """
        Load structure by fetching from RCSB PDB

        Parameters
        ----------
        pdb_id
            4-letter PDB identifier code to fetch

        Returns
        -------
        Loaded structure
        """
        # fetch as bCIF by default for quicker fetching/loading
        pdb_data = rcsb.fetch(pdb_id, format="cif")
        return cls(pdb_data, format="cif")

    def assemblies(self) -> dict[str, str | None]:
        """
        List biological assemblies for structure

        Returns
        -------
        Mapping from assembly identifier to description (description
        will be None for old PDB format)
        """
        if self.pdbx:
            return pdbx.list_assemblies(self.data)
        else:
            # for pdb, we only get list of assembly identifiers, so turn into dictionary
            return {
                id_: None for id_ in pdb.list_assemblies(self.data)
            }

    def model_count(self) -> int:
        """
        Return number of models contained in structure file

        Returns
        -------
        Number of models
        """
        if self.pdbx:
            return pdbx.get_model_count(self.data)
        else:
            return pdb.get_model_count(self.data)

    def get_model(
        self,
        model: int = 1,
        altloc: Literal["first", "occupancy", "all"] = "occupancy",
        use_author_fields: bool = True,
        include_bonds: bool = False,
        add_all_fields: bool = False,
    ) -> Structure:
        """
        Extract one model from asymmetric unit

        Parameters
        ----------
        model
            Number of model to extract (numbering starts from 1,
            check model_count() for total number of models)
        altloc
            Multiple location (altloc) per atom resolution strategy (see biotite documentation for details)
        use_author_fields
            If True, use author chain and residue numbering (possibly containing insertion codes),
            otherwise use label_seq_id and label_asym_id (the latter option available for PDBx-based formats
            (cif/bcif) only)
        include_bonds
            If True, include bond list (see biotite documentation for details)
        add_all_fields
            Extract all identifier columns (author and label ids), not just main numbering
            selected with use_author_fields. Available for PDBx-based formats (cif/bcif) only.

        Returns
        -------
        Extracted model
        """

        if (not use_author_fields or add_all_fields) and not self.pdbx:
            raise ValueError(
                "Legacy PDB format only supports use_author_fields = True and add_all_fields = False"
            )

        if self.pdbx:
            coords = pdbx.get_structure(
                self.data,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields_pdbx if add_all_fields else self._extra_fields,
                use_author_fields=use_author_fields,
                include_bonds=include_bonds
            )

            # remove insertion code, this belongs to author numbering and
            # only creates confusion if label_seq_ids are used
            if not use_author_fields:
                coords.ins_code[:] = ""
        else:
            coords = pdb.get_structure(
                self.data,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields,
                include_bonds=include_bonds
            )

        return Structure(coords)

    def get_assembly_model(
        self,
        assembly_id: str | None = None,
        model: int = 1,
        altloc: Literal["first", "occupancy", "all"] = "occupancy",
        use_author_fields: bool = True,
        include_bonds: bool = False,
        add_all_fields: bool = False,
        sym_id_to_chain_id: bool = True,
    ) -> Structure:
        """
        Extract one model from biological assembly

        Parameters
        ----------
        assembly_id
            Assembly to extract, check for available assemblies with assemblies() method
        model
            Number of model to extract (numbering starts from 1,
            check model_count() for total number of models)
        altloc
            Multiple location (altloc) per atom resolution strategy (see biotite documentation for details)
        use_author_fields
            If True, use author chain and residue numbering (possibly containing insertion codes),
            otherwise use label_seq_id and label_asym_id (the latter option available for PDBx-based formats
            (cif/bcif) only)
        include_bonds
            If True, include bond list (see biotite documentation for details)
        add_all_fields
            Extract all identifier columns (author and label ids), not just main numbering
            selected with use_author_fields. Available for PDBx-based formats (cif/bcif) only.
        sym_id_to_chain_id
            If True, merge sym_id into chain ID so chain_ids become A, A-2, ...
            instead of listing all coordinates for copies under chain A

        Returns
        -------
        Extracted model from biological assembly
        """
        if (not use_author_fields or add_all_fields) and not self.pdbx:
            raise ValueError(
                "Legacy PDB format only supports use_author_fields = True and add_all_fields = False"
            )

        if self.pdbx:
            coords = pdbx.get_assembly(
                self.data,
                assembly_id=assembly_id,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields_pdbx if add_all_fields else self._extra_fields,
                use_author_fields=use_author_fields,
                include_bonds=include_bonds
            )

            # remove insertion code, this belongs to author numbering and
            # only creates confusion if label_seq_ids are used
            if not use_author_fields:
                coords.ins_code[:] = ""
        else:
            coords = pdb.get_assembly(
                self.data,
                assembly_id=assembly_id,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields,
                include_bonds=include_bonds
            )

        if sym_id_to_chain_id and "sym_id" in coords.get_annotation_categories():
            coords.chain_id[:] = [
                (f"{chain_id}-{sym_id + 1}" if sym_id > 0 else chain_id)
                for chain_id, sym_id in zip(coords.chain_id, coords.sym_id)
            ]

        return Structure(coords)

    def sequences(self) -> dict[str, str]:
        """
        Extract biopolymer chain sequences

        Returns
        -------
        Map from chain identifier to sequences
        """
        if self.pdbx:
            return {
                id_: str(seq) for id_, seq in pdbx.get_sequence(self.data).items()
            }
        else:
            raise NotImplementedError(
                "Sequences not available for legacy PDB format (not implemented in biotite)"
            )

