##############################################
# SimPrep the OE way to setup a simulation   #
#                                            #
# Copyright Mey-research 2021                #
#                                            #
#  Adapted from KinoML                       #
#                                            #
# Author: Antonia Mey <antonia.mey@ed.ac.uk> #
##############################################

import logging
from typing import List, Set, Union, Iterable

from openeye import oechem, oegrid, oespruce
import pandas as pd

def read_smiles(smiles: str) -> oechem.OEGraphMol:
    """
    Read molecule from a smiles string.
    Parameters
    ----------
    smiles: str
        Smiles string.
    Returns
    -------
    molecule: oechem.OEGraphMol
        A molecule as OpenEye molecules.
    """
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles)

    molecules = []
    for molecule in ims.GetOEMols():
        molecules.append(oechem.OEGraphMol(molecule))

    return molecules[0]

def read_molecules(path: str) -> List[oechem.OEGraphMol]:
    """
    Read molecules from a file.
    Parameters
    ----------
    path: str
        Path to molecule file.
    Returns
    -------
    molecules: list of oechem.OEGraphMol
        A List of molecules as OpenEye molecules.
    """
    from pathlib import Path

    path = str(Path(path).expanduser().resolve())
    suffix = path.split(".")[-1]
    molecules = []
    with oechem.oemolistream(path) as ifs:
        if suffix == "pdb":
            ifs.SetFlavor(
                oechem.OEFormat_PDB,
                oechem.OEIFlavor_PDB_Default
                | oechem.OEIFlavor_PDB_DATA
                | oechem.OEIFlavor_PDB_ALTLOC,
            )
        # add more flavors here if behavior should be different from `Default`
        for molecule in ifs.GetOEGraphMols():
            molecules.append(oechem.OEGraphMol(molecule))

    # TODO: returns empty list if something goes wrong
    return molecules

def read_electron_density(path: str) -> Union[oegrid.OESkewGrid, None]:
    """
    Read electron density from a file.
    Parameters
    ----------
    path: str
        Path to electron density file.
    Returns
    -------
    electron_density: oegrid.OESkewGrid or None
        A List of molecules as OpenEye molecules.
    """
    from pathlib import Path

    path = str(Path(path).expanduser().resolve())
    electron_density = oegrid.OESkewGrid()
    # TODO: different map formats
    if not oegrid.OEReadMTZ(path, electron_density, oegrid.OEMTZMapType_Fwt):
        electron_density = None

    # TODO: returns None if something goes wrong
    return electron_density

def write_molecules(molecules: List[oechem.OEGraphMol], path: str):
    """
    Save molecules to file.
    Parameters
    ----------
    molecules: list of oechem.OEGraphMol
        A list of OpenEye molecules for writing.
    path: str
        File path for saving molecules.
    """
    from pathlib import Path

    path = str(Path(path).expanduser().resolve())
    with oechem.oemolostream(path) as ofs:
        for molecule in molecules:
            oechem.OEWriteMolecule(ofs, molecule)
    return

def select_chain(molecule, chain_id):
    """
    Select a chain from an OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    chain_id: str
        Chain identifier.
    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the selected chain.
    """
    # do not change input mol
    selection = molecule.CreateCopy()

    # delete other chains
    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if residue.GetChainID() != chain_id:
            selection.DeleteAtom(atom)

    return selection

def select_altloc(molecule, altloc_id):
    """
    Select an alternate location from an OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    altloc_id: str
        Alternate location identifier.
    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the selected alternate location.
    """
    # External libraries
    from openeye import oechem

    # do not change input mol
    selection = molecule.CreateCopy()

    allowed_altloc_ids = [" ", altloc_id]

    # delete other alternate location
    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if oechem.OEResidue.GetAlternateLocation(residue) not in allowed_altloc_ids:
            selection.DeleteAtom(atom)

    return selection

def remove_non_protein(
    molecule: oechem.OEGraphMol,
    exceptions: Union[None, List[str]] = None,
    remove_water: bool = False,
) -> oechem.OEGraphMol:
    """
    Remove non-protein atoms from an OpenEye molecule.
    Parameters
    ----------
    molecule: oechem.OEGraphMol
        An OpenEye molecule holding a molecular structure.
    exceptions: None or list of str
        Exceptions that should not be removed.
    remove_water: bool
        If water should be removed.
    Returns
    -------
    selection: oechem.OEGraphMol
        An OpenEye molecule holding the filtered structure.
    """
    if exceptions is None:
        exceptions = []
    if remove_water is False:
        exceptions.append("HOH")

    # do not change input mol
    selection = molecule.CreateCopy()

    for atom in selection.GetAtoms():
        residue = oechem.OEAtomGetResidue(atom)
        if residue.IsHetAtom():
            if residue.GetName() not in exceptions:
                selection.DeleteAtom(atom)

    return selection

def _prepare_structure(
    structure: oechem.OEGraphMol,
    has_ligand: bool = False,
    electron_density: Union[oegrid.OESkewGrid, None] = None,
    loop_db: Union[str, None] = None,
    ligand_name: Union[str, None] = None,
    cap_termini: bool = True,
    real_termini: Union[List[int], None] = None,
) -> Union[oechem.OEDesignUnit, None]:
    """
    Prepare an OpenEye molecule holding a protein ligand complex for docking.
    Parameters
    ----------
    structure: oechem.OEGraphMol
        An OpenEye molecule holding a structure with protein and optionally a ligand.
    has_ligand: bool
        If structure contains a ligand that should be used in design unit generation.
    electron_density: oegrid.OESkewGrid
        An OpenEye grid holding the electron density.
    loop_db: str or None
        Path to OpenEye Spruce loop database. You can request a copy at
        https://www.eyesopen.com/database-downloads. A testing subset (3TPP) is available
        at https://docs.eyesopen.com/toolkits/python/sprucetk/examples_make_design_units.html.
    ligand_name: str or None
        The name of the ligand located in the binding pocket of interest.
    cap_termini: bool
        If termini should be capped with ACE and NME.
    real_termini: list of int or None
        Residue numbers of biologically real termini will not be capped with ACE and NME.
    Returns
    -------
    design_unit: oechem.OEDesignUnit or None
        An OpenEye design unit holding the prepared structure with the highest quality among all identified design
        units.
    """

    def _has_residue_number(atom, residue_numbers=real_termini):
        """Return True if atom matches any given residue number."""
        residue = oechem.OEAtomGetResidue(atom)
        return any(
            [
                True if residue.GetResidueNumber() == residue_number else False
                for residue_number in residue_numbers
            ]
        )

    # set design unit options
    structure_metadata = oespruce.OEStructureMetadata()
    design_unit_options = oespruce.OEMakeDesignUnitOptions()
    if cap_termini is False:
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    if loop_db is not None:
        from pathlib import Path

        loop_db = str(Path(loop_db).expanduser().resolve())
        design_unit_options.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    # cap all termini but biologically real termini
    if real_termini is not None and cap_termini is True:
        oespruce.OECapTermini(structure, oechem.PyAtomPredicate(_has_residue_number))
        # already capped, preserve biologically real termini
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
        design_unit_options.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)

    # make design units
    if has_ligand:
        if electron_density is None:
            design_units = list(
                oespruce.OEMakeDesignUnits(
                    structure, structure_metadata, design_unit_options
                )
            )
            # filter design units for ligand of interest
            if ligand_name is not None:
                design_units = [
                    design_unit
                    for design_unit in design_units
                    if ligand_name in design_unit.GetTitle()
                ]

        else:
            design_units = list(
                oespruce.OEMakeDesignUnits(
                    structure, electron_density, structure_metadata, design_unit_options
                )
            )
    else:
        design_units = list(
            oespruce.OEMakeBioDesignUnits(
                structure, structure_metadata, design_unit_options
            )
        )

    if len(design_units) >= 1:
        design_unit = design_units[0]
    else:
        # TODO: Returns None if something goes wrong
        return None

    # fix loop issues and missing OXT
    # _OEFixLoopIssues(design_unit)
    # _OEFixMissingOXT(design_unit)

    return design_unit