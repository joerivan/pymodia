import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .fragment import MoDiaFragment
from .molecule import MoDiaMolecule

SUBSCRIPT_TRANS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPT_TRANS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _load_atom_data() -> Dict[str, Dict[str, object]]:
    """Load atomic reference data from the bundled JSON file."""
    data_path = os.path.join(os.path.dirname(__file__), 'atom_data.json')
    with open(data_path, "r") as f:
        return json.load(f)


def _find_by_atomic_number(
    data: Dict[str, Dict[str, object]],
    number: int,
) -> Optional[Dict[str, object]]:
    """
    Find an element dictionary by its atomic number.
    """
    return next(
        (value for value in data.values()
         if value.get("atomic_number") == number),
        None,
    )


def _extract_atoms(res: Dict[str, object]) -> List[int]:
    """
    Extract unique atom identifiers from a result payload.
    """
    atoms, idx = np.unique([n[1] for n in res['nuclei']], return_index=True)
    return list(atoms[np.argsort(idx)])


def _is_homonuclear_diatomic(res: Dict[str, object],
                             atoms: List[int]) -> bool:
    """Return True for a homonuclear diatomic molecule."""
    return len(atoms) == 1 and len(res['nuclei']) == 2


def _build_basis_mapping(res: Dict[str, object],
                         atoms: Iterable[int]
                         ) -> Tuple[Dict[int, Dict[int, int]],
                                    Dict[int, List[List[object]]]]:
    """
    Build basis-function mappings per atom.
    """
    mapping = {i: {} for i in atoms}
    ao_ids = {i: [] for i in atoms}

    for basis_index, cgf in enumerate(res['cgfs']):
        for nucleus in res['nuclei']:
            if np.allclose(cgf.p, nucleus[0]):
                c_ident = []
                for gto in cgf.gtos:
                    c_ident.extend([gto.c, gto.alpha, gto.l, gto.m, gto.n])

                found = False
                for ao_index, existing in enumerate(ao_ids[nucleus[1]]):
                    if np.allclose(existing, c_ident):
                        found = True
                        mapping[nucleus[1]][basis_index] = ao_index
                        break

                if not found:
                    mapping[nucleus[1]][basis_index] = len(ao_ids[nucleus[1]])
                    ao_ids[nucleus[1]].append(c_ident)
                break

    return mapping, ao_ids


def autobuild_from_pyqint(res: Dict[str, object],
                          name: Optional[str] = None
                          ) -> Tuple[MoDiaMolecule,
                                     MoDiaFragment,
                                     MoDiaFragment]:
    """
    Try to auto-build Molecule and Fragments from results object
    """
    atoms = _extract_atoms(res)

    # Sneaky hack to handle homonuclear diatomic molecules
    homonuclear_diatomic = _is_homonuclear_diatomic(res, atoms)
    if homonuclear_diatomic:
        atoms = [atoms[0], -1]
        res['nuclei'][1][1] = -1

    if len(atoms) != 2:
        raise ValueError(
            'Cannot autobuild from molecule with more than 2 distinct elements.'
        )

    data = _load_atom_data()
    mapping, _ = _build_basis_mapping(res, atoms)

    # in the case of homonuclear system, unspoof the second atom
    if homonuclear_diatomic:
        res['nuclei'][1][1] = res['nuclei'][0][1]

    # fragment 1
    el1 = _find_by_atomic_number(data, atoms[0])
    if el1 is None:
        raise ValueError(f"Unknown atom with atomic number {atoms[0]}")
    f1 = MoDiaFragment(
        el1['symbol'],
        el1['ao_energy'],
        el1['atomic_number'],
        mapping[atoms[0]],
        sublabel=superscript(el1['configuration']),
    )
                       
    # fragment 2
    atom2 = atoms[1] if atoms[1] != -1 else atoms[0]
    el2 = _find_by_atomic_number(data, atom2)  # for homonuclear systems
    if el2 is None:
        raise ValueError(f"Unknown atom with atomic number {atom2}")
    f2 = MoDiaFragment(
        el2['symbol'],
        el2['ao_energy'],
        el2['atomic_number'],
        mapping[atoms[1]],
        sublabel=superscript(el2['configuration']),
    )

    # molecule
    mol = MoDiaMolecule(name, res['orbe'], res['orbc'], res['nelec'])

    return mol, f1, f2


def subscript(string_in: str) -> str:
    """
    Function to turn all numbers in string to subscript
    """
    string_out = string_in.translate(SUBSCRIPT_TRANS)

    return string_out


def superscript(string_in: str) -> str:
    """
    Function to turn all numbers with ^in front in string to superscript
    """
    sup = False
    string_out = ""
    for element in string_in:
        if sup is True:
            sup_element = element.translate(SUPERSCRIPT_TRANS)
            sup = False
            string_out += sup_element
        elif element == "^":
            sup = True
        else:
            string_out += element

    return string_out
