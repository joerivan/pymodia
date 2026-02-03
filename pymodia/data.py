from typing import Any, Iterable, List
from .molecule import MoDiaMolecule
from .fragment import MoDiaFragment


class MoDiaData():
    """
    Class that combines the data to make the molecular orbital diagram
    """
    def __init__(
        self,
        molecule: MoDiaMolecule,
        fragment1: MoDiaFragment,
        fragment2: MoDiaFragment,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a MoDia data container.

        Parameters
        ----------
        molecule
            The molecule object containing molecular orbital energies,
            coefficients, and electron count.
        fragment1
            First molecular fragment (e.g. atom or group of atoms).
        fragment2
            Second molecular fragment.
        **kwargs
            Optional keyword arguments used to customize internal behavior
            or override default settings.
        """
        allowed_data = {'name', 'moe', 'orbc'}
        self.__dict__.update((k, v) for k, v in kwargs.items()
                             if k in allowed_data)

        self.molecule = molecule
        self.fragment1 = fragment1
        self.fragment2 = fragment2

        # When loading the molecular orbital (MO) energies, a copy is made of the
        # raw energies and used to set the energy labels. The user can now
        # overwrite the MO energies which will adjust the position where the
        # MOs are being plotted, but will conserve the energies. This allows
        # the user to make small adjustment to the energy levels to avoid
        # any form of overlapping.
        self.moe_labels: List[float] = [
            float(e) for e in self.molecule.state_energies
        ]

    def set_ao_energy(self, fragment_index: int,
                      energies: Iterable[float]) -> "MoDiaData":
        """
        Set the atomic orbital energies of fragment 1 or fragment 2.

        Parameters
        ----------
        fragment_index
            Either 0 or 1 corresponding to setting energies of fragment 1
            or fragment 2, respectively.
        energies
            Iterable containing the fragment-local orbital energies.

        """
        if fragment_index == 0:
            self.fragment1.state_energies = list(energies)
        elif fragment_index == 1:
            self.fragment2.state_energies = list(energies)
        else:
            raise ValueError("fragment_index must be either 0 or 1")

        return self

    def set_ao_energies(self, energies: Iterable[Iterable[float]]
                        ) -> "MoDiaData":
        """
        Set the atomic orbital energies of fragment 1 and fragment 2.

        Parameters
        ----------
        energies
            Iterable with two entries containing the energies of
            fragment 1 and fragment 2.

        """
        energies = list(energies)
        if len(energies) != 2:
            raise ValueError("energies must contain two entries")

        self.fragment1.state_energies = list(energies[0])
        self.fragment2.state_energies = list(energies[1])

        return self

    def set_moe(self, moe: Iterable[float]) -> "MoDiaData":
        """
        Overwrite MO energies. Used to adjust the position of the MO energies
        in the diagram while retaining the actual energy values.

        Parameters
        ----------
        moe
            Iterable of molecular orbital energies.
        """
        self.molecule.state_energies = list(moe)

        return self

    def from_json(self, json):
        """
        Reads json file and import data

        Parameters
        ----------
        json
            file path to json file with data

        """
        pass

    def save_json(self, path):
        """
        Saves data from MoDiaData object to json file

        Parameters
        ----------
        path
            path to save json file to

        """
        pass
