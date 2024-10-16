import numpy as np
from typing import Optional
from qutip import destroy, num, tensor
from qutip_qip.pulse import Pulse, Drift
from qutip_qip.noise import Noise

__all__ = ["DecoherenceNoise", "ZZCrossTalk"]

class DecoherenceNoise(Noise):
    """
    The decoherence on each qubit characterized by two time scales t1 and t2.

    Parameters
    ----------
    # TODO fix docstring

    Attributes
    ----------
    num_qubits: int
        Number of qubits.
    decoherence: dict
        Dictionary with t1 and t2 values for corresponing qubits.
    """

    def __init__(self, decoherence_dict:dict):
        self.decoherence = decoherence_dict

    def get_noisy_pulses(self, dims:list, pulses:Optional[Pulse]=None, systematic_noise:Optional[Pulse]=None):
        """
        Return the input pulses list with noise added and
        the pulse independent noise in a dummy :class:`.Pulse` object.

        Parameters
        ----------
        dims: list
            The dimension of the components system.
        pulses : list of :class:`.Pulse`
            The input pulses. The noise will be added to pulses in this list.
        systematic_noise : :class:`.Pulse`
            The dummy pulse with no ideal control element.

        Returns
        -------
        noisy_pulses: list of :class:`.Pulse`
            Noisy pulses.
        systematic_noise : :class:`.Pulse`
            The dummy pulse representing pulse-independent noise.
        """
        if systematic_noise is None:
            systematic_noise = Pulse(None, None, label="system")

        for key, value in self.decoherence.items():
            t1 = value["t1"]
            t2 = value["t2"]
            if t1 is not None:
                op = 1 / np.sqrt(t1) * destroy(dims[key])
                systematic_noise.add_lindblad_noise(op, key, coeff=True)
            if t2 is not None:
                # Keep the total dephasing ~ exp(-t/t2)
                if t1 is not None:
                    if 2 * t1 < t2:
                        raise ValueError(
                            "t1={}, t2={} does not fulfill " "2*t1>t2".format(t1, t2)
                        )
                    T2_eff = 1.0 / (1.0 / t2 - 1.0 / 2.0 / t1)
                else:
                    T2_eff = t2
                op = 1 / np.sqrt(2 * T2_eff) * 2 * num(dims[key])
                systematic_noise.add_lindblad_noise(op, key, coeff=True)
        return pulses, systematic_noise


class ZZCrossTalk(Noise):
    """
    An always-on ZZ cross talk noise with the corresponding coefficient
    on each pair of qubits.

    Parameters
    ----------
    cross_talk_dict: dict
        Cross-talk dictionary where key (i,j) corresponds to the
        cross-talk strength between qubit `i` and `j`.
    dims: list, optional
        The dimension of the components system, the default value is
        [3,3...,3].

    Attributes
    ----------
    cross_talk_dict: dict
        Cross-talk matrix.
    num_qubits: int
        Number of qubits.
    """

    def __init__(self, cross_talk_dict: dict):
        self.cross_talk_dict = cross_talk_dict

    def get_noisy_pulses(self, dims:list, pulses:Optional[Pulse]=None, systematic_noise:Optional[Pulse]=None):
        """
        Return the input pulses list with noise added and
        the pulse independent noise in a dummy :class:`.Pulse` object.

        Parameters
        ----------
        dims: list
            The dimension of the components system, the default value is
            [3,3...,3] for qutrit system.
        pulses : list of :class:`.Pulse`
            The input pulses. The noise will be added to pulses in this list.
        systematic_noise : :class:`.Pulse`
            The dummy pulse with no ideal control element.

        Returns
        -------
        pulses: list of :class:`.Pulse`
            Noisy pulses.
        systematic_noise : :class:`.Pulse`
            The dummy pulse representing pulse-independent noise.
        """
        if systematic_noise is None:
            systematic_noise = Pulse(None, None, label="system")

        for (key1, key2), value in self.cross_talk_dict.items():
            d1 = dims[key1]
            d2 = dims[key2]

            zz_op = tensor(num(d1), num(d2))
            zz_coeff = 2*np.pi*value

            systematic_noise.add_control_noise(
                zz_coeff * zz_op,
                targets=[key1, key2],
                tlist=None,
                coeff=True,
            )
        return pulses, systematic_noise
