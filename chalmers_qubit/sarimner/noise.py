import numpy as np
from typing import Optional
from qutip import destroy, num, tensor
from qutip_qip.pulse import Pulse, Drift
from qutip_qip.noise import Noise

__all__ = ["DecoherenceNoise", "ZZCrossTalk"]


class DecoherenceNoise(Noise):
    """
    Represents decoherence noise in a quantum system, characterized by T1 and T2 times for each qubit.

    This class models the decoherence effects on qubits in a quantum system. It allows for
    specifying individual T1 (relaxation) and T2 (dephasing) times for each qubit in the system.

    Parameters
    ----------
    decoherence_dict : dict
        A dictionary specifying the decoherence parameters for each qubit. The keys should be
        qubit identifiers, and the values should be dictionaries containing 't1' and 't2' keys
        with their respective time values in nanoseconds.

    Attributes
    ----------
    decoherence : dict
        A dictionary storing the T1 and T2 values for each qubit in the system.

    Example
    -------
    ```python
    decoherence_dict = {
        0: {'t1': 50e3, 't2': 70e3}, # values for the 0th qubit
        1: {'t1': 45e3, 't2': 60e3} # values for the 1st qubit
    }
    noise = [DecoherenceNoise(decoherence_dict)]
    ```

    Notes
    -----
    - T1 represents the relaxation time, which characterizes the time scale for energy dissipation.
    - T2 represents the dephasing time, which characterizes the time scale for loss of phase coherence.
    - T2 is always less than or equal to 2*T1.
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
    A noise model representing always-on ZZ cross talk between qubits,
    characterized by a cross-talk coefficient for each pair of qubits.

    Parameters
    ----------
    cross_talk_dict : dict
        A dictionary specifying the cross-talk strength between qubits.
        The keys are tuples (i, j), where `i` and `j` are qubit indices,
        and the corresponding value is the ZZ interaction strength
        between qubit `i` and qubit `j`.

    Attributes
    ----------
    cross_talk_dict : dict
        A dictionary representing the ZZ cross-talk interaction strengths
        between qubit pairs.

    Notes
    -----
    - This model assumes that cross talk is always on, meaning it
      continuously affects the qubits throughout their operation.

    Example
    -------
    ```python	
    cross_talk_dict = {
        (0, 1): 0.01,  # ZZ interaction strength between qubit 0 and qubit 1
        (1, 2): 0.02,  # ZZ interaction strength between qubit 1 and qubit 2
    }
    noise = [ZZCrossTalk(cross_talk)]
    ```
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
