import numpy as np

from qutip import destroy, num
from qutip_qip.pulse import Pulse
from chalmers_qubit.base.noise import Noise

class DecoherenceNoise(Noise):
    """
    The decoherence on each qubit characterized by two time scales t1 and t2.

    Parameters
    ----------
    t1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits

    Attributes
    ----------
    t1: float or list
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list
        The indices of qubits that are acted on.
    """

    def __init__(self, num_qubits, dims, t1, t2):
        self.num_qubits = num_qubits
        self.dims = dims
        self.t1 = t1
        self.t2 = t2

    def get_noisy_pulses(self, dims, pulses=None, systematic_noise=None):
        """
        Return the input pulses list with noise added and
        the pulse independent noise in a dummy :class:`.Pulse` object.

        Parameters
        ----------
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.
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
        systematic_noise = Pulse(
            None, None, label="systematic_noise", spline_kind=None
        )
        for qu_ind in range(self.num_qubits):
            t1 = self.t1[qu_ind]
            t2 = self.t2[qu_ind]
            if t1 is not None:
                op = 1 / np.sqrt(t1) * destroy(dims[qu_ind])
                systematic_noise.add_lindblad_noise(op, qu_ind, coeff=True)
            if t2 is not None:
                # Keep the total dephasing ~ exp(-t/t2)
                if t1 is not None:
                    if 2 * t1 < t2:
                        raise ValueError(
                            "t1={}, t2={} does not fulfill "
                            "2*t1>t2".format(t1, t2)
                        )
                    T2_eff = 1.0 / (1.0 / t2 - 1.0 / 2.0 / t1)
                else:
                    T2_eff = t2
                op = 1 / np.sqrt(2 * T2_eff) * 2 * num(dims[qu_ind])
                systematic_noise.add_lindblad_noise(op, qu_ind, coeff=True)
        return pulses, systematic_noise