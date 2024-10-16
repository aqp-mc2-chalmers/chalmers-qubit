import numpy as np
from typing import Optional, Union
from qutip import destroy, tensor
from qutip_qip.device import Model

__all__ = ["SarimnerModel"]


class SarimnerModel(Model):
    """
    Initializes a new quantum system simulation configuration.

    This method sets up the essential parameters and defaults needed for simulating a quantum system
    with specified qubit characteristics and interactions. It also initializes the internal state
    required for managing the system's dynamics, such as drift and controls, and prepares an empty noise
    model.

    Parameters
    ----------
    qubit_frequencies : list of float
        Frequencies of each qubit in GHz, defining the energy level spacings.
    anharmonicities : list of float
        Anharmonicities of each qubit in GHz, indicating the deviation from harmonic oscillator behavior.
    rotating_frame_frequencies : list of float, optional
        Frequencies defining the rotating frame for each qubit. Defaults to the frequencies of the qubits
        themselves if not provided.
    coupling_matrix : np.ndarray or int, optional
        Coupling matrix between qubits. If an integer is provided, it initializes a matrix filled with this
        integer in the upper triangular part. If not provided, the coupling effect is considered absent.
    dims : list of int, optional
        Dimensions for the state space of each qubit, defaulting to three levels (qutrits) per qubit if not specified.

    Raises
    ------
    ValueError
        If the lengths of `anharmonicities` does not match the number of qubits.
        If `coupling_matrix` is provided but is neither an integer nor a numpy.ndarray.

    Attributes
    ----------
    num_qubits : int
        Number of qubits.
    qubit_frequencies : list of float
        Qubit frequencies stored.
    anharmonicities : list of float
        Stored anharmonicities of each qubit.
    rotating_frame_frequencies : list of float
        Rotating frame frequencies used.
    coupling_matrix : np.ndarray
        Coupling matrix used for the simulation.
    dims : list of int
        Dimensions of each qubit's state space.
    spline_kind : str, optional
        Type of the coefficient interpolation. Default is "step_func"
        Note that they have different requirements for the length of ``coeff``.

        -"step_func":
        The coefficient will be treated as a step function.
        E.g. ``tlist=[0,1,2]`` and ``coeff=[3,2]``, means that the coefficient
        is 3 in t=[0,1) and 2 in t=[1,2). It requires
        ``len(coeff)=len(tlist)-1`` or ``len(coeff)=len(tlist)``, but
        in the second case the last element of ``coeff`` has no effect.

        -"cubic": Use cubic interpolation for the coefficient. It requires
        ``len(coeff)=len(tlist)``
    params : dict
        Dictionary holding system parameters for easy access.
    _drift : list
        Internal representation of the system's drift.
    _controls : dict
        Internal setup for system controls.
    _noise : list
        List initialized for adding noise models.
    """
    def __init__(self, transmon_dict: dict, coupling_dict: Optional[dict] = None, dim: int = 3):
        self.num_qubits = int(len(transmon_dict))
        # dimension of each subsystem
        self.dims = [dim] * self.num_qubits
        self.params = {
            "wq": self.qubit_frequencies,
            "alpha": self.anharmonicity,
            "wr": self.rotating_frame_frequencies,
            "coupling_matrix": self.coupling_matrix
        }

        # setup drift, controls an noise
        self._drift = self._set_up_drift()
        self._controls = self._set_up_controls()
        self._noise = []

    @staticmethod
    def _parse_dict(d: dict) -> dict:
        """Multiply the values of the dict with 2*pi.

        Args:
            d (dict): dictionary with frequencies.

        Returns:
            dict: dictionary where the frequencies that have been converted to radial frequencies.
        """
        def multiply_values(d: dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    multiply_values(value)
                elif isinstance(value, (int, float)):
                    # Multiply value by 2*pi to get radial frequency.
                    d[key] = 2 * np.pi * value
        # Create copies of the dictionaries to avoid modifying the original one
        new_dict = deepcopy(d)
        # Apply multiplication
        multiply_values(new_dict)
        return new_dict

    def _set_up_drift(self):
        drift = []
        for key, value in self.params["transmons"].items():
            destroy_op = destroy(self.dims[key])
            alpha = value["anharmonicity"]
            # We are simulating qubits in the rotating frame
            drift.append((alpha / 2 * destroy_op.dag()**2 * destroy_op**2, [key]))
        return drift

    def _set_up_controls(self):
        """
        Generate the Hamiltonians and save them in the attribute `controls`.
        """
        dims = self.dims
        controls = {}

        for m in range(num_qubits):
            destroy_op = destroy(dims[m])
            controls["x" + str(m)] = (destroy_op.dag() + destroy_op, [m])
            controls["y" + str(m)] = (1j*(destroy_op.dag() - destroy_op), [m])

        if self.coupling_matrix is not None:
            # Looping through non-zero elements of the coupling matrix
            for (m, n), value in np.ndenumerate(self.coupling_matrix):
                if value != 0:
                    destroy_op1 = destroy(dims[m])
                    destroy_op2 = destroy(dims[n])
                    op1 = tensor(destroy_op1.dag(), destroy_op2)
                    op2 = tensor(destroy_op1, destroy_op2.dag())
                    controls["(xx+yy)" + str(m) + str(n)] = (op1+op2, [m, n])
                    controls["(yx-xy)" + str(m) + str(n)] = (1j*(op1-op2), [m, n])

        return controls

    def get_control_latex(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        num_qubits = self.num_qubits
        labels = [
            {f"x{n}": r"$a_{" + f"{n}" + r"}^\dagger + a_{"
                + f"{n}" + r"}$" for n in range(num_qubits)},
            {f"y{n}": r"$i(a_{" + f"{n}" + r"}^\dagger - a_{"
             + f"{n}" + r"}$)" for n in range(num_qubits)},
        ]
        label_zz = {}

        for m in range(num_qubits - 1):
            for n in range(m + 1, num_qubits):
                label_zz[f"iswap{m}{n}"] = r"$iswap_{"+f"{m}{n}"+"}$"
                label_zz[f"cz_real{m}{n}"] = r"$cz_{" + f"{m}{n}" + "}$"
                label_zz[f"cz_imag{m}{n}"] = r"$\Im cz_{" + f"{m}{n}" + "}$"

        labels.append(label_zz)
        return labels
