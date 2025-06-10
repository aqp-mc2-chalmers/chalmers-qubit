import numpy as np
from copy import deepcopy
from typing import Optional, Union
from qutip import destroy, tensor, basis
from qutip_qip.device import Model

__all__ = ["SarimnerModel"]


class SarimnerModel(Model):
    """
    Initializes a new quantum system simulation configuration for the Sarimner model.

    This class sets up the essential parameters and structure needed for simulating a quantum system
    with specified transmon qubit characteristics and couplings. It initializes the internal state
    required for managing the system's dynamics, including drift and controls.

    Parameters
    ----------
    transmon_dict : dict
        A dictionary containing the parameters for each transmon qubit. The keys are qubit
        identifiers, and the values are dictionaries with qubit properties (e.g., frequency,
        anharmonicity).
    coupling_dict : Optional[dict], default=None
        A dictionary specifying the couplings between qubits. If None, no couplings are considered.
    dim : int, default=3
        The dimension of the Hilbert space for each qubit (default is 3 for qutrit).

    Attributes
    ----------
    num_qubits : int
        The number of qubits in the system.
    dims : list of int
        A list specifying the dimension of each qubit's Hilbert space.
    params : dict
        A dictionary containing all system parameters, including transmon properties and couplings.
    _drift : list
        Internal representation of the system's drift Hamiltonian.
    _controls : dict
        Internal setup for system control Hamiltonians.
    _noise : list
        An empty list initialized for potential future addition of noise models.

    Methods
    -------
    _parse_dict(input_dict: dict) -> dict
        Internal method to parse and validate input dictionaries.
    _set_up_drift() -> list
        Internal method to set up the drift Hamiltonian.
    _set_up_controls() -> dict
        Internal method to set up the control Hamiltonians.

    Notes
    -----
    - The `transmon_dict` should contain parameters for each qubit, such as frequency and anharmonicity.
    - The `coupling_dict`, if provided, should specify the coupling strengths between qubit pairs.
    - The model uses a fixed dimension (`dim`) for all qubits, defaulting to qutrit (3-level) systems.
    """
    def __init__(self, transmon_dict: dict, coupling_dict: Optional[dict] = None, dim: int = 3):
        self.num_qubits = int(len(transmon_dict))
        # dimension of each subsystem
        self.dims = [dim] * self.num_qubits
        self.params = {
            "transmons": self._parse_dict(transmon_dict),
            "couplings": self._parse_dict(coupling_dict) if coupling_dict is not None else None,
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

        for key in self.params["transmons"].keys():
            destroy_op = destroy(dims[key])
            controls["x" + str(key)] = (destroy_op.dag() + destroy_op, [key])
            controls["y" + str(key)] = (1j*(destroy_op.dag() - destroy_op), [key])

        if self.params["couplings"] is not None:
            for (key1, key2), value in self.params["couplings"].items():
                # Create basis states
                ket01 = tensor(basis(self.dims[key1],0), basis(self.dims[key2],1))
                ket10 = tensor(basis(self.dims[key1],1), basis(self.dims[key2],0))
                ket11 = tensor(basis(self.dims[key1],1), basis(self.dims[key2],1))
                ket20 = tensor(basis(self.dims[key1],2), basis(self.dims[key2],0))

                g = value # coupling strength
                iswap_op = g * (ket01*ket10.dag() + ket10*ket01.dag())
                cz_op_real = np.sqrt(2) * g * (ket11*ket20.dag() + ket20*ket11.dag())
                cz_op_imag = 1j * np.sqrt(2) * g * (ket11*ket20.dag() - ket20*ket11.dag())

                controls["iswap" + str(key1) + str(key2)] = (iswap_op, [key1, key2])
                controls["cz_real" + str(key1) + str(key2)] = (cz_op_real, [key1, key2])
                controls["cz_imag" + str(key1) + str(key2)] = (cz_op_imag, [key1, key2])

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
            {f"x{n}": "$sx_{" + f"{n}" + "}$" for n in range(num_qubits)},
            {f"y{n}": "$sy_{" + f"{n}" + "}$" for n in range(num_qubits)},
        ]
        label_zz = {}

        for m in range(num_qubits - 1):
            for n in range(m + 1, num_qubits):
                label_zz[f"iswap{m}{n}"] = r"$iswap_{"+f"{m}{n}"+"}$"
                label_zz[f"cz_real{m}{n}"] = r"$cz_{" + f"{m}{n}" + "}$"
                label_zz[f"cz_imag{m}{n}"] = r"$\Im cz_{" + f"{m}{n}" + "}$"

        labels.append(label_zz)
        return labels
