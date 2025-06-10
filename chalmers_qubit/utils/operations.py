import numpy as np
from qutip import Qobj


def project_on_qubit(rho: Qobj) -> Qobj:
    """
    Project a given quantum object (density matrix or state vector) onto the computational basis of qubits.

    Parameters
    ----------
    rho : Qobj
        The input quantum object, either a density matrix ('oper' type) or a state vector.

    Returns
    -------
    qubit_state : Qobj
        The projected quantum object onto the computational basis (0 and 1 states) of qubits.
    """
    if rho.dims is None:
        raise ValueError("The 'dims' attribute of the input Qobj is None. Please ensure it is correctly initialized.")

    dims = rho.dims
    num_qubits = len(dims[0])
    N = dims[0][0]
    # Generate all computational basis states in base N
    base_N = [np.base_repr(i, base=N) for i in range(N**num_qubits)]
    # Generate all computational basis states in base 2
    base_2 = [np.base_repr(i, base=2) for i in range(2**num_qubits)]
    # Find indices of computational basis states in base N list
    l = [base_N.index(i) for i in base_2]
    if rho.type == 'oper':
        # If rho is a density matrix, project it onto the computational basis
        qubit_state = Qobj([[rho.full()[j, i] for i in l] for j in l], dims=[
                           [2] * num_qubits, [2] * num_qubits])
    else:
        # If rho is a state vector, project it onto the computational basis
        qubit_state = Qobj([rho.full()[j] for j in l], dims=[
                           [2] * num_qubits, [1] * num_qubits])
    return qubit_state