import numpy as np
from qutip import Qobj
from itertools import product
from functools import reduce

# Predefined Pauli matrices
SIGMAX = np.array([[0, 1], [1, 0]])
SIGMAY = np.array([[0, -1j], [1j, 0]])
SIGMAZ = np.array([[1, 0], [0, -1]])
PAULI_MATRICES = [np.eye(2), SIGMAX, SIGMAY, SIGMAZ]

# TODO: Write Test cases for the functions in this module
def pauli_basis(n_qubits: int) -> list:
    """Generate n-qubit Pauli basis using Kronecker product."""
    basis = []
    for indices in product(range(4), repeat=n_qubits):
        ops = [PAULI_MATRICES[i] for i in indices]
        kron_prod = reduce(np.kron, ops)
        basis.append(kron_prod)
    return basis

# TODO: Write Test cases for the functions in this module
def unitary_to_ptm(U: Qobj) -> np.ndarray:
    """Convert a unitary Qobj to its Pauli Transfer Matrix (PTM)."""
    U_np = U.full()
    n_qubits = int(np.log2(U.shape[0]))
    d = 2 ** n_qubits
    pauli_ops = pauli_basis(n_qubits)

    ptm = np.zeros((d**2, d**2), dtype=float)
    for i, Pi in enumerate(pauli_ops):
        for j, Pj in enumerate(pauli_ops):
            q_map = U_np @ Pj @ U_np.conj().T
            ptm[i, j] = np.trace(Pi @ q_map).real / d

    return ptm