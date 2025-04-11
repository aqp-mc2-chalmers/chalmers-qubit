import numpy as np
from .pauli_transfer_matrices import I, X, Y, Z, S, S2, H

__all__ = [
    "single_qubit_clifford_group",
    "epstein_efficient_decomposition",
    "S1",
]

"""
Decomposition of the single qubit clifford group as per
Epstein et al. Phys. Rev. A 89, 062321 (2014)
"""

epstein_efficient_decomposition = [[] for _ in range(24)]
# explicitly reversing order because order of operators is order in time
epstein_efficient_decomposition[0] = ["I"]
epstein_efficient_decomposition[1] = ["Y90", "X90"]
epstein_efficient_decomposition[2] = ["mX90", "mY90"]
epstein_efficient_decomposition[3] = ["X180"]
epstein_efficient_decomposition[4] = ["mY90", "mX90"]
epstein_efficient_decomposition[5] = ["X90", "mY90"]
epstein_efficient_decomposition[6] = ["Y180"]
epstein_efficient_decomposition[7] = ["mY90", "X90"]
epstein_efficient_decomposition[8] = ["X90", "Y90"]
epstein_efficient_decomposition[9] = ["X180", "Y180"]
epstein_efficient_decomposition[10] = ["Y90", "mX90"]
epstein_efficient_decomposition[11] = ["mX90", "Y90"]
epstein_efficient_decomposition[12] = ["Y90", "X180"]
epstein_efficient_decomposition[13] = ["mX90"]
epstein_efficient_decomposition[14] = ["X90", "mY90", "mX90"]
epstein_efficient_decomposition[15] = ["mY90"]
epstein_efficient_decomposition[16] = ["X90"]
epstein_efficient_decomposition[17] = ["X90", "Y90", "X90"]
epstein_efficient_decomposition[18] = ["mY90", "X180"]
epstein_efficient_decomposition[19] = ["X90", "Y180"]
epstein_efficient_decomposition[20] = ["X90", "mY90", "X90"]
epstein_efficient_decomposition[21] = ["Y90"]
epstein_efficient_decomposition[22] = ["mX90", "Y180"]
epstein_efficient_decomposition[23] = ["X90", "Y90", "mX90"]

# The single qubit clifford group where each element is a 4x4 pauli transfer matrix
single_qubit_clifford_group = [np.empty([4, 4])] * (24)
# explictly reversing order because order of operators is order in time
single_qubit_clifford_group[0] = np.linalg.multi_dot([I, I, I][::-1])
single_qubit_clifford_group[1] = np.linalg.multi_dot([I, I, S][::-1])
single_qubit_clifford_group[2] = np.linalg.multi_dot([I, I, S2][::-1])
single_qubit_clifford_group[3] = np.linalg.multi_dot([X, I, I][::-1])
single_qubit_clifford_group[4] = np.linalg.multi_dot([X, I, S][::-1])
single_qubit_clifford_group[5] = np.linalg.multi_dot([X, I, S2][::-1])
single_qubit_clifford_group[6] = np.linalg.multi_dot([Y, I, I][::-1])
single_qubit_clifford_group[7] = np.linalg.multi_dot([Y, I, S][::-1])
single_qubit_clifford_group[8] = np.linalg.multi_dot([Y, I, S2][::-1])
single_qubit_clifford_group[9] = np.linalg.multi_dot([Z, I, I][::-1])
single_qubit_clifford_group[10] = np.linalg.multi_dot([Z, I, S][::-1])
single_qubit_clifford_group[11] = np.linalg.multi_dot([Z, I, S2][::-1])
single_qubit_clifford_group[12] = np.linalg.multi_dot([I, H, I][::-1])
single_qubit_clifford_group[13] = np.linalg.multi_dot([I, H, S][::-1])
single_qubit_clifford_group[14] = np.linalg.multi_dot([I, H, S2][::-1])
single_qubit_clifford_group[15] = np.linalg.multi_dot([X, H, I][::-1])
single_qubit_clifford_group[16] = np.linalg.multi_dot([X, H, S][::-1])
single_qubit_clifford_group[17] = np.linalg.multi_dot([X, H, S2][::-1])
single_qubit_clifford_group[18] = np.linalg.multi_dot([Y, H, I][::-1])
single_qubit_clifford_group[19] = np.linalg.multi_dot([Y, H, S][::-1])
single_qubit_clifford_group[20] = np.linalg.multi_dot([Y, H, S2][::-1])
single_qubit_clifford_group[21] = np.linalg.multi_dot([Z, H, I][::-1])
single_qubit_clifford_group[22] = np.linalg.multi_dot([Z, H, S][::-1])
single_qubit_clifford_group[23] = np.linalg.multi_dot([Z, H, S2][::-1])

# S1 is a subgroup of C1 (single qubit Clifford group) and is used when generating C2 (two qubit Clifford group)
S1 = [
    single_qubit_clifford_group[0],
    single_qubit_clifford_group[1],
    single_qubit_clifford_group[2],
]