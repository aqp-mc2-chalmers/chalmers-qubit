import numpy as np
from .clifford_group import C1, epstein_efficient_decomposition, S1
from .pauli_transfer_matrices import CZ
from typing import Optional, List, Tuple

class Clifford:
    """Base class for Clifford gates"""


class SingleQubitClifford:
    """Single Qubit Clifford gate class"""

    def __init__(self, idx: int) -> None:
        """Initialize the SingleQubitClifford object with a given index

        Args:
            idx (int): Index of the single qubit Clifford operation
            
        Raises:
            ValueError: If the index is not valid (0 <= idx < 24)
        """
        if not 0 <= idx < 24: # 24 elements in the single-qubit Clifford group
            raise ValueError(f"Invalid Clifford index: {idx}. Must be 0 <= idx < 24")
        self.idx = idx
        self.pauli_transfer_matrix = C1[idx]
    
    def __mul__(self, other: "SingleQubitClifford") -> "SingleQubitClifford":
        """Multiply two Clifford operations"""
        net_op = np.dot(self.pauli_transfer_matrix, other.pauli_transfer_matrix)
        idx = find_clifford_index(net_op)
        return SingleQubitClifford(idx)
    
    def __repr__(self) -> str:
        return f"SingleQubitClifford(idx={self.idx})"
    
    def get_inverse(self) -> "SingleQubitClifford":
        """Get the inverse of this Clifford operation"""
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = find_clifford_index(inverse_ptm)
        return SingleQubitClifford(idx)
    
    @classmethod
    def get_clifford_idx(cls, ptm: np.ndarray) -> int:
        """Get the index of a given Clifford operation"""
        idx = find_clifford_index(ptm)
        return idx
    
    @property
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the single qubit Clifford group
        according to the decomposition by Epstein et al.
        
        Returns:
            List of tuples where each tuple contains (gate_name, qubit_identifier)
        """
        return epstein_efficient_decomposition[self.idx]
    

gate_decomposition = epstein_efficient_decomposition
# used to transform the S1 subgroup
X90 = C1[16]
Y90 = C1[21]
mY90 = C1[15]

class TwoQubitClifford:
    """Two Qubit Clifford gate class"""
    # class constants
    GRP_SIZE_CLIFFORD = 24
    GRP_SIZE_SINGLE_QUBIT = GRP_SIZE_CLIFFORD**2
    GRP_SIZE_S1 = 3  # the S1 subgroup of SingleQubitClifford
    GRP_SIZE_CNOT = GRP_SIZE_SINGLE_QUBIT * GRP_SIZE_S1**2
    GRP_SIZE_ISWAP = GRP_SIZE_CNOT
    GRP_SIZE_SWAP = GRP_SIZE_SINGLE_QUBIT
    GRP_SIZE = GRP_SIZE_SINGLE_QUBIT + GRP_SIZE_CNOT + GRP_SIZE_ISWAP + GRP_SIZE_SWAP

    assert GRP_SIZE_SINGLE_QUBIT == 576
    assert GRP_SIZE_CNOT == 5184
    assert GRP_SIZE == 11_520

    _gate_decompositions = [[] for _ in range(GRP_SIZE)]
    ptm_cache: List[Optional[np.ndarray]] = [None] * GRP_SIZE

    def __init__(self, idx: int) -> None:
        """Initialize the TwoQubitClifford object with a given index

        Args:
            idx (int): Index of the single qubit Clifford operation
            
        Raises:
            ValueError: If the index is not valid (0 <= idx < 11520)
        """
        if not 0 <= idx < 11_520:
            raise ValueError(f"Invalid Clifford index: {idx}. Must be 0 <= idx < 11520")
        self.idx = idx

    def __mul__(self, other: "TwoQubitClifford") -> "TwoQubitClifford":
        """Multiply two Clifford operations"""
        net_op = np.dot(self.pauli_transfer_matrix, other.pauli_transfer_matrix)
        idx = find_clifford_index(net_op)
        return TwoQubitClifford(idx)
    
    def __repr__(self) -> str:
        return f"TwoQubitClifford(idx={self.idx})"
    
    def get_inverse(self) -> "TwoQubitClifford":
        """Get the inverse of this Clifford operation"""
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = find_clifford_index(inverse_ptm)
        return TwoQubitClifford(idx)

    @property  # FIXME: remove
    def pauli_transfer_matrix(self):
        # check cache
        if self.ptm_cache[self.idx] is None:
            # compute
            if self.idx < 576:
                ptm = self.single_qubit_like_PTM(self.idx)
            elif self.idx < 576 + 5184:
                ptm = self.CNOT_like_PTM(self.idx - 576)
            elif self.idx < 576 + 2 * 5184:
                ptm = self.iSWAP_like_PTM(self.idx - (576 + 5184))
            else:  # NB: GRP_SIZE checked upon construction
                ptm = self.SWAP_like_PTM(self.idx - (576 + 2 * 5184))

            # store in cache
            self.ptm_cache[self.idx] = ptm

        return self.ptm_cache[self.idx]
    
    @property  # FIXME: remove
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the two qubit Clifford group.

        Single qubit Cliffords are decomposed according to Epstein et al.
        """

        # check cache
        if self._gate_decompositions[self.idx] is None:
            # compute
            if self.idx < 576:
                _gate_decomposition = self.single_qubit_like_gates(self.idx)
            elif self.idx < 576 + 5184:
                _gate_decomposition = self.CNOT_like_gates(self.idx - 576)
            elif self.idx < 576 + 2 * 5184:
                _gate_decomposition = self.iSWAP_like_gates(self.idx - (576 + 5184))
            else:  # NB: GRP_SIZE checked upon construction
                _gate_decomposition = self.SWAP_like_gates(self.idx - (576 + 2 * 5184))

            # store in cache
            self._gate_decompositions[self.idx] = _gate_decomposition

        return self._gate_decompositions[self.idx]
    
    @classmethod
    def single_qubit_like_PTM(cls, idx: int):
        """
        Returns the pauli transfer matrix for gates of the single qubit like class
            (q0)  -- C1 --
            (q1)  -- C1 --
        """
        assert idx < cls.GRP_SIZE_SINGLE_QUBIT
        idx_q0 = idx % 24
        idx_q1 = idx // 24
        pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
        return pauli_transfer_matrix
    
    @classmethod
    def single_qubit_like_gates(cls, idx: int):
        """
        Returns the gates for Cliffords of the single qubit like class
            (q0)  -- C1 --
            (q1)  -- C1 --
        """
        assert idx < cls.GRP_SIZE_SINGLE_QUBIT
        idx_q0 = idx % 24
        idx_q1 = idx // 24

        g_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_q0]]
        g_q1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_q1]]
        gates = g_q0 + g_q1
        return gates

    @classmethod
    def CNOT_like_PTM(cls, idx: int):
        """
        Returns the pauli transfer matrix for gates of the cnot like class
            (q0)  --C1--•--S1--      --C1--•--S1------
                        |        ->        |
            (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
        """
        assert idx < cls.GRP_SIZE_CNOT
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = np.kron(np.eye(4), C1[idx_0])
        C1_q1 = np.kron(C1[idx_1], np.eye(4))
        # CZ
        S1_q0 = np.kron(np.eye(4), S1[idx_2])
        S1y_q1 = np.kron(np.dot(C1[idx_3], Y90), np.eye(4))
        return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1, CZ, S1_q0, S1y_q1])))

    @classmethod
    def CNOT_like_gates(cls, idx: int):
        """
        Returns the gates for Cliffords of the cnot like class
            (q0)  --C1--•--S1--      --C1--•--S1------
                        |        ->        |
            (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
        """
        assert idx < cls.GRP_SIZE_CNOT
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = [(g, "q0") for g in gate_decomposition[idx_0]]
        C1_q1 = [(g, "q1") for g in gate_decomposition[idx_1]]
        CZ = [("CZ", ["q0", "q1"])]

        idx_2s = SingleQubitClifford.get_clifford_idx(S1[idx_2])
        S1_q0 = [(g, "q0") for g in gate_decomposition[idx_2s]]
        # FIXME: precomputation of these 3 entries would be more efficient (more similar occurrences in this file):
        idx_3s = SingleQubitClifford.get_clifford_idx(np.dot(C1[idx_3], Y90))
        S1_yq1 = [(g, "q1") for g in gate_decomposition[idx_3s]]

        gates = C1_q0 + C1_q1 + CZ + S1_q0 + S1_yq1
        return gates

    @classmethod
    def iSWAP_like_PTM(cls, idx: int):
        """
        Returns the pauli transfer matrix for gates of the iSWAP like class
            (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                        |       ->        |        |
            (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
        """
        assert idx < cls.GRP_SIZE_ISWAP
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = np.kron(np.eye(4), C1[idx_0])
        C1_q1 = np.kron(C1[idx_1], np.eye(4))
        # CZ
        sq_swap_gates = np.kron(mY90, Y90)
        # CZ
        S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
        S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

        return np.linalg.multi_dot(
            list(reversed([C1_q0, C1_q1, CZ, sq_swap_gates, CZ, S1_q0, S1y_q1]))
        )

    @classmethod
    def iSWAP_like_gates(cls, idx: int):
        """
        Returns the gates for Cliffords of the iSWAP like class
            (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                        |       ->        |        |
            (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
        """
        assert idx < cls.GRP_SIZE_ISWAP
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = [(g, "q0") for g in gate_decomposition[idx_0]]
        C1_q1 = [(g, "q1") for g in gate_decomposition[idx_1]]
        CZ = [("CZ", ["q0", "q1"])]

        sqs_idx_q0 = SingleQubitClifford.get_clifford_idx(Y90)
        sqs_idx_q1 = SingleQubitClifford.get_clifford_idx(mY90)
        sq_swap_gates_q0 = [(g, "q0") for g in gate_decomposition[sqs_idx_q0]]
        sq_swap_gates_q1 = [(g, "q1") for g in gate_decomposition[sqs_idx_q1]]

        idx_2s = SingleQubitClifford.get_clifford_idx(np.dot(S1[idx_2], Y90))
        S1_q0 = [(g, "q0") for g in gate_decomposition[idx_2s]]
        idx_3s = SingleQubitClifford.get_clifford_idx(np.dot(C1[idx_3], X90))
        S1y_q1 = [(g, "q1") for g in gate_decomposition[idx_3s]]

        gates = (
            C1_q0
            + C1_q1
            + CZ
            + sq_swap_gates_q0
            + sq_swap_gates_q1
            + CZ
            + S1_q0
            + S1y_q1
        )
        return gates

    @classmethod
    def SWAP_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Returns the pauli transfer matrix for gates of the SWAP like class

        (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
        """
        assert idx < cls.GRP_SIZE_SWAP
        idx_q0 = idx % 24
        idx_q1 = idx // 24
        sq_like_cliff = np.kron(C1[idx_q1], C1[idx_q0])
        sq_swap_gates_0 = np.kron(Y90, mY90)
        sq_swap_gates_1 = np.kron(mY90, Y90)
        sq_swap_gates_2 = np.kron(Y90, np.eye(4))

        return np.linalg.multi_dot(
            list(
                reversed(
                    [
                        sq_like_cliff,
                        CZ,
                        sq_swap_gates_0,
                        CZ,
                        sq_swap_gates_1,
                        CZ,
                        sq_swap_gates_2,
                    ]
                )
            )
        )

    @classmethod
    def SWAP_like_gates(cls, idx: int):
        """
        Returns the gates for Cliffords of the SWAP like class

        (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
        """
        assert idx < cls.GRP_SIZE_SWAP
        idx_q0 = idx % 24
        idx_q1 = idx // 24
        C1_q0 = [(g, "q0") for g in gate_decomposition[idx_q0]]
        C1_q1 = [(g, "q1") for g in gate_decomposition[idx_q1]]
        CZ = [("CZ", ["q0", "q1"])]

        # sq_swap_gates_0 = np.kron(Y90, mY90)

        sqs_idx_q0 = SingleQubitClifford.get_clifford_idx(mY90)
        sqs_idx_q1 = SingleQubitClifford.get_clifford_idx(Y90)
        sq_swap_gates_0_q0 = [(g, "q0") for g in gate_decomposition[sqs_idx_q0]]
        sq_swap_gates_0_q1 = [(g, "q1") for g in gate_decomposition[sqs_idx_q1]]

        sqs_idx_q0 = SingleQubitClifford.get_clifford_idx(Y90)
        sqs_idx_q1 = SingleQubitClifford.get_clifford_idx(mY90)
        sq_swap_gates_1_q0 = [(g, "q0") for g in gate_decomposition[sqs_idx_q0]]
        sq_swap_gates_1_q1 = [(g, "q1") for g in gate_decomposition[sqs_idx_q1]]

        sqs_idx_q1 = SingleQubitClifford.get_clifford_idx(Y90)
        sq_swap_gates_2_q0 = [(g, "q0") for g in gate_decomposition[0]]
        sq_swap_gates_2_q1 = [(g, "q1") for g in gate_decomposition[sqs_idx_q1]]

        gates = (
            C1_q0
            + C1_q1
            + CZ
            + sq_swap_gates_0_q0
            + sq_swap_gates_0_q1
            + CZ
            + sq_swap_gates_1_q0
            + sq_swap_gates_1_q1
            + CZ
            + sq_swap_gates_2_q0
            + sq_swap_gates_2_q1
        )
        return gates

    
def hash_matrix(matrix: np.ndarray) -> int:
    """Create a hash value for a matrix using NumPy's internal representation
        
       Use the byte representation of the rounded integer matrix
    """
    return hash(matrix.round().astype(int).tobytes())

# Generate a hash table/dictionary for quick lookups
CLIFFORD_HASH_TABLE = {}
for idx, matrix in enumerate(C1):
    hash_value = hash_matrix(matrix)
    CLIFFORD_HASH_TABLE[hash_value] = idx

def find_clifford_index(matrix: np.ndarray) -> int:
    """Find the index of a Clifford matrix using hash lookup"""
    hash_value = hash_matrix(matrix)
    # Look up the index in our hash table
    if hash_value in CLIFFORD_HASH_TABLE:
        return CLIFFORD_HASH_TABLE[hash_value]
    
    # Fallback to direct comparison if hash not found (shouldn't happen with valid Clifford matrices)
    for idx, clifford_matrix in enumerate(C1):
        if np.allclose(matrix, clifford_matrix):
            return idx
    raise ValueError("Clifford index not found.")