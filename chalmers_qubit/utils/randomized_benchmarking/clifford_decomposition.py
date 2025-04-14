import numpy as np
from .clifford_group import C1, epstein_efficient_decomposition, S1
from .pauli_transfer_matrices import CZ
from typing import Optional, List, Tuple, Dict


class Clifford:
    """Base class for Clifford"""
    
    GROUP_SIZE: int # Size of the Clifford gGroup

    def __init__(self, idx: int) -> None:
        """Initialize the Clifford object with a given index
        
        Args:
            idx (int): Index of the Clifford operation
            
        Raises:
            ValueError: If the index is not valid
        """
        if not 0 <= idx < self.GROUP_SIZE:
            raise ValueError(f"Invalid Clifford index: {idx}. Must be 0 <= idx < {self.GROUP_SIZE}")
        self.idx = idx
        self.pauli_transfer_matrix: np.ndarray
    
    def __mul__(self, other: "Clifford") -> "Clifford":
        """Multiply two Clifford operations"""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot multiply {self.__class__.__name__} with {other.__class__.__name__}")
        
        net_op = np.dot(self.pauli_transfer_matrix, other.pauli_transfer_matrix)
        idx = self.find_clifford_index(net_op)
        return self.__class__(idx)
    
    def get_inverse(self) -> "Clifford":
        """Get the inverse of this Clifford operation"""
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = self.find_clifford_index(inverse_ptm)
        return self.__class__(idx)
    
    @property
    def gate_decomposition(self) -> List:
        """Returns the gate decomposition of the Clifford gate"""
        raise NotImplementedError("Subclasses must implement gate_decomposition")
    
    @classmethod
    def find_clifford_index(cls, matrix: np.ndarray) -> int:
        """Find the Clifford index of a given Clifford Pauli transfer matrix"""
        raise NotImplementedError("Subclasses must implement find_clifford_index")
    
    @staticmethod
    def hash_matrix(matrix: np.ndarray) -> int:
        """Create a hash value for a matrix using NumPy's internal representation
            
        Use the byte representation of the rounded integer matrix
        """
        return hash(matrix.round().astype(int).tobytes())
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(idx={self.idx})"


class SingleQubitClifford(Clifford):
    """Single Qubit Clifford gate class"""

    GROUP_SIZE = 24  # Size of the single qubit Clifford group
    CLIFFORD_HASH_TABLE: Dict[int, int] = {}

    def __init__(self, idx: int) -> None:
        """Initialize the SingleQubitClifford object with a given index

        Args:
            idx (int): Index of the single qubit Clifford operation
            
        Raises:
            ValueError: If the index is not valid (0 <= idx < 24)
        """
        super().__init__(idx)
        self.pauli_transfer_matrix = C1[idx]
    
    @classmethod
    def find_clifford_index(cls, matrix: np.ndarray) -> int:
        """Find the index of a Clifford matrix using hash lookup"""
        # Create Hash Table if it is empty
        if not cls.CLIFFORD_HASH_TABLE:
            print("Creating Hash Table")
            for idx, clifford_matrix in enumerate(C1):
                hash_value = cls.hash_matrix(clifford_matrix)
                cls.CLIFFORD_HASH_TABLE[hash_value] = idx

        hash_value = cls.hash_matrix(matrix)
        # Look up if the hash values is in our hash table
        if hash_value in cls.CLIFFORD_HASH_TABLE:
            return cls.CLIFFORD_HASH_TABLE[hash_value]
        
        raise ValueError("Clifford index not found.")
    
    @property
    def gate_decomposition(self) -> List[Tuple[str, str]]:
        """
        Returns the gate decomposition of the single qubit Clifford group
        according to the decomposition by Epstein et al.
        
        Returns:
            List of tuples where each tuple contains (gate_name, qubit_identifier)
        """
        gates = [(g, "q0") for g in epstein_efficient_decomposition[self.idx]]
        return gates
    
# used to transform the S1 subgroup
X90 = C1[16]
Y90 = C1[21]
mY90 = C1[15]

class TwoQubitClifford(Clifford):
    """Two Qubit Clifford gate class"""

    CLIFFORD_HASH_TABLE: Dict[int, int] = {}

    # Class Constants
    GROUP_SIZE_CLIFFORD = 24
    GROUP_SIZE_SINGLE_QUBIT = GROUP_SIZE_CLIFFORD**2
    GROUP_SIZE_S1 = 3 # the S1 subgroup of SingleQubitClifford
    GROUP_SIZE_CNOT = GROUP_SIZE_SINGLE_QUBIT * GROUP_SIZE_S1**2
    GROUP_SIZE_ISWAP = GROUP_SIZE_CNOT
    GROUP_SIZE_SWAP = GROUP_SIZE_SINGLE_QUBIT
    GROUP_SIZE = GROUP_SIZE_SINGLE_QUBIT + GROUP_SIZE_CNOT + GROUP_SIZE_ISWAP + GROUP_SIZE_SWAP

    assert GROUP_SIZE_SINGLE_QUBIT == 576
    assert GROUP_SIZE_CNOT == 5184
    assert GROUP_SIZE == 11_520

    # class variables
    _gate_decompositions = [[] for _ in range(GROUP_SIZE)]
    _pauli_transfer_matrices: List[Optional[np.ndarray]] = [None] * GROUP_SIZE

    def __init__(self, idx: int) -> None:
        """Initialize the TwoQubitClifford object with a given index

        Args:
            idx (int): Index of the single qubit Clifford operation
            
        Raises:
            ValueError: If the index is not valid (0 <= idx < 11520)
        """
        super().__init__(idx)

    @property  # FIXME: remove
    def pauli_transfer_matrix(self) -> np.ndarray:
        # check cache
        if self._pauli_transfer_matrices[self.idx] is None:
            # compute
            if self.idx < 576:
                _pauli_transfer_matrices = self.single_qubit_like_PTM(self.idx)
            elif self.idx < 576 + 5184:
                _pauli_transfer_matrices = self.CNOT_like_PTM(self.idx - 576)
            elif self.idx < 576 + 2 * 5184:
                _pauli_transfer_matrices = self.iSWAP_like_PTM(self.idx - (576 + 5184))
            else:  # NB: GROUP_SIZE checked upon construction
                _pauli_transfer_matrices = self.SWAP_like_PTM(self.idx - (576 + 2 * 5184))

            # store in cache
            self._pauli_transfer_matrices[self.idx] = _pauli_transfer_matrices

        return self._pauli_transfer_matrices[self.idx]
    
    @property  # FIXME: remove
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the two qubit Clifford group.

        Single qubit Cliffords are decomposed according to Epstein et al.
        """

        # check cache
        if not self._gate_decompositions[self.idx]:
            # compute
            if self.idx < 576:
                _gate_decomposition = self.single_qubit_like_gates(self.idx)
            elif self.idx < 576 + 5184:
                _gate_decomposition = self.CNOT_like_gates(self.idx - 576)
            elif self.idx < 576 + 2 * 5184:
                _gate_decomposition = self.iSWAP_like_gates(self.idx - (576 + 5184))
            else:  # NB: GROUP_SIZE checked upon construction
                _gate_decomposition = self.SWAP_like_gates(self.idx - (576 + 2 * 5184))
            # store in cache
            self._gate_decompositions[self.idx] = _gate_decomposition
        return self._gate_decompositions[self.idx]
    
    @classmethod
    def find_clifford_index(cls, matrix: np.ndarray) -> int:
        """Find the index of a Clifford matrix using hash lookup"""
        # Create Hash Table if it is empty
        if not cls.CLIFFORD_HASH_TABLE:
            print("Creating Hash Table")
            for idx in range(cls.GROUP_SIZE):
                matrix = cls(idx=idx).pauli_transfer_matrix
                hash_value = cls.hash_matrix(matrix)
                cls.CLIFFORD_HASH_TABLE[hash_value] = idx

        hash_value = cls.hash_matrix(matrix)
        # Look up if the hash values is in our hash table
        if hash_value in cls.CLIFFORD_HASH_TABLE:
            return cls.CLIFFORD_HASH_TABLE[hash_value]
        
        raise ValueError("Clifford index not found.")
    
    @classmethod
    def single_qubit_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Returns the pauli transfer matrix for gates of the single qubit like class
            (q0)  -- C1 --
            (q1)  -- C1 --
        """
        assert idx < cls.GROUP_SIZE_SINGLE_QUBIT
        idx_q0 = idx % 24
        idx_q1 = idx // 24
        pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
        return pauli_transfer_matrix
    
    @classmethod
    def single_qubit_like_gates(cls, idx: int) -> List[Tuple[str, str]]:
        """
        Returns the gates for Cliffords of the single qubit like class
            (q0)  -- C1 --
            (q1)  -- C1 --
        """
        assert idx < cls.GROUP_SIZE_SINGLE_QUBIT
        idx_q0 = idx % 24
        idx_q1 = idx // 24

        g_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_q0]]
        g_q1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_q1]]
        gates = g_q0 + g_q1
        return gates

    @classmethod
    def CNOT_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Returns the pauli transfer matrix for gates of the cnot like class
            (q0)  --C1--•--S1--      --C1--•--S1------
                        |        ->        |
            (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
        """
        assert idx < cls.GROUP_SIZE_CNOT
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
    def CNOT_like_gates(cls, idx: int) -> List[Tuple[str, str]]:
        """
        Returns the gates for Cliffords of the cnot like class
            (q0)  --C1--•--S1--      --C1--•--S1------
                        |        ->        |
            (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
        """
        assert idx < cls.GROUP_SIZE_CNOT
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_0]]
        C1_q1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_1]]
        CZ = [("CZ", ["q0", "q1"])]

        idx_2s = SingleQubitClifford.find_clifford_index(S1[idx_2])
        S1_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_2s]]
        # FIXME: precomputation of these 3 entries would be more efficient (more similar occurrences in this file):
        idx_3s = SingleQubitClifford.find_clifford_index(np.dot(C1[idx_3], Y90))
        S1_yq1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_3s]]

        gates = C1_q0 + C1_q1 + CZ + S1_q0 + S1_yq1
        return gates

    @classmethod
    def iSWAP_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Returns the pauli transfer matrix for gates of the iSWAP like class
            (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                        |       ->        |        |
            (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
        """
        assert idx < cls.GROUP_SIZE_ISWAP
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
    def iSWAP_like_gates(cls, idx: int) -> List[Tuple[str, str]]:
        """
        Returns the gates for Cliffords of the iSWAP like class
            (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
                        |       ->        |        |
            (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
        """
        assert idx < cls.GROUP_SIZE_ISWAP
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_0]]
        C1_q1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_1]]
        CZ = [("CZ", ["q0", "q1"])]

        sqs_idx_q0 = SingleQubitClifford.find_clifford_index(Y90)
        sqs_idx_q1 = SingleQubitClifford.find_clifford_index(mY90)
        sq_swap_gates_q0 = [(g, "q0") for g in epstein_efficient_decomposition[sqs_idx_q0]]
        sq_swap_gates_q1 = [(g, "q1") for g in epstein_efficient_decomposition[sqs_idx_q1]]

        idx_2s = SingleQubitClifford.find_clifford_index(np.dot(S1[idx_2], Y90))
        S1_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_2s]]
        idx_3s = SingleQubitClifford.find_clifford_index(np.dot(C1[idx_3], X90))
        S1y_q1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_3s]]

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
        assert idx < cls.GROUP_SIZE_SWAP
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
    def SWAP_like_gates(cls, idx: int) -> List[Tuple[str, str]]:
        """
        Returns the gates for Cliffords of the SWAP like class

        (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
        """
        assert idx < cls.GROUP_SIZE_SWAP
        idx_q0 = idx % 24
        idx_q1 = idx // 24
        C1_q0 = [(g, "q0") for g in epstein_efficient_decomposition[idx_q0]]
        C1_q1 = [(g, "q1") for g in epstein_efficient_decomposition[idx_q1]]
        CZ = [("CZ", ["q0", "q1"])]

        # sq_swap_gates_0 = np.kron(Y90, mY90)

        sqs_idx_q0 = SingleQubitClifford.find_clifford_index(mY90)
        sqs_idx_q1 = SingleQubitClifford.find_clifford_index(Y90)
        sq_swap_gates_0_q0 = [(g, "q0") for g in epstein_efficient_decomposition[sqs_idx_q0]]
        sq_swap_gates_0_q1 = [(g, "q1") for g in epstein_efficient_decomposition[sqs_idx_q1]]

        sqs_idx_q0 = SingleQubitClifford.find_clifford_index(Y90)
        sqs_idx_q1 = SingleQubitClifford.find_clifford_index(mY90)
        sq_swap_gates_1_q0 = [(g, "q0") for g in epstein_efficient_decomposition[sqs_idx_q0]]
        sq_swap_gates_1_q1 = [(g, "q1") for g in epstein_efficient_decomposition[sqs_idx_q1]]

        sqs_idx_q1 = SingleQubitClifford.find_clifford_index(Y90)
        sq_swap_gates_2_q0 = [(g, "q0") for g in epstein_efficient_decomposition[0]]
        sq_swap_gates_2_q1 = [(g, "q1") for g in epstein_efficient_decomposition[sqs_idx_q1]]

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