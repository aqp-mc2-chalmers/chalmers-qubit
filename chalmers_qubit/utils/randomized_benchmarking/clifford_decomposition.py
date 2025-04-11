from typing import Optional, List, Tuple
import numpy as np

from .clifford_group import single_qubit_clifford_group, epstein_efficient_decomposition

class SingleQubitClifford:
    """Single Qubit Clifford gate class"""
    
    # class variablesps
    _gate_decompositions: List[Optional[List[Tuple[str, str]]]] = [None] * 24

    def __init__(self, idx: int, qubit_idx: int = 0) -> None:
        """Initialize the SingleQubitClifford object with a given index

        Args:
            idx (int): Index of the single qubit Clifford operation
            qubit_idx (int): Index of the qubit (default is 0)
            
        Raises:
            ValueError: If the index is not valid (0 <= idx < 24)
        """
        if not 0 <= idx < 24: # 24 elements in the single-qubit Clifford group
            raise ValueError(f"Invalid Clifford index: {idx}. Must be 0 <= idx < 24")
        self.idx = idx
        self.qubit_idx = qubit_idx
        self.pauli_transfer_matrix = single_qubit_clifford_group[idx]
    
    def __mul__(self, other: "SingleQubitClifford") -> "SingleQubitClifford":
        """Multiply two Clifford operations"""
        net_op = np.dot(self.pauli_transfer_matrix, other.pauli_transfer_matrix)
        idx = find_clifford_index(net_op)
        return SingleQubitClifford(idx)
    
    def get_inverse(self) -> "SingleQubitClifford":
        """Get the inverse of this Clifford operation"""
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = find_clifford_index(inverse_ptm)
        return SingleQubitClifford(idx)
    
    @property
    def gate_decomposition(self):
        """
        Returns the gate decomposition of the single qubit Clifford group
        according to the decomposition by Epstein et al.
        
        Returns:
            List of tuples where each tuple contains (gate_name, qubit_identifier)
        """
        # Check if we've already computed this decomposition
        if self._gate_decompositions[self.idx] is None:
            # Get the basic decomposition from the Epstein table
            gates = epstein_efficient_decomposition[self.idx]
            # Add qubit identifier to each gate
            decomposition = [(g, f"q{self.qubit_idx}") for g in gates]
            # Cache the result in the class variable
            self._gate_decompositions[self.idx] = decomposition
        
        return SingleQubitClifford._gate_decompositions[self.idx]
    
    def __repr__(self) -> str:
        return f"SingleQubitClifford(idx={self.idx})"
    
def hash_matrix(matrix: np.ndarray) -> int:
    """Create a hash value for a matrix using NumPy's internal representation
        
       Use the byte representation of the rounded integer matrix
    """
    return hash(matrix.round().astype(int).tobytes())

# Generate a hash table/dictionary for quick lookups
CLIFFORD_HASH_TABLE = {}
for idx, matrix in enumerate(single_qubit_clifford_group):
    hash_value = hash_matrix(matrix)
    CLIFFORD_HASH_TABLE[hash_value] = idx

def find_clifford_index(matrix: np.ndarray) -> int:
    """Find the index of a Clifford matrix using hash lookup"""
    hash_value = hash_matrix(matrix)
    # Look up the index in our hash table
    if hash_value in CLIFFORD_HASH_TABLE:
        return CLIFFORD_HASH_TABLE[hash_value]
    
    # Fallback to direct comparison if hash not found (shouldn't happen with valid Clifford matrices)
    for idx, clifford_matrix in enumerate(single_qubit_clifford_group):
        if np.allclose(matrix, clifford_matrix):
            return idx
    raise ValueError("Clifford index not found.")