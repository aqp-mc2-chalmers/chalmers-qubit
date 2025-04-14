import numpy as np
from typing import Optional, Union, Literal, Type, List, TypeAlias
from .clifford_decomposition import SingleQubitClifford, TwoQubitClifford, Clifford
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import RX, RY, CZ

__all__ = [
    "randomized_benchmarking_sequence",
    "randomized_benchmarking_circuit",
    "calculate_net_clifford",
] 

def calculate_net_clifford(
    clifford_indices: np.ndarray,
    CliffordClass: Type["Clifford"],
) -> "Clifford":
    """
    Calculate the net-clifford from a list of cliffords indices.

    Args:
        clifford_indices (np.ndarray): Array of integers specifying the Cliffords.
        CliffordClass: Clifford class used to determine the inversion technique
                  and valid indices. Valid choices are `SingleQubitClifford`
                  and `TwoQubitClifford`.

    Returns:
        net_clifford: a `Clifford` object containing the net-clifford.
            the Clifford index is contained in the Clifford.idx attribute.

    Note: the order corresponds to the order in a pulse sequence but is
        the reverse of what it would be in a chained dot product.
    """

    # Calculate the net clifford
    net_clifford = CliffordClass(0) # assumes element 0 is the Identity
    for idx in clifford_indices:
        clifford = CliffordClass(idx)

        # order of operators applied in is right to left, therefore
        # the new operator is applied on the left side.
        net_clifford = clifford * net_clifford

    return net_clifford

def add_interleaved_clifford(clifford_sequence: np.ndarray, interleaved_clifford: int) -> np.ndarray:
    """
    Adds an interleaved Clifford gate to the sequence.

    Args:
        clifford_sequence (np.ndarray): Array of Clifford indices.

    Returns:
        np.ndarray: Array with interleaved Clifford.
    """
    interleaved_sequence = np.empty(clifford_sequence.size * 2, dtype=int)
    interleaved_sequence[0::2] = clifford_sequence
    interleaved_sequence[1::2] = interleaved_clifford
    return interleaved_sequence

def add_inverse_clifford(
    clifford_sequence: np.ndarray,
    CliffordClass: Type["Clifford"],
) -> np.ndarray:
    """
    Adds the inverse of the total sequence to the end of the sequence.

    Args:
        clifford_sequence (np.ndarray): Array of Clifford indices.
        CliffordClass: The class of the Clifford group used.

    Returns:
        np.ndarray: Array with appended inverse Clifford.
    """

    # Calculate the net Clifford
    net_clifford = calculate_net_clifford(clifford_sequence, CliffordClass)
    # Get the inverse of the net clifford to find the Clifford that inverts the sequence
    inverse_clifford = CliffordClass(net_clifford.idx).get_inverse() 
    return np.append(clifford_sequence, inverse_clifford.idx)

def randomized_benchmarking_sequence(
    number_of_cliffords: int,
    apply_inverse: bool = True,
    clifford_group: Literal[1, 2] = 1,
    interleaved_clifford_idx: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a randomized benchmarking sequence using the one- or two-qubit Clifford group.

    Args:
        number_of_cliffords (int): Number of Clifford gates in the sequence (excluding the optional inverse).
        apply_inverse (bool): Whether to append the recovery Clifford that inverts the total sequence.
        clifford_group (int): Specifies which Clifford group to use. 
                              1 for single-qubit (24 elements), 2 for two-qubit (11,520 elements).
        interleaving_clifford_idx (Optional[int]): Optional ID for interleaving a specific Clifford gate.
        seed (Optional[int]): Optional seed for reproducibility.

    Returns:
        np.ndarray: Array of Clifford indices representing the randomized benchmarking sequence.

    Raises:
        ValueError: If `number_of_cliffords` is negative.
        NotImplementedError: If an unsupported Clifford group is specified.
    """
    if number_of_cliffords < 0:
        raise ValueError("Number of Cliffords can't be negative.")

    clifford_classes = {
        1: (24, SingleQubitClifford),
        2: (11_520, TwoQubitClifford),
    }

    if clifford_group not in clifford_classes:
        raise NotImplementedError("Only one- and two-qubit Clifford groups (1 or 2) are supported.")

    group_size, CliffordClass = clifford_classes[clifford_group]

    rng = np.random.default_rng(seed)
    clifford_indices = rng.integers(low=0, high=group_size, size=number_of_cliffords)

    # Add interleaving Clifford if applicable
    if interleaved_clifford_idx is not None:
        clifford_indices = add_interleaved_clifford(clifford_indices, interleaved_clifford_idx)

    # Add inverse Clifford if applicable
    if apply_inverse:
        clifford_indices = add_inverse_clifford(clifford_indices, CliffordClass)

    return clifford_indices

def randomized_benchmarking_circuit(
    clifford_indices: np.ndarray, 
    num_qubits: int=1, 
    clifford_group: Literal[1, 2] = 1,
    targets: Union[List[int],int]=0,
) -> QubitCircuit:
    """
    Generates a randomized benchmarking circuit from a sequence of Clifford indices.

    Args:
        clifford_indices (np.ndarray): Array of Clifford indices.
        num_qubits (int): Number of qubits in the circuit.
        targets (Union[List[int],int]): Target qubits for the Clifford sequence.

    Returns:
        QubitCircuit: The randomized benchmarking circuit.
    """
    
    clifford_groups = {
        1: SingleQubitClifford,
        2: TwoQubitClifford,
    }
    
    
    if clifford_group not in clifford_groups:
        raise NotImplementedError("Only one- and two-qubit Clifford groups (1 or 2) are supported.")
    
    if num_qubits < 1:
        raise ValueError("Number of Qubits must be >=1.")
    

    CliffordClass = clifford_groups[clifford_group]
    
    if not isinstance(targets, list):
        targets = [targets] # Make targets into a list
    
    qubit_map = {f"q{idx}": target for idx, target in enumerate(targets)}
    operation_map = {
        "X180": lambda q: RX(targets=qubit_map[q], arg_value=np.pi),
        "X90": lambda q: RX(targets=qubit_map[q], arg_value=np.pi/2),
        "Y180": lambda q: RY(targets=qubit_map[q], arg_value=np.pi),
        "Y90": lambda q: RY(targets=qubit_map[q], arg_value=np.pi/2),
        "mX90": lambda q: RX(targets=qubit_map[q], arg_value=-np.pi/2),
        "mY90": lambda q: RY(targets=qubit_map[q], arg_value=-np.pi/2),
        "CZ": lambda q: CZ(controls=qubit_map[q[0]], targets=qubit_map[q[1]]),
    }
    

    # Initialize the circuit
    circuit = QubitCircuit(num_qubits)

    # Decompose Clifford sequence into physical gates
    for clifford_idx in clifford_indices:
        cl_decomp = CliffordClass(clifford_idx).gate_decomposition
        
        if cl_decomp is None:
            raise ValueError(f"Clifford gate {clifford_idx} has no decomposition.")
        for gate, q in cl_decomp:
            if gate == "I":
                continue
            
            operation = operation_map[gate](q)
            circuit.add_gate(operation)

    return circuit
    

if __name__ == "__main__":
    # Example usage
    sequence, net_clifford = randomized_benchmarking_sequence(10, apply_inverse=True)
    print("Randomized Benchmarking Sequence:", sequence)
    print("Net Clifford Index:", net_clifford.idx)