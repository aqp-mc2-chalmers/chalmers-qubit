import numpy as np
from typing import Optional, Union, Literal, Type
from .clifford_decomposition import SingleQubitClifford
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import RX, RY, CZ

__all__ = [
    "randomized_benchmarking_sequence",
    "randomized_benchmarking_circuit",
]

def calculate_net_clifford(
    clifford_indices: np.ndarray,
    Clifford: Type[SingleQubitClifford] = SingleQubitClifford,
) -> "SingleQubitClifford":
    """
    Calculate the net-clifford from a list of cliffords indices.

    Args:
        clifford_indices (np.ndarray): Array of integers specifying the Cliffords.
        Clifford: Clifford class used to determine the inversion technique
                  and valid indices. Valid choices are `SingleQubitClifford`
                  and `TwoQubitClifford`.

    Returns:
        net_clifford: a `Clifford` object containing the net-clifford.
            the Clifford index is contained in the Clifford.idx attribute.

    Note: the order corresponds to the order in a pulse sequence but is
        the reverse of what it would be in a chained dot product.
    """

    # Calculate the net clifford
    net_clifford = Clifford(0) # assumes element 0 is the Identity

    for idx in clifford_indices:
        clifford = Clifford(idx)

        # order of operators applied in is right to left, therefore
        # the new operator is applied on the left side.
        net_clifford = clifford * net_clifford

    return net_clifford

def randomized_benchmarking_sequence(
    number_of_cliffords: int,
    apply_inverse_gate: bool = True,
    clifford_group: Literal[1, 2] = 1,
    interleaved_clifford_idx: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a randomized benchmarking sequence using the one- or two-qubit Clifford group.

    Args:
        number_of_cliffords (int): Number of Clifford gates in the sequence (excluding the optional inverse).
        apply_inverse_gate (bool): Whether to append the recovery Clifford that inverts the total sequence.
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

    group_sizes = {1: 24, 2: 11_520} # Size of single and two-qubit Clifford groups
    if clifford_group not in group_sizes:
        raise NotImplementedError("Only one- and two-qubit Clifford groups (1 or 2) are supported.")

    group_size = group_sizes[clifford_group]
    rng = np.random.default_rng(seed)
    clifford_indices = rng.integers(low=0, high=group_size, size=number_of_cliffords)

    # Add interleaving Clifford if applicable
    if interleaved_clifford_idx is not None:
        clifford_indices_interleaved = np.empty(clifford_indices.size * 2, dtype=int)
        clifford_indices_interleaved[0::2] = clifford_indices
        clifford_indices_interleaved[1::2] = interleaved_clifford_idx
        clifford_indices = clifford_indices_interleaved

    if apply_inverse_gate:
        # Calculate the net Clifford
        net_clifford = calculate_net_clifford(clifford_indices)
        # Get the inverse of the net clifford to find the Clifford that inverts the sequence
        recovery_clifford = SingleQubitClifford(net_clifford.idx).get_inverse() 
        # Append the recovery Clifford to the sequence
        clifford_indices = np.append(clifford_indices, recovery_clifford.idx)

    return clifford_indices

def randomized_benchmarking_circuit(clifford_indices: np.ndarray) -> QubitCircuit:
    """
    Generates a randomized benchmarking circuit from a sequence of Clifford indices.

    Args:
        clifford_indices (np.ndarray): Array of Clifford indices.

    Returns:
        QubitCircuit: The randomized benchmarking circuit.
    """
    operation_map = {
        "X180": lambda q: RX(targets=q, arg_value=np.pi),
        "X90": lambda q: RX(targets=q, arg_value=np.pi/2),
        "Y180": lambda q: RY(targets=q, arg_value=np.pi),
        "Y90": lambda q: RY(targets=q, arg_value=np.pi/2),
        "mX90": lambda q: RX(targets=q, arg_value=-np.pi/2),
        "mY90": lambda q: RY(targets=q, arg_value=-np.pi/2),
    }

    circuit = QubitCircuit(1)

    # Decompose Clifford sequence into physical gates
    for clifford_gate_idx in clifford_indices:
        cl_decomp = SingleQubitClifford(clifford_gate_idx).gate_decomposition
        
        if cl_decomp is None:
            raise ValueError(f"Clifford gate {clifford_gate_idx} has no decomposition.")
        
        for gate, q in cl_decomp:
            if gate == "I":
                continue
            
            qubit_index = int(q[1:]) # Extract qubit index from string, e,.g., "q0" -> 0
            operation = operation_map[gate](qubit_index)
            circuit.add_gate(operation)

    return circuit
    


if __name__ == "__main__":
    # Example usage
    sequence, net_clifford = randomized_benchmarking_sequence(10, apply_inverse_gate=True)
    print("Randomized Benchmarking Sequence:", sequence)
    print("Net Clifford Index:", net_clifford.idx)