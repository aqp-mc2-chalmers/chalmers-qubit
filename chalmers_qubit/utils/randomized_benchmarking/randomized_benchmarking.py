# Modifications copyright (c) 2025 Pontus Vikstål
#
# MIT License
# 
# Copyright (c) 2016 DiCarlo lab-QuTech-Delft University of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Optional, Union, Literal, Dict, List
from chalmers_qubit.utils.randomized_benchmarking.clifford_group import SingleQubitClifford, TwoQubitClifford, Clifford
from chalmers_qubit.utils.transformation import unitary_to_ptm
from qutip import Qobj
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import RX, RY, CZ, Gate

__all__ = [
    "RandomizedBenchmarking",
]

class RandomizedBenchmarking:
    """
    Implementation of Clifford-based Randomized Benchmarking (RB) for quantum gates.

    This class supports standard and interleaved randomized benchmarking for one and 
    two-qubit Clifford groups. It generates random Clifford sequences and converts them 
    to quantum circuits with physical gates.

    Attributes
    ----------
    clifford_group : int
        Specifies which Clifford group to use (1 for single-qubit, 2 for two-qubit).
    CliffordClass : Type
        The class representing Clifford operations for the chosen group.
    """

    def __init__(self, clifford_group: Literal[1, 2] = 1) -> None:
        """
        Initialize the RandomizedBenchmarking class.

        Parameters
        ----------
        clifford_group : Literal[1, 2], optional
            Specifies which Clifford group to use.
            1 for single-qubit (24 elements), 2 for two-qubit (11,520 elements).

        Raises
        ------
        NotImplementedError
            If an unsupported Clifford group is specified.
        """
        clifford_classes = {
            1: SingleQubitClifford,
            2: TwoQubitClifford,
        }
        if clifford_group not in clifford_classes:
            raise NotImplementedError("Only one- and two-qubit Clifford groups (1 or 2) are supported.")

        self.CliffordClass = clifford_classes[clifford_group]
        self.clifford_group = clifford_group

    def __repr__(self) -> str:
        """
        Return a string representation of the RandomizedBenchmarking instance.

        Returns
        -------
        str
            String representation of the instance.
        """
        return f"RandomizedBenchmarking(clifford_group={self.clifford_group})"

    def _calculate_net_clifford(
            self,    
            clifford_indices: np.ndarray,
        ) -> "Clifford":
        """
        Calculate the net Clifford from a list of Clifford indices.

        Parameters
        ----------
        clifford_indices : np.ndarray
            Array of integers specifying the Cliffords.

        Returns
        -------
        Clifford
            A `Clifford` object containing the net Clifford. The Clifford index is contained in the `Clifford.idx` attribute.

        Notes
        -----
        The order corresponds to the order in a pulse sequence but is
        the reverse of what it would be in a chained dot product.
        """
        net_clifford = self.CliffordClass(0) # assumes element 0 is the Identity
        for idx in clifford_indices:
            clifford = self.CliffordClass(idx)
            net_clifford = clifford * net_clifford
        return net_clifford

    def _add_interleaved_clifford_idx(
            self,
            clifford_sequence: np.ndarray, 
            interleaved_clifford_idx: int
        ) -> np.ndarray:
        """
        Add the interleaved Clifford gate index to the sequence.

        Parameters
        ----------
        clifford_sequence : np.ndarray
            Array of Clifford indices.
        interleaved_clifford_idx : int
            Interleaved Clifford index.

        Returns
        -------
        np.ndarray
            Clifford sequence with interleaved Clifford.
        """
        interleaved_sequence = np.empty(clifford_sequence.size * 2, dtype=int)
        interleaved_sequence[0::2] = clifford_sequence
        interleaved_sequence[1::2] = interleaved_clifford_idx
        return interleaved_sequence

    def _add_inverse_clifford_idx(
        self,
        clifford_sequence: np.ndarray,
    ) -> np.ndarray:
        """
        Find and add the inverse of the total sequence to the end of the sequence.

        Parameters
        ----------
        clifford_sequence : np.ndarray
            Array of Clifford indices.

        Returns
        -------
        np.ndarray
            Array with appended inverse Clifford index.
        """
        net_clifford = self._calculate_net_clifford(clifford_sequence)
        inverse_clifford = net_clifford.get_inverse() 
        return np.append(clifford_sequence, inverse_clifford.idx)
    
    def _get_interleaved_clifford_idx(
        self,
        interleaved_clifford_gate: Optional[Union[Gate, QubitCircuit]],
    ) -> int:
        """
        Get the index of the interleaved Clifford gate.

        Parameters
        ----------
        interleaved_clifford_gate : Optional[Union[Gate, QubitCircuit]]
            The interleaved Clifford gate.

        Returns
        -------
        int
            The index of the interleaved Clifford gate.

        Raises
        ------
        ValueError
            If the input is not a Gate or QubitCircuit, or if the unitary dimension does not match the Clifford group.
        """
        # Get the unitary representation of the interleaved Clifford gate
        if isinstance(interleaved_clifford_gate, QubitCircuit):
            interleaved_clifford_unitary = interleaved_clifford_gate.compute_unitary()
        elif isinstance(interleaved_clifford_gate, Gate):
            interleaved_clifford_unitary = interleaved_clifford_gate.get_compact_qobj()
        else:
            raise ValueError(f"interleaved_clifford must be {QubitCircuit} or {Gate}.")
        
        # Check that the unitary dimension matches the clifford group
        if interleaved_clifford_unitary.shape[0] != 2**self.clifford_group:
            raise ValueError(f"Interleaved Clifford unitary dimension {interleaved_clifford_unitary.shape[0]} does not match the dimension of clifford group {self.clifford_group}.")
        
        # Convert the unitary to a PTM and find the corresponding Clifford index
        ptm = unitary_to_ptm(interleaved_clifford_unitary)
        interleaved_clifford_idx = self.CliffordClass.find_clifford_index(ptm)
        return interleaved_clifford_idx
    
    def _add_interleaved_gate(self, circuit: QubitCircuit, gate: Union[Gate, QubitCircuit]):
        """
        Add an interleaved gate or circuit to the given QubitCircuit.

        Parameters
        ----------
        circuit : QubitCircuit
            The circuit to which the gate will be added.
        gate : Union[Gate, QubitCircuit]
            The gate or circuit to add.

        Raises
        ------
        ValueError
            If the input is not a Gate or QubitCircuit.
        """
        if isinstance(gate, QubitCircuit):
            circuit.add_circuit(gate)
        elif isinstance(gate, Gate):
            circuit.add_gate(gate)
        else:
            raise ValueError(f"interleaved_clifford_gate must be a {Gate} or {QubitCircuit}.")
    
    def _gate_decomposition(
            self,
            clifford_indices: np.ndarray,
            interleaved_clifford_gate: Optional[Union[Gate, QubitCircuit]] = None,
    ) -> QubitCircuit:
        """
        Decompose a sequence of Clifford indices into physical gates.

        Parameters
        ----------
        clifford_indices : np.ndarray
            Array of Clifford indices.
        interleaved_clifford_gate : Optional[Union[Gate, QubitCircuit]], optional
            Interleaved Clifford gate or circuit.

        Returns
        -------
        QubitCircuit
            The physical circuit with gates.

        Raises
        ------
        ValueError
            If a Clifford gate has no decomposition.
        """
        # Create a mapping of qubit indices to qubit names
        qubit_map = {f"q{idx}": idx for idx in range(self.clifford_group)}
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
        circuit = QubitCircuit(self.clifford_group)

        for i, clifford_idx in enumerate(clifford_indices):
            if interleaved_clifford_gate is not None and (i+1) % 2 == 0:
                self._add_interleaved_gate(circuit, interleaved_clifford_gate)
                continue
 
            cl_decomp = self.CliffordClass(clifford_idx).gate_decomposition
            if cl_decomp is None:
                raise ValueError(f"Clifford gate {clifford_idx} has no decomposition.")
            for gate, q in cl_decomp:
                if gate == "I":
                    continue
                operation = operation_map[gate](q)
                circuit.add_gate(operation)
        return circuit

    def _randomized_benchmarking_sequence(
        self,
        number_of_cliffords: int,
        apply_inverse: bool = True,
        interleaved_clifford_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a randomized benchmarking sequence using the one- or two-qubit Clifford group.

        Parameters
        ----------
        number_of_cliffords : int
            Number of Clifford gates in the sequence (excluding the optional inverse).
        apply_inverse : bool, optional
            Whether to append the recovery Clifford that inverts the total sequence.
        interleaved_clifford_idx : Optional[int], optional
            Optional ID for interleaving a specific Clifford gate.
        seed : Optional[int], optional
            Optional seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of Clifford indices representing the randomized benchmarking sequence.

        Raises
        ------
        ValueError
            If `number_of_cliffords` is negative.
        """
        if number_of_cliffords < 1:
            raise ValueError("Number of Cliffords must at least be 1.")
        
        group_size = self.CliffordClass.GROUP_SIZE
        rng = np.random.default_rng(seed)
        clifford_indices = rng.integers(low=0, high=group_size, size=number_of_cliffords)

        if interleaved_clifford_idx is not None:
            clifford_indices = self._add_interleaved_clifford_idx(
                clifford_sequence=clifford_indices, 
                interleaved_clifford_idx=interleaved_clifford_idx
            )

        if apply_inverse:
            clifford_indices = self._add_inverse_clifford_idx(clifford_sequence=clifford_indices)

        return clifford_indices

    def randomized_benchmarking_circuit(
        self,
        number_of_cliffords: int,
        apply_inverse: bool = True,
        interleaved_clifford_gate: Optional[Union[Gate, QubitCircuit]] = None,
        seed: Optional[int] = None,
    ) -> QubitCircuit:
        """
        Generate a randomized benchmarking circuit from a sequence of Clifford indices.

        Parameters
        ----------
        number_of_cliffords : int
            Number of Cliffords in the sequence.
        apply_inverse : bool, optional
            Whether to append the recovery Clifford that inverts the total sequence.
        interleaved_clifford_gate : Optional[Union[Gate, QubitCircuit]], optional
            Optional interleaved Clifford gate or circuit.
        seed : Optional[int], optional
            Optional seed for reproducibility.

        Returns
        -------
        QubitCircuit
            The randomized benchmarking circuit.
        """
        if interleaved_clifford_gate is not None:
            interleaved_clifford_idx = self._get_interleaved_clifford_idx(interleaved_clifford_gate)
        else:
            interleaved_clifford_idx = None

        clifford_indices = self._randomized_benchmarking_sequence(
            number_of_cliffords=number_of_cliffords,
            apply_inverse=apply_inverse,
            interleaved_clifford_idx=interleaved_clifford_idx,
            seed=seed,
        )

        circuit = self._gate_decomposition(
            clifford_indices=clifford_indices, 
            interleaved_clifford_gate=interleaved_clifford_gate,
        )
        return circuit