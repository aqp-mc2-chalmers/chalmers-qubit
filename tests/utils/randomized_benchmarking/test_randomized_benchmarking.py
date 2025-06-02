import unittest
import numpy as np
from qutip import qeye, average_gate_fidelity
from qutip_qip.operations import CZ, H

from chalmers_qubit.utils import RandomizedBenchmarking
from chalmers_qubit.utils.randomized_benchmarking.clifford_group import SingleQubitClifford, TwoQubitClifford

class TestRandomizedBenchmarking(unittest.TestCase):

    def setUp(self) -> None:
        # Test cases for different Clifford groups
        self.test_cases = [
            (1, H(0)),  # (clifford_group, interleaved_clifford_idx)
            (2, CZ(controls=0, targets=1)) # Example interleaved indices
        ]

    def test_init(self):
        """Test RandomizedBenchmarking initialization."""
        # Test valid initialization
        rb1 = RandomizedBenchmarking(clifford_group=1)
        self.assertEqual(rb1.clifford_group, 1)
        self.assertEqual(rb1.CliffordClass, SingleQubitClifford)
        
        rb2 = RandomizedBenchmarking(clifford_group=2)
        self.assertEqual(rb2.clifford_group, 2)
        self.assertEqual(rb2.CliffordClass, TwoQubitClifford)
        
        # Test invalid initialization
        with self.assertRaises(NotImplementedError):
            RandomizedBenchmarking(clifford_group=3)

    def test_randomized_benchmarking_sequence(self):
        """Test that the randomized benchmarking sequence is generated correctly."""
        for clifford_group, _ in self.test_cases:
            with self.subTest(clifford_group=clifford_group):
                rb = RandomizedBenchmarking(clifford_group=clifford_group)
                number_of_cliffords = 100
    
                # Test without inverse
                sequence = rb._randomized_benchmarking_sequence(
                    number_of_cliffords=number_of_cliffords,
                    apply_inverse=False,
                    interleaved_clifford_idx=None,
                    seed=123
                )
                self.assertEqual(len(sequence), number_of_cliffords)
                
                # Test with inverse
                sequence_with_inverse = rb._randomized_benchmarking_sequence(
                    number_of_cliffords=number_of_cliffords,
                    apply_inverse=True,
                    interleaved_clifford_idx=None,
                    seed=123
                )
                self.assertEqual(len(sequence_with_inverse), number_of_cliffords + 1)


    def test_recovery_clifford_produces_identity(self):
        """Test that recovery gate in clifford sequence yields identity."""
        for clifford_group, _ in self.test_cases:
            with self.subTest(clifford_group=clifford_group):
                rb = RandomizedBenchmarking(clifford_group=clifford_group)
                sequence = rb._randomized_benchmarking_sequence(
                    number_of_cliffords=10,  # Using smaller number for faster testing
                    apply_inverse=True,
                    seed=123
                )
                
                # Get the recovery Clifford (last element)
                recovery_clifford = rb.CliffordClass(sequence[-1])
                
                # Compute net Clifford from the sequence (excluding recovery)
                net_clifford = rb._calculate_net_clifford(sequence[:-1])
                
                # Compute the product of the net clifford and the recovery clifford
                total_clifford = net_clifford * recovery_clifford
                identity_clifford = rb.CliffordClass(0)  # Identity is at index 0
                
                self.assertTrue(
                    total_clifford == identity_clifford, 
                    "The product of a sequence and its recovery should be the identity"
                )

    def test_interleaved_randomized_benchmarking(self):
        """Test that recovery gate in interleaved clifford sequence yields identity."""
        for clifford_group, interleaved_clifford_gate in self.test_cases:
            with self.subTest(clifford_group=clifford_group):
                rb = RandomizedBenchmarking(clifford_group=clifford_group)
                interleaved_clifford_idx = rb._get_interleaved_clifford_idx(interleaved_clifford_gate)
                sequence = rb._randomized_benchmarking_sequence(
                    number_of_cliffords=10,  # Using smaller number for faster testing
                    apply_inverse=True,
                    interleaved_clifford_idx=interleaved_clifford_idx,
                    seed=123
                )
                
                # Get the recovery Clifford (last element)
                recovery_clifford = rb.CliffordClass(sequence[-1])
                
                # Compute net Clifford from the sequence (excluding recovery)
                net_clifford = rb._calculate_net_clifford(sequence[:-1])
                
                # Compute the product of the net clifford and the recovery clifford
                total_clifford = net_clifford * recovery_clifford
                identity_clifford = rb.CliffordClass(0)  # Identity is at index 0
                
                self.assertTrue(
                    total_clifford == identity_clifford, 
                    "The product of an interleaved sequence and its recovery should be the identity"
                )

    def test_randomized_benchmarking_circuit(self):
        """Test that the randomized benchmarking circuit produces high fidelity."""
        for clifford_group, _ in self.test_cases:
            with self.subTest(clifford_group=clifford_group):
                rb = RandomizedBenchmarking(clifford_group=clifford_group)
                
                # Generate a short RB circuit for testing
                circuit = rb.randomized_benchmarking_circuit(
                    number_of_cliffords=5,  # Short sequence for faster testing
                    apply_inverse=True,
                    seed=123
                )
                
                # Check that circuit has the correct number of qubits
                self.assertEqual(circuit.N, clifford_group)
                
                # Compute unitary and check fidelity with identity
                U = circuit.compute_unitary()
                identity = qeye(dimensions=[2]*clifford_group)
                
                f = average_gate_fidelity(U, identity)
                self.assertAlmostEqual(
                    1, f, places=10, 
                    msg="RB circuit should have high fidelity with identity when recovery is applied"
                )

if __name__ == '__main__':
    unittest.main()