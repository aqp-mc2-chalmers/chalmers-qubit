import unittest
import numpy as np
from qutip import qeye, average_gate_fidelity
from chalmers_qubit.utils import randomized_benchmarking_sequence, randomized_benchmarking_circuit, calculate_net_clifford
from chalmers_qubit.utils.randomized_benchmarking.clifford_decomposition import SingleQubitClifford, TwoQubitClifford, Clifford

class TestRandomizedBenchmarkingSequence(unittest.TestCase):

    def setUp(self) -> None:
        self.sequence = randomized_benchmarking_sequence(
            number_of_cliffords=100,
            apply_inverse=True,
            clifford_group=1,
            seed=123
        )
        self.interleaved_sequence = randomized_benchmarking_sequence(
            number_of_cliffords=4,
            apply_inverse=True,
            clifford_group=2,
            interleaved_clifford_idx=10_4368,
            seed=123
        )
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_single_qubit_clifford_equality(self):
        c1 = SingleQubitClifford(5)
        c2 = SingleQubitClifford(5)
        c3 = SingleQubitClifford(6)

        self.assertEqual(c1, c2, "Cliffords with the same index should be equal")
        self.assertNotEqual(c1, c3, "Cliffords with different indices should not be equal")

    def test_two_qubit_clifford_equality(self):
        c1 = TwoQubitClifford(5)
        c2 = TwoQubitClifford(5)
        c3 = TwoQubitClifford(6)

        self.assertEqual(c1, c2, "Cliffords with the same index should be equal")
        self.assertNotEqual(c1, c3, "Cliffords with different indices should not be equal")

    def test_single_qubit_identity_clifford(self):
        ptm = SingleQubitClifford(0).pauli_transfer_matrix
        eye = np.identity(4)
        self.assertTrue(np.array_equal(eye, ptm), "The identity Clifford should be equal to itself")

    def test_two_qubit_identity_clifford(self):
        ptm = TwoQubitClifford(0).pauli_transfer_matrix
        eye = np.identity(16)
        self.assertTrue(np.array_equal(eye, ptm), "The identity Clifford should be equal to itself")

    def test_get_inverse(self):
        idx = 10
        c = SingleQubitClifford(idx)
        c_inv = c.get_inverse()
        # Check that the inverse is correct
        result = c_inv * c
        identity = SingleQubitClifford(0)
        # Check that the inverse is correct
        self.assertTrue(result == identity, "The product of a Clifford and its inverse should be the identity")

    def test_inverse_gate_creates_identity_transfer_matrix(self):
        # The last gate is the recovery gate
        recovery_idx = self.sequence[-1]
        sequence_without_recovery = self.sequence[:-1]

        # Compute net Clifford from the sequence
        net_clifford = calculate_net_clifford(sequence_without_recovery, SingleQubitClifford)
        recovery_clifford = SingleQubitClifford(recovery_idx)

        composed_matrix = np.dot(net_clifford.pauli_transfer_matrix, recovery_clifford.pauli_transfer_matrix)
        identity = np.eye(composed_matrix.shape[0])
        np.testing.assert_allclose(composed_matrix, identity, atol=1e-10)

    def test_randomized_benchmarking_circuit(self):
        circuit = randomized_benchmarking_circuit(self.sequence)
        U = circuit.compute_unitary()
        f = average_gate_fidelity(U, qeye(2))

        self.assertAlmostEqual(1, f, places=5, msg="Precision of randomized benchmarking circuit failed.")

    def test_interleaved_randomized_benchmarking(self):
        sequence = randomized_benchmarking_sequence(
            number_of_cliffords=100,
            apply_inverse=True,
            clifford_group=2,
            interleaved_clifford_idx=10_4368,
            seed=123
        )
        # sequence = self.interleaved_sequence
        sequence_without_recovery = sequence[:-1]
        recovery_idx = sequence[-1]
        net_clifford = calculate_net_clifford(sequence_without_recovery, TwoQubitClifford)
        recovery_clifford = TwoQubitClifford(recovery_idx)

        composed_matrix = np.dot(net_clifford.pauli_transfer_matrix, recovery_clifford.pauli_transfer_matrix)

        identity = np.eye(composed_matrix.shape[0])
        np.testing.assert_allclose(composed_matrix, identity, atol=1e-8)


class TestCliffordCaching(unittest.TestCase):

    def setUp(self):
        TwoQubitClifford._PTM_CACHE.clear()
        TwoQubitClifford._GATE_DECOMP_CACHE.clear()

    def test_two_qubit_ptm_cache(self):
        idx1, idx2 = 0, 5

        # Compute PTMs
        c1 = TwoQubitClifford(idx=idx1).pauli_transfer_matrix

        self.assertIn(idx1, TwoQubitClifford._PTM_CACHE)

        np.testing.assert_array_equal(c1, TwoQubitClifford._PTM_CACHE[idx1])

        # Check that a new instance has the share the same cash
        c2 = TwoQubitClifford(idx=idx2)
        np.testing.assert_array_equal(c2._PTM_CACHE[idx1], c1)

    def test_two_qubit_gate_decomp_cache(self):
        idx1, idx2 = 0, 5

        decomp1 = TwoQubitClifford(idx=idx1).gate_decomposition

        self.assertIn(idx1, TwoQubitClifford._GATE_DECOMP_CACHE)
        self.assertEqual(decomp1, TwoQubitClifford._GATE_DECOMP_CACHE[idx1])

        # Check that a new instance has the share the same cash
        decomp2 = TwoQubitClifford(idx=idx2)
        self.assertEqual(decomp2._GATE_DECOMP_CACHE[idx1], decomp1)

class TestCliffordHashTable(unittest.TestCase):

    def setUp(self):
        SingleQubitClifford.CLIFFORD_HASH_TABLE.clear()
        TwoQubitClifford.CLIFFORD_HASH_TABLE.clear()

    def test_clifford_hash_table_generation(self):
        """
        Test the generation of the hash table for both SingleQubitClifford and TwoQubitClifford classes.
        This test checks if the hash table is correctly populated with the expected values.
        """

        idx = 5 # Some arbitrary Clifford idx
        test_cases = [
            (SingleQubitClifford, idx),
            (TwoQubitClifford, idx),
        ]

        for CliffordClass, idx in test_cases:
            with self.subTest(CliffordClass=CliffordClass.__name__):
                c = CliffordClass(idx)
                ptm = c.pauli_transfer_matrix

                # Trigger hash table population
                c.find_clifford_index(ptm)

                # Check if the hash table was correctly populated
                hash_value = c._hash_matrix(ptm)
                # Assert that the hash value is in the hash table
                self.assertIn(hash_value, CliffordClass.CLIFFORD_HASH_TABLE)
                # Assert that the index in the hash table matches the original index
                self.assertEqual(CliffordClass.CLIFFORD_HASH_TABLE[hash_value], idx)

if __name__ == '__main__':
    unittest.main()