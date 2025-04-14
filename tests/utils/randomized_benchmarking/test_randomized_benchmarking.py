import unittest
import numpy as np
from qutip import qeye, average_gate_fidelity
from qutip_qip.circuit import QubitCircuit
from chalmers_qubit.utils import randomized_benchmarking_sequence, randomized_benchmarking_circuit, calculate_net_clifford
from chalmers_qubit.utils.randomized_benchmarking.clifford_decomposition import SingleQubitClifford

class TestRandomizedBenchmarkingSequence(unittest.TestCase):

    def setUp(self) -> None:
        self.sequence = randomized_benchmarking_sequence(
            number_of_cliffords=10,
            apply_inverse=True,
            clifford_group=1,
            seed=123
        )
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_inverse_gate_creates_identity_transfer_matrix(self):
        # The last gate is the recovery gate
        recovery_idx = self.sequence[-1]
        sequence_without_recovery = self.sequence[:-1]

        # Compute net Clifford from the sequence
        net_clifford = calculate_net_clifford(sequence_without_recovery)
        recovery_clifford = SingleQubitClifford(recovery_idx)

        composed_matrix = np.dot(net_clifford.pauli_transfer_matrix, recovery_clifford.pauli_transfer_matrix)
        identity = np.eye(composed_matrix.shape[0])
        np.testing.assert_allclose(composed_matrix, identity, atol=1e-10)

    def test_randomized_benchmarking_circuit(self):
        circuit = randomized_benchmarking_circuit(self.sequence)
        U = circuit.compute_unitary()
        f = average_gate_fidelity(U, qeye(2))

        self.assertAlmostEqual(1, f, places=5, msg="Precision of randomized benchmarking circuit failed.")


if __name__ == '__main__':
    unittest.main()
