import unittest
from qutip import tensor, basis
from chalmers_qubit.utils.operations import project_on_qubit

class TestQuditProjection(unittest.TestCase):

    def test_project_single_qudit(self):
        # Create a 3-dimensional qudit state |1>
        qudit = basis(3, 1)
        # Expected qubit state |1>
        expected_qubit = basis(2, 1)
        # Project the qudit onto the qubit subspace
        projected_qubit = project_on_qubit(qudit)
        # Assert that the projection is correct
        self.assertTrue(
            projected_qubit == expected_qubit,
            "The projected qudit does not match the expected qubit state.",
        )

    def test_project_two_qudits(self):
        # Create a tensor product of two 3-dimensional qudits |1> ⊗ |1>
        qudit_tensor = tensor(basis(3, 1), basis(3, 1))
        # Expected tensor product of qubits |1> ⊗ |1>
        expected_qubit_tensor = tensor(basis(2, 1), basis(2, 1))
        # Project the tensor product of qudits onto the qubit subspace
        projected_qubit_tensor = project_on_qubit(qudit_tensor)
        # Assert that the projection is correct
        self.assertTrue(
            projected_qubit_tensor == expected_qubit_tensor,
            "The projected qudit tensor does not match the expected qubit tensor state.",
        )


if __name__ == "__main__":
    unittest.main()
