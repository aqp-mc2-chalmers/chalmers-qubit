import unittest
import numpy as np
from qutip import basis, tensor, fidelity, average_gate_fidelity
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import rz
from chalmers_qubit.sarimner import SarimnerModel, SarimnerProcessor, SarimnerCompiler
from chalmers_qubit.base.operations import project_on_qubit
from chalmers_qubit.base.gates import cczs

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class TestSingleQubitGates(unittest.TestCase):
    def setUp(self) -> None:
        self.num_qubits = 1
        transmon_dict = {0: {"frequency": 5.0, "anharmonicity": 0.3}}
        self.model = SarimnerModel(transmon_dict=transmon_dict)
        self.processor = SarimnerProcessor(model=self.model)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_global_phase_gate(self):
        qc = QubitCircuit(self.num_qubits)
        arg_value = np.pi
        qc.add_gate("RX", targets=0, arg_value=np.pi/2)
        qc.add_gate("GLOBALPHASE", targets=0, arg_value=arg_value)
        qc.add_gate("RZ", targets=0, arg_value=np.pi/2)
        self.processor.load_circuit(qc)
        global_phase = self.processor.global_phase
        self.assertEqual(global_phase, arg_value)

    def test_rx_gate(self):
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("RX", targets=0, arg_value=np.pi/2)
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc)
        final_prop = project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        # We want 99.99% Average Gate Fidelity
        self.assertAlmostEqual(1, f, places=4, msg="Precision of RX-gate failed.")

    def test_ry_gate(self):
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("RY", targets=0, arg_value=np.pi/2)
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc)
        final_prop = project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        # We want 99.99% Average Gate Fidelity
        self.assertAlmostEqual(1, f, places=4, msg="Precision of RY-gate failed.")

    def test_rz_gate(self):
        qc = QubitCircuit(self.num_qubits)
        # Since the RZ-gate is virtual we need at least
        # one physical gate to test it.
        qc.add_gate("RZ", targets=0, arg_value=np.pi / 2)
        qc.add_gate("RX", targets=0, arg_value=np.pi / 2)
        # Prepare "plus"-state as initial state
        init_state = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
        result = self.processor.run_state(init_state, qc=qc)
        final_state = project_on_qubit(result.states[-1])
        ideal_state = basis(2, 0)
        f = fidelity(final_state, ideal_state)
        # We want 99.99% average state fidelity
        self.assertAlmostEqual(1, f, places=4, msg="Precision of RZ-gate failed.")

    def test_no_physical_pulse(self):
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("RZ", targets=0, arg_value=np.pi / 2)
        init_state = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
        # Raise value error if quantum circuit contain no physical pulses.
        with self.assertRaises(ValueError):
            self.processor.run_state(init_state, qc=qc)

    def test_x_gate(self):
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("X", targets=0)
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc)
        final_prop = project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        # We want 99.99% Average Gate Fidelity
        self.assertAlmostEqual(1, f, places=4, msg="Precision of X-gate failed.")

    def test_hadamard_gate(self):
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("H", targets=0)
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc)
        # Since we have Virtually rotated the Bloch sphere 180-degrees
        # we need to rotate it back to get the correct propagator.
        final_prop = rz(np.pi) * project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        # We want 99.99% Average Gate Fidelity
        self.assertAlmostEqual(1, f, places=4, msg="Precision of H-gate failed")

    def test_idling_gate(self):
        t_total = 1e3 # time in (ns)
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("IDLE", targets=0, arg_value=t_total)
        # Simulate
        init_state = (basis(3, 0) + basis(3, 1)).unit()
        result = self.processor.run_state(init_state, qc=qc)
        final_state = result.states[-1]
        # Compute the state fidelity
        f = fidelity(final_state, init_state)
        self.assertAlmostEqual(1, f, places=7)

    def test_single_qubit_circuit(self):
        # Create a somewhat random qubit circuit
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("RX", targets=0, arg_value=np.pi/2)
        qc.add_gate("RZ", targets=0, arg_value=np.pi/2)
        qc.add_gate("RY", targets=0, arg_value=np.pi/2)
        qc.add_gate("RZ", targets=0, arg_value=-np.pi/2)
        qc.add_gate("RX", targets=0, arg_value=np.pi)
        # Simulate
        init_state = basis(3, 0)
        result = self.processor.run_state(init_state, qc=qc)
        final_state = result.states[-1]
        # Project final state to the qubit subspace
        qubit_state = project_on_qubit(final_state)
        qubit_init_state = project_on_qubit(init_state)
        # Get the ideal target state
        target_state = qc.compute_unitary() * qubit_init_state
        # Compute the fidelity
        f = fidelity(target_state, qubit_state)
        self.assertAlmostEqual(1, f, places=3)


class TestTwoQubitGates(unittest.TestCase):
    def setUp(self) -> None:
        self.num_qubits = 2
        transmon_dict = {
            0: {"frequency": 5.0, "anharmonicity": 0.3},
            1: {"frequency": 5.4, "anharmonicity": 0.3},
        }
        coupling_dict = {(0,1): 1e-3}
        options = {
            "dt": 0.1,  # time step in (ns)
            "two_qubit_gate": {
                "buffer_time": 0,
                "rise_fall_time": 0,
            },
        }
        self.model = SarimnerModel(transmon_dict=transmon_dict, 
                                   coupling_dict=coupling_dict)
        compiler = SarimnerCompiler(model=self.model, options=options)
        self.processor = SarimnerProcessor(model=self.model, compiler=compiler)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_cz_gate(self):
        # Create a circuit
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("CZ", controls=0, targets=1)
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc, options={"nsteps": 1e5})
        final_prop = project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        # We want 99.99% Average Gate Fidelity
        self.assertAlmostEqual(1, f, places=4)

    def test_iswap_gate(self):
        # Create a circuit
        qc = QubitCircuit(self.num_qubits)
        qc.add_gate("ISWAP", controls=0, targets=1)
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc, options={"nsteps": 1e5})
        # Need to append the "Virtual RZ-gates"
        final_prop = tensor(rz(np.pi), rz(np.pi)) * project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        self.assertAlmostEqual(1, f, places=4)


class TestThreeQubitGates(unittest.TestCase):
    def setUp(self) -> None:
        self.num_qubits = 3
        transmon_dict = {
            0: {"frequency": 5.3, "anharmonicity": 0.3},
            1: {"frequency": 5.1, "anharmonicity": 0.3},
            2: {"frequency": 5.2, "anharmonicity": 0.3},
        }
        coupling_dict = {(0, 1): 1e-3,
                         (0, 2): 1e-3}
        # Compiler options
        options = {
            "dt": 0.1, # time step in (ns)
            "two_qubit_gate": {
                "buffer_time": 0,
                "rise_fall_time": 0,
            },
        }
        self.model = SarimnerModel(transmon_dict=transmon_dict,
                                   coupling_dict=coupling_dict)
        compiler = SarimnerCompiler(model=self.model, options=options)
        self.processor = SarimnerProcessor(model=self.model, compiler=compiler)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_cczs_gate(self):
        # Create a circuit
        qc = QubitCircuit(self.num_qubits)
        qc.user_gates = {"CCZS": cczs}
        qc.add_gate("CCZS", targets=[0,1,2], arg_value=[np.pi/2,np.pi,0])
        # Compute the ideal propagator
        U = qc.compute_unitary()
        # Compute the propagator
        prop = self.processor.run_propagator(qc=qc, options={"nsteps": 1e5})
        final_prop = project_on_qubit(prop)
        # Compute the average gate fidelity
        f = average_gate_fidelity(U, final_prop)
        # We want 99.99% Average Gate Fidelity
        self.assertAlmostEqual(1, f, places=4)

if __name__ == "__main__":
    unittest.main()
