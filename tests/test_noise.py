import unittest
import numpy as np
from qutip import basis, destroy, Options, Qobj, fidelity, tensor
from qutip_qip.circuit import QubitCircuit
from chalmers_qubit.sarimner import SarimnerModel, SarimnerProcessor, DecoherenceNoise, ZZCrossTalk
from chalmers_qubit.base.operations import project_on_qubit


def _bloch_redfield(psi: Qobj, t1, t2, t: float) -> Qobj:
    psi_data = psi.full()
    alpha = psi_data[0, 0]
    beta = psi_data[1, 0]
    Gamma_1 = 1 / t1
    Gamma_2 = 1 / t2
    rho_00 = 1 + (abs(alpha) ** 2 - 1) * np.exp(-Gamma_1 * t)
    rho_01 = alpha * np.conj(beta) * np.exp(-Gamma_2 * t)
    rho_10 = np.conj(alpha) * beta * np.exp(-Gamma_2 * t)
    rho_11 = abs(beta) ** 2 * np.exp(-Gamma_1 * t)
    rho = np.array([[rho_00, rho_01], [rho_10, rho_11]])
    return Qobj(rho)

class TestSingleQubitNoise(unittest.TestCase):
    def setUp(self) -> None:
        transmon_dict = {
            0: {"frequency": 5.0, "anharmonicity": -0.30},
        }
        self.decoherence_dict = {
            0: {"t1": 50e3, "t2": 40e3},
        }
        # Create the processor with the given hardware parameters
        # Load the physical parameters onto the model
        self.model = SarimnerModel(transmon_dict=transmon_dict)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def _spin_echo(self, tau:float):
        # Circuit for measuring Hahn-echo
        circuit = QubitCircuit(1)
        circuit.add_gate("RX", targets=0, arg_value=np.pi / 2)
        circuit.add_gate("IDLE", targets=0, arg_value=tau / 2)
        circuit.add_gate("RX", targets=0, arg_value=np.pi)
        circuit.add_gate("IDLE", targets=0, arg_value=tau / 2)
        circuit.add_gate("RX", targets=0, arg_value=np.pi / 2)
        return circuit

    def test_bloch_redfield(self):
        init_state = (basis(3,0) + basis(3,1)).unit()
        t_total = 80*1e3 # ns
        tlist = np.linspace(0, t_total, 10)
        # Add noise
        noise = [DecoherenceNoise(self.decoherence_dict)]
        # Create the processor with the given hardware parameters
        sarimner = SarimnerProcessor(model=self.model, noise=noise)
        # Master equation simulation
        result = sarimner.run_state(init_state, tlist=tlist)
        final_state = project_on_qubit(result.states[-1])
        t1 = self.decoherence_dict[0]["t1"]
        t2 = self.decoherence_dict[0]["t2"]
        predicted_state = _bloch_redfield(init_state, t1, t2, t_total)
        f = fidelity(final_state, predicted_state)
        self.assertAlmostEqual(1, f, places=6)

    def test_relaxation(self):
        # Add noise
        noise = [DecoherenceNoise(self.decoherence_dict)]
        # Create the processor with the given hardware parameters
        sarimner = SarimnerProcessor(model=self.model, noise=noise)
        # master equation simulation
        init_state = basis(3, 1)
        a = destroy(3)
        e_ops = [a.dag() * a]
        t_total = 200*1e3 # ns
        tlist = np.linspace(0, t_total, 30)
        result = sarimner.run_state(init_state, tlist=tlist, e_ops=e_ops)
        # Get the population of the |0> state
        simulated_relaxation = result.expect[0]
        t1 = self.decoherence_dict[0]["t1"]
        theoretical_relaxation = np.exp(-tlist / t1)
        # Verify that simulated relaxation follows the theoretical prediction within a tolerance
        np.testing.assert_allclose(
            simulated_relaxation,
            theoretical_relaxation,
            rtol=0.05,
            atol=0.0,
            err_msg="Simulated relaxation does not match theoretical prediction.",
        )

    def test_spin_echo(self):
        # Do spin echo / Hahn echo experiment
        noise = [DecoherenceNoise(self.decoherence_dict)]  # Create noise
        # Initial state for the simulation
        init_state = basis(3, 0)
        # Idling time list in (ns)
        idle_tlist = np.linspace(0,100*1e3,5)
        # Expectation value of number operator
        e_ops = [destroy(3).dag()*destroy(3)]
        # Create the processor with the given hardware parameters
        sarimner = SarimnerProcessor(model=self.model, noise=noise)
        # Population list
        simulated_decay = []
        # Total time list
        total_time = []
        # Loop over times
        for tau in idle_tlist:
            times, coeffs = sarimner.load_circuit(self._spin_echo(tau))
            # Get the total time of the simulation
            t_total = sarimner.get_full_tlist()[-1]
            # Run master equation simulation and save only start and end times
            result = sarimner.run_state(init_state, tlist=[0,t_total] , e_ops=e_ops, options=Options(nsteps=5e6))
            # Save results
            simulated_decay.append(result.expect[0][-1])
            total_time.append(t_total)
        # Theoretical decay for spin echo
        t2 = self.decoherence_dict[0]["t2"]
        theoretical_decay = (np.exp(-np.array(total_time) / t2) + 1) / 2
        np.testing.assert_allclose(
            1-np.array(simulated_decay),
            theoretical_decay,
            rtol=0.05,
            atol=0.0,
            err_msg="Simulated decay does not match theoretical prediction.",
        )


class TestTwoQubitNoise(unittest.TestCase):
    def setUp(self) -> None:
        transmon_dict = {
            0: {"frequency": 5.0, "anharmonicity": -0.30},
            1: {"frequency": 5.4, "anharmonicity": -0.30},
        }
        coupling_dict = {(0,1): 1e-3}
        self.model = SarimnerModel(
            transmon_dict=transmon_dict,
            coupling_dict=coupling_dict
        )
        # Cross talk matrix (GHz)
        self.cross_talk_dict = {(0,1): 2e-4}
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_zz_cross_talk(self):
        # Add noise
        noise = [ZZCrossTalk(self.cross_talk_dict)]
        # Create the processor with the given hardware parameters
        sarimner = SarimnerProcessor(model=self.model, noise=noise)
        # Prepare qubit in 11 state
        rho = tensor(basis(3, 1), basis(3, 1))
        # Transmon Hamiltonian with a slight detuning
        t_total = 1000  # in (ns)
        tlist = np.linspace(0, t_total, 50)
        # Master equation simulation
        result = sarimner.run_state(rho, tlist=tlist)
        final_state = project_on_qubit(result.states[-1])
        simulated_phase = np.angle(final_state.full()[3,0])
        zz = 2*np.pi*self.cross_talk_dict[(0,1)]
        predicted_phase = np.angle(np.exp(-1j * zz * t_total))
        # Check that predicted phase is equal to simulated phase
        self.assertAlmostEqual(predicted_phase, simulated_phase, places=5)

if __name__ == "__main__":
    unittest.main()
