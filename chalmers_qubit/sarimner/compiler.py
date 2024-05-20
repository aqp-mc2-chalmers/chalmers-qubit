import numpy as np
from qutip_qip.device.processor import Model
from qutip_qip.compiler import GateCompiler, Instruction
from qutip_qip.operations import Gate

__all__ = ["SarimnerCompiler"]

class SarimnerCompiler(GateCompiler):
    """
    Compiler for :class:`.ChalmersQubits`.
    Compiled pulse strength is in the unit of GHz.

    Parameters
    ----------
    model: Model
        model object of the superconducting qubit device with hardware parameters

    Attributes
    ----------
    num_qubits: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as qubit frequency, anharmonicity etc.

    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    phase: list
        List with values of how much we have virtually rotated the Bloch sphere of each qubit.

    global_phase: float
        The global phase of the quantum state.
    """

    def __init__(self, model:Model):
        self.num_qubits = model.num_qubits
        self.params = model.params

        super().__init__(num_qubits=self.num_qubits, params=self.params)
        self.gate_compiler.update(
            {
                "RZ": self.rz_compiler,
                "RX": self.rx_compiler,
                "RY": self.ry_compiler,
                "X": self.x_compiler,
                "H": self.h_compiler,
                "CZ": self.cz_compiler,
                "ISWAP": self.iswap_compiler,
                "CCZS": self.cczs_compiler,
                "IDLE": self.idle_compiler,
                "GLOBALPHASE": self.globalphase_compiler,
            }
        )
        # initialise the phase for all the qubits to 0.
        self.phase = [0] * self.num_qubits
        # initialise the global phase to 0.
        self.global_phase = 0

    def _coupling(self, t, op: str, args: dict) -> np.ndarray:
        omega_drive = args['drivefreq']
        delta = args['detuning']
        phi = args['phase']
        g = args['coupling_strength']
        if op == "(xx+yy)":
            coeff = g * np.cos(omega_drive * t + phi) * np.cos(delta * t)
        elif op == "(yx-xy)":
            coeff = g * np.cos(omega_drive * t + phi) * np.sin(delta * t)
        return coeff

    def _drag(self, t: np.ndarray, op: str, args: dict) -> np.ndarray:
        # Amplitude, needs to be optimized to get perfect pi-pulse or pi/2-pulse
        amp = args['amp']
        # DRAG-parameter, needs to be optimized to get no phase errors for a pi/2-pulse
        qscale = args['qscale']
        drive_freq = args['freq']
        phase = args['phase']
        L = args['gatetime']
        sigma = args['sigma']
        Omega_x = amp * np.exp(-pow(t - 0.5 * L, 2)
                               / (2 * pow(sigma, 2)))
        Omega_y = qscale * (t - 0.5 * L) / pow(sigma, 2) * Omega_x

        if op == "x":
            coeff = Omega_x * np.cos(drive_freq * t + phase) \
                  + Omega_y * np.sin(drive_freq * t + phase)
        elif op == "y":
            coeff = Omega_x * np.sin(drive_freq * t + phase) \
                  - Omega_y * np.cos(drive_freq * t + phase)
        return coeff

    def _rotation_gate(self, gate, phase):
        """
        Compiler for the rotational gate that lies along the equator gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """

        # target qubit
        target = gate.targets[0]

        # Gaussian width in ns
        sigma = 5

        # amplitude
        amp = 0.12533148558448476 * gate.arg_value / np.pi

        # parameters
        alpha = self.params["alpha"][target]
        omega_qubit = self.params["wq"][target]
        rotating_freq = self.params["wr"][target]
        omega_drive = rotating_freq - omega_qubit

        # Gate time in ns
        t_total = 50
        # Full width half maximum of gaussian
        fwhm = int(2.354 * sigma)
        # Number of samples
        n_steps = 3 * (omega_drive * t_total) / (2*np.pi) + fwhm

        tlist = np.linspace(0, t_total, int(n_steps))

        # arguments
        args = {
            "amp": amp,
            "qscale": -0.5 / alpha,
            "phase": phase,
            "freq": omega_drive,
            "gatetime": t_total,
            "sigma": sigma,
        }

        # set start and end to zero
        coeff_x = self._drag(tlist, "x", args)
        coeff_y = self._drag(tlist, "y", args)

        pulse_info = [
            # (control label, coeff)
            ("x" + str(target), coeff_x),
            ("y" + str(target), coeff_y),
        ]

        return [Instruction(gate, tlist, pulse_info, t_total)]

    def rz_compiler(self, gate, args):
        """
        Compiler for the Virtual-RZ gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        """
        q = gate.targets[0]  # target qubit
        # This corresponds to clockwise rotation of the Bloch-sphere around the Z-axis.
        self.phase[q] -= gate.arg_value

    def ry_compiler(self, gate, args):
        """
        Compiler for the RY gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        # target qubit
        target = gate.targets[0]
        # phase
        phase = self.phase[target] + np.pi / 2
        return self._rotation_gate(gate=gate, phase=phase)

    def rx_compiler(self, gate, args):
        """
        Compiler for the RX gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """

        # target qubit
        target = gate.targets[0]
        # phase
        phase = self.phase[target]
        return self._rotation_gate(gate=gate, phase=phase)

    def x_compiler(self, gate, args):
        """
        Compiler for the Hadamard gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        """
        q = gate.targets[0]  # target qubit
        rx_gate = self.gate_compiler["RX"](
            Gate("RX", [q], None, arg_value=np.pi), 
            None)
        self.gate_compiler["GLOBALPHASE"](
            Gate("GLOBALPHASE", [q], None, arg_value=np.pi / 2), 
            None)
        # we only return the physical gate
        return rx_gate

    def h_compiler(self, gate, args):
        """
        Compiler for the Hadamard gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        """
        q = gate.targets[0]  # target qubit

        self.gate_compiler["RZ"](
            Gate("RZ", [q], None, arg_value=np.pi), 
            None)
        ry_gate = self.gate_compiler["RY"](
            Gate("RY", [q], None, arg_value=np.pi/2), 
            None)
        self.gate_compiler["GLOBALPHASE"](
            Gate("GLOBALPHASE", [q], None, arg_value=np.pi/2),
            None)
        # we only return the physical gate
        return ry_gate

    def iswap_compiler(self, gate, args):
        """
        Compiler for ISWAP gate.

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        q1 = gate.controls[0]
        q2 = gate.targets[0]

        omega1 = self.params["wq"][q1]
        omega1_rot = self.params["wr"][q1]
        omega2 = self.params["wq"][q2]
        omega2_rot = self.params["wr"][q2]
        coupling_strength = self.params["coupling_matrix"][q1,q2]

        # Drive frequency
        omega_d = abs(omega1 - omega2)
        # Rotating frame detuning
        detuning = (omega1_rot - omega2_rot)
        # Total time of gate
        t_total = np.pi / coupling_strength
        # sampling frequency should be at least two times the signal frequency
        # (Here we have choosen 2.1 times, but we could play around with this value)
        f_s = 2.1 * (omega_d + abs(detuning)) / (2 * np.pi)
        n_steps = int(t_total * f_s)
        tlist = np.linspace(0, t_total, n_steps)

        # Check if qubit q1 and q2 are coupled together
        if coupling_strength == 0:
            raise ValueError(f"Qubit {q1} and {q2} are not coupled together.")

        args = {
            "coupling_strength": coupling_strength,
            "drivefreq": omega_d,
            "detuning": detuning,
            "phase": 0,
        }

        pulse_info = [
            ("(xx+yy)" + str(q1) + str(q2), self._coupling(tlist, "(xx+yy)", args)),
            ("(yx-xy)" + str(q1) + str(q2), self._coupling(tlist, "(yx-xy)", args)),
        ]
        # ADD VIRTUAL-Z GATES TO CORRECT THE PHASE
        self.phase[q1] -= np.pi
        self.phase[q2] -= np.pi
        return [Instruction(gate, tlist, pulse_info, t_total)]

    def cz_compiler(self, gate, args):
        """
        Compiler for CZ gate.

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        q1 = gate.controls[0]
        q2 = gate.targets[0]

        omega1 = self.params["wq"][q1]
        omega1_rot = self.params["wr"][q1]
        omega2 = self.params["wq"][q2]
        omega2_rot = self.params["wr"][q2]
        alpha1 = self.params["alpha"][q1]
        coupling_strength = self.params["coupling_matrix"][q1,q2]

        # Drive frequency
        omega_d = abs(omega1 + alpha1 - omega2)
        # Rotating frame detuning
        detuning = omega1_rot - omega2_rot

        # Time of gate
        t_total = np.sqrt(2) * np.pi / coupling_strength
        # sampling frequency should be at least two times the maximum signal frequency
        # (Here we have choosen 2.1 times, but we could play around with this value)
        f_s = 2.1 * (omega_d + abs(detuning)) / (2*np.pi)
        n_steps = int(t_total * f_s)
        tlist = np.linspace(0, t_total, n_steps)

        # Check if qubit q1 and q2 are coupled together
        if coupling_strength == 0:
            raise ValueError(f"Qubit {q1} and {q2} are not coupled together.")

        args = {'coupling_strength': coupling_strength,
                'drivefreq': omega_d,
                'detuning': detuning,
                'phase': 0}

        pulse_info = [("(xx+yy)" + str(q1) + str(q2), self._coupling(tlist, "(xx+yy)", args)),
                      ("(yx-xy)" + str(q1) + str(q2), self._coupling(tlist, "(yx-xy)", args))]

        return [Instruction(gate, tlist, pulse_info, t_total)]

    def cczs_compiler(self, gate, args):
        """
        Compiler for CCZS gate.

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        q1 = gate.targets[0]  # this is the control qubit
        q2 = gate.targets[1]
        q3 = gate.targets[2]
        theta, phi, gamma = gate.arg_value

        # define parameters
        omega1 = self.params["wq"][q1]
        omega2 = self.params["wq"][q2]
        omega3 = self.params["wq"][q3]
        omega1_rot = self.params["wr"][q1]
        omega2_rot = self.params["wr"][q2]
        omega3_rot = self.params["wr"][q3]
        alpha1 = self.params["alpha"][q1]
        g1 = self.params["coupling_matrix"][q1,q2]
        g2 = self.params["coupling_matrix"][q1,q3]

        # Check if qubit q1 and q2 and q1 and q3 are coupled together
        if g1 == 0:
            raise ValueError(f"Qubit {q1} and {q2} are not coupled together.")
        elif g2 == 0:
            raise ValueError(f"Qubit {q1} and {q3} are not coupled together.")

        omega1_drive = abs(omega1 + alpha1 - omega2)
        omega2_drive = abs(omega1 + alpha1 - omega3)

        detuning1 = omega1_rot - omega2_rot
        detuning2 = omega1_rot - omega3_rot

        fs_1 = omega1_drive + abs(detuning1)
        fs_2 = omega2_drive + abs(detuning2)
        # sampling frequency should be at least two times the maximum signal frequency
        # (Here we have choosen 2.1 times, but we could play around with this value)
        fs = 2.1 * max([fs_1, fs_2]) / (2 * np.pi)

        # We do simultaneous drive
        args1 = {"coupling_strength": g1,
                 "drivefreq": omega1_drive,
                 "detuning": detuning1,
                 "phase": 0}
        args2 = {"coupling_strength": g2,
                 "drivefreq": omega2_drive,
                 "detuning": detuning2,
                 "phase": phi - np.pi}

        # gate time
        g = np.sqrt(np.abs(g1)**2 + np.abs(g2)**2)
        t_total = np.sqrt(2) * np.pi / g
        n_steps = int(t_total * fs)
        tlist = np.linspace(0, t_total, n_steps)

        op_1 = "(xx+yy)"
        op_2 = "(yx-xy)"
        pulse_info = [
            (op_1 + str(q1) + str(q2), self._coupling(tlist, op_1, args1)),
            (op_2 + str(q1) + str(q2), self._coupling(tlist, op_2, args1)),
            (op_1 + str(q1) + str(q3), self._coupling(tlist, op_1, args2)),
            (op_2 + str(q1) + str(q3), self._coupling(tlist, op_2, args2))
        ]

        return [Instruction(gate, tlist, pulse_info, t_total)]

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
        return [Instruction(gate, self.global_phase, [])]

    def idle_compiler(self, gate, args):
        """
        Compiler for the IDLE gate
        """
        # target qubit
        q = gate.targets[0]
        idle_time = gate.arg_value

        if idle_time < 0:
            ValueError("Error: Idle time cannot be negative.")
        elif idle_time == 0:
            # Skip the gate if idle-time is zero.
            pass
        else:
            # Take one time step for every nano second
            n_steps = int(np.ceil(idle_time))
            # Make sure that we at least take two steps
            if n_steps < 2:
                n_steps = 2
            tlist = np.linspace(0, idle_time, n_steps)
            coeff = np.zeros(n_steps)
            # Create a pulse with zero amplitude
            pulse_info = [
                ("x" + str(q), coeff),
                ("y" + str(q), coeff)
            ]
            return [Instruction(gate, tlist, pulse_info, idle_time)]
