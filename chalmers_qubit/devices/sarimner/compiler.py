import numpy as np
from typing import Optional
from qutip_qip.device.processor import Model
from qutip_qip.compiler import GateCompiler, Instruction
from qutip_qip.operations import Gate

__all__ = ["SarimnerCompiler"]


class SarimnerCompiler(GateCompiler):
    """
    Compiler for :class:`.SarimnerModel`.
    Frequencies are the unit of GHz and times in ns.

    Parameters
    ----------
    model : Model
        Model object of the superconducting qubit device with hardware parameters.
    options : Optional[dict], optional
        A dictionary of compiler options. If not provided, default options will be used.

    Attributes
    ----------
    num_qubits : int
        The number of component systems (qubits).
    params : dict
        A Python dictionary containing the name and value of hardware parameters,
        such as qubit frequency, anharmonicity, etc.
    gate_compiler : dict
        A Python dictionary in the form of {gate_name: decompose_function}.
        It stores the decomposition scheme for each supported gate.
    phase : list
        List of values indicating how much we have virtually rotated the Bloch sphere of each qubit.
    global_phase : float
        The global phase of the quantum state.
    dt : float
        Time step in nanoseconds.
    two_qubit_gate_options : dict
        Options specific to two-qubit gates, including buffer_time and rise_fall_time.
    single_qubit_gate_options : dict
        Options specific to single-qubit gates, including type, gate_time, pi-pulse amplitude, and std.

    Notes
    -----
    The compiler supports various gates including RZ, RX, RY, X, H, CZ, ISWAP, CCZS, IDLE, and GLOBALPHASE.
    Default options are provided for dt, two-qubit gates, and single-qubit gates, which can be overridden
    by passing custom options during initialization.
    """

    def __init__(self, model: Model, options: Optional[dict] = None):
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

        # Parse options
        default_options = {
            "dt": 0.1,  # in (ns)
            "two_qubit_gate": {
                "buffer_time": 1,  # in (ns)
                "rise_fall_time": 1,  # in (ns)
            },
            "single_qubit_gate": {
                "type": "gaussian",
                "gate_time": 50,  # in (ns)
                "pi_pulse_amplitude": 0.12533148558448476,
                "std": 5,  # in (ns)
            },
        }

        if options is None:
            options = {}

        self.options = self._merge_options(default_options, options)

        # Set attributes based on parsed options
        self.dt = self.options["dt"]  # time step in (ns)
        self.two_qubit_gate_options = self.options["two_qubit_gate"]
        self.single_qubit_gate_options = self.options["single_qubit_gate"]
        # initialise the phase for all the qubits to 0.
        self.phase = [0] * self.num_qubits
        # initialise the global phase to 0.
        self.global_phase = 0

    def _merge_options(self, default_options, user_options):
        """Recursively merge user options into default options."""
        for key, value in user_options.items():
            if isinstance(value, dict) and key in default_options:
                default_options[key] = self._merge_options(default_options[key], value)
            else:
                default_options[key] = value
        return default_options

    @staticmethod
    def _envelope_function(t, t_gate, t_rise_fall, t_buffer):
        # Total gate time including rise and fall times
        total_time = t_gate + 2 * t_rise_fall + 2 * t_buffer
        if t_rise_fall > 0:
            # Calculate rise window
            rise_window = np.sin(np.pi * (t - t_buffer) / (2 * t_rise_fall)) ** 2
            # Calculate fall window
            fall_window = np.cos(np.pi * (t - (total_time - t_rise_fall - t_buffer)) / (2 * t_rise_fall)) ** 2
        else:
            rise_window = 0
            fall_window = 0

        # Initialize array with zeros
        arr = np.zeros_like(t)

        # Set rise window values
        arr = np.where((t >= t_buffer) & (t < (t_buffer + t_rise_fall)), rise_window, arr)
        # Set constant window values
        arr = np.where((t >= (t_buffer + t_rise_fall)) & (t < (total_time - t_rise_fall - t_buffer)), 1, arr)
        # Set fall window values
        arr = np.where((t >= (total_time - t_rise_fall - t_buffer)) & (t < (total_time - t_buffer)), fall_window, arr)
        return arr

    @staticmethod
    def _gaussian_drag(t: np.ndarray, op: str, args: dict) -> np.ndarray:
        # Amplitude, needs to be optimized to get perfect pi-pulse or pi/2-pulse
        amp = args['amp']
        # DRAG-parameter, needs to be optimized to get no phase errors for a pi/2-pulse
        qscale = args['qscale']
        drive_freq = args['freq']
        phase = args['phase']
        L = args['gatetime']
        sigma = args['sigma']
        Omega_x = amp * np.exp(-pow(t - 0.5 * L, 2) / (2 * pow(sigma, 2)))
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

        # Use the parsed options for single qubit gates
        sigma = self.single_qubit_gate_options["std"]
        amp = self.single_qubit_gate_options["pi_pulse_amplitude"] * gate.arg_value / np.pi
        t_gate = self.single_qubit_gate_options["gate_time"]

        # parameters
        alpha = self.params["transmons"][target]["anharmonicity"]
        omega_qubit = self.params["transmons"][target]["frequency"]
        rotating_freq = omega_qubit
        omega_drive = rotating_freq - omega_qubit

        # arguments
        args = {
            "amp": amp,
            "qscale": -0.5 / alpha,
            "phase": phase,
            "freq": omega_drive,
            "gatetime": t_gate,
            "sigma": sigma,
        }

        # time list
        tlist = np.arange(0, t_gate + self.dt, self.dt)

        # set start and end to zero
        coeff_x = self._gaussian_drag(tlist, "x", args)
        coeff_y = self._gaussian_drag(tlist, "y", args)

        pulse_info = [
            # (control label, coeff)
            ("x" + str(target), coeff_x),
            ("y" + str(target), coeff_y),
        ]

        return [Instruction(gate, tlist, pulse_info, t_gate)]

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

        # Coupling strength
        g = self.params["couplings"][q1, q2]

        # Get frequencies
        alpha1 = self.params["transmons"][q1]["anharmonicity"]
        delta = alpha1

        # Total time of gate
        t_gate = np.pi / (np.sqrt(2) * abs(g))
        # Rise and fall time of envelope
        rise_fall_time = self.two_qubit_gate_options["rise_fall_time"]
        # Buffer time
        t_buffer = self.two_qubit_gate_options["buffer_time"]
        # Total time
        t_total = t_gate + 2 * rise_fall_time + 2 * t_buffer
        # Time list
        tlist = np.arange(0, t_total + self.dt, self.dt)

        coeff = self._envelope_function(tlist, t_gate, rise_fall_time, t_buffer)

        pulse_info = [
            ("cz_real" + str(q1) + str(q2), coeff * np.cos(delta * tlist)),
            ("cz_imag" + str(q1) + str(q2), coeff * np.sin(delta * tlist)),
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
            Gate("RY", [q], None, arg_value=np.pi / 2),
            None)
        self.gate_compiler["GLOBALPHASE"](
            Gate("GLOBALPHASE", [q], None, arg_value=np.pi / 2),
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

        # Coupling strength
        g = self.params["couplings"][q1, q2]

        # Total time of gate
        t_gate = np.pi / (2 * abs(g))
        # Rise and fall time of envelope
        rise_fall_time = self.two_qubit_gate_options["rise_fall_time"]
        # Buffer time
        t_buffer = self.two_qubit_gate_options["buffer_time"]
        # Total time
        t_total = t_gate + 2 * rise_fall_time + 2 * t_buffer
        # Time list
        tlist = np.arange(0, t_total + self.dt, self.dt)

        coeff = self._envelope_function(tlist, t_gate, rise_fall_time, t_buffer)

        pulse_info = [("iswap" + str(q1) + str(q2), coeff)]
        # ADD VIRTUAL-Z GATES TO CORRECT THE PHASE
        self.phase[q1] -= np.pi
        self.phase[q2] -= np.pi
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
        alpha1 = self.params["transmons"][q1]["anharmonicity"]

        g1 = self.params["couplings"][q1, q2]
        g2 = self.params["couplings"][q1, q3]
        g = np.sqrt(np.abs(g1) ** 2 + np.abs(g2) ** 2)

        # Total time of gate
        t_gate = np.pi / (np.sqrt(2) * g)
        # Rise and fall time of envelope
        rise_fall_time = self.two_qubit_gate_options["rise_fall_time"]
        # Buffer time
        t_buffer = self.two_qubit_gate_options["buffer_time"]
        # Total time
        t_total = t_gate + 2 * rise_fall_time + 2 * t_buffer
        # Time list
        tlist = np.arange(0, t_total + self.dt, self.dt)

        coeff = self._envelope_function(tlist, t_gate, rise_fall_time, t_buffer)

        pulse_info = [
            ("cz_real" + str(q1) + str(q2), coeff * np.cos(alpha1 * tlist)),
            ("cz_imag" + str(q1) + str(q2), coeff * np.sin(alpha1 * tlist)),
            ("cz_real" + str(q1) + str(q3), coeff * np.cos(alpha1 * tlist)),
            ("cz_imag" + str(q1) + str(q3), coeff * np.sin(alpha1 * tlist)),
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
        elif (0 <= idle_time < self.dt):
            # Skip the gate if idle-time is less than dt.
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
