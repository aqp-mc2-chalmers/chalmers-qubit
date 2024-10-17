import warnings
from copy import deepcopy
from typing import Optional
import numpy as np

import qutip
from qutip import propagator, Qobj, QobjEvo
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import Processor, Model
from qutip_qip.device.processor import _pulse_interpolate
from qutip_qip.compiler import GateCompiler

from chalmers_qubit.sarimner.compiler import SarimnerCompiler


class SarimnerProcessor(Processor):
    """
    Initialize a new SarimnerProcessor instance with a quantum model, an optional compiler, and noise models.

    Parameters
    ----------
    model : Model
        The quantum model that defines the physical properties and capabilities of the processor.
    compiler : GateCompiler, optional
        The compiler used to translate quantum gates into executable operations. If not provided,
        a default compiler specific to the model (SarimnerCompiler) is instantiated and used.
    noise : list, optional
        A list of noise models to be added to the processor. Each element in the list should be compatible
        with the processor's noise handling methods.

    Attributes
    ----------
    model : Model
        The model of the quantum processor, storing physical properties.
    _default_compiler : GateCompiler
        Holds the compiler instance being used, either the provided one or a default SarimnerCompiler.
    native_gates : None
        Initially set to None, to be configured with the gate set natively supported by the processor.
    spline_kind: str
        Type of the coefficient interpolation.
    global_phase : float
        The global phase of the quantum state managed by the processor, initialized to 0.
    """

    def __init__(self,
                 model: Model,
                 compiler: Optional[GateCompiler] = None,
                 noise: Optional[list] = None):

        self.model = model

        if compiler is None:
            self._default_compiler = SarimnerCompiler(model=model)
        else:
            self._default_compiler = compiler

        if noise is not None:
            for elem in noise:
                self.add_noise(elem)

        super(SarimnerProcessor, self).__init__(model=self.model)
        self.native_gates = None
        self.spline_kind = "cubic"
        self.global_phase = 0

    def load_circuit(self, qc: QubitCircuit, schedule_mode: str = "ASAP", compiler: Optional[GateCompiler] = None):
        """
        The default routine of compilation.
        It first calls the :meth:`.transpile` to convert the circuit to
        a suitable format for the hardware model.
        Then it calls the compiler and save the compiled pulses.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            Takes the quantum circuit to be implemented.

        schedule_mode: string
            "ASAP" or "ALAP" or None.

        compiler: subclass of :class:`.GateCompiler`
            The used compiler.

        Returns
        -------
        tlist, coeffs: dict of 1D NumPy array
            A dictionary of pulse label and the time sequence and
            compiled pulse coefficients.
        """
        # Choose a compiler and compile the circuit
        if compiler is None and self._default_compiler is not None:
            compiler = self._default_compiler
        if compiler is not None:
            tlist, coeffs = compiler.compile(
                qc.gates, schedule_mode=schedule_mode
            )
        else:
            raise ValueError("No compiler defined.")

        # Update global phase
        self.global_phase = compiler.global_phase

        # Save compiler pulses
        if coeffs is None and tlist is None:
            raise ValueError("The compiled quantum circuit contains no physical pulses.")
        else:
            self.set_coeffs(coeffs)
            self.set_tlist(tlist)
        return tlist, coeffs

    def run_state(
            self,
            init_state=None,
            analytical=False,
            states=None,
            noisy=True,
            solver="mesolve",
            qc=None,
            **kwargs):
        if qc is not None:
            self.load_circuit(qc)
        return super().run_state(init_state, analytical, states, noisy, solver, **kwargs)

    def run_propagator(self, qc: Optional[QubitCircuit] = None, noisy: bool = False, **kwargs):
        """
        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`, optional
            A quantum circuit. If given, it first calls the ``load_circuit``
            and then calculate the evolution.
        noisy: bool, optional
            If noise are included. Default is False.
        **kwargs
            Keyword arguments for the qutip solver.
        Returns
        -------
        prop: list of Qobj or Qobj
            Returns the propagator(s) calculated at times t.
        """
        if qc is not None:
            self.load_circuit(qc)

        # construct qobjevo
        noisy_qobjevo, sys_c_ops = self.get_qobjevo(noisy=noisy)
        drift_qobjevo = self._get_drift_obj().get_ideal_qobjevo(self.dims)
        H = QobjEvo.__add__(noisy_qobjevo, drift_qobjevo)

        # add collpase operators into kwargs
        if "c_ops" in kwargs:
            if isinstance(kwargs["c_ops"], (Qobj, QobjEvo)):
                kwargs["c_ops"] += [kwargs["c_ops"]] + sys_c_ops
            else:
                kwargs["c_ops"] += sys_c_ops
        else:
            kwargs["c_ops"] = sys_c_ops

        # set time
        if "t" in kwargs:
            t = kwargs["t"]
            del kwargs["t"]
        else:
            tlist = self.get_full_tlist()
            if tlist is None:
                raise ValueError("tlist is None.")
            else:
                t = tlist[-1]

        options = kwargs.get("options", qutip.Options())
        if options.get("max_step", 0.0) == 0.0:
            options["max_step"] = self._get_max_step()
        options["progress_bar"] = False
        kwargs["options"] = options

        # compute the propagator
        prop = propagator(H=H, t=t, **kwargs)

        return prop

    def plot_pulses(
            self,
            title=None,
            figsize=(12, 6),
            dpi=None,
            show_axis=False,
            rescale_pulse_coeffs=True,
            num_steps=1000,
            pulse_labels=None,
            use_control_latex=True,
    ):
        """
        Plot the ideal pulse coefficients.

        Parameters
        ----------
        title: str, optional
            Title for the plot.

        figsize: tuple, optional
            The size of the figure.

        dpi: int, optional
            The dpi of the figure.

        show_axis: bool, optional
            If the axis are shown.

        rescale_pulse_coeffs: bool, optional
            Rescale the hight of each pulses.

        num_steps: int, optional
            Number of time steps in the plot.

        pulse_labels: list of dict, optional
            A map between pulse labels and the labels shown in the y axis.
            E.g. ``[{"sx": "sigmax"}]``.
            Pulses in each dictionary will get a different color.
            If not given and ``use_control_latex==False``,
            the string label defined in each :obj:`.Pulse` is used.

        use_control_latex: bool, optional
            Use labels defined in ``Processor.model.get_control_latex``.

        pulse_labels: list of dict, optional
            A map between pulse labels and the labels shown on the y axis.
            E.g. ``["sx", "sigmax"]``.
            If not given and ``use_control_latex==False``,
            the string label defined in each :obj:`.Pulse` is used.

        use_control_latex: bool, optional
            Use labels defined in ``Processor.model.get_control_latex``.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The `Figure` object for the plot.

        axis: list of ``matplotlib.axes._subplots.AxesSubplot``
            The axes for the plot.

        Notes
        -----
        :meth:.Processor.plot_pulses` only works for array_like coefficients.
        """
        if hasattr(self, "get_operators_labels"):
            warnings.warn(
                "Using the get_operators_labels to provide labels "
                "for plotting is deprecated. "
                "Please use get_control_latex instead."
            )
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # choose labels
        if pulse_labels is None:
            if use_control_latex and not hasattr(self.model, "get_control_latex"):
                warnings.warn(
                    "No method get_control_latex defined in the model. "
                    "Switch to using the labels defined in each pulse."
                    "Set use_control_latex=False to turn off the warning."
                )
            if use_control_latex:  # use control labels in the model
                control_labels = deepcopy(self.get_control_latex())
                pulse_labels = control_labels
            else:
                pulse_labels = [{pulse.label: pulse.label for pulse in self.pulses}]

        # If it is a nested list instead of a list of dict, we assume that
        if isinstance(pulse_labels[0], list):
            for ind, pulse_group in enumerate(pulse_labels):
                pulse_labels[ind] = {i: latex for i, latex in enumerate(pulse_group)}

        # create a axis for each pulse
        fig = plt.figure(figsize=figsize, dpi=dpi)
        grids = gridspec.GridSpec(sum([len(d) for d in pulse_labels]), 1)
        grids.update(wspace=0.0, hspace=0.0)

        tlist = np.linspace(0.0, self.get_full_tlist()[-1], num_steps)
        dt = tlist[1] - tlist[0]

        # make sure coeffs start and end with zero, for ax.fill
        tlist = np.hstack(([-dt * 1.0e-20], tlist, [tlist[-1] + dt * 1.0e-20]))
        coeffs = []
        for pulse in self.pulses:
            coeffs.append(_pulse_interpolate(pulse, tlist))

        pulse_ind = 0
        axis = []
        for i, label_group in enumerate(pulse_labels):
            for j, (label, latex_str) in enumerate(label_group.items()):
                try:
                    if "cz_real" in label:  # Combine real and imaginaru part of CZ and plot them togehter
                        real_label = label
                        imag_label = real_label.replace("cz_real", "cz_imag")  # Find corresponding cz_imag
                        pulse_real = self.find_pulse(real_label)
                        pulse_imag = self.find_pulse(imag_label)
                        coeff_real = _pulse_interpolate(pulse_real, tlist)
                        coeff_imag = _pulse_interpolate(pulse_imag, tlist)
                        # Combine real and imaginary parts
                        coeff = np.abs(coeff_real + 1j * coeff_imag)
                    elif "cz_imag" in label:
                        coeff = 0
                    else:
                        pulse = self.find_pulse(label)
                        coeff = _pulse_interpolate(pulse, tlist)
                except KeyError:
                    coeff = np.zeros(tlist.shape)
                if ~np.all(coeff == 0):  # only plot pulse if it is non-zero
                    grid = grids[pulse_ind]
                    ax = plt.subplot(grid)
                    axis.append(ax)
                    ax.fill(tlist, coeff, color_list[i], alpha=0.7)
                    ax.plot(tlist, coeff, color_list[i])
                    if rescale_pulse_coeffs:
                        ymax = np.max(np.abs(coeff)) * 1.1
                    else:
                        ymax = np.max(np.abs(coeffs)) * 1.1
                    if ymax != 0.0:
                        ax.set_ylim((-ymax, ymax))

                    # disable frame and ticks
                    if not show_axis:
                        ax.set_xticks([])
                        ax.spines["bottom"].set_visible(False)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.set_yticks([])
                    ax.set_ylabel(latex_str, rotation=0)
                    pulse_ind += 1
                if i == 0 and j == 0 and title is not None:
                    ax.set_title(title)
        fig.tight_layout()
        return fig, axis
