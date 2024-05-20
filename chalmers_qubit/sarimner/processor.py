from typing import Optional

import qutip
from qutip import propagator, Qobj, QobjEvo
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import Processor, Model
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
                 model:Model,
                 compiler:Optional[GateCompiler] = None,
                 noise:Optional[list] = None):

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

    def load_circuit(self, qc:QubitCircuit, schedule_mode:str="ASAP", compiler:Optional[GateCompiler]=None):
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
        return super().run_state(init_state,analytical,states,noisy,solver,**kwargs)

    def run_propagator(self, qc:Optional[QubitCircuit]=None, noisy:bool=False, **kwargs):
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
