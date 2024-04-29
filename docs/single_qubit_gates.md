# Single-qubit gates

## Rotational X and Y-gates
On this page we are describing how a single-qubit gate that implements rotation around the $x$- or $y$-axis on the Bloch sphere can be performed.

We consider a driven weakly anharmonic qubit whose Hamiltonian in lab frame can be written as

\begin{equation}
    \label{eq:Transmon}
    \frac{H}{\hbar} = \omega_q a^\dagger a+\frac{\alpha}{2} a^\dagger a^\dagger a a +E(t)a^\dagger+E(t)^*a,
\end{equation}

where $\omega_q\equiv \omega_q^{0\rightarrow 1}$ is the qubit frequency and $\alpha = \omega_q ^{1\rightarrow 2}-\omega_q^{0\rightarrow 1}$ is the anharmonicity. The driving and control is given by

\begin{equation}
    E(t)= \begin{cases} 
        \Omega^x(t)\cos(\omega_d t)+\Omega^y(t)\sin(\omega_d t),& 0<t<t_g, \\ 0, & \text{otherwise}.
    \end{cases}
\end{equation}

Here $\Omega^x(t)$ and $\Omega^y(t)$ are two independent quadrature controls, $t_g$ is the total gate-time, and $\omega_d$ is the drive frequency. Next we move into the rotating frame of the drive by performing the following unitary transformation $U(t)=e^{i\omega_r t a^\dagger a}$, where $\omega_r$ is the rotating frame frequency. The Hamiltonian in the rotating frame after having performed the rotating wave approximation reads

\begin{multline}
    \frac{H^R}{\hbar}=
    \Delta a^\dagger a + \frac{\alpha}{2} a^{\dagger 2}a^2 + 
    (\frac{\Omega^x(t)}{2}\cos([\omega_r-\omega_d]t)-\frac{\Omega^y(t)}{2}\sin([\omega_r-\omega_d]t))(a^\dagger + a) \\
    + (\frac{\Omega^x(t)}{2}\sin([\omega_r-\omega_d]t)+\frac{\Omega^y(t)}{2}\cos([\omega_r-\omega_d]t))(ia^\dagger - ia),
\end{multline}

where $\Delta \equiv \omega_q - \omega_r$ is the qubit detuning. 

As a concrete example, assume that we apply a pulse at the qubit frequency $\omega_d=\omega_q$, and choose the rotating frame of the drive $\omega_r=\omega_d$. Then,

\begin{equation}
    \frac{H^R}{\hbar} =
    \frac{\alpha}{2} a^{\dagger 2}a^2
    + \frac{\Omega^x(t)}{2}(a^\dagger + a)
    + \frac{\Omega^y(t)}{2}(ia^\dagger - ia).
\end{equation}

If we treat the Hamiltonian as an effective two level system (ignoring the anharmonic term) and make the replacement $(a^\dagger + a)\rightarrow \sigma_x$ and $(ia^\dagger-ia)\rightarrow \sigma_y$, we obtain

\begin{equation}
    \frac{H^R}{\hbar} = \frac{\Omega^x(t)}{2}\sigma_x + \frac{\Omega^y(t)}{2}\sigma_y,
\end{equation}

showing that an in-phase pulse (i.e. the $\Omega^x(t)$ quadrature component) corresponds to a rotation around the $x$-axis while the out-of-phase pulse (i.e. the $\Omega^y(t)$ quadrature component), corresponds to rotations about the $y$-axis. As a concrete example of an in-phase pulse, writing out the unitary evolution operator yields,

\begin{equation}
    U^R(t)=\exp([-\frac{i}{2}\int_0^t\Omega^x(t')\mathrm{d}t']\sigma_x).
\end{equation}

By defining the angle

\begin{equation}
    \Theta(t)=\int_0^t\Omega^x(t')\mathrm{d}t',
\end{equation}

which is the angle a state is rotated given a waveform envelope $\Omega^x(t)$. This means that to implement a $\pi$-pulse on the $x$-axis one would solve $\Theta(t)=\pi$ and output the signal in-phase with the qubit drive.

In this simple example we assumed that we could ignore the higher levels of the qubit. In general leakage errors which take the qubit out of the computational subspace as well as phase errors can occur. To combat theses errors the so-called DRAG[@motzoi2009simple] procedure (Derivative Reduction by Adiabatic Gate) is used. In doing so we apply an extra signal in the out-of-phase component, such that

\begin{align}
    \Omega^x(t) = B e^{-\frac{(t-t_g)^2}{2\sigma^2}},\quad
    \Omega^y(t) = q\sigma\frac{d\Omega^x(t)}{dt}
\end{align}

where $q$ is a scale parameter that needs to be optimized with respect to a $\pi/2$-pulse. Interchanging $\Omega^x(t)$ and $\Omega^y(t)$ in the equation above corresponds to DRAG pulsing the $\Omega^y(t)$ component. The amplitude $B$ is fixed such that

\begin{equation}
    \Big|\int_{0}^{t}[\Omega^x(t')+i\Omega^y(t')]\mathrm{d}t'\Big|=\pi.
\end{equation}

for a $\pi$-pulse with DRAG.

### Example
The following example shows how an $R_X(\pi/2)$-gate is implemented on the `Sarimner` using drag.

```py
import numpy as np
import matplotlib.pyplot as plt
from qutip_qip.circuit import QubitCircuit
from chalmers_qubit.sarimner import (
    SarimnerProcessor, SarimnerModel, SarimnerCompiler
)

# Create circuit with a sngle RX-gate
circuit = QubitCircuit(1)
circuit.add_gate("RX", targets=0, arg_value=np.pi/2)

# Qubit frequencies in (GHz)
qubit_frequencies = [2 * np.pi * 5.0]
# Anharmonicity in (GHz)
anharmonicities = [- 2 * np.pi * 0.3]

# Load the physical parameters onto the model
model = SarimnerModel(qubit_frequencies=qubit_frequencies, 
                      anharmonicities=anharmonicities)

# Choose compiler
compiler = SarimnerCompiler(model=model)

# Create the processor with the given hardware parameters
sarimner = SarimnerProcessor(model=model, compiler=compiler, noise=[])

# Load circuit
tlist, coeffs = sarimner.load_circuit(circuit)

# Plot pulses
fig, axis = sarimner.plot_pulses(show_axis=True)
plt.xlabel("Time (ns)")
```

![Drag Pulse](figures/drag.png "drag")

### Virtual Z-gate
The virtual-z gate, we change our drive to include a phase $\phi$ so. All qubits are initialized with a phase of $\phi=0$

\begin{equation}
    E(t)= \begin{cases}
        \Omega^x(t)\cos(\omega_d t + \phi)+\Omega^y(t)\sin(\omega_d t + \phi),& 0<t<t_g, \\ 0, & \text{otherwise}.
    \end{cases}
\end{equation}

Upon initialization, all qubit drives have a phase of $\phi=0$. Now, when a $R_Z(\theta)$-gate is performed we simply update the phase of the corresponding qubit drive, such that $\phi$ shifts to $-\theta$. The minus sign here signifies that we do a rotation of the Bloch sphere rather than the state itself.

**Note:** If a $R_Z(\phi)$-gate is performed at the end of a quantum circuit, this gate will not have any effect on the quantum state.

