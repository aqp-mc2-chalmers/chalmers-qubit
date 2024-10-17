## Transmons

Quantum processors operate based on the manipulation of qubits. To simulate how quantum circuits behave on such devices, we model their Hamiltonian. The Hamiltonian for a transmon qubit, is given by:

\begin{equation}
    \label{eq:transmon}
    H = \sum_{i=1}^N(\omega_{q_i} a^\dagger a +  \frac{\alpha_i}{2} a^{\dagger 2} a^2),
\end{equation}

where $N$ is the number of qubits, $\omega_q$ is the qubit frequency and $\alpha$ is the anhamronicity. We often work in a rotating frame to simplify calculations. This involves applying a transformation $U(t) = e^{i\omega_r t a^\dagger a}$, where $\omega_r$ is the rotating frame frequency. In this frame, the Hamiltonian becomes:

\begin{equation}
    \label{eq:transmon_rotating}
    H^R = \sum_{i=1}^N(\Delta_i a^\dagger a +  \frac{\alpha_i}{2} a^{\dagger 2} a^2).
\end{equation}

Here $\Delta_i = \omega_{q_i} - \omega_{r_i}$ represents the detuning of the i-th qubit, which is the difference between its intrinsic frequency and the rotating frame frequency.

### Example: Initializing a Model with Hardware Parameters

The `SarimnerModel` allows us to create a model with specific hardware parameters. Here's an example of how to initialize such a model:

```python
from chalmers_qubit.sarimner import SarimnerModel
# Qubit frequencies are given in (GHz)
transmon_dict = {
    0: {"frequency": 5.0, "anharmonicity": -0.30},
}

# Load the physical parameters onto the model
model = SarimnerModel(transmon_dict=transmon_dict)
```
This code defines qubit frequencies and anharmonicities for a single qubit system (N=1) and creates a `SarimnerModel` object to represent the physical system. 

All simulations are performed in the rotating frame of the individual qubits.