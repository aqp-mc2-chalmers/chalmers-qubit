# Transmons

To simulate the quantum circuits running on an actual quantum processor we model our Hamiltonian as

\begin{equation}
    \label{eq:transmon}
    H = \sum_{i=1}^N(\omega_{q_i} a^\dagger a +  \frac{\alpha_i}{2} a^{\dagger 2} a^2),
\end{equation}

where $N$ is the number of qubits, $\omega_q$ is the qubit frequency and $\alpha$ is the anhamronicity. In the rotating frame of $U(t)=e^{i\omega_r t a^\dagger a}$, where $\omega_r$ is the rotating frame frequency. The Hamiltonian reads

\begin{equation}
    \label{eq:transmon_rotating}
    H^R = \sum_{i=1}^N(\Delta_i a^\dagger a +  \frac{\alpha_i}{2} a^{\dagger 2} a^2),
\end{equation}

where $\Delta_i = \omega_{q_i} - \omega_{r_i}$ is the qubit detuning

## Example
To create a we simply initialize a model

```py
# Qubit frequencies in (GHz)
qubit_frequencies = [2 * np.pi * 5]
# Anharmonicity in (GHz)
anharmonicities = [-2 * np.pi * 0.3]
# Load the physical parameters onto the model
model = SarimnerModel(
    qubit_frequencies=qubit_frequencies, anharmonicities=anharmonicities
)
print(model.drift)
```
We can print the corresponding drift terms for each qubit.