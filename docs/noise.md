# Noise

## Decoherence

### Longitudinal decoherence
The $T_1$ relaxation time describes the strength of amplitude damping and can be described, in a two-level system, by a collapse operator $\sqrt{\Gamma _1} a$, where $a$ is the annihilation operator and $\Gamma_1=1/T_1$. This leads to an exponential decay of the population of excited states proportional to $\exp(-\Gamma _1 t)$.

### Transversal decoherence
The $T_2$ time describes the dephasing process. Here one has to be careful that the amplitude damping channel characterized by $T_1$ will also lead to a dephasing proportional to $\exp(-\frac{t}{2T_1})$. To make sure that the overall phase damping is $\exp(-\frac{t}{T_2})$, the processor (internally) uses an collapse operator $\sqrt{\frac{\Gamma_\varphi}{2}} \sigma_z = \frac{1}{\sqrt{2T_\varphi}} \sigma_z$ with 

\begin{equation}
    \frac{1}{T_\varphi} = \frac{1}{T_2} - \frac{1}{2T_1}
\end{equation}

to simulate the dephasing. (This also indicates that $T_2\leq 2T_1$).

## ZZ-Crosstalk