# Noise

## Decoherence
Decoherence refers to the loss of quantum information due to interactions with the environment. There are two main types of decoherence: longitudinal decoherence and transversal decoherence.

### Longitudinal decoherence
The $T_1$ **relaxation time** describes the decay of a qubit's excited state population. It is characterized by the collapse operator $\sqrt{\Gamma _1} \hat a$, where $\hat a$ is the annihilation operator and $\Gamma_1 = 1/T_1$ is the decay rate. This leads to an exponential decay of the population of first excited state proportional of $\exp(-\Gamma _1 t)$.

### Transversal decoherence
The $T_2$ **dephasing time** describes the loss of phase coherence, meaning the information about the qubits' relative phase is lost. Here, it's important to note that $T_1$ relaxation also contributes to dephasing with a rate proportional to $\exp(-\frac{t}{2T_1})$. 

To make sure that the overall phase damping is $\exp(-\frac{t}{T_2})$, we use the collapse operator $\sqrt{\frac{\Gamma_\varphi}{2}} 2 \hat n = \frac{1}{\sqrt{2T_\varphi}} 2 \hat n$, where $\hat n$ is the number operator and

\begin{equation}
    \Gamma_\varphi \equiv \frac{1}{T_\varphi} = \frac{1}{T_2} - \frac{1}{2T_1}.
\end{equation}

This equation also highlights why $T_2$ is always lesss than or equal to twice $T_1$ $(T_2\leq 2T_1)$.


### Example
To simulate with $T_1$ and $T_2$ noise

```py
import numpy as np
import matplotlib.pyplot as plt
from qutip_qip.circuit import QubitCircuit
from chalmers_qubit.sarimner import (
    SarimnerProcessor, SarimnerModel, SarimnerCompiler
)

# Qubit frequencies in (GHz)
transmon_dict = {
    0: {"frequency": 5.0, "anharmonicity": -0.30},
}
# Relaxation time in (ns)
decoherence_dict = {
    0: {"t1": 60e3, "t2": 80e3},
}

# Load the physical parameters onto the model
model = SarimnerModel(transmon_dict=transmon_dict)

# Choose compiler
compiler = SarimnerCompiler(model=model)

# Add noise
noise = [DecoherenceNoise(decoherence_dict=decoherence_dict)]

# Create the processor with the given hardware parameters and noise
sarimner = SarimnerProcessor(model=model, compiler=compiler, noise=noise)
```

## ZZ-Crosstalk
**ZZ-crosstalk** is a phenomenon in quantum computing, particularly relevant in systems with superconducting qubits, where the interaction between two or more qubits leads to unwanted phase shifts in the qubits that are not being directly operated upon. This effect arises due to the residual coupling between qubits, even when they are not intentionally interacting. The "ZZ" refers to the interaction type, denoting the direct coupling between the Z components of the qubit state.

Mathematically, ZZ-crosstalk between two qubits is represented by a constant drift term of the form $\zeta(\hat n \otimes \hat n)$, where $\zeta$ represents the strength of the crosstalk and $\hat n$ is the number operator.

### Example
To simulate with ZZ-crosstalk

```py
import numpy as np
import matplotlib.pyplot as plt
from qutip_qip.circuit import QubitCircuit
from chalmers_qubit.sarimner import (
    SarimnerProcessor, SarimnerModel, SarimnerCompiler
)

# Qubit frequencies in (GHz)
transmon_dict = {
    0: {"frequency": 5.0, "anharmonicity": -0.30},
    1: {"frequency": 5.4, "anharmonicity": -0.30},
}
# Cross-talk in (GHz)
cross_talk_dict = {(0,1): 1e-3}

# Create the processor with the given hardware parameters
model = SarimnerModel(transmon_dict=transmon_dict)

# Choose compiler
compiler = SarimnerCompiler(model=model)

# Add noise
noise = [ZZCrossTalk(cross_talk_dict=cross_talk_dict)]

# Create the processor with the given hardware parameters
sarimner = SarimnerProcessor(model=model, compiler, noise=noise)
```

## Example

For examples on how to use these noise models check out [relaxation.ipynb](https://github.com/aqp-mc2-chalmers/chalmers-qubit/blob/main/docs/examples/relaxation.ipynb).
