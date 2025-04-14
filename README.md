# chalmers-qubit

[![Tests](https://github.com/aqp-mc2-chalmers/chalmers-qubit/actions/workflows/tests.yml/badge.svg)](https://github.com/aqp-mc2-chalmers/chalmers-qubit/actions/workflows/tests.yml) [![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://squidfunk.github.io/mkdocs-material/) [![license](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause) <a href="https://pypi.org/project/chalmers-qubit/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/chalmers-qubit"></a>


A simulation framework for Chalmers devices that can be used to simulate the running of quantum algorithms with realistic noise. We follow [qutip-qip](https://qutip-qip.readthedocs.io/en/stable/) to build a processor that can take in a quantum circuit (e.g., a QASM cicruit) and performs a master equation simulation adding noise such as T1 and T2. It is also possible to perform a Monte-Carlo trajectory simulation and customize the processor to add various types of noise such as [ZZCrossTalk](https://qutip-qip.readthedocs.io/en/latest/apidoc/qutip_qip.noise.html#qutip_qip.noise.ZZCrossTalk).

The package is under development and testing. 

## Installation

The main requirement to use this package is [qutip-qip](https://qutip-qip.readthedocs.io/en/stable/) based on [qutip](https://qutip-qip.readthedocs.io/en/stable/): The Quantum Toolbox in Python. The requirements are already specified in the `setup.cfg` file and you can install the package `chalmers_qubit` simply by downloading this folder or cloning this repository and running:

``` zsh
pip install .
```

to get the minimal installation. You can instead use `'.[full]'` to install the package with all optional dependencies, such as matplotlib. Moreover, it might be beneficial to install an editable version. In the editable version, changes to the code are reflected system-wide without requiring a reinstallation.

``` zsh
pip install -e '.[full]'
```

If you do not care about making changes to the source code and just want to try out the package (e.g., from Google Colab), you can do a git+ install with

``` zsh
pip install git+https://github.com/aqp-mc2-chalmers/chalmers-qubit.git
```

## Usage

The usage of the package follows [qutip-qip](https://qutip-qip.readthedocs.io/en/stable/) where first, a quantum circuit is defined using [`qutip-qip`](https://qutip-qip.readthedocs.io/en/stable/qip-simulator.html) and then run on one of the custom Chalmers processors, e.g., the processor called sarimner. The custom processor is defined in `chalmers_qubit.sarimner.processor` and can be initialized with a `model`, `compiler` and `noise`. 

Note that only gates with compilation instructions in `chalmers_qubit/sarimner/compiler.py` will work for this particular processor.

Notebooks exploring the usage of the simulator is available in `docs/examples/`. 

``` python
import numpy as np
from qutip import basis, tensor
from qutip_qip.circuit import QubitCircuit
from chalmers_qubit.sarimner import (
    SarimnerProcessor,
    SarimnerModel,
    SarimnerCompiler,
    DecoherenceNoise,
    ZZCrossTalk,
)

# Define a quantum circuit
qc = QubitCircuit(2)
qc.add_gate("RX", targets=0, arg_value=np.pi / 2)
qc.add_gate("RY", targets=1, arg_value=np.pi / 2)
qc.add_gate("CZ", controls=0, targets=1)

# All frequencies are defined in GHz, and times in ns.
transmon_dict = {
    0: {"frequency": 5.0, "anharmonicity": -0.30},
    1: {"frequency": 5.4, "anharmonicity": -0.30},
}
coupling_dict = {
    (0, 1): 0.04,
}
# Construct model
model = SarimnerModel(transmon_dict=transmon_dict,
                      coupling_dict=coupling_dict)

# Load a compiler
compiler = SarimnerCompiler(model=model)

# Define all the noise objects as a list.
decoherence_dict = {
    0: {"t1": 60e3, "t2": 80e3},
    1: {"t1": 100e3, "t2": 105e3},
}
cross_talk_dict = {
    (0, 1): 1e-4,
}
noise = [
    DecoherenceNoise(decoherence_dict=decoherence_dict),
    ZZCrossTalk(cross_talk_dict=cross_talk_dict),
]

# Initialize the processor
processor = SarimnerProcessor(model=model, compiler=compiler, noise=noise)

# Load the circuit that generates the pulses to run the simulation
tlist, coeffs = processor.load_circuit(qc)

# Initial state for the simulation.
# The default assumptions is that each transmon is a qudit with 3 levels.
init_state = tensor(basis(3, 1), basis(3, 1))

# Run master equation simulation
result = processor.run_state(init_state)
print("Final state", result.states[-1])

# Run the same circuit but with mcsolve using 100 trajectories.
result = processor.run_state(init_state, solver="mcsolve", ntraj=100)
print("Final state", result.states[-1])
```

It is also possible to import [QASM circuits](https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v5/quantum-circuits/qasm.ipynb).

## Development

In order to add new custom pulses or modify the device, edit the processor, or compiler the tutorials and detailed instructions in [qutip-qip](https://qutip-qip.readthedocs.io/en/stable/).

The [tutorials](https://qutip.org/qutip-tutorials/) show examples of how to customize the processor. If you have installed the package in the develop mode, any changes to the processor, e.g., adding a new gate will be reflected immediately system-wide without requiring a reinstallation of the package.

## Support

This package was built from contributions by Pontus Vikstål and Shahnawaz Ahmed.

Contact vikstal@chalmers.se, shahnawaz.ahmed95@gmail.com or anton.frisk.kockum@chalmers.se for help and support.
