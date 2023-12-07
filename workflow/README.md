# LDMI | Workflow

`run.sh` is the main file used for executing the workflow procedures.

* `run.sh -m generate`: Generate dataset of dyanmic mean field transmembrane currents $I$ with uniformly sampled external thalamic ($I_{th}$) and cortico-cortical ($I_{cc}$) input currents.
* `run.sh -m map`: Map transmembrane currents $I$ to synaptic locations along depth axis of cortex ($I_k = \sum_{i=1}^N I_k^i$).
* `run.sh -m protocol`: Build a protocol for the experiment.
* `run.sh -m hemodynamic`: Generate hemodynamic response from mapped currents (baseline is substracted).

Other arguments:
* `-n --name`: Experiment name.
* `-k --depths`: Number of depths.
* `-t --threads`: Number of threads.
* `-a --area`: Area (default: V1).
* `-s --sigma`: Standard deviation of mapping kernel.
* `-d --device`: Device (default: cpu).
* `-r --ratio`: Training/Test ratio (default: 0.8).

# TODO:
* Sensitivity and Specificity analysis of *systemic* and *physiological* parameters.
* *Systemic* parameters are the external input $I_{th}$ and $I_{cc}$, and recurrent connectivity $G_{rec}$.
* *Physiological* parameters are all parameters related to cortical physiology, i.e. how vasculature affects the laminar BOLD response.