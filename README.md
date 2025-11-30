# CS 8803 MLS Final Project

## Installation

For this project, the `uv` package manager is used to manage the packages. Ensure you run `uv sync` to initialze the project.

For running the actual project, go through the following steps:

1. Initialize the same environment on the PACE cluster, and use `jupyter_gpu.job` to start a jupyter server and use an ssh tunnel to interact with the cluster. More information about setting up a jupyter kernel on PACE can be found [here](https://github.com/McWilliamsCenter/slurm_jupyter/tree/master)

2. Run `generate_sv.ipynb` to generate steering vectors for your respective model and datasets. If you don't have access to PACE, you can find already generated steering vectors in the `caa_sv` directory.

3. After editing the necessary config parameters in `epo_bruteforce.py` regarding `MODEL_NAME`, `BEHAVIORS`, and `COEFFS`, run `uv run epo_bruteforce.py`. Ensure that you do this on the PACE cluster as the process is resource intensive and requires the use of `submitit` to ensure the algorithm runs on a timely manner. 

4. After generating the `histories` folder, go to `plot.ipynb` to generate the respective plots for the folder in `histories`. Edit the `FOLDER` parameter to reflect where your runs are stored. This repository has this folder populated for ease of use and analysis.