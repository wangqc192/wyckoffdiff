# Installation

Perform these steps to install WyckoffDiff and its dependencies.

1. Clone the WyckoffDiff repository:
    ```
    git clone https://github.com/httk/wyckoffdiff.git
    cd wyckoffdiff
    ```

2. Create a conda environment called ```wyckoffdiff``` with Python 3.12 and activate it using
    ```
    conda create -n wyckoffdiff python=3.12
    conda activate wyckoffdiff
    ```

    **Alternatively**, if your system Python version is recent enough, you can instead use a Python venv:
    ```
    mkdir -p data/deps
    python3 -m venv data/deps/venv
    source data/deps/venv/bin/activate
    ```

3. Install [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), and [PyTorch Scatter](https://pypi.org/project/torch-scatter/) by
    ```
    pip install numpy==1.26.4 torch==2.3.0 torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+{COMPUTE_PLATFORM}.html
    ```
    where `{COMPUTE_PLATFORM}` should be `cu118`, `cu121` or `cpu`.
    For CUDA, the version of your installed driver can be seen in the output of `nvidia-smi`, and it may be a good idea to pick the highest version of 1.18 or 1.21 that is lower or equal to the version supported by the driver.

4. Install [Aviary](https://github.com/CompRhys/aviary) by
    ```
    pip install -U git+https://github.com/CompRhys/aviary@v1.1.1
    ```

5. Install wyckoffdiff as a package with its dependencies by
    ```
    pip install -e .
    ```

6. Install pre-commit hooks by
    ```
    pre-commit install
    ```

At this point you are ready to continue with the usage instructions in README.md.

(To activate the environment created above in a new shell, just do: `conda activate wyckoffdiff` or `source data/deps/venv/bin/activate`.)
