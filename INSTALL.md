# Installation
Perform these three steps to make the necessary installations.

1. Create a conda environment called ```wyckoff``` with Python 3.12 using
```
conda create -n wyckoff python=3.12
```

2. Install package and dependencies by running
```
pip install -e .
```
3. Install pre-commit hooks by running
```
pre-commit install
```
4. Install ```torch-scatter``` using
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+{CUDA}.html
```
where ```{CUDA}``` is ```cu118``` or ```cu121``` depending on your system

5. Install ```aviery``` using
```
pip install -U git+https://github.com/CompRhys/aviary
```
