# Data
Below you find some information about the three different datasets that are implemented in this repo. However, we have only used WBM in our experiments.

For all datasets, we found the protostructures using spglib. Download URLs to our versions are provided for convenience. However, if using our code, these will be downloaded automatically and placed in data/ if not already present, and hence, **no separate/manual download is required for using the codebase**.

Note that although train/val/test splits exist for all datasets, we have only used the train split. 

## WBM
Dataset from [Predicting stable crystalline compounds using chemical similarity](https://www.nature.com/articles/s41524-020-00481-6).

```
@article{wang_predicting_2021,
  title = {Predicting Stable Crystalline Compounds Using Chemical Similarity},
  author = {Wang, Hai-Chen and Botti, Silvana and Marques, Miguel A. L.},
  year = {2021},
  month = jan,
  journal = {npj Computational Materials},
  volume = {7},
  number = {1},
  pages = {1--9},
  publisher = {Nature Publishing Group},
  issn = {2057-3960},
  doi = {10.1038/s41524-020-00481-6},
  copyright = {2021 The Author(s)},
  langid = {english},
}

@software{Riebesell_Matbench_Discovery_2023,
author = {Riebesell, Janosh and Goodall, Rhys and Benner, Philipp and Chiang, Yuan and Deng, Bowen and Lee, Alpha and Jain, Anubhav and Persson, Kristin},
doi = {10.48550/arXiv.2308.14920},
license = {MIT},
month = aug,
title = {{Matbench Discovery}},
url = {https://github.com/janosh/matbench-discovery},
version = {1.0.0},
year = {2023}
}

```
The curated dataset can be obtained from [matbench-discovery](https://github.com/janosh/matbench-discovery/tree/main/data/wbm). The original data can be found on [Materials Cloud](https://doi.org/10.24435/materialscloud:96-09) under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. 

Our processed version of this dataset can be downloaded from
* train: <https://public.openmaterialsdb.se/wyckoffdiff/data/wbm/raw/wbm_train.csv.bz2>
* val: <https://public.openmaterialsdb.se/wyckoffdiff/data/wbm/raw/wbm_val.csv.bz2>
* test: <https://public.openmaterialsdb.se/wyckoffdiff/data/wbm/raw/wbm_test.csv.bz2>

## MP20
Based on the Materials Project, which provides data under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

```
@article{jain2013commentary,
  title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and others},
  journal={APL materials},
  volume={1},
  number={1},
  pages={011002},
  year={2013},
  publisher={American Institute of PhysicsAIP}
}
```
Our processed version is based on data originally downloaded from the [DiffCSP++ repository](https://github.com/jiaor17/DiffCSP-PP/tree/main/data/mp_20). 

Our processed version of this dataset can be downloaded from
* train: <https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw/mp20_train.csv.bz2>
* val: <https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw/mp20_val.csv.bz2>
* test: <https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw/mp20_test.csv.bz2>

## Carbon24
The original data is from
```
Chris J. Pickard, AIRSS data for carbon at 10GPa and the C+N+H+O system at 1GPa, Materials Cloud Archive 2020.0026/v1 (2020), https://doi.org/10.24435/materialscloud:2020.0026/v1
```
where it is provided under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
```
@misc{carbon2020data,
  doi = {10.24435/MATERIALSCLOUD:2020.0026/V1},
  url = {https://archive.materialscloud.org/record/2020.0026/v1},
  author = {Pickard,  Chris J.},
  keywords = {DFT,  ab initio random structure searching,  carbon},
  language = {en},
  title = {AIRSS data for carbon at 10GPa and the C+N+H+O system at 1GPa},
  publisher = {Materials Cloud},
  year = {2020},
  copyright = {info:eu-repo/semantics/openAccess}
}
```
Our processed data is based on data from the [DiffCSP++ repository](https://github.com/jiaor17/DiffCSP-PP/tree/main/data/mp_20). 

Our processed version of this data set can be downloaded from
* train: <https://public.openmaterialsdb.se/wyckoffdiff/data/carbon24/raw/carbon24_train.csv.bz2>
* val: <https://public.openmaterialsdb.se/wyckoffdiff/data/carbon24/raw/carbon24_val.csv.bz2>
* test: <https://public.openmaterialsdb.se/wyckoffdiff/data/carbon24/raw/carbon24_test.csv.bz2>