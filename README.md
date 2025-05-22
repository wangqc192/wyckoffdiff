# WyckoffDiff
![](assets/wyckoffdiff_graphical_abstract.png)

This is the offical public repository for the paper [_WyckoffDiff -- A Generative Diffusion Model for Crystal Symmetry_](https://arxiv.org/abs/2502.06485) by Filip Ekström Kelvinius, Oskar B. Andersson, Abhijith S. Parackal, Dong Qian, Rickard Armiento, and Fredrik Lindsten. See [Citation](#citation) for how to cite this work.

## Data
The code already supports the dataset used in the paper (WBM), in addition to MP20 and Carbon24. Download is automatic, and using the codebase **does not require any manual download**. Please see the [data README](data/README.md) for more information on the data.

## Installation
See [INSTALL.md](INSTALL.md) for instructions on how to install required packages

## Usage
### Train
To train a WyckoffDiff model on WBM, a minimal example is
```
python main.py --mode train_d3pm --d3pm_transition [uniform/marginal/zeros_init] --logger [none/model_only/local_only/tensorboard/wandb]
```
Warning: using logger ```none``` will not save any checkpoints (or anything else), but can be used for, e.g., debugging.

This command will use the default values for all other parameters, which are the ones used in the paper.

### Generate
To generate new data, a minimal example is
```
python main.py --mode generate --num_samples [num_samples] --load [path/to/parameters.pt]
```

### Parse generated data
To convert generated data to protostructures and prototypes, run
```
python main.py --mode post_process --enrich_data --save_protostructures --load [path/to/checkpoint/dir]
```
## Questions and issues
If you have any questions or issues, please feel free to open an issue, or send an email to any of the authors (contact information in the paper).

## Citation
If you have used this code, please cite the WyckoffDiff paper
```
@misc{kelvinius2025wyckoffdiff,
      title={WyckoffDiff -- A Generative Diffusion Model for Crystal Symmetry},
      author={Filip Ekström Kelvinius and Oskar B. Andersson and Abhijith S. Parackal and Dong Qian and Rickard Armiento and Fredrik Lindsten},
      year={2025},
      eprint={2502.06485},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.06485},
}
```

## License
This code is licensed with a MIT License, but please note that [`wyckoff_generation/models/d3pm`](wyckoff_generation/models/d3pm) uses code from the official public D3PM implementation <https://github.com/google-research/google-research/tree/master/d3pm>, which is under and Apache 2.0 license. See [`wyckoff_generation/models/d3pm/README`](wyckoff_generation/models/d3pm/README)
