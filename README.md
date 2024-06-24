

# Time Series Analysis toolbox

This repository contains a toolbox to classify time series data.

Given some time series data (i.e. DLC outputs), this codebase allows you to: 
- train a model to reconstruct the time series 
- train a model to segment the time series into classes (i.e. actions)
  - in a supervised setting, i.e. if there are labels
  - in a semi-supervised setting, i.e. if there are no ground truth labels but there is some weak signal
- combine information across multiple time series.

## Data and Code structure:
- data structure:

```
data_directory
├── labels-hand
│   ├── <sess_id_0>.csv
│   └── <sess_id_1>.csv
├── labels-heuristic
│   ├── <sess_id_0>.csv
│   └── <sess_id_1>.csv
├── markers
│   ├── <sess_id_0>.csv
│   └── <sess_id_1>.csv
└── videos
    ├── <sess_id_0>.mp4
    └── <sess_id_1>.mp4
```
- Experiment configuration: 
  see scripts/script_configs/fly_daart.py
```
data_cfg:  data configuration
trainer_cfg: trainer configuration
eval_cfg: evaluation configuration
module_cfg: module configuration
  classifier_cfg: architecture configuration
```

- modules:
  - ClassifierModule: vanilla classification module.
  - ClassifierSeqModule: vanilla sequence classification module.
  - ClassifierSeqModuleBS: example with specific loss added to sequence module.
  - SegmenterModule: sequence module with reconstruction and classification.

## Installation instructions:

See [INSTALLATION.md](docs/INSTALLATION.md)

## Scripts:
1. Train a simple supervised model:
```
python run.py --config-name="fly_daart" eval_cfg.eval_only=0
```
2. Reproduce results daart demo:
```
python run.py --config-name="fly_daart"
```

# References:
- [Daart](https://github.com/themattinthehatt/daart)