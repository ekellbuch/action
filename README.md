

# Action recognition toolbox

## Scripts:
1. Train a simple supervised model:
```
python run.py --config-name="fly_daart" eval_cfg.eval_only=0
```
2. Reproduce results daart demo:
```
python run.py --config-name="fly_daart"
```

## Experiments

|                                                             Wandb Experiment                                                             |                              parameters                              |                                                                                                                                     comments                                                                                                                                      |
|:----------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                   [zv5odluq](https://wandb.ai/ekellbuch/uncategorized/sweeps/zv5odluq)                                   |              train vanilla classifier on top of dataset              | running on lion
|                                      [35r08dx1](https://wandb.ai/ekellbuch/data_fly/runs/35r08dx1)                                       |                     reproduce fly_daart results                      | we can reproduce them when loading checkpoints
| [pojyktyr](https://wandb.ai/ekellbuch/uncategorized/sweeps/pojyktyr) |                compare segmenter w/o loss of interest                | no difference for labeled dataset
| [ub5ahdos](https://wandb.ai/ekellbuch/uncategorized/sweeps/ub5ahdos) |        compare segmenter w weak_bsoftmax, turn off grad_clip         | no difference for labeled dataset
| [07zg6fd8](https://wandb.ai/ekellbuch/uncategorized/sweeps/07zg6fd8) | compare segmenter w weak_bsoftmax for unlabeled using simba features | worse performance in all classes



# Code and Data structure:
- data structure:
```
data_directory
├── features-simba
│   ├── <sess_id_0>.csv
│   └── <sess_id_1>.csv
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
- modules:
  - ClassifierModule: classifier module
  - ClassifierSeqModule: sequence classifier module
  - ClassifierSeqModuleBS: sequence classifier module with specific loss
  - SegmenterModule: segmenter module


# References:
- [Daart](https://github.com/themattinthehatt/daart)