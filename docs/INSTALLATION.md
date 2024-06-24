# Installation instructions:

First create a Conda environment in which this package and its dependencies will be installed.
```
conda create --name <YOUR_ENVIRONMENT_NAME> python=3.8
```
and activate it:
```
conda activate <YOUR_ENVIRONMENT_NAME>
```
Download the repository from GitHub and install it in the conda environment:
```
cd <SOME_FOLDER/segment>
git clone https://github.com/ekellbuch/action
```

Then move into the newly-created repository folder, and install dependencies:
```
cd action
pip install -r requirements/txt
cd src
pip install -e .
```

To run the code:
```
# move back to the main directory where action package is installed:
cd ../
# export where the package is installed (see usage in script_configs/fly_daart.yaml)
export LOCAL_PROJECTS_DIR="SOME_FOLDER"
python run.py --config-name="fly_daart" trainer_cfg.fast_dev_run=1
```

