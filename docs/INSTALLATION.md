# Installation instructions:

First create a Conda environment in which this package and its dependencies will be installed.
```
conda create --name <YOUR_ENVIRONMENT_NAME> python=3.8
```
and activate it:
```
conda activate <YOUR_ENVIRONMENT_NAME>
```
Download the repository from GitHub and install it in the Conda environment:
```
cd <SOME_FOLDER>
git clone https://github.com/ekellbuch/action
```
Then move into the newly-created repository folder, and install dependencies:
```
cd action/src
pip install -e .
```