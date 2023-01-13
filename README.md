# intro-to-ml-project
Final project of training a computer vision neural network on the CIFAR-10 dataset.

## Run

If you have poetry https://python-poetry.org/, just do in the project root
```bash
poetry install
```

Then you can start the jupyter lab with
```bash
poetry run jupyter-lab
```

If not, the dependencies can be found in `pyproject.toml` under `[tool.poetry.dependencies]` and you can install them manually with `pip`.


## Project structure

* `src` contains source code
* `data` contains cached CIFAR-10 dataset
* `notebooks` contains jupyter notebooks with experiments + wandb data (checkpoints, models, ..)
* `logs` + `artifacts` are not used at the moment

## Wandb training report

The report can be accessed at https://api.wandb.ai/report/pkantek/5ljf1nvk
