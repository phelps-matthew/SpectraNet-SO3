# &#127922; implicit-pdf
Pytorch implementation of [Implicit Representation of Probability Distributions on the Rotation Manifold](https://github.com/google-research/google-research/tree/master/implicit_pdf) (ICML 2021)


## Installation
```bash
# dependencies - uses mlflow for logging artifacts/metrics and pyrallis for easy
# config management
pip install torch torchvision
pip install -U mlflow pyrallis pillow matplotlib tqdm

# install repo
pip install -e implicit-pdf/implicit_pdf
```

## Download raw Symmetric Solids dataset (SymSol 1.0.0)
```bash
# 3.1 GB download
symsol_path="~/data/datasets/symsol_1.0.0"
mkdir $symsol_path
cd $symsol_path
curl -O https://storage.googleapis.com/gresearch/implicit-pdf/symsol_dataset.zip
unzip symsol_dataset.zip
```

## Usage
* Train implicit pdf model on GPU 0
```python
python train.py --gpus [0]
```
* View train configuration options
```python
python train.py --help
```
* Train from yaml configuration, with CLI override
```python
python train.py --config_path train_cfg.yaml --lr 0.001 --gpus [4]
```
* Start mlflow ui to visualize results
```
# navgiate to implicit_pdf root directory containing `mlruns`
mlflow ui
# to set host and port
mlflow ui --host 0.0.0.0 --port 8080
```
* Serialize dataclass train config to yaml, outputting `train_cfg.yaml`
```python
python cfg.py
```
