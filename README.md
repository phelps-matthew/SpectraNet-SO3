# 🛰 SpectraNet-SO(3) 
SpectraNet-SO(3): Learning Satellite Orientation from Optical Spectra By Implicitly Modeling Mutually Exclusive Probability Distributions on the Rotation Manifold (Official Repo)

## Installation
* Create conda environment
```
conda create -n ipdf python=3.9 pip
conda activate ipdf
```
* Install torch and dependencies. Uses mlflow for logging artifacts/metrics and pyrallis for easy config management
```
pip install -U pip
# cuda version >= 11.0
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# cuda version < 11.0
pip install torch torchvision
pip install mlflow pyrallis pandas tqdm pillow
```

* Install implicit-pdf repo
```
git clone https://github.com/phelps-matthew/implicit-pdf.git
cd implicit-pdf
pip install -e .
```

## Download raw Symmetric Solids dataset (SymSol 1.0.0)
```
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
