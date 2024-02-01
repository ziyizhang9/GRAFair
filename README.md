# GRAFair
## Requirements:
ipdb==0.13.13
matplotlib==3.7.2
multiset==3.0.2
networkx==3.1
numba==0.57.1
numpy==1.24.4
pandas==2.0.3
Pillow==10.0.0
Pillow==10.2.0
Requests==2.31.0
scikit_learn==1.3.0
scipy==1.12.0
seaborn==0.13.2
setuptools==68.0.0
skimage==0.0
sympy==1.12
tensorboardX==2.6.2.2
tensorly==0.8.1
texttable==1.7.0
torch==1.8.0+cu111
torch_geometric==1.5.0
torch_scatter==2.0.6
torch_sparse==0.6.10
torchvision==0.9.0+cu111
torchviz==0.0.2
tqdm==4.66.1

In addition, we need to install DeepRobust locally.
```
cd DeepRobust
pip install -e .
```
## Usage
Run GRAFair on the bail dataset:
```
sh run_GRAFair_bail.sh
```
Run GRAFair on the credit dataset:
```
sh run_GRAFair_credit.sh
```
Run GRAFair on the german dataset:
```
sh run_GRAFair_german.sh
```
