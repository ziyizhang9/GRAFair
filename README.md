# GRAFair
## Requirements
- torch==1.5.0
- torchvision==0.6.0
- torch-scatter==2.0.5
- torch-sparse==0.6.7
- torch-cluster==1.5.7
- torch-spline-conv==1.2.0
- torch-geometric==1.5.0

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
