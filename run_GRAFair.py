from itertools import product,chain
import os
import subprocess

# Fixed

output_emd_dim = 20

# Choices

# model_type_choices = ["Cheb","GCN","SAGE"]

model_type_choices = ["GCN","SAGE","GAT","GIN"]
#model_type_choices = ["GAT"]

beta_choices0 = [-1]
is_private_choices0 = [False]
sensitive_attr_choices0 = ['gender']
choices0 = product(
    beta_choices0,
    is_private_choices0,
    sensitive_attr_choices0,
    model_type_choices,
)

beta_choices1 = [0.001,0.005,0.01,0.05,0.1,0.5,0.75]
is_private_choices1 = [True]
lr_choices1 = [0.01,0.005,0.001]
epochs_choices1 = [100,200,300,400,500]
num_layers_choices1 = [1,2,3]
#dataset_choices1 = ["bail","credit","german"]
dataset_choices1 = ["credit"]
choices1 = product(
    beta_choices1,
    is_private_choices1,
    dataset_choices1,
    lr_choices1,
    epochs_choices1,
    num_layers_choices1,
    model_type_choices,
)

choices = chain(choices1)


for index, (beta, is_private, dataset, lr, epochs, num_layers, model_type) in enumerate(choices):    # if idx <= 1:
    log_path_name = f"../log/GRAFair_{dataset}_{num_layers}_{model_type}_{beta}_{lr}_{epochs}_1"
    subprocess.run(f"python main_VGPF.py --beta {beta} --is_private {is_private} --model_type {model_type} --log_path_name {log_path_name} --output_emd_dim {output_emd_dim} --dataset {dataset} --lr {lr} --epochs {epochs} --num_layers {num_layers}" 
     ,shell=True, check=True)

# bail encoder: GCN/GAT/GIN 2-layer, lr: 0.01, epochs: 100, clf: 1-layer
#      encoder: SAGE 1-layer 
#      encoder: GAT 3-layer (x)
#      encoder: GIN 1-layer

# credit encoder: GCN/GAT/SAGE 2-layer, lr: 0.005, epochs:100, clf: 1-layer
#        encoder: GIN 1-layer

# german encoder: GCN/GIN 2-layer, lr: 0.005, epochs: 200, clf: 2-layer
#        encoder: SAGE 1-layer, lr: 0.005, epochs: 200, clf: 2-layer
#        encoder: GAT 3-layer
#        encoder: GAT beta=1 (running) 3-layer


