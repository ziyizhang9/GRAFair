from itertools import product,chain
import os
import subprocess

# Fixed

output_emd_dim = 20

# Choices

# model_type_choices = ["Cheb","GCN","SAGE"]

model_type_choices = ["GCN"]

beta1_choices0 = [-1]
is_private_choices0 = [False]
sensitive_attr_choices0 = ['gender']
choices0 = product(
    beta1_choices0,
    is_private_choices0,
    sensitive_attr_choices0,
    model_type_choices,
)


beta1_choices1 = [0.5,0.1,0.05,0.01,0.005]
is_private_choices1 = [True]
sensitive_attr_choices1 = ['gender']
choices1 = product(
    beta1_choices1,
    is_private_choices1,
    sensitive_attr_choices1,
    model_type_choices,
)

choices = chain(choices1)

for index, (beta1, is_private, sensitive_attr, model_type) in enumerate(choices):    # if idx <= 1:
    log_path_name = f"../log/PGIB_ml_{model_type}_{beta1}_{is_private}_{sensitive_attr}_{output_emd_dim}"
    subprocess.run(f"python main_VGPF_pokec.py --beta1 {beta1} --is_private {is_private} --sensitive_attr {sensitive_attr} --model_type {model_type} --log_path_name {log_path_name} --output_emd_dim {output_emd_dim}" 
     ,shell=True, check=True)