#!/bin/bash
echo "==============GCN================"
python main_GRAFair.py --beta 0.05 --is_private True --model_type GCN --log_path_name ./log/GRAFair_GCN_german --dataset german --lr 0.01 --epochs 100 --num_layers 2 --num_classifier 2

echo "==============GIN================"
python main_GRAFair.py --beta 0.05 --is_private True --model_type GIN --log_path_name ./log/GRAFair_GIN_german --dataset german --lr 0.01 --epochs 100 --num_layers 3 --num_classifier 2

echo "==============SAGE================"
python main_GRAFair.py --beta 0.05 --is_private True --model_type SAGE --log_path_name ./log/GRAFair_SAGE_german --dataset german --lr 0.001 --epochs 200 --num_layers 3 --num_classifier 2

echo "==============Cheb================"
python main_GRAFair.py --beta 0.1 --is_private True --model_type Cheb --log_path_name ./log/GRAFair_Cheb_german --dataset german --lr 0.001 --epochs 100 --num_layers 2 --num_classifier 2
