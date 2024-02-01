#!/bin/bash
echo "==============GCN================"
python main_GRAFair.py --beta 0.001 --is_private True --model_type GCN --log_path_name ./log/GRAFair_GCN_bail --dataset bail --lr 0.005 --epochs 200 --num_layers 1 --num_classifier 1

echo "==============GIN================"
python main_GRAFair.py --beta 0.1 --is_private True --model_type GIN --log_path_name ./log/GRAFair_GIN_bail --dataset bail --lr 0.005 --epochs 200 --num_layers 1 --num_classifier 2

echo "==============SAGE================"
python main_GRAFair.py --beta 0.01 --is_private True --model_type SAGE --log_path_name ./log/GRAFair_SAGE_bail --dataset bail --lr 0.005 --epochs 200 --num_layers 2 --num_classifier 1

echo "==============Cheb================"
python main_GRAFair.py --beta 0.001 --is_private True --model_type Cheb --log_path_name ./log/GRAFair_Cheb_bail --dataset bail --lr 0.001 --epochs 100 --num_layers 2 --num_classifier 2
