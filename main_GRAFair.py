import argparse
from copy import deepcopy
import datetime
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling, 
                                   remove_self_loops,
                                   add_self_loops,
                                   train_test_split_edges)
import random
import sklearn
import sys, os
import gc
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from model import GRAFair, train_GRAFair
from pytorch_net.util import str2bool, eval_tuple

from itertools import product,chain
# from dataset_bail import load_data, load_data_cf, load_data_rb
from tqdm import tqdm
from pathlib import Path

date_time = "{0}-{1}".format(datetime.datetime.now().month, datetime.datetime.now().day)  # Today's month and day. Used for the directory name saving the experiment result files.

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', default="exp id", help='experiment ID')
parser.add_argument('--data_type', default="MovieLens",help='Data type: choose from PROTEINS.')
parser.add_argument('--model_type', default='GCN', help='Model type: GCN, Cheb, or SAGE')
parser.add_argument('--train_fraction', type=float, default=1., help='train_fraction')
parser.add_argument('--beta', type=float, default=-1, help='beta value')
parser.add_argument('--sample_size', type=int, default=1, help='sample_size')
parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
parser.add_argument('--output_emd_dim', type=int, default=20, help='output_emd_dim')
parser.add_argument('--heads', type=int, default=2, help='heads')
parser.add_argument('--reparam_mode', default="diag", help='diag, diagg, or full')
parser.add_argument('--prior_mode', default="mixGau-100", help='prior mode for VIB')
parser.add_argument('--val_use_mean', type=str2bool, nargs='?', const=True, default=True, help='Whether to use mean of Z during validation.')
parser.add_argument('--reparam_all_layers', type=str, default="(-1,)", help='Whether to reparameterize all layers.')
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs.")
parser.add_argument('--batch_size', type=int, default=8192, help="batch size.")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay.")
parser.add_argument('--threshold', type=float, default=0.05, help='threshold for GCNJaccard')
parser.add_argument('--save_best_model', type=str2bool, nargs='?', const=True, default=False, help='Whether to save the best model.')
parser.add_argument('--skip_previous', type=str2bool, nargs='?', const=True, default=False, help='Whether to skip previously trained model in the same directory.')
parser.add_argument('--date_time', default=date_time, help="Current date and time.")
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--idx', default="0", help='idx')
parser.add_argument('--is_private', type=str2bool, nargs='?', const=True, default=False, help='Whether to use private mode.')
parser.add_argument('--log_path_name', default="../log/GRAFair", help='log_path_name')
parser.add_argument('--emb_save_name', default="./checkpoints/GRAFair", help='emb_save_name')
parser.add_argument('--retrain', type=str2bool, nargs='?', const=True, default=True, help='Whether to retrain the model.')
parser.add_argument('--dataset', type=str, default= "bail", help = 'Please choose from bail, credit and german.')

args = parser.parse_args()


if "args" in locals():
    exp_id = args.exp_id
    data_type = args.data_type
    model_type = args.model_type
    train_fraction = args.train_fraction
    beta = args.beta
    output_emd_dim = args.output_emd_dim
    heads = args.heads
    latent_size = output_emd_dim * 2 // heads # Latent dimension for GCN-based or GAT-based models.
    sample_size = args.sample_size
    num_layers = args.num_layers
    reparam_mode = args.reparam_mode
    prior_mode = args.prior_mode
    val_use_mean = args.val_use_mean
    reparam_all_layers = eval_tuple(args.reparam_all_layers)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    threshold = args.threshold
    save_best_model = args.save_best_model
    skip_previous = args.skip_previous
    date_time = args.date_time
    seed = args.seed
    idx = args.idx
    is_private = args.is_private
    sensitive_attr = args.sensitive_attr
    is_cuda = "cuda:0"
    use_sensitive_mlp = False
    log_path_name = args.log_path_name
    att_train_rate = args.att_train_rate
    retrain = args.retrain
    emb_save_name = args.emb_save_name
    dataset = args.dataset
    args = vars(args)


if dataset == 'bail':
    from dataset_bail import load_data, load_data_cf, load_data_rb
    sens_attr = 'white'
elif dataset == 'credit':
    from dataset_credit import load_data, load_data_cf, load_data_rb
    sens_attr = 'age'
elif dataset == 'german':
    from dataset_german import load_data, load_data_cf, load_data_rb
    sens_attr = 'gender'

device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")
Path("log").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)

if lr == -1:
    lr = None
if weight_decay == -1:
    weight_decay = None
if beta == -1:
    beta = None
if beta is None:
    beta_list, reparam_mode, prior_mode = None, None, None
else:
    beta_list = np.ones(epochs + 1) * beta


best_metrics_list = []
best_metrics_att_list = []

# sens_attr = 'white'

for t, seed in enumerate([100,200,300,400,500]):
    print(f"Current args settings:")
    for key in ["model_type", 
    "prior_mode", "beta", "is_private", 
    "sensitive_attr", "weight_decay", 
    "batch_size", "reparam_all_layers","output_emd_dim"]:
        print(f"{key}: {args[key]}")
    print(f"Current round {t} seed {seed}")
    # Setting the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Round {t}:")
    print("Loading data")
    data = load_data()
    num_ent = data.x.size(0)
    num_rel = 2
    data.s = data[sens_attr].float()
    data.s = torch.cat([data.s,torch.zeros([num_ent - data.s.size(0), data.s.size(1)]).cuda()])
    
# load counterfactual data
    data_cf = load_data_cf()
    data_cf.s = data_cf[sens_attr].float()
    data_cf.s = torch.cat([data_cf.s,torch.zeros([num_ent - data_cf.s.size(0), data_cf.s.size(1)]).cuda()])

# load robustness data
    data_rb = load_data_rb()
    data_rb.s = data_rb[sens_attr].float()
    data_rb.s = torch.cat([data_rb.s,torch.zeros([num_ent - data_rb.s.size(0), data_rb.s.size(1)]).cuda()])

    print("Initiating task model")
    sensitive_dim = 0
    if is_private:
        if use_sensitive_mlp:
            sensitive_dim = output_emd_dim
        else:
            sensitive_dim = data.s.size(1)
    
    emb_file = f"{emb_save_name}_{seed}.pth"
    is_model_trained = False
    if (retrain) or ((not retrain) and (not os.path.exists(emb_file))):
        # For GIB-GAT, GAT or GCN:
        model = GRAFair(
            model_type=model_type,
            num_features=data.x.size(1),
            num_classes=2,
            normalize=True,
            reparam_mode=reparam_mode,
            prior_mode=prior_mode,
            latent_size=latent_size,
            num_sensitive=data.s.size(1),
            sample_size=sample_size,
            num_layers=num_layers,
            dropout=False, # dont use dropout
            with_relu=False, # dont use relu
            with_bias=True, # always use bias in GCN
            val_use_mean=val_use_mean,
            reparam_all_layers=reparam_all_layers,
            is_cuda=is_cuda,
            is_private=is_private,
            heads=heads,
            use_sensitive_mlp=use_sensitive_mlp,
        )
        print(model)
        print("Training task model")

        data_record, embeddings = train_GRAFair(
                    model=model,
                    data=data,
                    data_type=data_type,
                    model_type=model_type,
                    beta_list=beta_list,
                    epochs=epochs,
                    inspect_interval=20,
                    verbose=True,
                    isplot=False,
                    compute_metrics=None,
                    lr=lr,
                    weight_decay=weight_decay,
                    save_best_model=save_best_model,
                )
        gc.collect()
        
        model.eval()
        # load counterfactual data
        logits, _ = model(data)
        test_proba = torch.softmax(logits, 1)[data['test_id_feat'][0]].detach().cpu().numpy()
        test_pred = test_proba.argmax(1)
        x_cf = model.encode(data_cf)[0]
        cf_01 = 0
        if model.is_private:
            if model.use_sensitive_mlp:
                x_cf = torch.cat([x_cf,model.sensitive_mlp(data_cf.s)],dim=1)
            else:
                x_cf = torch.cat([x_cf,data_cf.s],dim=1)
        logits_cf = model.classifier(x_cf)
        test_proba_cf = torch.softmax(logits_cf, 1)[data['test_id_feat'][0]].detach().cpu().numpy()
        test_pred_cf = test_proba_cf.argmax(1)
        for i in range(data['test_id_feat'][0].size(0)):
            if test_pred_cf[i]==test_pred[i]:
                cf_01 += 1
        counterfactual = 1- cf_01 / data['test_id_feat'][0].size(0)
        # load robustness data
        x_rb = model.encode(data_rb)[0]
        rb = 0
        if model.is_private:
            if model.use_sensitive_mlp:
                x_rb = torch.cat([x_rb,model.sensitive_mlp(data_rb.s)],dim=1)
            else:
                x_rb = torch.cat([x_rb,data_rb.s],dim=1)
        logits_rb = model.classifier(x_rb)
        test_proba_rb = torch.softmax(logits_rb, 1)[data['test_id_feat'][0]].detach().cpu().numpy()
        test_pred_rb = test_proba_rb.argmax(1)
        for i in range(data['test_id_feat'][0].size(0)):
            if test_pred_rb[i]==test_pred[i]:
                rb += 1
        robustness = 1- rb / data['test_id_feat'][0].size(0)
    
        data_record['unfairness'] = counterfactual
        data_record['robustness'] = robustness

        
        best_metrics_list.append(data_record)
        is_model_trained = True

    # load robustness data
    


if is_model_trained:
    best_metrics_df = pd.DataFrame(best_metrics_list).mean().to_frame().T
    #best_metrics_df = best_metrics_df.drop(columns=['val_rmse'])


if is_model_trained:
    log_df = pd.concat([best_metrics_df],axis=1)
    log_df['dataset'] = dataset
    log_df['num_layers'] = num_layers
    log_df['model_type'] = model_type
    log_df['beta'] = -1 if beta is None else beta
    log_df['lr'] = lr
    log_df['epochs'] = epochs
    log_df['clf_layers'] = 1
    log_df['is_private'] = is_private
    log_df['prior_mode'] = prior_mode
    log_df['reparam_all_layers'] = reparam_all_layers


    log_df_name = f"{log_path_name}.csv"
    log_df.to_csv(log_df_name)

# std
if is_model_trained:
    best_metrics_std = pd.DataFrame(best_metrics_list).std().to_frame().T
    #best_metrics_std = best_metrics_std.drop(columns=['val_rmse'])


if is_model_trained:
    log_std = pd.concat([best_metrics_std],axis=1)
    log_std['dataset'] = dataset
    log_std['num_layers'] = num_layers
    log_std['model_type'] = model_type
    log_std['beta'] = -1 if beta is None else beta
    log_std['lr'] = lr
    log_std['epochs'] = epochs
    log_std['clf_layers'] = 1
    log_std['is_private'] = is_private
    log_std['prior_mode'] = prior_mode
    log_std['reparam_all_layers'] = reparam_all_layers

    log_std_name = f"{log_path_name}_std.csv"
    log_std.to_csv(log_std_name)

if is_model_trained:
    log_task_all = f"{log_path_name}_task.csv"
    pd.DataFrame(best_metrics_list).to_csv(log_task_all)
