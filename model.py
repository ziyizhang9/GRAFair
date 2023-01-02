from collections import OrderedDict
from copy import deepcopy
import itertools
import matplotlib.pylab as plt
import numpy as np
import math
import os.path as osp
import pickle
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import sklearn
from sklearn.manifold import TSNE
import torch
from torch.nn import Parameter, Linear, BatchNorm1d, Embedding, NLLLoss, MSELoss
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, softmax, negative_sampling, degree, to_undirected
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
from pytorch_net.util import sample, to_cpu_recur, to_np_array, to_Variable, record_data, make_dir, remove_duplicates, update_dict, get_list_elements, to_string, filter_filename
from util import get_reparam_num_neurons, sample_lognormal, scatter_sample, uniform_prior, compose_log, edge_index_2_csr, COLOR_LIST, LINESTYLE_LIST, process_data_for_nettack, parse_filename, add_distant_neighbors
from DeepRobust.deeprobust.graph.targeted_attack import Nettack

from tqdm import tqdm
import time

# ## GCNConv:


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True,
                 reparam_mode=None, prior_mode=None, sample_size=1, val_use_mean=True,
                 **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.reparam_mode = None if reparam_mode == "None" else reparam_mode
        self.prior_mode = prior_mode
        self.val_use_mean = val_use_mean
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, self.out_neurons))

        if bias: 
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)

        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    

    def set_cache(self, cached):
        self.cached = cached


    def to_device(self, device):
        self.to(device)
        if self.cached and self.cached_result is not None:
            edge_index, norm = self.cached_result
            self.cached_result = edge_index.to(device), norm.to(device)
        return self


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.float()
        self.weight = self.weight.float()
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        out = self.propagate(edge_index, x=x, norm=norm)

        if self.reparam_mode is not None:
            # Reparameterize:
            self.dist, _ = reparameterize(model=None, input=out, 
                                          mode=self.reparam_mode, 
                                          size=self.out_channels
                                         )  # [B, Z]
            Z = sample(self.dist, self.sample_size)  # [S, B, Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(x.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(x.size(0), self.out_channels).to(x.device),
                                           )  # [B, Z]

            # Calculate prior loss:
            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1)
            else:
                Z_logit = self.dist.log_prob(Z).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z)  # [S, B]
                prior_logit = self.feature_prior.log_prob(Z).sum(-1)  # [S, B]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)  # [B, Z]
            else:
                out = out[:, :self.out_channels]  # [B, Z]
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)  # [B]

        structure_kl_loss = torch.zeros([]).to(x.device)
        return out, ixz, structure_kl_loss
            

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# ## SAGEConv:
class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 reparam_mode=None, prior_mode=None, sample_size=1, val_use_mean=True,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.reparam_mode = None if reparam_mode == "None" else reparam_mode
        self.prior_mode = prior_mode
        self.val_use_mean = val_use_mean
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, self.out_neurons, bias=bias)
        self.lin_root = Linear(in_channels, self.out_neurons, bias=False)

        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def set_cache(self, cached):
        self.cached = cached


    def to_device(self, device):
        self.to(device)
        if self.cached and self.cached_result is not None:
            edge_index, norm = self.cached_result
            self.cached_result = edge_index.to(device), norm.to(device)
        return self

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.float()
        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_rel(out) + self.lin_root(x[1])

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        if self.reparam_mode is not None:
            # Reparameterize:
            self.dist, _ = reparameterize(model=None, input=out, 
                                          mode=self.reparam_mode, 
                                          size=self.out_channels
                                         )  # [B, Z]
            Z = sample(self.dist, self.sample_size)  # [S, B, Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(x[1].size(0), self.out_channels).to(x[1].device),
                                            scale=torch.ones(x[1].size(0), self.out_channels).to(x[1].device),
                                           )  # [B, Z]

            # Calculate prior loss:
            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1)
            else:
                Z_logit = self.dist.log_prob(Z).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z)  # [S, B]
                prior_logit = self.feature_prior.log_prob(Z).sum(-1)  # [S, B]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)  # [B, Z]
            else:
                out = out[:, :self.out_channels]  # [B, Z]
        else:
            ixz = torch.zeros(x[1].size(0)).to(x[1].device)  # [B]

        structure_kl_loss = torch.zeros([]).to(x[1].device)
        return out, ixz, structure_kl_loss

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# ## GAT
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    `_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        struct_dropout_mode (tuple, optional): Choose from: None, ("standard", prob), ("info", ${MODE}),
            where ${MODE} chooses from "subset", "lognormal", "loguniform".
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, reparam_mode=None, prior_mode=None,
                 struct_dropout_mode=None, sample_size=1,
                 val_use_mean=True,
                 bias=True,
                 **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * self.out_neurons))
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_neurons))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)
            
        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        
        x = x.float()
        self.weight = self.weight.float()

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x)

        if self.reparam_mode is not None:
            # Reparameterize:
            out = out.view(-1, self.out_neurons)
            self.dist, _ = reparameterize(model=None, input=out,
                                          mode=self.reparam_mode,
                                          size=self.out_channels,
                                         )  # dist: [B * head, Z]
            Z_core = sample(self.dist, self.sample_size)  # [S, B * head, Z]
            Z = Z_core.view(self.sample_size, -1, self.heads * self.out_channels)  # [S, B, head * Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(out.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(out.size(0), self.out_channels).to(x.device),
                                           )  # feature_prior: [B * head, Z]

            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1).view(-1, self.heads).mean(-1)
            else:
                Z_logit = self.dist.log_prob(Z_core).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z_core)  # [S, B * head]
                prior_logit = self.feature_prior.log_prob(Z_core).sum(-1)  # [S, B * head]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0).view(-1, self.heads).mean(-1)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)
            else:
                out = out[:, :self.out_channels].contiguous().view(-1, self.heads * self.out_channels)
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)

        structure_kl_loss = torch.zeros([]).to(x.device)

        return out, ixz, structure_kl_loss

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_neurons)  # [N_edge, heads, out_channels]
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_neurons:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_neurons)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [N_edge, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_neurons)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


    def to_device(self, device):
        self.to(device)
        return self

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# ## GINConv
class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, eps=0, train_eps=False, 
                 reparam_mode=None, prior_mode=None, sample_size=1, val_use_mean=True,
                 **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.out_neurons), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.out_neurons),
            torch.nn.Linear(self.out_neurons, self.out_neurons), 
        )
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.nn.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
        if self.reparam_mode is not None:
            # Reparameterize:
            self.dist, _ = reparameterize(model=None, input=out, 
                                          mode=self.reparam_mode, 
                                          size=self.out_channels
                                         )  # [B, Z]
            Z = sample(self.dist, self.sample_size)  # [S, B, Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(x.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(x.size(0), self.out_channels).to(x.device),
                                           )  # [B, Z]

            # Calculate prior loss:
            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1)
            else:
                Z_logit = self.dist.log_prob(Z).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z)  # [S, B]
                prior_logit = self.feature_prior.log_prob(Z).sum(-1)  # [S, B]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)  # [B, Z]
            else:
                out = out[:, :self.out_channels]  # [B, Z]
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)  # [B]

        structure_kl_loss = torch.zeros([]).to(x.device)
        return out, ixz, structure_kl_loss


    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

# ## GRAFair:

class GRAFair(torch.nn.Module):
    def __init__(
        self,
        model_type,
        num_features,
        num_classes,
        reparam_mode,
        prior_mode,
        latent_size,
        num_sensitive,
        sample_size=1,
        num_layers=2,
        dropout=True,
        with_relu=True,
        with_bias=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
        is_cuda=False,
        is_private=False,
        heads=8,
        use_sensitive_mlp=False,
    ):
        """Class implementing a general GNN, which can realize GAT, GIB-GAT, GCN.
        
        Args:
            model_type:   name of the base model. Choose from "GAT", "GCN".
            num_features: number of features of the data.x.
            num_classes:  number of classes for data.y.
            reparam_mode: reparameterization mode for XIB. Choose from "diag" and "full". Default "diag" that parameterizes the mean and diagonal element of the Gaussian
            prior_mode:   distribution type for the prior. Choose from "Gaussian" or "mixGau-{Number}", where {Number} is the number of components for mixture of Gaussian.
            latent_size:  latent size for each layer of GNN. If model_type="GAT", the true latent size is int(latent_size/2)
            sample_size=1:how many Z to sample for each feature X.
            num_layers=2: number of layers for the GNN
            dropout:      whether to use dropout on features.
            with_relu:    whether to use nonlinearity for GCN.
            val_use_mean: Whether during evaluation use the parameter value instead of sampling. If True, during evaluation,
                          XIB will use mean for prediction, and AIB will use the parameter of the categorical distribution for prediction.
            reparam_all_layers: Which layers to use XIB, e.g. (1,2,4). Default (-2,), meaning the second last layer. If True, use XIB for all layers.
            normalize:    whether to normalize for GCN (only effective for GCN)
            is_cuda:      whether to use CUDA, and if so, which GPU to use. Choose from False, True, "CUDA:{GPU_ID}", where {GPU_ID} is the ID for the CUDA.
        """
        super(GRAFair, self).__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.normalize = normalize
        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.dropout = dropout
        self.latent_size = latent_size
        self.sample_size = sample_size
        self.num_layers = num_layers
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.val_use_mean = val_use_mean
        self.reparam_all_layers = reparam_all_layers
        self.is_cuda = is_cuda
        self.is_private = is_private
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")

        self.heads = heads
        self.num_sensitive = num_sensitive
        self.use_sensitive_mlp = use_sensitive_mlp
        self.init()


    def init(self):
        """Initialize the layers for the GNN."""
        self.reparam_layers = []
        if self.model_type in ["GCN","SAGE","GAT","GIN"]:
            for i in range(self.num_layers):
                if self.reparam_all_layers is True:
                    is_reparam = True
                elif isinstance(self.reparam_all_layers, tuple):
                    reparam_all_layers = tuple([kk + self.num_layers if kk < 0 else kk for kk in self.reparam_all_layers])
                    is_reparam = i in reparam_all_layers
                else:
                    raise
                if is_reparam:
                    self.reparam_layers.append(i)
                if self.model_type == "GCN":
                    setattr(self, "conv{}".format(i + 1),
                            GCNConv(self.num_features if i == 0 else self.latent_size,
                                    self.latent_size,
                                    cached=True,
                                    reparam_mode=self.reparam_mode if is_reparam else None,
                                    prior_mode=self.prior_mode if is_reparam else None,
                                    sample_size=self.sample_size,
                                    # bias=True if self.with_relu else False,
                                    bias=self.with_bias,
                                    val_use_mean=self.val_use_mean,
                                    normalize=self.normalize,
                    ))
                elif self.model_type == "SAGE":
                    setattr(self, "conv{}".format(i + 1),
                            SAGEConv(self.num_features if i == 0 else self.latent_size,
                                     self.latent_size,
                                    #  cached=True,
                                     reparam_mode=self.reparam_mode if is_reparam else None,
                                     prior_mode=self.prior_mode if is_reparam else None,
                                     sample_size=self.sample_size,
                                     # bias=True if self.with_relu else False,
                                     bias=self.with_bias,
                                     val_use_mean=self.val_use_mean,
                                     normalize=self.normalize,

                    ))
                elif self.model_type == "GAT":
                    setattr(self, "conv{}".format(i + 1),
                            GATConv(self.num_features if i == 0 else self.latent_size,
                                     self.latent_size,
                                    #  cached=True,
                                     reparam_mode=self.reparam_mode if is_reparam else None,
                                     prior_mode=self.prior_mode if is_reparam else None,
                                     sample_size=self.sample_size,
                                     # bias=True if self.with_relu else False,
                                     bias=self.with_bias,
                                     val_use_mean=self.val_use_mean,
                                    #  normalize=self.normalize,

                    ))   
                elif self.model_type == "GIN":
                    setattr(self, "conv{}".format(i + 1),
                            GINConv(self.num_features if i == 0 else self.latent_size,
                                     self.latent_size,
                                    #  cached=True,
                                     reparam_mode=self.reparam_mode if is_reparam else None,
                                     prior_mode=self.prior_mode if is_reparam else None,
                                     sample_size=self.sample_size,
                                     # bias=True if self.with_relu else False,
                                     val_use_mean=self.val_use_mean,
                                    #  normalize=self.normalize,

                    ))           
        else:
            raise Exception("Model_type {} is not valid!".format(self.model_type))

        self.reparam_layers = sorted(self.reparam_layers)

        if self.use_sensitive_mlp:
            self.sensitive_mlp = torch.nn.Sequential(
                        torch.nn.Linear(self.num_sensitive, self.latent_size * self.heads // 2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(self.latent_size * self.heads // 2, self.latent_size * self.heads // 2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(self.latent_size * self.heads // 2, self.latent_size * self.heads // 2)).cuda()
        

        # control beta
        self.classifier = torch.nn.Linear(self.latent_size+2 , self.num_classes).cuda()
        #self.classifier = torch.nn.Linear(self.latent_size , self.num_classes).cuda()
        # self.classifier = torch.nn.Sequential(torch.nn.Linear(self.latent_size+2 , 32),
        #         torch.nn.LeakyReLU(),
        #         torch.nn.Linear(32, 32),
        #         torch.nn.LeakyReLU(),
        #         torch.nn.Linear(32, self.num_classes)).cuda()

        if self.model_type in ["GCN","SAGE","GAT","GIN"]:
            if self.with_relu:
                raise NotImplementedError # Modify the parameters
                reg_params = [getattr(self, "conv{}".format(i+1)).parameters() for i in range(self.num_layers - 1)]
                self.reg_params = itertools.chain(*reg_params)
                self.non_reg_params = getattr(self, "conv{}".format(self.num_layers)).parameters()
            else:
                self.reg_params = OrderedDict()
                self.non_reg_params = self.parameters()
        else:
            self.reg_params = self.parameters()
            self.non_reg_params = OrderedDict()

        self.to(self.device)


    def set_cache(self, cached):
        """Set cache for GCN."""
        for i in range(self.num_layers):
            if hasattr(getattr(self, "conv{}".format(i+1)), "set_cache"):
                getattr(self, "conv{}".format(i+1)).set_cache(cached)


    def to_device(self, device):
        """Send all the layers to the specified device."""
        for i in range(self.num_layers):
            getattr(self, "conv{}".format(i+1)).to_device(device)
        self.to(device)
        return self

    def encode(self, data, record_Z=False, isplot=False):
        reg_info = {}
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if self.model_type in ["GCN","SAGE","GAT","GIN"]:
            for i in range(self.num_layers - 1):
                layer = getattr(self, "conv{}".format(i + 1))
                x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
                # Record:
                record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
                if layer.reparam_mode is not None:
                    record_data(reg_info, [layer.Z_std], ["Z_std"])
                if record_Z:
                    record_data(reg_info, [to_np_array(x)], ["Z_{}".format(i)], nolist=True)
                if self.with_relu:
                    x = F.relu(x)
                    if self.dropout is True:
                        x = F.dropout(x, training=self.training)
            layer = getattr(self, "conv{}".format(self.num_layers))
            x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
            # Record:
            record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
            if layer.reparam_mode is not None:
                record_data(reg_info, [layer.Z_std], ["Z_std"])
            if record_Z:
                record_data(reg_info, [to_np_array(x)], ["Z_{}".format(self.num_layers - 1)], nolist=True)
        return x, reg_info

    def forward(self, data, record_Z=False, isplot=False):
        """Main forward function.
    
        Args:
            data: the pytorch-geometric data class.
            record_Z: whether to record the standard deviation for the representation Z.
            isplot:   whether to plot.
        
        Returns:
            x: output
            reg_info: other information or metrics.
        """
        x, reg_info = self.encode(data, record_Z=False, isplot=False)
        # decoder
        if self.is_private:
            if self.use_sensitive_mlp:
                x = torch.cat([x,self.sensitive_mlp(data.s)],dim=1)
            else:
                x = torch.cat([x,data.s],dim=1)

        res = self.classifier(x)
        return res, reg_info

def train_model(model, data, optimizer, beta=None):
    model.train()
    optimizer.zero_grad()
    nodes_logits, reg_info = model(data)
    output_clf = nodes_logits[data.train_id_feat[0]]
    labels_clf = data.train_id_feat[2].cuda()
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(output_clf, labels_clf)
        # Add IB loss:
    if beta is not None and beta != 0:
        ixz = torch.stack(reg_info["ixz_list"], 1).mean(0).sum()
        loss = loss + ixz * beta
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test_model(
    model,
    data,
):
    metrics_all = {}
    model.eval()
    nodes_logits, _ = model(data)
    val_proba = torch.softmax(nodes_logits, 1)[data['val_id_feat'][0]].detach().cpu().numpy()
    val_pred = val_proba.argmax(1)
    val_labels = data['val_id_feat'][2].detach().cpu().numpy()

    metrics_all['val_auc_binary'] = 0
    metrics_all['val_auc_macro'] = 0
    metrics_all['val_auc_weighted'] = 0
    try:
        if data.s.size(1) == 2:
            metrics_all['val_auc_binary'] = sklearn.metrics.roc_auc_score(val_labels, val_proba[:,1])
        else:
            metrics_all['val_auc_macro'] = sklearn.metrics.roc_auc_score(val_labels, val_proba, average='macro',multi_class='ovr')
            metrics_all['val_auc_weighted'] = sklearn.metrics.roc_auc_score(val_labels, val_proba, average='weighted',multi_class='ovr')
    except:
        pass
    metrics_all['val_f1_micro'] = sklearn.metrics.f1_score(val_labels, val_pred, average='micro')
    metrics_all['val_f1_macro'] = sklearn.metrics.f1_score(val_labels, val_pred , average='macro')
    metrics_all['val_f1_weighted'] = sklearn.metrics.f1_score(val_labels, val_pred, average='weighted')
 
    test_proba = torch.softmax(nodes_logits, 1)[data['test_id_feat'][0]].detach().cpu().numpy()
    test_pred = test_proba.argmax(1)
    test_labels = data['test_id_feat'][2].detach().cpu().numpy()

    metrics_all['test_sparity'] = 0
    metrics_all['test_equality'] = 0
    metrics_all['unfairness'] = 0
    metrics_all['robustness'] = 0

# add the sparity
    att_0 = 0
    att_1 = 0
    test_nodes = np.arange(data['test_id_feat'][0].size(0))
    for i in range(data['test_id_feat'][0].size(0)):
        if data['test_id_feat'][1][i] == 0:
            test_nodes[i] = 0
            att_0 += 1
        else:
            test_nodes[i] = 1
            att_1 += 1
    test_pred_0 = test_pred[np.where(test_nodes==0)]
    test_pred_1 = test_pred[np.where(test_nodes==1)]
    sparity = abs( sum(test_pred_0) / att_0 - sum(test_pred_1) / att_1 )
    
# add the equality
    att_0_1 = 0
    att_1_1 = 0
    test_nodes = np.arange(data['test_id_feat'][0].size(0))
    for i in range(data['test_id_feat'][0].size(0)):
        if data['test_id_feat'][1][i] == 0 and data['test_id_feat'][2][i] == 1:
            test_nodes[i] = 0
            att_0_1 += 1
        elif data['test_id_feat'][1][i] == 1 and data['test_id_feat'][2][i] == 1:
            test_nodes[i] = 1
            att_1_1 += 1
        else:
            test_nodes[i] = 2
    test_pred_0_1 = test_pred[np.where(test_nodes==0)]
    test_pred_1_1 = test_pred[np.where(test_nodes==1)]
    equality = abs( sum(test_pred_0_1) / att_0_1 - sum(test_pred_1_1) / att_1_1 )


    metrics_all['test_sparity'] = sparity
    metrics_all['test_equality'] = equality

    metrics_all['test_auc_binary'] = 0
    metrics_all['test_auc_macro'] = 0
    metrics_all['test_auc_weighted'] = 0
    
    try:
        if data.s.size(1) == 2:
            metrics_all['test_auc_binary'] = sklearn.metrics.roc_auc_score(test_labels, test_proba[:,1])
        else:
            metrics_all['test_auc_macro'] = sklearn.metrics.roc_auc_score(test_labels, test_proba, average='macro',multi_class='ovr')
            metrics_all['test_auc_weighted'] = sklearn.metrics.roc_auc_score(test_labels, test_proba, average='weighted',multi_class='ovr')
    except:
        pass
    metrics_all['test_f1'] = sklearn.metrics.f1_score(test_labels, test_pred)
    metrics_all['test_acc'] = sklearn.metrics.accuracy_score(test_labels, test_pred)
    metrics_all['test_f1_micro'] = sklearn.metrics.f1_score(test_labels, test_pred, average='micro')
    metrics_all['test_f1_macro'] = sklearn.metrics.f1_score(test_labels, test_pred , average='macro')
    metrics_all['test_f1_weighted'] = sklearn.metrics.f1_score(test_labels, test_pred, average='weighted')

    return metrics_all

def train_GRAFair(
    model,
    data,
    data_type,
    model_type,
    beta_list,
    epochs,
    verbose=True,
    inspect_interval=10,
    isplot=True,
    filename=None,
    compute_metrics=None, # "Silu", "DB", "CH"
    lr=None,
    weight_decay=None,
    save_best_model=False,
):
    """Training multiple epochs."""
    if lr is None:
        if model_type == "GCN":
            lr = 0.01
        elif model_type == "GAT":
            lr = 0.01 if data_type.startswith("Pubmed") else 0.005
        else:
            lr = 0.01
    if weight_decay is None:
        if model_type == "GCN":
            weight_decay = 5e-4 
        elif model_type == "GAT":
            weight_decay = 1e-3 if data_type.startswith("Pubmed") else 5e-4
        else:
            weight_decay = 5e-4  

    # Training:
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=lr)
    best_metrics = {}
    best_metrics["val_f1_macro"] = 0
    best_metrics["val_f1_micro"] = 0
    best_metrics["val_f1_weighted"] = 0
    best_metrics["val_auc_binary"] = 0
    best_metrics["val_auc_macro"] = 0
    best_metrics["val_auc_weighted"] = 0
    best_metrics["test_f1_macro"] = 0
    best_metrics["test_f1_micro"] = 0
    best_metrics["test_f1"] = 0
    best_metrics["test_acc"] = 0
    best_metrics["test_f1_weighted"] = 0
    best_metrics["test_auc_binary"] = 0
    best_metrics["test_auc_macro"] = 0
    best_metrics["test_auc_weighted"] = 0
    best_metrics['test_sparity'] = 0
    best_metrics['test_equality'] = 0
    best_metrics['unfairness'] = 0
    best_metrics['robustness'] = 0
    
    #data_record = {"num_layers": model.num_layers}

    # Train:
    mean_time = []
    for epoch in range(1, epochs + 1):
        beta = beta_list[epoch] if beta_list is not None else None
        time1 = time.time()
        loss = train_model(
            model,
            data,
            optimizer,
            beta=beta,
        )
        mean_time.append(time.time() - time1)
        metrics = test_model(model, data)
        if metrics["val_f1_macro"] > best_metrics["val_f1_macro"]:
            best_embeddings = model.encode(data)[0].detach().clone()
            for key in best_metrics.keys():
                best_metrics[key] = metrics[key]
            #if save_best_model:
            #    data_record["best_model_dict"] = deepcopy(model.model_dict)

        #record_data(data_record, list(metrics.values()), list(metrics.keys()))
        if verbose and epoch % inspect_interval == 0:
            print(f"Inspect Epoch:{epoch} \nCurrent Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v} ", end=" ")
            print(f"\nloss: {loss}")    
            print(f"\nBest Metrics so far:")
            for k, v in best_metrics.items():
                print(f"{k}: {v} ", end=" ")
            # #record_data(data_record, [epoch], ["inspect_epoch"])
            # log = 'Epoch: {:03d}:'.format(epoch) + '\tF1 micro: ({:.4f}, {:.4f}, {:.4f})'.format(metrics["train_f1_micro"], best_val_f1, b_test_f1_micro)
            # log += compose_log(metrics, "f1_macro", 2)
            # log += compose_log(metrics, "acc", tabs=2, newline=True) + compose_log(metrics, "loss", 7)
            # if beta is not None:
            #     log += "\n\t\tixz: ({:.4f}, {:.4f}, {:.4f})".format(metrics["train_ixz"], metrics["val_ixz"], metrics["test_ixz"])
            #     if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            #         log += " " * 7 + "ixz_DN: ({:.4f}, {:.4f}, {:.4f})".format(metrics["train_ixz_DN"], metrics["val_ixz_DN"], metrics["test_ixz_DN"])
            #     if "Z_std" in metrics:
            #         log += "\n\t\tZ_std: {}".format(to_string(metrics["Z_std"], connect=", ", num_digits=4))
            # if beta2 is not None:
            #     log += "\n\t\tstruct_kl: {:.4f}".format(metrics["structure_kl"])
            # if compute_metrics is not None:
            #     for metric in compute_metrics:
            #         log += "\n\t"
            #         for kk in range(model.num_layers):
            #             List = [metrics["{}{}_{}".format(id, metric, kk)] for id in ["", "train_", "val_", "test_"]]
            #             log += "\t{}_{}:\t({})".format(metric, kk, "{:.4f}; ".format(List[0]) + to_string(List[1:], connect=", ", num_digits=4))
            # log += "\n"
            # print(log)
            # try:
            #     sys.stdout.flush()
            # except:
            #     pass


        # Saving:
        # if epoch % 200 == 0:
        #     data_record["model_dict"] = model.model_dict
        #     if filename is not None:
        #         make_dir(filename)
        #         pickle.dump(data_record, open(filename + ".p", "wb"))

    # Plotting:
    #if isplot:
    #    plot(data_record, compute_metrics=compute_metrics)
    best_metrics["time"] = sum(mean_time) / len(mean_time)
    return best_metrics, best_embeddings


def remove_edge_random(data, remove_edge_fraction):
    """Randomly remove a certain fraction of edges."""
    data_c = deepcopy(data)
    num_edges = int(data_c.edge_index.shape[1] / 2)
    num_removed_edges = int(num_edges * remove_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data_c.edge_index.T)]
    for i in range(num_removed_edges):
        idx = np.random.choice(len(edges))
        edge = edges[idx]
        edge_r = (edge[1], edge[0])
        edges.pop(idx)
        try:
            edges.remove(edge_r)
        except:
            pass
    data_c.edge_index = torch.LongTensor(np.array(edges).T).to(data.edge_index.device)
    return data_c


def add_random_edge(data, added_edge_fraction=0):
    """Add random edges to the original data's edge_index."""
    if added_edge_fraction == 0:
        return data
    data_c = deepcopy(data)
    num_edges = int(data.edge_index.shape[1] / 2)
    num_added_edges = int(num_edges * added_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data.edge_index.T)]
    added_edges = []
    for i in range(num_added_edges):
        while True:
            added_edge_cand = tuple(np.random.choice(data.x.shape[0], size=2, replace=False))
            added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
            if added_edge_cand in edges or added_edge_cand in added_edges:
                if added_edge_cand in edges:
                    assert added_edge_r_cand in edges
                if added_edge_cand in added_edges:
                    assert added_edge_r_cand in added_edges
                continue
            else:
                added_edges.append(added_edge_cand)
                added_edges.append(added_edge_r_cand)
                break

    added_edge_index = torch.LongTensor(np.array(added_edges).T).to(data.edge_index.device)
    data_c.edge_index = torch.cat([data.edge_index, added_edge_index], 1)
    return data_c

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    
    Adapted from https://github.com/danielzuegner/nettack/blob/master/nettack/utils.py

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return nodes_to_keep

