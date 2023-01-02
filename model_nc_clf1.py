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

# ## ChebConv
class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a scalar when
            operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 reparam_mode=None, prior_mode=None, sample_size=1, val_use_mean=True,
                 bias=True, **kwargs):
        super(ChebConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.reparam_mode = None if reparam_mode == "None" else reparam_mode
        self.prior_mode = prior_mode
        self.val_use_mean = val_use_mean
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_channels, self.out_neurons))

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
    
    def set_cache(self, cached):
        self.cached = cached


    def to_device(self, device):
        self.to(device)
        if self.cached and self.cached_result is not None:
            edge_index, norm = self.cached_result
            self.cached_result = edge_index.to(device), norm.to(device)
        return self

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, normalization, lambda_max,
             dtype=None, batch=None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and torch.is_tensor(lambda_max):
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight[edge_weight == float('inf')] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1,
                                                 num_nodes=num_nodes)

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None,
                lambda_max=None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                     edge_weight, self.normalization,
                                     lambda_max, dtype=x.dtype, batch=batch)

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

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
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

# ## VGPF:

class VGPF(torch.nn.Module):
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
        super(VGPF, self).__init__()
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
        if self.model_type in ["GCN","SAGE","Cheb","GAT","GIN"]:
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
                elif self.model_type == "Cheb":
                    setattr(self, "conv{}".format(i + 1),
                            ChebConv(self.num_features if i == 0 else self.latent_size,
                                     self.latent_size,
                                     K=2,
                                    #  cached=True,
                                     reparam_mode=self.reparam_mode if is_reparam else None,
                                     prior_mode=self.prior_mode if is_reparam else None,
                                     sample_size=self.sample_size,
                                     # bias=True if self.with_relu else False,
                                     bias=self.with_bias,
                                     val_use_mean=self.val_use_mean,
                                    #  normalize=self.normalize,

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

        if self.model_type in ["GCN","SAGE","Cheb","GAT","GIN"]:
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
        if self.model_type in ["GCN","SAGE","Cheb","GAT","GIN"]:
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


class VGPF_ml(torch.nn.Module):
    def __init__(
        self,
        model_type,
        num_features,
        num_ent,
        reparam_mode,
        prior_mode,
        latent_size,
        decoder, 
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
        super(VGPF_ml, self).__init__()
        self.model_type = model_type
        self.num_features = num_features
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

        self.decoder = decoder
        self.num_ent = num_ent
        self.heads = heads
        self.num_sensitive = num_sensitive
        self.use_sensitive_mlp = use_sensitive_mlp
        self.init()


    def init(self):
        """Initialize the layers."""
        self.batchnorm = BatchNorm1d(self.latent_size)
        r = 6 / np.sqrt(self.latent_size)
        self.emb_layer = Embedding(self.num_ent, self.latent_size, \
                                    max_norm=1, norm_type=2)
        self.emb_layer.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)

        self.reparam_layers = []
        if self.model_type in ["GCN","SAGE","Cheb"]:
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
                            GCNConv(self.latent_size,
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
                            SAGEConv(self.latent_size,
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
                elif self.model_type == "Cheb":
                    setattr(self, "conv{}".format(i + 1),
                            ChebConv(self.latent_size,
                                     self.latent_size,
                                     K=2,
                                    #  cached=True,
                                     reparam_mode=self.reparam_mode if is_reparam else None,
                                     prior_mode=self.prior_mode if is_reparam else None,
                                     sample_size=self.sample_size,
                                     # bias=True if self.with_relu else False,
                                     bias=self.with_bias,
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
   
        if self.model_type in ["GCN","SAGE","Cheb"]:
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

        x = self.emb_layer(x)
        x = self.batchnorm(x)

        if self.model_type in ["GCN","SAGE","Cheb"]:
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

    def forward(self, data, pos_edges, record_Z=False, isplot=False):
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
        pos_head_embeds = x[pos_edges[:, 0]]
        pos_tail_embeds = x[pos_edges[:, -1]]
        res = self.decoder(pos_head_embeds, pos_tail_embeds)

        # res = torch.einsum("ef,ef->e", x_i, x_j)
        return res, reg_info


    def compute_metrics_fun(self, data, metrics, mask=None, mask_id=None):
        """Compute metrics for measuring clustering performance.
        Choices: "Silu", "CH", "DB".
        """
        _, info_dict = self(data, record_Z=True)
        y = to_np_array(data.y)
        info_metrics = {}
        if mask is not None:
            mask = to_np_array(mask)
            mask_id += "_"
        else:
            mask_id = ""
        for k in range(self.num_layers):
            if mask is not None:
                Z_i = info_dict["Z_{}".format(k)][mask]
                y_i = y[mask]
            else:
                Z_i = info_dict["Z_{}".format(k)]
                y_i = y
            for metric in metrics:
                if metric == "Silu":
                    score = sklearn.metrics.silhouette_score(Z_i, y_i, metric='euclidean')
                elif metric == "DB":
                    score = sklearn.metrics.davies_bouldin_score(Z_i, y_i)
                elif metric == "CH":
                    score = sklearn.metrics.calinski_harabasz_score(Z_i, y_i)
                info_metrics["{}{}_{}".format(mask_id, metric, k)] = score
        return info_metrics


class SharedBilinearDecoder(torch.nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """
    def __init__(self, num_relations, num_weights, embed_dim):
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = Embedding(num_weights, embed_dim * embed_dim)
        self.weight_scalars = Parameter(torch.Tensor(num_weights, num_relations))
        stdv = 1. / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        # self.nll = NLLLoss()
        # self.mse = MSELoss()

    def predict(self, embeds1, embeds2):
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, \
                                                     self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        preds = 0
        for j in range(0, self.num_relations):
            index = (torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return preds

    def forward(self, embeds1, embeds2):
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, \
                                                     self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        return logit

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                            neg_edge_index.size(1)).float().cuda()
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train_model_lp(model, data, optimizer, beta1=None):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()
    x, pos_edge_index, all_edge_index = data.x, data.train_pos_edge_index, data.edge_index

    _edge_index, _ = remove_self_loops(all_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                    num_nodes=x.size(0))
    # negative_sampling in all edges
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=pos_edge_index.size(1))

    edge_logits, reg_info = model(data, pos_edge_index, neg_edge_index)
    edge_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
    

    # Add IB loss:
    if beta1 is not None and beta1 != 0:
        ixz = torch.stack(reg_info["ixz_list"], 1).mean(0).sum()
#        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
#            ixz = ixz + torch.stack(reg_info["ixz_DN_list"], 1).mean(0).sum()
        loss = loss + ixz * beta1
    
    loss.backward()
    optimizer.step()
    return loss

def train_model_nc(model, data, optimizer, beta1=None):
    model.train()
    optimizer.zero_grad()
    nodes_logits, reg_info = model(data)
    output_clf = nodes_logits[data.train_id_feat[0]]
    labels_clf = data.train_id_feat[2].cuda()
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(output_clf, labels_clf)
        # Add IB loss:
    if beta1 is not None and beta1 != 0:
        ixz = torch.stack(reg_info["ixz_list"], 1).mean(0).sum()
        loss = loss + ixz * beta1
    loss.backward()
    optimizer.step()
    return loss

def train_model_ml(model, data, optimizer, beta1=None, batch_size=-1):
    """Train the model for one epoch."""
    model.train()
    train_size = data.train_pos_edges.size(0)
    if batch_size == -1:
        batch_size = train_size
    batch_start = 0
    batch_end = batch_size
    while batch_start < train_size:
        if batch_end > train_size:
            batch_end = train_size

        optimizer.zero_grad()
        rels = data.train_pos_edges[batch_start:batch_end, 1]

        output, reg_info = model(data, data.train_pos_edges[batch_start:batch_end])
        loss = F.cross_entropy(output, rels)
        
        # Add IB loss:
        if beta1 is not None and beta1 != 0:
            ixz = torch.stack(reg_info["ixz_list"], 1).mean(0).sum()
#            if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
#                ixz = ixz + torch.stack(reg_info["ixz_DN_list"], 1).mean(0).sum()
            loss = loss + ixz * beta1
       
        loss.backward()
        optimizer.step()

        batch_start += batch_size
        batch_end += batch_size
    return loss

@torch.no_grad()
def test_model_lp(
    model,
    data,
):
    metrics_all_m = {}
    metrics_all_f = {}
    model.eval()
    
    x, val_pos_edge_index, all_edge_index = data.x, data.val_pos_edge_index, data.edge_index

    _edge_index, _ = remove_self_loops(all_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                    num_nodes=x.size(0))
    # negative_sampling in all edges
    data['val_neg_edge_index'] = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=val_pos_edge_index.size(1))
    val_split_nodes = data['val_pos_edge_index'][0, :]
    val_split_nodes2 = data['val_neg_edge_index'][0, :]
    val_gender_nodes = np.arange(val_split_nodes.size(0) + val_split_nodes2.size(0))
    for i in range(val_split_nodes.size(0)):
            if data["age"][val_split_nodes[i]][0] == 0:
                val_gender_nodes[i] = 0
            else:
                val_gender_nodes[i] = 1
    for i in range(val_split_nodes.size(0), val_split_nodes.size(0)+val_split_nodes2.size(0)):
            if data["age"][val_split_nodes2[i-val_split_nodes.size(0)]][0] == 0:
                val_gender_nodes[i] = 0
            else:
                val_gender_nodes[i] = 1
    val_proba = torch.sigmoid(model(data, data.val_pos_edge_index, data.val_neg_edge_index)[0]).detach().cpu().numpy()
    val_labels = get_link_labels(data.val_pos_edge_index, data.val_neg_edge_index).detach().cpu().numpy()

    val_proba_m = val_proba[np.where(val_gender_nodes==0)]
    val_proba_f = val_proba[np.where(val_gender_nodes==1)]
    val_labels_m = val_labels[np.where(val_gender_nodes==0)]
    val_labels_f = val_labels[np.where(val_gender_nodes==1)]
    print(val_labels_m)
    print(val_labels_f)
    metrics_all_m['val_auc'] = sklearn.metrics.roc_auc_score(val_labels_m, val_proba_m)
    metrics_all_f['val_auc'] = sklearn.metrics.roc_auc_score(val_labels_f, val_proba_f)

    val_pred = val_proba
    val_pred[val_pred>0.5] = 1
    val_pred[val_pred<=0.5] = 0
    val_pred_m = val_pred[np.where(val_gender_nodes==0)]
    val_pred_f = val_pred[np.where(val_gender_nodes==1)]
    metrics_all_m['val_f1'] = sklearn.metrics.f1_score(val_labels_m, val_pred_m)
    metrics_all_f['val_f1'] = sklearn.metrics.f1_score(val_labels_f, val_pred_f)
    metrics_all_m['val_f1_micro'] = sklearn.metrics.f1_score(val_labels_m, val_pred_m, average='micro')
    metrics_all_f['val_f1_micro'] = sklearn.metrics.f1_score(val_labels_f, val_pred_f, average='micro')
    metrics_all_m['val_f1_macro'] = sklearn.metrics.f1_score(val_labels_m, val_pred_m, average='macro')
    metrics_all_f['val_f1_macro'] = sklearn.metrics.f1_score(val_labels_f, val_pred_f, average='macro')
    metrics_all_m['val_f1_weighted'] = sklearn.metrics.f1_score(val_labels_m, val_pred_m, average='weighted')
    metrics_all_f['val_f1_weighted'] = sklearn.metrics.f1_score(val_labels_f, val_pred_f, average='weighted')

    x, test_pos_edge_index, all_edge_index = data.x, data.test_pos_edge_index, data.edge_index

    _edge_index, _ = remove_self_loops(all_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                    num_nodes=x.size(0))
    # negative_sampling in all edges
    data['test_neg_edge_index'] = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=test_pos_edge_index.size(1))

    test_split_nodes = data['test_pos_edge_index'][0, :]
    test_split_nodes2 = data['test_neg_edge_index'][0, :]
    test_gender_nodes = np.arange(test_split_nodes.size(0)+test_split_nodes2.size(0))
    for i in range(test_split_nodes.size(0)):
            if data["age"][test_split_nodes[i]][0] == 0:
                test_gender_nodes[i] = 0
            else:
                test_gender_nodes[i] = 1
    for i in range(test_split_nodes.size(0),test_split_nodes.size(0)+test_split_nodes2.size(0)):
            if data["age"][test_split_nodes2[i-test_split_nodes.size(0)]][0] == 0:
                test_gender_nodes[i] = 0
            else:
                test_gender_nodes[i] = 1

    test_proba = torch.sigmoid(model(data, data.test_pos_edge_index, data.test_neg_edge_index)[0]).detach().cpu().numpy()
    test_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index).detach().cpu().numpy()
    test_proba_m = test_proba[np.where(test_gender_nodes==0)]
    test_proba_f = test_proba[np.where(test_gender_nodes==1)]
    test_labels_m = test_labels[np.where(test_gender_nodes==0)]
    test_labels_f = test_labels[np.where(test_gender_nodes==1)]
    metrics_all_m['test_auc'] = sklearn.metrics.roc_auc_score(test_labels_m, test_proba_m)
    metrics_all_f['test_auc'] = sklearn.metrics.roc_auc_score(test_labels_f, test_proba_f)
    
    print(test_labels_m)
    print(test_labels_f)

    test_pred = test_proba
    test_pred[test_pred>0.5] = 1
    test_pred[test_pred<=0.5] = 0

    test_pred_m = test_pred[np.where(test_gender_nodes==0)]
    test_pred_f = test_pred[np.where(test_gender_nodes==1)]
    metrics_all_m['test_f1'] = sklearn.metrics.f1_score(test_labels_m, test_pred_m)
    metrics_all_f['test_f1'] = sklearn.metrics.f1_score(test_labels_f, test_pred_f)
    metrics_all_m['test_f1_micro'] = sklearn.metrics.f1_score(test_labels_m, test_pred_m, average='micro')
    metrics_all_f['test_f1_micro'] = sklearn.metrics.f1_score(test_labels_f, test_pred_f, average='micro')
    metrics_all_m['test_f1_macro'] = sklearn.metrics.f1_score(test_labels_m, test_pred_m, average='macro')
    metrics_all_f['test_f1_macro'] = sklearn.metrics.f1_score(test_labels_f, test_pred_f, average='macro')
    metrics_all_m['test_f1_weighted'] = sklearn.metrics.f1_score(test_labels_m, test_pred_m, average='weighted')
    metrics_all_f['test_f1_weighted'] = sklearn.metrics.f1_score(test_labels_f, test_pred_f, average='weighted')

    return metrics_all_m, metrics_all_f

@torch.no_grad()
def test_model_nc(
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

@torch.no_grad()
def test_embs_lp(
    embs,
    data,
    is_private
):
    metrics_all = {}
    
    # decoder
    if is_private:
        embs = torch.cat([embs,data.s],dim=1)
    total_edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=-1)
    x_j = torch.index_select(embs, 0, total_edge_index[0])
    x_i = torch.index_select(embs, 0, total_edge_index[1])
    res = torch.einsum("ef,ef->e", x_i, x_j)
    val_proba = torch.sigmoid(res).detach().cpu().numpy()
    val_labels = get_link_labels(data.val_pos_edge_index, data.val_neg_edge_index).detach().cpu().numpy()
    metrics_all['val_auc'] = sklearn.metrics.roc_auc_score(val_labels, val_proba)
    
    val_pred = val_proba
    val_pred[val_pred>0.5] = 1
    val_pred[val_pred<=0.5] = 0

    metrics_all['val_f1'] = sklearn.metrics.f1_score(val_labels, val_pred)
    metrics_all['val_f1_micro'] = sklearn.metrics.f1_score(val_labels, val_pred, average='micro')
    metrics_all['val_f1_macro'] = sklearn.metrics.f1_score(val_labels, val_pred , average='macro')
    metrics_all['val_f1_weighted'] = sklearn.metrics.f1_score(val_labels, val_pred, average='weighted')

    # decoder
    if is_private:
        embs = torch.cat([embs,data.s],dim=1)
    total_edge_index = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1)
    x_j = torch.index_select(embs, 0, total_edge_index[0])
    x_i = torch.index_select(embs, 0, total_edge_index[1])
    res = torch.einsum("ef,ef->e", x_i, x_j)
    test_proba = torch.sigmoid(res).detach().cpu().numpy()
    test_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index).detach().cpu().numpy()
    metrics_all['test_auc'] = sklearn.metrics.roc_auc_score(test_labels, test_proba)
    
    test_pred = test_proba
    test_pred[test_pred>0.5] = 1
    test_pred[test_pred<=0.5] = 0

    metrics_all['test_f1'] = sklearn.metrics.f1_score(test_labels, test_pred)
    metrics_all['test_f1_micro'] = sklearn.metrics.f1_score(test_labels, test_pred, average='micro')
    metrics_all['test_f1_macro'] = sklearn.metrics.f1_score(test_labels, test_pred , average='macro')
    metrics_all['test_f1_weighted'] = sklearn.metrics.f1_score(test_labels, test_pred, average='weighted')

    return metrics_all


def test_model_ml(
    model,
    data,
):
    metrics_all = {}
    model.eval()
    for split in ['val','test']:
        outputs,_ = model(data, data[f'{split}_pos_edges'])

        proba_torch = torch.softmax(outputs, dim=1)
        proba = proba_torch.detach().cpu().numpy()

        labels_torch = data[f'{split}_pos_edges'][:, 1]
        labels = labels_torch.detach().cpu().numpy()
        # print(labels)
        print(proba)
        metrics_all[f'{split}_auc_macro'] = sklearn.metrics.roc_auc_score(labels, proba, average = "macro", multi_class = "ovr")
        metrics_all[f'{split}_auc_weighted'] = sklearn.metrics.roc_auc_score(labels, proba, average = "weighted", multi_class = "ovr")

        pred_labels = proba.argmax(1)

        metrics_all[f'{split}_f1_micro'] = sklearn.metrics.f1_score(labels, pred_labels, average='micro')
        metrics_all[f'{split}_f1_macro'] = sklearn.metrics.f1_score(labels, pred_labels , average='macro')
        metrics_all[f'{split}_f1_weighted'] = sklearn.metrics.f1_score(labels, pred_labels, average='weighted')

        pred_ratings = 0
        for j in range(0, 5):
            index = (torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            pred_ratings += (j + 1) * torch.index_select(proba_torch, 1, index)

        metrics_all[f'{split}_rmse'] = torch.sqrt(F.mse_loss(pred_ratings.view(-1), labels_torch + 1)).item()
        print(pred_ratings.view(-1))
        print(labels_torch + 1)

    return metrics_all

def train_VGPF(
    model,
    data,
    data_type,
    model_type,
    beta1_list,
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
        beta1 = beta1_list[epoch] if beta1_list is not None else None
        time1 = time.time()
        loss = train_model_nc(
            model,
            data,
            optimizer,
            beta1=beta1,
        )
        mean_time.append(time.time() - time1)
        metrics = test_model_nc(model, data)
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
            # if beta1 is not None:
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

def eval_VGPF_embs(
    embs,
    data,
    is_private
):
    
    best_metrics = {}
    best_metrics["val_f1"] = 0
    best_metrics["val_auc"] = 0
    best_metrics["test_auc"] = 0
    best_metrics["test_f1"] = 0
    best_metrics["test_acc"] = 0
    best_metrics["test_f1_micro"] = 0
    best_metrics["test_f1_macro"] = 0
    best_metrics["test_f1_weighted"] = 0
    
    
    metrics = test_embs_lp(embs, data, is_private)
    for key in best_metrics.keys():
        best_metrics[key] = metrics[key]

    return best_metrics



def train_VGPF_ml(
    model,
    data,
    data_type,
    model_type,
    beta1_list,
    epochs,
    verbose=True,
    inspect_interval=10,
    isplot=True,
    compute_metrics=None, # "Silu", "DB", "CH"
    lr=None,
    weight_decay=None,
    save_best_model=False,
    batch_size=-1,
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

    best_metrics["val_rmse"] = np.inf
    best_metrics["test_rmse"] = 0
    best_metrics["test_auc_macro"] = 0
    best_metrics["test_auc_weighted"] = 0
    best_metrics["test_f1_micro"] = 0
    best_metrics["test_f1_macro"] = 0
    best_metrics["test_f1_weighted"] = 0
    
    #data_record = {"num_layers": model.num_layers}

    # Train:
    mean_time = []
    all_metrics = []
    train_iter = tqdm(range(1, epochs + 1),dynamic_ncols=True)
    for epoch in train_iter:
        beta1 = beta1_list[epoch] if beta1_list is not None else None
        time1 = time.time()
        loss = train_model_ml(
            model,
            data,
            optimizer,
            beta1=beta1,
            batch_size=batch_size
        )
        mean_time.append(time.time() - time1)
        metrics = test_model_ml(model, data)
        all_metrics.append(metrics)
        if metrics["val_rmse"] < best_metrics["val_rmse"]:
            best_embeddings = model.encode(data)[0].detach().clone()
            for key in best_metrics.keys():
                best_metrics[key] = metrics[key]
            #if save_best_model:
            #    data_record["best_model_dict"] = deepcopy(model.model_dict)

        #record_data(data_record, list(metrics.values()), list(metrics.keys()))
        train_iter.set_postfix({'loss': loss.item(), 
            'ctr': metrics["test_rmse"],
            'ctam': metrics["test_auc_macro"],
            'cvr': metrics["val_rmse"],
            'btr': best_metrics["test_rmse"],
            'btam': best_metrics["test_auc_macro"],
            })
            # #record_data(data_record, [epoch], ["inspect_epoch"])
            # log = 'Epoch: {:03d}:'.format(epoch) + '\tF1 micro: ({:.4f}, {:.4f}, {:.4f})'.format(metrics["train_f1_micro"], best_val_f1, b_test_f1_micro)
            # log += compose_log(metrics, "f1_macro", 2)
            # log += compose_log(metrics, "acc", tabs=2, newline=True) + compose_log(metrics, "loss", 7)
            # if beta1 is not None:
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
    return best_metrics, all_metrics, best_embeddings

def get_data(
    data_type,
    train_fraction=1,
    added_edge_fraction=0,
    feature_noise_ratio=0,
    **kwargs):
    """Get the pytorch-geometric data object.
    
    Args:
        data_type: Data type. Choose from "Cora", "Pubmed", "citeseer". If want the feature to be binarized, include "-bool" in data_type string.
                   if want to use largest connected components, include "-lcc" in data_type. If use random splitting with train:val:test=0.1:0.1:0.8,
                   include "-rand" in the data_type string.
        train_fraction: Fraction of training labels preserved for the training set.
        added_edge_fraction: Fraction of added (or deleted) random edges. Use positive (negative) number for randomly adding (deleting) edges.
        feature_noise_ratio: Noise ratio for the additive independent Gaussian noise on the features.

    Returns:
        A pytorch-geometric data object containing the specified dataset.
    """
    def to_mask(idx, size):
        mask = torch.zeros(size).bool()
        mask[idx] = True
        return mask
    path = osp.join(osp.dirname(osp.realpath("__file__")), 'data', data_type)
    # Obtain the mode if given:
    data_type_split = data_type.split("-")
    
    data_type_full = data_type
    data_type = data_type_split[0]
    mode = "lcc" if "lcc" in data_type_split else None
    boolean = True if "bool" in data_type_split else False
    split = "rand" if "rand" in data_type_split else None
    
    # Load data:
    info = {}
    if data_type in ["Cora", "Pubmed", "citeseer"]:
        dataset = Planetoid(path, data_type, transform=T.NormalizeFeatures())
        data = dataset[0]
        info["num_features"] = dataset.num_features
        info["num_classes"] = dataset.num_classes
        info['loss'] = 'softmax'
    else:
        raise Exception("data_type {} is not valid!".format(data_type))

    # Process the dataset according to the mode given:
    if mode is not None:
        if mode == "lcc":
            data = get_data_lcc(dataset.data)
        else:
            raise

    if boolean:
        data.x = data.x.bool().float()
    
    if split == "rand":
        unlabeled_share = 0.8
        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share

        split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(data.x.shape[0]),
                                                                               train_size=train_share,
                                                                               val_size=val_share,
                                                                               test_size=unlabeled_share,
                                                                               stratify=to_np_array(data.y),
                                                                               random_state=kwargs["seed"] if "seed" in kwargs else None,
                                                                              )
        data.train_mask = to_mask(split_train, data.x.shape[0])
        data.val_mask = to_mask(split_val, data.x.shape[0])
        data.test_mask = to_mask(split_unlabeled, data.x.shape[0])

    # Reduce the number of training examples by randomly choosing some of the original training examples:
    if train_fraction != 1:
        try:
            train_mask_file = "../attack_data/{}/train_mask_tr_{}_seed_{}.p".format(data_type_full, train_fraction, kwargs["seed"] % 10)
            new_train_mask = pickle.load(open(train_mask_file, "rb"))
            data.train_mask = torch.BoolTensor(new_train_mask).to(data.y.device)
            print("Load train_mask at {}".format(train_mask_file))
        except:
            raise
            ids_chosen = []
            n_per_class = int(to_np_array(data.train_mask.sum()) * train_fraction / info["num_classes"])
            train_ids = torch.where(data.train_mask)[0]
            for i in range(info["num_classes"]):
                class_id_train = to_np_array(torch.where(((data.y == i) & data.train_mask))[0])
                ids_chosen = ids_chosen + np.random.choice(class_id_train, size=n_per_class, replace=False).tolist()
            new_train_mask = torch.zeros(data.train_mask.shape[0]).bool().to(data.y.device)
            new_train_mask[ids_chosen] = True
            data.train_mask = new_train_mask
            make_dir("../attack_data/{}/".format(data_type_full))
            pickle.dump(to_np_array(new_train_mask), open("../attack_data/{}/train_mask_tr_{}_seed_{}.p".format(data_type_full, train_fraction, kwargs["seed"] % 10), "wb"))

    # Add random edges for untargeted attacks:
    if added_edge_fraction > 0:
        data = add_random_edge(data, added_edge_fraction=added_edge_fraction)
    elif added_edge_fraction < 0:
        data = remove_edge_random(data, remove_edge_fraction=-added_edge_fraction)

    # Perturb features for untargeted attacks:
    if feature_noise_ratio > 0:
        x_max_mean = data.x.max(1)[0].mean()
        data.x = data.x + torch.randn(data.x.shape) * x_max_mean * feature_noise_ratio

    # For adversarial attacks:
    data.data_type = data_type
    if "attacked_nodes" in kwargs:
        attack_path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'attack_data', data_type_full) 
        if not os.path.exists(attack_path):
            os.makedirs(attack_path)
        try:
            with open(os.path.join(attack_path, "test-node.pkl"), 'rb') as f:
                node_ids = pickle.load(f)
                info['node_ids'] = node_ids
                print("Load previous attacked node_ids saved in {}.".format(attack_path))
        except:
            test_ids = to_np_array(torch.where(data.test_mask)[0])
            node_ids = get_list_elements(test_ids, kwargs['attacked_nodes'])
            with open(os.path.join(attack_path, "test-node.pkl"), 'wb') as f:
                pickle.dump(node_ids, f)
            info['node_ids'] = node_ids
            print("Save attacked node_ids into {}.".format(attack_path))
    return data, info


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


def get_data_lcc(data):
    """Return a new data object consisting of the largest connected component."""
    data_lcc = deepcopy(data)
    edge_index_sparse = edge_index_2_csr(data.edge_index, data.num_nodes)
    lcc = largest_connected_components(edge_index_sparse)
    edge_index_lcc_sparse = edge_index_sparse[lcc][:, lcc].tocoo()
    data_lcc.edge_index = torch.stack(list(to_Variable(edge_index_lcc_sparse.row, edge_index_lcc_sparse.col))).long()

    data_lcc.x = data.x[lcc]
    data_lcc.y = data.y[lcc]
    data_lcc.train_mask = data.train_mask[lcc]
    data_lcc.val_mask = data.val_mask[lcc]
    data_lcc.test_mask = data.test_mask[lcc]
    return data_lcc

