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
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels,
                                                              n_components=n_components)

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
        x = x.float()
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
                Z_logit = self.dist.log_prob(Z).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(
                    Z)  # [S, B]
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
        num_classifier=2,
    ):
        """Class implementing a general GNN.
        
        Args:
            model_type:   name of the base model. Choose from "GCN", "SAGE", "Cheb", "GIN".
            num_features: number of features of the data.x.
            num_classes:  number of classes for data.y.
            reparam_mode: reparameterization mode for XIB. Choose from "diag" and "full". Default "diag" that parameterizes the mean and diagonal element of the Gaussian
            prior_mode:   distribution type for the prior. Choose from "Gaussian" or "mixGau-{Number}", where {Number} is the number of components for mixture of Gaussian.
            latent_size:  latent size for each layer of GNN.
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
        self.num_classifier=num_classifier
        self.init()


    def init(self):
        """Initialize the layers for the GNN."""
        self.reparam_layers = []
        if self.model_type in ["GCN","SAGE","Cheb","GIN"]:
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
        
        if self.num_classifier == 1:
            self.classifier = torch.nn.Linear(self.latent_size+2, self.num_classes).cuda()
        else:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(self.latent_size + 2, 32),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Linear(32, 32),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Linear(32, self.num_classes)).cuda()

        # self.classifier = torch.nn.Linear(self.latent_size , self.num_classes).cuda()

        if self.model_type in ["GCN","SAGE","Cheb","GIN"]:
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
        if self.model_type in ["GCN","SAGE","Cheb","GIN"]:
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

    metrics_all['sp'] = 0
    metrics_all['eo'] = 0
    metrics_all['cf'] = 0
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


    metrics_all['sp'] = sparity
    metrics_all['eo'] = equality

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
    model_type,
    beta_list,
    epochs,
    verbose=True,
    inspect_interval=10,
    lr=None,
    weight_decay=None,
    save_best_model=False,
):
    """Training multiple epochs."""
    if lr is None:
        if model_type == "GCN":
            lr = 0.01
        else:
            lr = 0.01
    if weight_decay is None:
        if model_type == "GCN":
            weight_decay = 5e-4
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
    best_metrics['sp'] = 0
    best_metrics['eo'] = 0
    best_metrics['cf'] = 0
    best_metrics['robustness'] = 0

    # data_record = {"num_layers": model.num_layers}

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
            # if save_best_model:
            #    data_record["best_model_dict"] = deepcopy(model.model_dict)

        # record_data(data_record, list(metrics.values()), list(metrics.keys()))
        if verbose and epoch % inspect_interval == 0:
            print(f"Inspect Epoch:{epoch} \nCurrent Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v} ", end=" ")
            print(f"\nloss: {loss}")    
            print(f"\nBest Metrics so far:")
            for k, v in best_metrics.items():
                print(f"{k}: {v} ", end=" ")

    best_metrics["time"] = sum(mean_time) / len(mean_time)
    return best_metrics, best_embeddings


