from models.layers import HeteroRGCNLayer, HGTLayer, ieHGCNConv
from abc import ABCMeta
import dgl
from dgl.ops import edge_softmax
from dgl.nn import TypedLinear

import dgl.function as Fn
import torch.nn as nn
import torch.nn.functional as F 
import torch


class BaseModel(nn.Module, metaclass=ABCMeta):
    @classmethod
    def build_model_from_args(cls, args, hg):
        r"""
        Build the model instance from args and hg.
        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        r"""
        The model plays a role of encoder. So the forward will encoder original features into new features.
        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict[str, th.Tensor]
            the dict of heterogeneous feature
        Return
        -------
        out_dic : dict[str, th.Tensor]
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError

    def extra_loss(self):
        r"""
        Some model want to use L2Norm which is not applied all parameters.
        Returns
        -------
        th.Tensor
        """
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        r"""
        Return the embedding of a model for further analysis.
        Returns
        -------
        numpy.array
        """
        raise 

class HGT(nn.Module):
    def __init__(self, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for i in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads[i], use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        with G.local_scope():
            h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
            for i in range(self.n_layers):
                h = self.gcs[i](G, h, out_key)
            for key in out_key:
                h[key] = self.out(h[key])
            return [h[key] for key in out_key]

class SimpleHGN(BaseModel):
    r"""
    This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__
    The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.
    Calculating the coefficient:
    
    .. math::
        \alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\psi(<i,j>)}]))}{\Sigma_{k\in\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\psi(<i,k>)}]))}} \quad (1)
    
    Residual connection including Node residual:
    
    .. math::
        h_i^{(l)} = \sigma(\Sigma_{j\in \mathcal{N}_i} {\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)}) \quad (2)
    
    and Edge residual:
        
    .. math::
        \alpha_{ij}^{(l)} = (1-\beta)\alpha_{ij}^{(l)}+\beta\alpha_{ij}^{(l-1)} \quad (3)
        
    Multi-heads:
    
    .. math::
        h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (4)
    
    Residual:
    
        .. math::
            h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (5)
    
    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    hidden_dim: int
        the output dimension
    num_classes: int
        the number of the output classes
    num_layers: int
        the number of layers we used in the computing
    heads: list
        the list of the number of heads in each layer
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    beta: float
        the hyperparameter used in edge residual
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(args.edge_dim,
                   len(hg.etypes),
                   [args.hidden_dim],
                   args.h_dim,
                   args.out_dim,
                   args.num_layers,
                   heads,
                   args.feats_drop_rate,
                   args.slope,
                   True,
                   args.beta
                   )

    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                num_layers, heads, feat_drop=0.0, negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGN, self).__init__()
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim,
                hidden_dim,
                heads[0],
                num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )
        # output projection
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                hidden_dim * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the SimpleHGN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata = 'h')
            h = g.ndata['h']
            for l in range(self.num_layers):  
                h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                h = h.flatten(1)

        h_dict = {}
        for index, ntype in enumerate(hg.ntypes):
            h_dict[ntype] = h[torch.where(g.ndata['_TYPE'] == index)]
        # g.ndata['h'] = h
        # hg = dgl.to_heterogeneous(g, hg.ntypes, hg.etypes)
        # h_dict = hg.ndata['h']

        return h_dict['l'], h_dict['v']

class SimpleHGNConv(nn.Module):
    r"""
    The SimpleHGN convolution layer.
    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """
    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted = False):
        """
        The forward part of the SimpleHGNConv.
        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1,
                                                       self.num_heads, self.edge_dim)

        row = g.edges()[0]
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row.to(torch.long)]
        h_r = (self.a_r * emb).sum(dim=-1)[col.to(torch.long)]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * \
                             (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                         Fn.sum('m', 'emb'))
            # g.apply_edges(Fn.u_mul_e('emb', 'alpha', 'm'))
            h_output = g.ndata['emb'].view(-1, self.out_dim * self.num_heads)
            # h_prime = []
            # for i in range(self.num_heads):
            #     g.edata['alpha'] = edge_attention[:, i]
            #     g.srcdata.update({'emb': emb[i]})
            #     g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
            #                  Fn.sum('m', 'emb'))
            #     h_prime.append(g.ndata['emb'])
            # h_output = torch.cat(h_prime, dim=1)

        g.edata['alpha'] = edge_attention
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output



class ieHGCN(BaseModel):
    r"""
    ie-HGCN from paper `Interpretable and Efficient Heterogeneous Graph Convolutional Network
    <https://arxiv.org/pdf/2005.13183.pdf>`__.
    `Source Code Link <https://github.com/kepsail/ie-HGCN>`_
    
    The core part of ie-HGCN, the calculating flow of projection, object-level aggregation and type-level aggregation in
    a specific type block.
    Projection
    
    .. math::
        Y^{Self-\Omega }=H^{\Omega} \cdot W^{Self-\Omega} \quad (1)-1
        Y^{\Gamma - \Omega}=H^{\Gamma} \cdot W^{\Gamma - \Omega} , \Gamma \in N_{\Omega} \quad (1)-2
    Object-level Aggregation
    
    .. math::
        Z^{ Self - \Omega } = Y^{ Self - \Omega}=H^{\Omega} \cdot W^{Self - \Omega} \quad (2)-1
        Z^{\Gamma - \Omega}=\hat{A}^{\Omega-\Gamma} \cdot Y^{\Gamma - \Omega} = \hat{A}^{\Omega-\Gamma} \cdot H^{\Gamma} \cdot W^{\Gamma - \Omega} \quad (2)-2
    Type-level Aggregation
    
    .. math::
        Q^{\Omega}=Z^{Self-\Omega} \cdot W_q^{\Omega} \quad (3)-1
        K^{Self-\Omega}=Z^{Self -\Omega} \cdot W_{k}^{\Omega} \quad (3)-2
        K^{\Gamma - \Omega}=Z^{\Gamma - \Omega} \cdot W_{k}^{\Omega}, \quad \Gamma \in N_{\Omega} \quad (3)-3
    .. math::
        e^{Self-\Omega}={ELU} ([K^{ Self-\Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}) \quad (4)-1
        e^{\Gamma - \Omega}={ELU} ([K^{\Gamma - \Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}), \Gamma \in N_{\Omega} \quad (4)-2
    .. math::
        [a^{Self-\Omega}\|a^{1 - \Omega}\| \ldots . a^{\Gamma - \Omega}\|\ldots\| a^{|N_{\Omega}| - \Omega}] \\
        = {softmax}([e^{Self - \Omega}\|e^{1 - \Omega}\| \ldots\|e^{\Gamma - \Omega}\| \ldots \| e^{|N_{\Omega}| - \Omega}]) \quad (5)
    .. math::
        H_{i,:}^{\Omega \prime}=\sigma(a_{i}^{Self-\Omega} \cdot Z_{i,:}^{Self-\Omega}+\sum_{\Gamma \in N_{\Omega}} a_{i}^{\Gamma - \Omega} \cdot Z_{i,:}^{\Gamma - \Omega}) \quad (6)
    
    Parameters
    ----------
    num_layers: int
        the number of layers
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    out_dim: int
        the output dimension
    attn_dim: int
        the dimension of attention vector
    ntypes: list
        the node type of a heterogeneous graph
    etypes: list
        the edge type of a heterogeneous graph
    """
    @classmethod
    def build_model_from_args(cls, args, hg:dgl.DGLGraph):
        return cls(args.num_layers,
                   args.in_dim,
                   args.hidden_dim,
                   args.out_dim,
                   args.attn_dim,
                   hg.ntypes,
                   hg.etypes
                   )

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, attn_dim, ntypes, etypes):
        super(ieHGCN, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgcn_layers = nn.ModuleList()
        
        self.hgcn_layers.append(
            ieHGCNConv(
                in_dim,
                hidden_dim,
                attn_dim,
                ntypes,
                etypes,
                self.activation,
            )
        )

        for i in range(1, num_layers - 1):
            self.hgcn_layers.append(
                ieHGCNConv(
                    hidden_dim,
                    hidden_dim,
                    attn_dim,
                    ntypes,
                    etypes,
                    self.activation
                )
            )
        
        self.hgcn_layers.append(
            ieHGCNConv(
                hidden_dim,
                out_dim,
                attn_dim,
                ntypes,
                etypes,
                None,
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        with hg.local_scope(): 
            for l in range(self.num_layers):
                h_dict, attention = self.hgcn_layers[l](hg, h_dict)
            
            return h_dict['l'], h_dict['v'], attention


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G, out_key):
        input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        h_dict = self.layer1(G, input_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get appropriate logits
        return h_dict[out_key]       
