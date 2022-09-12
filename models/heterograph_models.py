from models.layers import HeteroRGCNLayer, HGTLayer, ieHGCNConv
from abc import ABCMeta
import dgl
import torch.nn as nn
import torch.nn.functional as F 


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
                h = self.gcs[i](G, h)
            return self.out(h[out_key])


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
                h_dict = self.hgcn_layers[l](hg, h_dict)
            
            return h_dict['l']


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