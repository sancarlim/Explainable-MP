from fractions import gcd
import torch
from torch import nn
import torch.nn.functional as F
import math
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax


# Conv layer with norm (gn or bn) and relu.
class Conv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride,
                              bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride,
                              bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


# Post residual layer
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, norm='GN', ng=32, act=True):
        super(PostRes, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm2d(n_out)
            self.bn2 = nn.BatchNorm2d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


def linear_interp(x, n_max):
    """Given a Tensor of normed positions, returns linear interplotion weights and indices.
    Example: For position 1.2, its neighboring pixels have indices 0 and 1, corresponding
    to coordinates 0.5 and 1.5 (center of the pixel), and linear weights are 0.3 and 0.7.

    Args:
        x: Normalizzed positions, ranges from 0 to 1, float Tensor.
        n_max: Size of the dimension (pixels), multiply x to get absolution positions.
    Returns: Weights and indices of left side and right side.
    """
    x = x * n_max - 0.5

    mask = x < 0
    x[mask] = 0
    mask = x > n_max - 1
    x[mask] = n_max - 1
    n = torch.floor(x)

    rw = x - n
    lw = 1.0 - rw
    li = n.long()
    ri = li + 1
    mask = ri > n_max - 1
    ri[mask] = n_max - 1

    return lw, li, rw, ri


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None, final_linear=False):
        super(MLP, self).__init__()
        self.final = final_linear
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)
        if final_linear:
            self.final_linear = nn.Linear(out_features, out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.leaky_relu(hidden_states)
        if self.final:
            return self.final_linear(hidden_states)
        return hidden_states




class GlobalGraph(nn.Module):
    r"""
    Global graph

    It's actually a self-attention.
    """

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, attention_decay=False, visualize=False):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_decay_arg = attention_decay
        self.visualize = visualize
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        if attention_decay:
            self.attention_decay = nn.Parameter(torch.ones(1) * 0.5)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, mapping=None, return_scores=False):
        mixed_query_layer = self.query(hidden_states) # (batch, max_vector_num, hidden_dim)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight) 
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch, num heads, max_vector_num, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # print(attention_scores.shape, attention_mask.shape)
        if attention_mask is not None:
            # turn 1 to 0, 0 to -10000.0 in att_mask for softmax - a way to enhance the attention?
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask) 
        # if utils.args.attention_decay and utils.second_span:
        #     attention_scores[:, 0, 0, 0] = attention_scores[:, 0, 0, 0] - self.attention_decay
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # (batch, num heads, max_vector_num, max_vector_num)
        if self.visualize and mapping is not None:
            for i, each in enumerate(attention_probs.tolist()):
                mapping[i]['attention_scores'] = np.array(each[0]) 
        context_layer = torch.matmul(attention_probs, value_layer) # (batch, num heads, max_vector_num, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert attention_probs.shape[1] == 1
            attention_probs = torch.squeeze(attention_probs, dim=1)
            assert len(attention_probs.shape) == 3
            return context_layer, attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, key_hidden_size=None,
                 query_hidden_size=None):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None, mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] \
                   and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        # hidden_states = self.global_graph(hidden_states, attention_mask, mapping) \
        #                 + self.global_graph2(hidden_states, attention_mask, mapping)
        hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask),
                                   self.global_graph2(hidden_states, attention_mask)], dim=-1)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h, out_key):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                if ntype in out_key:
                    '''
                        Step 3: Target-specific Aggregation
                        x = norm( W[node_type] * gelu( Agg(x) ) + x )
                    '''
                    n_id = node_dict[ntype]
                    alpha = torch.sigmoid(self.skip[n_id])
                    t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                    trans_out = self.drop(self.a_linears[n_id](t))
                    trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                    if self.use_norm:
                        new_h[ntype] = self.norms[n_id](trans_out)
                    else:
                        new_h[ntype] = trans_out
                else:
                    new_h[ntype] = h[ntype]
            return new_h



class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class ieHGCNConv(nn.Module):
    r"""
    The ieHGCN convolution layer.
    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the feature drop rate
    activation: str
        the activation function
    """
    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation = F.elu):
        super(ieHGCNConv, self).__init__()
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] =  attn_size
        self.W_self = dglnn.HeteroLinear(node_size, out_size)
        self.W_al = dglnn.HeteroLinear(attn_vector, 1)
        self.W_ar = dglnn.HeteroLinear(attn_vector, 1)
        
        # self.conv = dglnn.HeteroGraphConv({
        #     etype: dglnn.GraphConv(in_size, out_size, norm = 'right', weight = True, bias = True)
        #     for etype in etypes
        # })
        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        mods = {
            #etype: dglnn.GraphConv(in_size, out_size, norm = 'right', 
            #                       weight = True, bias = True, allow_zero_in_degree = True)
            etype: dglnn.GATv2Conv(in_size, out_size, num_heads = 1, feat_drop = 0.0, attn_drop = 0.0, 
                                   bias = True, allow_zero_in_degree = True)
            for etype in etypes
            }
        self.mods = nn.ModuleDict(mods)
        
        self.linear_q = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        
        self.activation = activation
        

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            # formulas (2)-1
            hg.ndata['z'] = self.W_self(hg.ndata['h'])
            query = {}
            key = {}
            attn = {}
            attention = {}
            edge_att = {}
            
            # formulas (3)-1 and (3)-2
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](hg.ndata['z'][ntype])
                key[ntype] = self.linear_k[ntype](hg.ndata['z'][ntype])
            # formulas (4)-1
            h_l = self.W_al(key)
            h_r = self.W_ar(query)
            for ntype in hg.dsttypes:
                attention[ntype] = F.elu(h_l[ntype] + h_r[ntype])
                attention[ntype] = attention[ntype].unsqueeze(0)
            
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                # formulas (2)-2
                # Retrieve edge attention from object-level aggregation
                dstdata, e_att = self.mods[etype](
                rel_graph,
                (h_dict[srctype], h_dict[dsttype]), 
                att
                )
                edge_att[etype] = e_att.squeeze()
                dstdata = dstdata.squeeze(-2)
                
                outputs[dsttype].append(dstdata)
                # formulas (3)-3
                attn[dsttype] = self.linear_k[dsttype](dstdata)
                # formulas (4)-2
                h_attn = self.W_al(attn)
                attn.clear()
                edge_attention = F.elu(h_attn[dsttype] + h_r[dsttype])
                attention[dsttype] = torch.cat((attention[dsttype], edge_attention.unsqueeze(0)))

            # formulas (5)
            for ntype in hg.dsttypes:
                attention[ntype] = F.softmax(attention[ntype], dim = 0)

            # formulas (6)
            rst = {ntype: 0 for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                data = [hg.ndata['z'][ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]
                
            # h = self.conv(hg, hg.ndata['h'], aggregate = self.my_agg_func)
        if self.activation is not None:
            for ntype in rst.keys():
                rst[ntype] = self.activation(rst[ntype])
            
        return rst, [attention, edge_att]
