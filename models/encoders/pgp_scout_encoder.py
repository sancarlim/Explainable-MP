from code import interact
from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Dict, List
from torch import Tensor
from models.layers import Res1d, Conv1d, MLP, GlobalGraph, LayerNorm
import torch.nn.functional as F 
from dgl import DGLError
from dgl.nn import GATv2Conv 
from dgl import function as fn
from dgl.nn.pytorch.conv.gatconv import edge_softmax, Identity, expand_as_pair
import scipy.sparse as spp

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False):
        super(GATv2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias)
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.
        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class GATv2(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(GATv2Conv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=False, share_weights=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(GATv2Conv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=False, share_weights=True))
        # output projection
        """self.gatv2_layers.append(GATv2Conv(
            num_hidden * heads[-2], out_dim, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=False, share_weights=True))
    """
    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        # logits = self.gatv2_layers[-1](g, h).mean(1)
        return h


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, merge='cat',  feat_drop=0., attn_drop=0.):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append( GATv2Conv(in_feats, out_feats, feat_drop, attn_drop, residual=False, activation=F.elu) ) 
        self.merge = merge

    def forward(self, g, h, e_w):
        if isinstance(h, list):
            head_outs = [attn_head(g, h_mode, e_w) for attn_head, h_mode in zip(self.heads, h)]
        else:
            head_outs = [attn_head(g, h, e_w) for attn_head in self.heads]
            
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1), for intermediate layers
            return torch.cat(head_outs, dim=1)
        elif self.merge == 'list':
            return head_outs
        else:
            # merge using average, for final layer
            return torch.mean(torch.stack(head_outs, dim=1),dim=1)


class PGP_SCOUTEncoder(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        Lane node features and agent histories encoded using GRUs.
        Additionally, agent-node attention layers infuse each node encoding with nearby agent context.
        Finally GAT layers aggregate local context at each node.

        args to include:

        target_agent_feat_size: int Size of target agent features
        target_agent_emb_size: int Size of target agent embedding
        taret_agent_enc_size: int Size of hidden state of target agent GRU encoder

        node_feat_size: int Size of lane node features
        node_emb_size: int Size of lane node embedding
        node_enc_size: int Size of hidden state of lane node GRU encoder

        nbr_feat_size: int Size of neighboring agent features
        nbr_enb_size: int Size of neighboring agent embeddings
        nbr_enc_size: int Size of hidden state of neighboring agent GRU encoders

        num_gat_layers: int Number of GAT layers to use.
        """

        super().__init__()

        # Target agent encoder
        self.target_agent_emb = nn.Linear(args['target_agent_feat_size'], args['target_agent_emb_size'])
        self.target_agent_enc = nn.GRU(args['target_agent_emb_size'], args['target_agent_enc_size'], batch_first=True)

        # Node encoders
        self.node_emb = nn.Linear(args['node_feat_size']+1, args['node_emb_size'])
        self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)
        self.node_encoder = GATv2(args['num_layers'],args['node_emb_size']*20, args['node_enc_size'], args['node_enc_size'],
                                         heads=args['num_heads_lanes'], activation=F.leaky_relu, feat_drop=args['feat_drop'], attn_drop=args['attn_drop'], 
                                         residual=True, negative_slope=0.2) 

        # Surrounding agent encoder
        self.nbr_emb = nn.Linear(args['nbr_feat_size'] + 1, args['nbr_emb_size'])
        self.nbr_enc = nn.GRU(args['nbr_emb_size'], args['nbr_enc_size'], batch_first=True)
        
        # Agent interaction (agent||veh_nbr||ped_nbr -» agent_nbr_context) ! 
        self.interaction_net = GATv2Conv(args['nbr_enc_size'], args['nbr_enc_size'], 
                                         num_heads=args['num_heads_agents'], feat_drop=0., attn_drop=0., share_weights=True,
                                         residual=True, activation=F.elu, allow_zero_in_degree=True)  

        # Agent-node attention
        self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.key_emb = nn.Linear(args['nbr_enc_size']*2, args['nbr_enc_size']*2)
        self.val_emb = nn.Linear(args['nbr_enc_size']*2, args['nbr_enc_size']*2)
        self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Non-linearities
        self.leaky_relu = nn.LeakyReLU()

        # GAT layers
        self.gat = nn.ModuleList([GAT(args['node_enc_size'], args['node_enc_size'])
                                  for _ in range(args['num_heads'])])

    def forward(self, inputs: Dict) -> Dict:
        """
        Forward pass for PGP encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor, shape [batch_size, t_h, target_agent_feat_size]
            map_representation: Dict with
                'lane_node_feats': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]
                'lane_node_masks': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]

                (Optional)
                's_next': Edge look-up table pointing to destination node from source node
                'edge_type': Look-up table with edge type

            surrounding_agent_representation: Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'vehicle_masks': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'pedestrians': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
                'pedestrian_masks': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
            agent_node_masks:  Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_nodes, max_vehicles]
                'pedestrians': torch.Tensor, shape [batch_size, max_nodes, max_pedestrians]

            Optionally may also include the following if edges are defined for graph traversal
            'init_node': Initial node in the lane graph based on track history.
            'node_seq_gt': Ground truth node sequence for pre-training

        :return:
        """

        # Encode target agent
        target_agent_feats = inputs['target_agent_representation']
        target_agent_embedding = self.leaky_relu(self.target_agent_emb(target_agent_feats))
        _, target_agent_enc = self.target_agent_enc(target_agent_embedding)
        target_agent_enc = target_agent_enc.squeeze(0) #B,32

        # Encode surrounding agents
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        nbr_vehicle_embedding = self.leaky_relu(self.nbr_emb(nbr_vehicle_feats))
        nbr_vehicle_enc = self.variable_size_gru_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.nbr_enc) #B,84,32
        _, masks_for_batching_veh = self.create_batched_input(nbr_vehicle_feats, nbr_vehicle_masks)
        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']
        nbr_ped_embedding = self.leaky_relu(self.nbr_emb(nbr_ped_feats))
        nbr_ped_enc = self.variable_size_gru_encode(nbr_ped_embedding, nbr_ped_masks, self.nbr_enc)
        _, masks_for_batching_ped = self.create_batched_input(nbr_ped_feats, nbr_ped_masks)


        interaction_feats = torch.cat((target_agent_enc.unsqueeze(1),nbr_vehicle_enc, nbr_ped_enc), dim=1)
        target_masks = torch.ones((target_agent_enc.shape[0], 1, 1), device=target_agent_enc.device).bool()
        interaction_masks = torch.cat((target_masks, masks_for_batching_veh.squeeze(-1),masks_for_batching_ped.squeeze(-1)), dim=1).repeat(1,1,interaction_feats.shape[-1])
        interaction_feats_batched = torch.masked_select(interaction_feats, interaction_masks!=0) 
        interaction_feats_batched = interaction_feats_batched.view(-1, interaction_feats.shape[2]) # BN,32

        # Use mask for batching 
        """ 
        sum=0
        for i, len_nbr in enumerate(inputs['surrounding_agent_representation']['len_adj']):
            interaction_feats = torch.cat((target_agent_enc[i].unsqueeze(0), nbr_enc[sum:sum+len_nbr-1]), dim=0) 
            sum+=len_nbr-1 """
        
        # Agent interaction (agent||veh_nbr||ped_nbr [32,162,32]-» agent_nbr_context) ! 
        interaction_graph = inputs['interaction_graphs'].to(target_agent_feats.device)
        #dst = inputs['surrounding_agent_representation']['dst_nodes']
        #interaction_graph = dgl.graph((src, dst))  
        interaction_graph.create_formats_()
        #interaction_feats = torch.cat((target_agent_enc, nbr_vehicle_enc, nbr_ped_enc), dim=0) 
        agent_nbr_context = self.interaction_net(interaction_graph, interaction_feats_batched, None)
        # Concatenate outputs of the different heads
        agent_nbr_context = agent_nbr_context.view(agent_nbr_context.shape[0],-1)

        # Concatenate agents encodings and agent_nbr_context
        interaction_feats_batched = torch.cat((interaction_feats_batched, agent_nbr_context), dim=-1) # BN,32
        interaction_feats = self.scatter_batched_input(interaction_feats_batched, interaction_masks[:,:,-1:].unsqueeze(-1)) # B, N, 32

        # nbr_vehicle_enc = torch.cat((nbr_vehicle_enc, agent_nbr_context[1:1+nbr_vehicle_enc.shape[0]]), dim=-1)
        # nbr_ped_enc = torch.cat((nbr_ped_enc, agent_nbr_context[1+nbr_vehicle_enc.shape[0]:]), dim=-1) 

        # Encode lane nodes
        """lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks'] 
        lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats))
        lane_node_enc = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.node_encoder) # B,164,32
        """
        # Encode lane nodes 
        lanes_graphs = inputs['lanes_graphs'].to(target_agent_feats.device) 
        lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks'] 
        lane_node_feats = torch.cat((lane_node_feats, lane_node_masks[:,:,:,:1]), dim=-1)
        lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats)) 
        lane_node_embedding = lane_node_embedding.view(lane_node_embedding.shape[0], lane_node_embedding.shape[1], -1) # B,164,32
        lane_node_enc = self.node_encoder(lanes_graphs, lane_node_embedding) 
        lane_node_enc = lane_node_enc.view(lane_node_embedding.shape[0], lane_node_embedding.shape[1], -1) # B,164,32

        # Agent-node attention (between nbrs and lanes)  
        veh_interaction_feats = torch.cuda.FloatTensor(interaction_feats.shape[0],nbr_vehicle_feats.shape[1], interaction_feats.shape[-1])
        ped_interaction_feats = torch.cuda.FloatTensor(interaction_feats.shape[0],nbr_ped_feats.shape[1], interaction_feats.shape[-1]) 
        for i, batch in enumerate(interaction_feats): 
            veh_interaction_feats[i] = interaction_feats[i,1:nbr_vehicle_feats.shape[1]+1]
            ped_interaction_feats[i] = interaction_feats[i,-nbr_ped_feats.shape[1]:]
        queries = self.query_emb(lane_node_enc).permute(1, 0, 2)
        keys = self.key_emb(torch.cat((veh_interaction_feats, ped_interaction_feats), dim=1)).permute(1, 0, 2)
        vals = self.val_emb(torch.cat((veh_interaction_feats, ped_interaction_feats), dim=1)).permute(1, 0, 2)
        attn_masks = torch.cat((inputs['agent_node_masks']['vehicles'],
                                inputs['agent_node_masks']['pedestrians']), dim=2)
        att_op, _ = self.a_n_att(queries, keys, vals, attn_mask=attn_masks)
        att_op = att_op.permute(1, 0, 2)

        # Concatenate with original node encodings and 1x1 conv 
        lane_node_enc = self.leaky_relu(self.mix(torch.cat((lane_node_enc, att_op), dim=2)))

        # GAT layers
        # Build adj matrix - treat succ and prox edges as equivalent and bidirectional
        """adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
        for gat_layer in self.gat:
            lane_node_enc += gat_layer(lane_node_enc, adj_mat) """

        # Lane node masks
        lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
        lane_node_masks = lane_node_masks.any(dim=2)
        lane_node_masks = ~lane_node_masks
        lane_node_masks = lane_node_masks.float()

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc, #before interaction
                     'context_encoding': {'combined': lane_node_enc,
                                          'combined_masks': lane_node_masks,
                                          'interaction'
                                          'map': None,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': None,
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None
                                          },
                     }

        # Pass on initial nodes and edge structure to aggregator if included in inputs
        if 'init_node' in inputs:
            encodings['init_node'] = inputs['init_node']
            encodings['node_seq_gt'] = inputs['node_seq_gt']
            encodings['s_next'] = inputs['map_representation']['s_next']
            encodings['edge_type'] = inputs['map_representation']['edge_type']

        return encodings

    
    @staticmethod
    def create_batched_input(input: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Create a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """
        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool() # B, N,20,6 
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3) # B, N,1,1
        input_batched = torch.masked_select(input, masks_for_batching) 
        input_batched = input_batched.view(-1, input.shape[2], input.shape[3]) # BN,20,16
        
        return input_batched, masks_for_batching

    @staticmethod
    def variable_size_gru_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """
        Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool() # B, 164,20,6
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3) # 32, 164,1,1
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching) 
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3]) # 970,20,16

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1) # B, 164
        seq_lens_batched = seq_lens[seq_lens != 0].cpu() # Bx164 != 0 =» 970
        if len(seq_lens_batched) != 0:
            feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                         batch_first=True, enforce_sorted=False)

            # Encode
            _, encoding_batched = gru(feat_embedding_packed)
            encoding_batched = encoding_batched.squeeze(0)

            # Scatter back to appropriate batch index
            masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
            encoding = torch.zeros(masks_for_scattering.shape, device=device)
            encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = gru.hidden_size
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding

    
    @staticmethod
    def scatter_batched_input(batched_input: torch.Tensor, masks_for_batching: torch.Tensor) -> torch.Tensor:
        """
        Create a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """
        # Scatter back to appropriate batch index
        masks_for_scattering = masks_for_batching.squeeze(-1).repeat(1, 1, batched_input.shape[-1])
        input = torch.zeros(masks_for_scattering.shape, device=device)
        input = input.masked_scatter(masks_for_scattering, batched_input)

        return input

    @staticmethod
    def build_adj_mat(s_next, edge_type):
        """
        Builds adjacency matrix for GAT layers.
        """
        batch_size = s_next.shape[0]
        max_nodes = s_next.shape[1]
        max_edges = s_next.shape[2]
        adj_mat = torch.diag(torch.ones(max_nodes, device=device)).unsqueeze(0).repeat(batch_size, 1, 1).bool()

        dummy_vals = torch.arange(max_nodes, device=device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges) # B, 164, 15 - src indices
        dummy_vals = dummy_vals.float()
        s_next[edge_type == 0] = dummy_vals[edge_type == 0] # set to 0 for non-edges
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_edges)
        src_indices = torch.arange(max_nodes).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        adj_mat[batch_indices[:, :, :-1], src_indices[:, :, :-1], s_next[:, :, :-1].long()] = True
        adj_mat = adj_mat | torch.transpose(adj_mat, 1, 2) # bidirectional 

        return adj_mat


class GAT(nn.Module):
    """
    GAT layer for aggregating local context at each lane node. Uses scaled dot product attention using pytorch's
    multihead attention module.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT layer.
        :param in_channels: size of node encodings
        :param out_channels: size of aggregated node encodings
        """
        super().__init__()
        self.query_emb = nn.Linear(in_channels, out_channels)
        self.key_emb = nn.Linear(in_channels, out_channels)
        self.val_emb = nn.Linear(in_channels, out_channels)
        self.att = nn.MultiheadAttention(out_channels, 1)

    def forward(self, node_encodings, adj_mat):
        """
        Forward pass for GAT layer
        :param node_encodings: Tensor of node encodings, shape [batch_size, max_nodes, node_enc_size]
        :param adj_mat: Bool tensor, adjacency matrix for edges, shape [batch_size, max_nodes, max_nodes]
        :return:
        """
        queries = self.query_emb(node_encodings.permute(1, 0, 2))
        keys = self.key_emb(node_encodings.permute(1, 0, 2))
        vals = self.val_emb(node_encodings.permute(1, 0, 2))
        att_op, _ = self.att(queries, keys, vals, attn_mask=~adj_mat)

        return att_op.permute(1, 0, 2)
