import torch.optim
from typing import Dict, Union
import torch
import numpy as np
import dgl
import scipy.sparse as spp
from torch.utils.data._utils.collate import default_collate


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_double_to_float(data: Union[Dict, torch.Tensor]):
    """
    Utility function to convert double tensors to float tensors in nested dictionary with Tensors
    """
    if type(data) is torch.Tensor and data.dtype == torch.float64:
        return data.float()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert_double_to_float(v)
        return data
    else:
        return data


def send_to_device(data: Union[Dict, torch.Tensor]):
    """
    Utility function to send nested dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to(device)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
        return data
    else:
        return data 


def convert2tensors(data):
    """
    Converts data (dictionary of nd arrays etc.) to tensor with batch_size 1
    """
    if type(data) is np.ndarray:
        return torch.as_tensor(data).unsqueeze(0)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert2tensors(v)
        return data
    else:
        return data



def collate_fn_dgl(batch): 
    # Collate function for dataloader. 
    graphs = []
    for element in batch:        
        adj = element['inputs']['surrounding_agent_representation']['adj_matrix'] 
        len_adj = element['inputs']['surrounding_agent_representation']['len_adj']  
        adj_matrix = adj[:len_adj, :len_adj]
        graphs.append(dgl.from_scipy(spp.coo_matrix(adj_matrix)).int())
    #[dgl.graph((s,d)) for s,d in zip(src,dst)] 
    # graphs = [dgl.add_self_loop(graph) for graph in graphs]
    interaction_batched_graph = dgl.batch(graphs)

    # Create lanes heterograph 
    """adj_matrix = [element['inputs']['map_representation']['adj_matrix'] for element in batch]
    lanes_heterograph = [dgl.heterograph(dgl.from_scipy(spp.coo_matrix(adj))).int() for adj in adj_matrix]
    lanes_heterograph = [dgl.add_self_loop(graph) for graph in graphs]
    lanes_batched_graph = dgl.batch(lanes_heterograph)"""

    data = default_collate(batch)
    data['inputs']['interaction_graphs'] = interaction_batched_graph
    #data['inputs']['lanes_graphs'] = lanes_batched_graph
    return data

def collate_fn_dgl_lanes(batch): 
    # Collate function for dataloader. 
    graphs = []
    lanes_graphs = []
    for element in batch:        
        # Create interaction graph
        adj = element['inputs']['surrounding_agent_representation']['adj_matrix'] 
        len_adj = element['inputs']['surrounding_agent_representation']['len_adj']  
        adj_matrix = adj[:len_adj, :len_adj]
        graphs.append(dgl.from_scipy(spp.coo_matrix(adj_matrix)).int())
        # Create lanes graph  
        adj_lanes = element['inputs']['map_representation']['adj_matrix']
        lane_masks = element['inputs']['map_representation']['lane_node_masks'] # 164 x 20
        len_adj_lanes = np.count_nonzero((~(lane_masks[:,:,0]!=0)).any(-1)) # 164
        adj_matrix_lanes = adj_lanes[:len_adj_lanes, :len_adj_lanes]
        lane_graph = dgl.add_self_loop(dgl.from_scipy(spp.coo_matrix(adj_matrix_lanes)).int())
        lanes_graphs.append(lane_graph)

    interaction_batched_graph = dgl.batch(graphs) 
    lanes_batched_graph = dgl.batch(lanes_graphs) 
    data = default_collate(batch)
    data['inputs']['interaction_graphs'] = interaction_batched_graph
    data['inputs']['lanes_graphs'] = lanes_batched_graph
    return data


def collate_fn_dgl_hetero(batch): 
    # Collate function for dataloader.
    interaction_graphs = []
    lanes_graphs = []
    for element in batch:        
        # Interaction graph
        adj = element['inputs']['surrounding_agent_representation']['adj_matrix'] 
        len_adj = element['inputs']['surrounding_agent_representation']['len_adj']  
        adj_matrix = adj[:len_adj, :len_adj]
        interaction_graphs.append(dgl.from_scipy(spp.coo_matrix(adj_matrix)).int())
        # Lane graph
        succ_adj_matrix = element['inputs']['map_representation']['succ_adj_matrix'] 
        prox_adj_matrix = element['inputs']['map_representation']['prox_adj_matrix']
        # TODO: Do it in preprocess, save succ_u, succ_v, prox_u, prox_v in data. For batching they need to have the same length.
        succ_u = np.array([])
        succ_v = np.array([])
        for i, count in enumerate(np.count_nonzero(succ_adj_matrix, axis=1)):
            if count != 0:
                succ_u = np.append(succ_u,[i]*count)
                succ_v = np.append(succ_v, np.nonzero(succ_adj_matrix[i])[0])
        if len(succ_u) == 0:
            # If no successors, add self loop - some sequences have no lanes!
            succ_u = np.array([0])
            succ_v = np.array([0])
        prox_u = np.array([])
        prox_v = np.array([])
        for i, count in enumerate(np.count_nonzero(prox_adj_matrix, axis=1)):
            if count != 0:
                prox_u = np.append(prox_u,[i]*count)
                prox_v = np.append(prox_v, np.nonzero(prox_adj_matrix[i])[0]) 
        if len(prox_u) == 0:
            prox_u = np.array([0])
            prox_v = np.array([0])
        lanes_graphs.append(
            dgl.heterograph({
            ('l','successor','l'): (torch.tensor(succ_u, dtype=torch.int), torch.tensor(succ_v, dtype=torch.int)),
            ('l','proximal','l'):  (torch.tensor(prox_u, dtype=torch.int), torch.tensor(prox_v, dtype=torch.int))
        }) )
        
    interaction_batched_graph = dgl.batch(interaction_graphs)
    lanes_batched_graph = dgl.batch(lanes_graphs) 

    data = default_collate(batch)
    data['inputs']['interaction_graphs'] = interaction_batched_graph
    data['inputs']['lanes_graphs'] = lanes_batched_graph
    return data