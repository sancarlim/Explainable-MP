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
    adj_matrix = [element['inputs']['surrounding_agent_representation']['adj_matrix'] for element in batch]
    len_adj = [element['inputs']['surrounding_agent_representation']['len_adj'] for element in batch]
    adj_matrix = [adj[:len, :len] for adj, len in zip(adj_matrix, len_adj)]
    graphs = [dgl.from_scipy(spp.coo_matrix(adj)).int() for adj in adj_matrix]
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
