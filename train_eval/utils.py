from sklearn.metrics import v_measure_score
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
        lane_veh_adj_matrix = element['inputs']['agent_node_masks']['vehicles'].transpose(1,0) # 84 x 164 
        lane_ped_adj_matrix = element['inputs']['agent_node_masks']['pedestrians'].transpose(1,0) # 84 x 164 
        lane_node_masks = element['inputs']['map_representation']['lane_node_masks']
        # create a mask for the lanes that are not empty
        lane_mask = (~(lane_node_masks[:,:,0]!=0)).any(-1) # 164
        # Add row of zeros to the adjacency matrix to account for the focal vehicle
        lane_veh_adj_matrix = np.vstack((~lane_mask*1, lane_veh_adj_matrix)) # 85 x 164
        veh_lane_u, veh_lane_v = np.where(lane_veh_adj_matrix==0)
        lane_ped_u, lane_ped_v = np.where(lane_ped_adj_matrix==0)
        veh_mask=element['inputs']['surrounding_agent_representation']['vehicle_masks']
        ped_mask=element['inputs']['surrounding_agent_representation']['pedestrian_masks']
        num_v = np.where(veh_mask[:,:,0]==0)[0].max()+2 if len(np.where(veh_mask[:,:,0]==0)[0])>0 else 0 # +1 to account for focal agent which is not present in veh_mask
        num_p = np.where(ped_mask[:,:,0]==0)[0].max()+1 if len(np.where(ped_mask[:,:,0]==0)[0])>0 else 0
        adj_matrix_v = adj[:num_v, :num_v] 
        adj_matrix_p = adj[num_v:num_v+num_p, num_v:num_v+num_p]
        veh_u, veh_v = np.nonzero(adj_matrix_v)
        ped_u, ped_v = np.nonzero(adj_matrix_p)
        ped_veh_u, ped_veh_v = np.where(adj[num_v:num_v+num_p, :num_v] == 1)  
        # mask those pedestrians that don't appear in the interaction graph v2p
        max_p_in_graph = ped_veh_u.max()+1 if len(ped_veh_u)>0 else 0
        ped_mask[max_p_in_graph:,:, :] = 1
        element['inputs']['surrounding_agent_representation']['pedestrian_masks'] =  ped_mask 
        # interaction_graphs.append(dgl.from_scipy(spp.coo_matrix(adj_matrix)).int())
        # Lane graph
        succ_adj_matrix = element['inputs']['map_representation']['succ_adj_matrix'] 
        prox_adj_matrix = element['inputs']['map_representation']['prox_adj_matrix']
        
        succ_u, succ_v = np.nonzero(succ_adj_matrix)[0], np.nonzero(succ_adj_matrix)[1] 
        prox_u, prox_v = np.nonzero(prox_adj_matrix)  
        lanes_graphs.append(
            dgl.heterograph({
            ('l','successor','l'): (torch.tensor(succ_u, dtype=torch.int), torch.tensor(succ_v, dtype=torch.int)),
            ('l','proximal','l'):  (torch.tensor(prox_u, dtype=torch.int), torch.tensor(prox_v, dtype=torch.int)),
            ('v', 'v_close_l','l'): (torch.tensor(veh_lane_u, dtype=torch.int), torch.tensor(veh_lane_v, dtype=torch.int)),
            ('v', 'v_interact_v','v'): (torch.tensor(veh_u, dtype=torch.int), torch.tensor(veh_v, dtype=torch.int)),  
            ('p', 'p_interact_v','v'): (torch.tensor(ped_veh_u, dtype=torch.int), torch.tensor(ped_veh_v, dtype=torch.int)),  
        }) )
        if len(veh_mask)+1-len(np.nonzero(veh_mask[:,0,0])[0]) != lanes_graphs[-1].num_nodes('v'):
            print('stop')
        
    #interaction_batched_graph = dgl.batch(interaction_graphs)
    lanes_batched_graph = dgl.batch(lanes_graphs) 

    data = default_collate(batch)
    #data['inputs']['interaction_graphs'] = interaction_batched_graph
    data['inputs']['lanes_graphs'] = lanes_batched_graph
    return data