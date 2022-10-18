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

class Collate_heterograph(object):
    def __init__(self, args):
        self.mask_frames = args['mask_frames_p']
        self.agent_mask_prob_v = args['agent_mask_p_veh']
        self.lane_mask_prob = args['lane_mask_p'] 
    def __call__(self,batch):
        # Collate function for dataloader. 
        lanes_graphs = []
        for element in batch: 
            adj = element['inputs']['surrounding_agent_representation']['adj_matrix'] 
            lane_node_masks = element['inputs']['map_representation']['lane_node_masks']
            veh_mask=element['inputs']['surrounding_agent_representation']['vehicle_masks']
            num_v = np.where(veh_mask[:,:,0]==0)[0].max()+2 if len(np.where(veh_mask[:,:,0]==0)[0])>0 else 0 # +1 to account for focal agent which is not present in veh_mask
            
            ### ROBUSTNESS ANALYSYS ###
            mask_out_lanes = []
            if 'mask_out_lanes' in element['inputs']['map_representation']:
                mask_out_lanes = element['inputs']['map_representation']['mask_out_lanes']
            elif self.lane_mask_prob > 0.:
                ###### Mask out lane_node_masks by lane p% of the time - 1 means mask out
                mask_out = np.tile(np.expand_dims((np.random.random((lane_node_masks.shape[0])) < self.lane_mask_prob), [-1,-2]), [ 1,lane_node_masks.shape[-2],lane_node_masks.shape[-1]]) 
                lane_node_masks =  lane_node_masks.astype(int) | mask_out.astype(int)
                # Indeces of masked out lanes
                mask_out_lanes = np.where(mask_out[:,0,0] == True)[0]
                element['inputs']['map_representation']['lane_node_masks'] = lane_node_masks
                # Update with masked out lanes
                element['inputs']['map_representation']['succ_adj_matrix'] = element['inputs']['map_representation']['succ_adj_matrix'] * (1-lane_node_masks[:,0,0])
                element['inputs']['map_representation']['prox_adj_matrix'] = element['inputs']['map_representation']['prox_adj_matrix'] * (1-lane_node_masks[:,0,0])
                element['inputs']['agent_node_masks']['vehicles'] = element['inputs']['agent_node_masks']['vehicles'].astype(int) | np.expand_dims((lane_node_masks[:,0,0]), -1).astype(int)
            
            if self.mask_frames > 0.0: 
                target_adj_matrix = element['inputs']['surrounding_agent_representation']['adj_matrix'][0,1:veh_mask.shape[0]+1] 
                target_adj_matrix = np.tile(np.expand_dims(target_adj_matrix,-1), [1,veh_mask.shape[-2]]) 
                target_adj_matrix *= (np.random.random((target_adj_matrix.shape[0], target_adj_matrix.shape[1])) > self.mask_frames).astype(int)                
                veh_mask = veh_mask.astype(int) | np.tile(np.expand_dims((1-target_adj_matrix),-1), [1,1,veh_mask.shape[-1]]).astype(int)  
                # Mask out frames of nearby agents with a 60% probability
                element['inputs']['surrounding_agent_representation']['vehicle_masks'] = veh_mask
                element['inputs']['agent_node_masks']['vehicles'] = element['inputs']['agent_node_masks']['vehicles'].astype(int) | np.tile(np.expand_dims(veh_mask[:,:,0].any(-1 ),0), [164,1])  
            #############################
            # Update with new masked out vehicles to update the graph
            v_nodes_mask = (veh_mask[:,:,0].sum(-1)==veh_mask.shape[-2]) == False # True where there is a vehicle
            v_nodes = np.where(v_nodes_mask)[0] # 0 to 83
            v_nodes = np.insert((v_nodes + np.ones((v_nodes.shape[0]))),0,0).astype(int) # 0 is the focal vehicle
            adj_matrix_v = adj[v_nodes][:,v_nodes] 
            veh_u, veh_v = np.nonzero(adj_matrix_v)
            
            # Pedestrians
            ped_mask=element['inputs']['surrounding_agent_representation']['pedestrian_masks']
            num_p = np.where(ped_mask[:,:,0]==0)[0].max()+1 if len(np.where(ped_mask[:,:,0]==0)[0])>0 else 0
            ped_veh_u, ped_veh_v = np.where(adj[num_v:num_v+num_p, v_nodes] == 1)   
            # mask those pedestrians that don't appear in the interaction graph v2
            max_p_in_graph = ped_veh_u.max()+1 if len(ped_veh_u)>0 else 0
            ped_mask[max_p_in_graph:,:, :] = 1
            element['inputs']['surrounding_agent_representation']['pedestrian_masks'] =  ped_mask

            # Objects 
            # obj_mask=element['inputs']['surrounding_agent_representation']['object_masks']
            # num_o = np.where(obj_mask[:,:,0]==0)[0].max()+1 if len(np.where(obj_mask[:,:,0]==0)[0])>0 else 0
            # obj_veh_u, obj_veh_v = np.where(adj[num_v+num_p:num_v+num_p+num_o, v_nodes] == 1) 
            # max_o_in_graph = obj_veh_u.max()+1 if len(obj_veh_u)>0 else 0 
            # obj_mask[max_o_in_graph:,:, :] = 1
            # element['inputs']['surrounding_agent_representation']['object_masks'] =  obj_mask 

            # Lane graph 
            lane_veh_adj_matrix = element['inputs']['agent_node_masks']['vehicles'].transpose(1,0) 
            # Remove masked lanes
            lane_veh_adj_matrix = np.delete(lane_veh_adj_matrix, mask_out_lanes, 1)
            # To keep the indexing consistent, we set to 0 the edge type of masked out lanes, i.e. no edge.
            if len(mask_out_lanes) > 0:
                for lane in mask_out_lanes:
                    element['inputs']['map_representation']['edge_type'] = np.where( element['inputs']['map_representation']['s_next'] == lane, 0, element['inputs']['map_representation']['edge_type'])
            element['inputs']['map_representation']['s_next'][mask_out_lanes] = 0
            # Update with new masked out vehicles 
            lane_veh_adj_matrix = lane_veh_adj_matrix[v_nodes_mask] #num_nbr_vehicles x 164
            # create a mask for the lanes that are not empty
            lane_mask = np.delete( (~(lane_node_masks[:,:,0]!=0)).any(-1),  mask_out_lanes) # 164
            # Add row of zeros to the adjacency matrix to account for the focal vehicle
            lane_veh_adj_matrix = np.vstack((~lane_mask*1, lane_veh_adj_matrix)) # num_veh x 164
            veh_lane_u, veh_lane_v = np.where(lane_veh_adj_matrix==0)  
            succ_adj_matrix = np.delete( np.delete( element['inputs']['map_representation']['succ_adj_matrix'], mask_out_lanes, 1), mask_out_lanes, 0)
            prox_adj_matrix = np.delete( np.delete( element['inputs']['map_representation']['prox_adj_matrix'], mask_out_lanes, 1), mask_out_lanes, 0)
            succ_u, succ_v = np.nonzero(succ_adj_matrix)[0], np.nonzero(succ_adj_matrix)[1] 
            prox_u, prox_v = np.nonzero(prox_adj_matrix) 

            # Create heterogeneous graph  
            lanes_graphs.append(dgl.heterograph({
                ('l','successor','l'): (torch.tensor(succ_u, dtype=torch.int), torch.tensor(succ_v, dtype=torch.int)),
                ('l','proximal','l'):  (torch.tensor(prox_u, dtype=torch.int), torch.tensor(prox_v, dtype=torch.int)),
                ('v', 'v_close_l','l'): (torch.tensor(veh_lane_u, dtype=torch.int), torch.tensor(veh_lane_v, dtype=torch.int)),
                ('v', 'v_interact_v','v'): (torch.tensor(veh_u, dtype=torch.int), torch.tensor(veh_v, dtype=torch.int)),  
                ('p', 'p_interact_v','v'): (torch.tensor(ped_veh_u, dtype=torch.int), torch.tensor(ped_veh_v, dtype=torch.int)),  
                #('o', 'o_interact_v','v'): (torch.tensor(obj_veh_u, dtype=torch.int), torch.tensor(obj_veh_v, dtype=torch.int)),
            }) )

            assert len(np.nonzero(veh_mask[:,:,0].sum(-1)<5)[0])+1 == lanes_graphs[-1].num_nodes('v') 

        lanes_batched_graph = dgl.batch(lanes_graphs)  
        data = default_collate(batch) 
        data['inputs']['lanes_graphs'] = lanes_batched_graph

        return data



def collate_fn_dgl_hetero(batch): 
    # Collate function for dataloader. 
    lanes_graphs = []
    for element in batch:        
        # Interaction graph
        adj = element['inputs']['surrounding_agent_representation']['adj_matrix']  
        lane_veh_adj_matrix = element['inputs']['agent_node_masks']['vehicles'].transpose(1,0) # 84 x 164 
        lane_node_masks = element['inputs']['map_representation']['lane_node_masks']
        # create a mask for the lanes that are not empty
        lane_mask = (~(lane_node_masks[:,:,0]!=0)).any(-1) # 164
        # Add row of zeros to the adjacency matrix to account for the focal vehicle
        lane_veh_adj_matrix = np.vstack((~lane_mask*1, lane_veh_adj_matrix)) # 85 x 164
        veh_lane_u, veh_lane_v = np.where(lane_veh_adj_matrix==0) 
        veh_mask=element['inputs']['surrounding_agent_representation']['vehicle_masks']
        ped_mask=element['inputs']['surrounding_agent_representation']['pedestrian_masks']
        num_v = np.where(veh_mask[:,:,0]==0)[0].max()+2 if len(np.where(veh_mask[:,:,0]==0)[0])>0 else 0 # +1 to account for focal agent which is not present in veh_mask
        num_p = np.where(ped_mask[:,:,0]==0)[0].max()+1 if len(np.where(ped_mask[:,:,0]==0)[0])>0 else 0
        adj_matrix_v = adj[:num_v, :num_v]  
        veh_u, veh_v = np.nonzero(adj_matrix_v) 
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

        
    lanes_batched_graph = dgl.batch(lanes_graphs) 

    data = default_collate(batch) 
    data['inputs']['lanes_graphs'] = lanes_batched_graph
    return data