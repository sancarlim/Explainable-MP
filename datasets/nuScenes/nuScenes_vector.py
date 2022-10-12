from datasets.nuScenes.nuScenes import NuScenesTrajectories
from nuscenes.prediction.input_representation.static_layers import correct_yaw , get_lanes_for_agent
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.arcline_path_utils import compute_segment_sign
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from nuscenes.prediction import PredictHelper
import numpy as np
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import pickle
import torch
from scipy import spatial 
import dgl
import scipy.sparse as spp

mapping_dict = {
            'YIELD':1,
            'STOP_SIGN':1,
            'PED_CROSSING':0, #truck_bus
            'TURN_STOP':2,
            'TRAFFIC_LIGHT': 3
        }


class NuScenesVector(NuScenesTrajectories):
    """
    NuScenes dataset class for single agent prediction, using the vector representation for maps and agents
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'compute_stats', 'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param helper: NuScenes PredictHelper
        :param args: Dataset arguments
        """
        super().__init__(mode, data_dir, args, helper)

        # Initialize helper and maps
        self.map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.helper.data.dataroot) for i in self.map_locs}

        # Vector map parameters
        self.map_extent = args['map_extent']
        self.polyline_resolution = args['polyline_resolution']
        self.polyline_length = args['polyline_length']

        # Load dataset stats (max nodes, max agents etc.)
        if self.mode == 'extract_data':
            stats = self.load_stats()
            self.max_nodes = stats['num_lane_nodes']
            self.max_vehicles = stats['num_vehicles']
            self.max_pedestrians = stats['num_pedestrians']

        # Whether to add random flips for data augmentation
        elif self.mode == 'load_data':
            self.random_flips = args['random_flips']

    def compute_stats(self, idx: int) -> Dict[str, int]:
        """
        Function to compute statistics for a given data point
        """
        num_lane_nodes = self.get_map_representation(idx)
        num_vehicles, num_pedestrians = self.get_surrounding_agent_representation(idx)
        stats = {
            'num_lane_nodes': num_lane_nodes,
            'num_vehicles': num_vehicles,
            'num_pedestrians': num_pedestrians
        }

        return stats

    def load_data(self, idx: int) -> Dict:
        """
        Perform random flips if lag is set to true.
        """
        data = super().load_data(idx)

        if self.random_flips:
            if torch.randint(2, (1, 1)).squeeze().bool().item():
                data = self.flip_horizontal(data)

        return data

    def get_target_agent_representation(self, idx: int) -> np.ndarray:
        """
        Extracts target agent representation
        :param idx: data index
        :return hist: track history for target agent, shape: [t_h * 2, 5]
        """
        i_t, s_t = self.token_list[idx].split("_")

        # x, y co-ordinates in agent's frame of reference
        hist = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=True)

        # Zero pad for track histories shorter than t_h
        hist_zeropadded = np.zeros((int(self.t_h) * 2 + 1, 2))

        # Flip to have correct order of timestamps
        hist = np.flip(hist, 0)
        hist_zeropadded[-hist.shape[0]-1: -1] = hist
        hist = hist_zeropadded

        # Get velocity, acc and yaw_rate over past t_h sec
        motion_states = self.get_past_motion_states(i_t, s_t)
        hist = np.concatenate((hist, motion_states), axis=1)

        return hist

    def get_lanes_for_agent(self, x: float, y: float, yaw: float, 
                        map_api: NuScenesMap) -> Dict[str, List[Tuple[float, float, float]]]:
        lanes, no_lane = map_api.get_closest_lane(x, y, yaw)
        # [N lanes, [n outgoing lanes per candidate lane]]
        candidates_paths = {}
        if len(lanes) != 0:
            for lane in lanes:
                candidates_paths[lane] = map_api.get_outgoing_lane_ids(lane)
        return candidates_paths

    def get_path(self, paths_ids, map_api, origin=None):
        lane_vector_paths = []
        for i, lane_candidate in enumerate(paths_ids):
            lane_dict = {}
            lane_points = map_api.discretize_lanes([lane_candidate]+[out for out in paths_ids[lane_candidate]], resolution_meters=1)  # { lane: [x,y,yaw], outgoing1:[], ... }
            if origin != None: 
                for lane in lane_points:
                    points = np.array(lane_points[lane])
                    if len(points.shape) != 2:
                        continue
                    lane_points[lane] = [self.global_to_local(origin, (pose[0], pose[1], 0)) for pose in points[:,:2] ]
            arcline_path = map_api.get_arcline_path(lane_candidate)[0]
            arc_vector = list(arcline_path.values())
            arc_vector[2] = list(compute_segment_sign(arcline_path)) 
            arc_vector[3] = [1/arc_vector[3]] # compute curvature 
            arc_vector.insert(0,[i,0]) # path number, lane number (0)
            arc_vector=np.concatenate(arc_vector)
            lane_vector = [np.concatenate((np.array(point[:2]), arc_vector)) for point in lane_points[lane_candidate] ]
            lane_vector_paths.extend(lane_vector)

            lane_dict['outgoing_arcline_path'] = []
            # Candidates for lanes in current position
            for outgoing in paths_ids[lane_candidate]:  
                try:
                    lane_dict['outgoing_arcline_path'].append(map_api.get_arcline_path(outgoing)[0]) 
                except:
                    pass 
            
            for nout, out_arc in enumerate(lane_dict['outgoing_arcline_path']):
                out_token = paths_ids[lane_candidate][nout]
                out_vector = list(out_arc.values())
                out_vector[2] =  list(compute_segment_sign(out_arc))
                out_vector[3] = [1/out_vector[3]] # compute curvature 
                out_vector.insert(0,[i,nout+1]) # number of lane in path
                out_vector_cat = np.concatenate(out_vector.copy()) 
                lane_points_out = np.array(lane_points[out_token])[:,:2]
                lane_vector = np.concatenate((lane_vector, np.array([np.concatenate((np.array(point), out_vector_cat)) for point in lane_points_out])))
                lane_vector_paths.extend([np.concatenate((np.array(point), out_vector_cat)) for point in lane_points_out])            

        return lane_vector_paths
        
    def get_map_representation(self, idx: int) -> Union[int, Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]

        # Get agent representation in global co-ordinates
        global_pose = self.get_target_agent_global_pose(idx)

        # Get path candidates
        # paths_ids = {i_t: self.get_lanes_for_agent(global_pose,map_api)[0] } 
        # paths_vectors = self.get_path(paths_ids, map_api)

        # Get lanes around agent within map_extent (80m)
        lanes = self.get_lanes_around_agent(global_pose, map_api)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose, map_api)

        # Get vectorized representation of lanes
        lane_node_feats, _ = self.get_lane_node_feats(global_pose, lanes, polygons, map_api)

        # Discard lanes outside map extent
        lane_node_feats = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 7))]

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':
            return len(lane_node_feats)

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, 7)

        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
            # 'paths_ids': paths_ids,
            # 'path_candidates': paths_vectors
        }

        return map_representation

    def get_surrounding_agent_representation(self, idx: int) -> \
            Union[Tuple[int, int], Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
        """

        # Get vehicles and pedestrian histories for current sample
        vehicles  = self.get_agents_of_type(idx, 'vehicle')
        # paths_ids_v, paths_vectors_v = self.split_lanes(paths_vectors_v, self.polyline_length, paths_ids_v)
        pedestrians = self.get_agents_of_type(idx, 'human')

        # Discard poses outside map extent
        vehicles = self.discard_poses_outside_extent(vehicles)
        # paths_vectors_v = self.discard_poses_outside_extent(paths_vectors_v)
        pedestrians = self.discard_poses_outside_extent(pedestrians)

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == 'compute_stats':
            return len(vehicles), len(pedestrians)

        # Convert to fixed size arrays for batching
        vehicles, vehicle_masks = self.list_to_tensor(vehicles, self.max_vehicles, self.t_h * 2 + 1, 5)
        pedestrians, pedestrian_masks = self.list_to_tensor(pedestrians, self.max_pedestrians, self.t_h * 2 + 1, 5) 
        # max_length = max([len(array) for array in paths_vectors_v])
        # paths_vectors_v, paths_vectors_v_masks = self.list_to_tensor(paths_vectors_v, self.max_vehicles, max_length , 17)

        # Get adjacency matrix
        adj_matrix, len_adj = self.get_adj_matrix(vehicles,vehicle_masks.any(-1),pedestrians,pedestrian_masks.any(-1))
        """ src, dst = np.nonzero(adj_matrix)
        if len(src) == 0 or len(dst) == 0:
            src, dst = np.array([0]), np.array([0]) """
        
        surrounding_agent_representation = {
            'vehicles': vehicles,
            'vehicle_masks': vehicle_masks,
            'pedestrians': pedestrians,
            'pedestrian_masks': pedestrian_masks,
            'adj_matrix': adj_matrix,  
            'len_adj': len_adj,   
        }

        return surrounding_agent_representation

    def get_adj_matrix(self, vehicles, veh_masks, pedestrians, ped_masks):
        """
        Get adjacency matrix for the interaction graph
        :param vehicles: ndarray with vehicle track histories, shape [max_vehicles, t_h * 2, 5]
        :param pedestrians: ndarray with pedestrian track histories, shape [max_pedestrians, t_h * 2, 5]
        :return: adjacency matrix 
        """
        # Remove empty rows 
        veh_masks = ~veh_masks[np.any(~veh_masks,-1)] 
        ped_masks = ~ped_masks[np.any(~ped_masks,-1)] 
        index_veh = np.array( [ np.where(mask_i)[0][-1] if len(np.where(mask_i)[0])!=0 else -1 for mask_i in veh_masks])
        index_ped = np.array( [ np.where(mask_i)[0][-1] if len(np.where(mask_i)[0])!=0 else -1 for mask_i in ped_masks])
        agents_last_positions = np.array([[0,0]])
        if len(veh_masks) > 0:
            vehicles_masked = vehicles[np.arange(veh_masks.shape[0]), index_veh,:2]
            agents_last_positions = np.concatenate((agents_last_positions,vehicles_masked),axis=0)
        else:
            vehicles_masked = np.array(([[]]))
        if len(ped_masks) > 0:
            ped_masked = pedestrians[np.arange(ped_masks.shape[0]), index_ped,:2]
            agents_last_positions = np.concatenate((agents_last_positions,ped_masked),axis=0)
        else:
            ped_masked = np.array(([[]])) 

        # Compute distance between any pair of objects
        dist_matrix = spatial.distance.cdist(agents_last_positions, agents_last_positions) 

        # If distance between last positions is less than 40m, then they are adjacent
        # Adj matrix between target agent and all neighbors
        adj_matrix = np.zeros_like(dist_matrix) 
        adj_matrix[dist_matrix < 20] = 1  

        # Convert to fixed size for batching 
        len_adj = len(adj_matrix)
        adj_array = np.zeros((self.max_vehicles+self.max_pedestrians+1, self.max_vehicles+self.max_pedestrians+1))
        adj_array[:len_adj, :len_adj] = adj_matrix

        return adj_array, len_adj


    def get_target_agent_global_pose(self, idx: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """
        i_t, s_t = self.token_list[idx].split("_")
        sample_annotation = self.helper.get_sample_annotation(i_t, s_t)
        x, y = sample_annotation['translation'][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw = correct_yaw(yaw)
        global_pose = (x, y, yaw)

        return global_pose

    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return lanes: Dictionary of lane polylines
        """
        x, y, _ = global_pose
        radius = max(self.map_extent)
        lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        lanes = map_api.discretize_lanes(lanes, self.polyline_resolution)

        return lanes

    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float], map_api: NuScenesMap) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, _ = global_pose
        radius = max(self.map_extent)
        record_tokens = map_api.get_records_in_radius(x, y, radius, ['ped_crossing','stop_line'])
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = map_api.get(k, record_token)['polygon_token']
                polygons[k].append({record_token: map_api.extract_polygon(polygon_token)})

        return polygons

    def get_lane_node_feats(self, origin: Tuple, lanes: Dict[str, List[Tuple]],
                            polygons: Dict[str, List[Polygon]], map_api: NuScenesMap) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates vector HD map representation in the agent centric frame of reference
        :param origin: (x, y, yaw) of target agent in global co-ordinates
        :param lanes: lane centerline poses in global co-ordinates
        :param polygons: stop-line and cross-walk polygons in global co-ordinates
        :return:
        """

        # Convert lanes to list
        lane_ids = [k for k, v in lanes.items()]
        lanes = [v for k, v in lanes.items()]

        # Get flags indicating whether a lane lies on stop lines or crosswalks
        lane_flags = self.get_lane_flags(lanes, polygons, map_api)

        # Convert lane polylines to local coordinates:
        lanes = [np.asarray([self.global_to_local(origin, pose) for pose in lane]) for lane in lanes]

        # Concatenate lane poses and lane flags
        lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

        return lane_node_feats, lane_node_ids

    def get_agents_of_type(self, idx: int, agent_type: str) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_type: 'human' or 'vehicle'
        :return: list of ndarrays of agent track histories.
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]

        # Get agent representation in global co-ordinates
        origin = self.get_target_agent_global_pose(idx)

        # Load all agents for sample
        agent_details = self.helper.get_past_for_sample(s_t, seconds=self.t_h, in_agent_frame=False, just_xy=False)
        agent_hist = self.helper.get_past_for_sample(s_t, seconds=self.t_h, in_agent_frame=False, just_xy=True)

        # Add present time to agent histories
        present_time = self.helper.get_annotations_for_sample(s_t)
        for annotation in present_time:
            ann_i_t = annotation['instance_token']
            if ann_i_t in agent_hist.keys():
                present_pose = np.asarray(annotation['translation'][0:2]).reshape(1, 2)
                if agent_hist[ann_i_t].any():
                    agent_hist[ann_i_t] = np.concatenate((present_pose, agent_hist[ann_i_t]))
                else:
                    agent_hist[ann_i_t] = present_pose

        # Filter for agent type
        agent_list = []
        agent_i_ts = []
        agent_yaw = []
        for k, v in agent_details.items():
            if v and agent_type in v[0]['category_name'] and v[0]['instance_token'] != i_t:
                agent_list.append(agent_hist[k])
                agent_i_ts.append(v[0]['instance_token'])
                yaw = quaternion_yaw(Quaternion(agent_details[v[0]['instance_token']][0]['rotation']))
                yaw = correct_yaw(yaw)
                agent_yaw.append(yaw)

        # Convert to target agent's frame of reference
        paths_ids = {}
        paths_vectors = []
        for k, agent in enumerate(agent_list):
            # if 'vehicle' in agent_type:
            #     # Get path candidates for vehicles
            #     paths_ids[agent_i_ts[k]] = self.get_lanes_for_agent(agent[0][0],agent[0][1],agent_yaw[k],map_api)  
            #     path = self.get_path(paths_ids[agent_i_ts[k]], map_api, origin)
            #     paths, paths_ids[agent_i_ts[k]] = self.split_lanes(path, self.polyline_length, paths_ids[agent_i_ts[k]])
            #     paths_vectors.append(paths)
            for n, pose in enumerate(agent):
                local_pose = self.global_to_local(origin, (pose[0], pose[1], 0))
                agent[n] = np.asarray([local_pose[0], local_pose[1]])

        # Flip history to have most recent time stamp last and extract past motion states
        for n, agent in enumerate(agent_list):
            xy = np.flip(agent, axis=0)
            motion_states = self.get_past_motion_states(agent_i_ts[n], s_t)
            motion_states = motion_states[-len(xy):, :]
            agent_list[n] = np.concatenate((xy, motion_states), axis=1)
        
        return agent_list 

    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],
                                     ids: List[str] = None) -> Union[List[np.ndarray],
                                                                     Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if self.map_extent[0] <= pose[0] <= self.map_extent[1] and \
                        self.map_extent[2] <= pose[1] <= self.map_extent[3]:
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    def load_stats(self) -> Dict[str, int]:
        """
        Function to load dataset statistics like max surrounding agents, max nodes, max edges etc.
        """
        filename = os.path.join(self.data_dir, 'stats.pickle')
        if not os.path.isfile(filename):
            raise Exception('Could not find dataset statistics. Please run the dataset in compute_stats mode')

        with open(filename, 'rb') as handle:
            stats = pickle.load(handle)

        return stats

    def get_past_motion_states(self, i_t, s_t):
        """
        Returns past motion states: v, a, yaw_rate for a given instance and sample token over self.t_h seconds
        """
        motion_states = np.zeros((2 * self.t_h + 1, 3))
        motion_states[-1, 0] = self.helper.get_velocity_for_agent(i_t, s_t)
        motion_states[-1, 1] = self.helper.get_acceleration_for_agent(i_t, s_t)
        motion_states[-1, 2] = self.helper.get_heading_change_rate_for_agent(i_t, s_t)
        hist = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=True, just_xy=False)

        for k in range(len(hist)):
            if k>3:
                print(f"k Is bigger than 3 {k}")
                break
            motion_states[-(k + 2), 0] = self.helper.get_velocity_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 1] = self.helper.get_acceleration_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 2] = self.helper.get_heading_change_rate_for_agent(i_t, hist[k]['sample_token'])

        motion_states = np.nan_to_num(motion_states)
        return motion_states

    @staticmethod
    def global_to_local(origin: Tuple, global_pose: Tuple) -> Tuple:
        """
        Converts pose in global co-ordinates to local co-ordinates.
        :param origin: (x, y, yaw) of origin in global co-ordinates
        :param global_pose: (x, y, yaw) in global co-ordinates
        :return local_pose: (x, y, yaw) in local co-ordinates
        """
        # Unpack
        global_x, global_y, global_yaw = global_pose
        origin_x, origin_y, origin_yaw = origin

        # Translate
        local_x = global_x - origin_x
        local_y = global_y - origin_y

        # Rotate
        global_yaw = correct_yaw(global_yaw)
        theta = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))

        r = np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                        [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]])
        local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

        local_pose = (local_x, local_y, theta)

        return local_pose

    @staticmethod
    def split_lanes(lanes: List[np.ndarray], max_len: int, lane_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses: (n+1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    @staticmethod
    def get_lane_flags(lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]], map_api: NuScenesMap) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """

        lane_flags = [np.zeros((len(lane), len(mapping_dict.keys())-1)) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list: 
                        if list(polygon.values())[0].contains(point):
                            if k == 'stop_line':
                                n = mapping_dict[map_api.get('stop_line',list(polygon.keys())[0] )['stop_line_type']] 
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags

    @staticmethod
    def list_to_tensor(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array

    @staticmethod
    def flip_horizontal(data: Dict):
        """
        Helper function to randomly flip some samples across y-axis for data augmentation
        :param data: Dictionary with inputs and ground truth values.
        :return: data: Dictionary with inputs and ground truth values fligpped along y-axis.
        """
        # Flip target agent
        hist = data['inputs']['target_agent_representation']
        hist[:, 0] = -hist[:, 0]  # x-coord
        hist[:, 4] = -hist[:, 4]  # yaw-rate
        data['inputs']['target_agent_representation'] = hist

        # Flip lane node features
        lf = data['inputs']['map_representation']['lane_node_feats']
        lf[:, :, 0] = -lf[:, :, 0]  # x-coord
        lf[:, :, 2] = -lf[:, :, 2]  # yaw
        data['inputs']['map_representation']['lane_node_feats'] = lf

        # Flip surrounding agents
        vehicles = data['inputs']['surrounding_agent_representation']['vehicles']
        vehicles[:, :, 0] = -vehicles[:, :, 0]  # x-coord
        vehicles[:, :, 4] = -vehicles[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['vehicles'] = vehicles

        peds = data['inputs']['surrounding_agent_representation']['pedestrians']
        peds[:, :, 0] = -peds[:, :, 0]  # x-coord
        peds[:, :, 4] = -peds[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['pedestrians'] = peds

        # Flip groud truth trajectory
        fut = data['ground_truth']['traj']
        fut[:, 0] = -fut[:, 0]  # x-coord
        data['ground_truth']['traj'] = fut

        return data
