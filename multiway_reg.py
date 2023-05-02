import numpy as np
import cv2
import os
import json
import open3d as o3d
from tqdm import tqdm
import copy

# From http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
def pairwise_registration(
        source, target, max_correspondence_distance_coarse,
        max_correspondence_distance_fine
    ):
    # Coarse ICP starting with identity matrix
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        max_correspondence_distance_coarse, 
        init = np.identity(4),
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    # Fine ICP using the result of coarse ICP as initialization
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        max_correspondence_distance_fine,
        init = icp_coarse.transformation,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    # Return the transformation matrix and information matrix
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(
        pcds, max_correspondence_distance_coarse,
        max_correspondence_distance_fine
    ):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in tqdm(range(n_pcds)):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], 
                pcds[target_id], 
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine,
            )
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph