# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import open3d as o3d

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# callback next frame function
def next_frame(visualizer):
    visualizer.close()


# callback exit function
def exit(visualizer):
    visualizer.destroy_window()


def show_pcl(pcl):
    """
    Visualize LIDAR point-cloud

    Parameters:
    pcl (2D numpy array): lidar point cloud to visualize
    cnt_frame (int): index current frame corresponding to pcl

    """

    print("Visualize lidar point-cloud")

    # initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window('LIDAR Point-Cloud')
    
    # create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3]) # [x,y,z,r] we don't need r. pcl[:, :3]

    # add the pcd instance to visualization using add_geometry
    vis.add_geometry(pcd)

    # visualize point cloud and keep window open until right-arrow or space-bar are pressed
    # right-arrow (key-code 262) move to next frame while space-bar (key-code 262) exits the process
    vis.register_key_callback(262, next_frame)
    vis.register_key_callback(32, exit)
    vis.run()


def show_range_image(frame, lidar_name):
    """
    Visualize range image for each frame
    Ref https://waymo.com/intl/en_us/dataset-data-format/

    Parameters:
    frame (tfrecord): current frame to visualize
    lidar_name (int): integer corresponding to lidar name

    Returns:
    img_range_intensity (unsigned 8-bit integer): numpy array containing the processed range and intensity channels

    """

    print("Visualize range image")

    # extract lidar data and range image for the roof-mounted lidar
    lidar = waymo_utils.get(frame.lasers, lidar_name)
    range_image, camera_projection, range_image_pose = waymo_utils.parse_range_image_and_camera_projection(lidar)  # Parse the top laser range image and get the associated projection.

    # extract the range [0] and the intensity [1] channel from the range image
    range_channel = range_image[:, :, 0]
    intensity_channel = range_image[:, :, 1]

    # set values <0 to zero
    range_channel[range_channel < 0] = 0
    intensity_channel[intensity_channel < 0] = 0
    
    # map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    range_channel = range_channel * 255 / (np.max(range_channel) - np.min(range_channel))
    range_channel = range_channel.astype(np.uint8)

    # map the intensity channel onto an 8-bit scale
    # normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    perc_1, perc_99 = np.percentile(intensity_channel, (1, 99))
    intensity_channel = 255 * (intensity_channel - perc_1) / (perc_99 - perc_1)
    intensity_channel = intensity_channel.astype(np.uint8)
    
    # stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((range_channel, intensity_channel)).astype(np.uint8)
    
    return img_range_intensity


def bev_from_pcl(lidar_pcl, configs):
    """
    Create birds-eye view of lidar data.
    1. Convert sensor coordinates to bev-map coordinates
    Ref http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/
    2. Compute intensity layer of the BEV map
    3. Compute height layer of the BEV map

    Parameters:
    lidar_pcl (2D numpy array): lidar point cloud which is to be converted (point = [x y z r])
    configs (edict): dictionary containing config info

    Returns:
    input_bev_maps ():

    """

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]

    ##################
    # Convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ##################

    print("Convert sensor coordinates to bev-map coordinates")

    # compute bev-map discretization (resolution) by dividing x-range by the bev-image height (see configs)
    pcl_res = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    # create a copy of the lidar pcl and transform all metrics x-coordinates into bev-image coordinates
    # (lidar_pcl is already in vehicle space!)
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / pcl_res))

    # perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    pcl_res = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / pcl_res) + (configs.bev_width + 1) / 2)
    lidar_pcl_cpy[:, 1] = np.abs(lidar_pcl_cpy[:, 1])

    # visualize point-cloud using the function
    # show_pcl(lidar_pcl_cpy)

    ##################
    # Compute intensity layer of the BEV map
    ##################

    # create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    idx = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idx]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    # also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    lidar_pcl_top, indices, counts = np.unique(lidar_pcl_top, axis=0, return_index=True, return_counts=True)

    # assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    # normalize intensity with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    perc_1, perc_99 = np.percentile(lidar_pcl_top[:, 3], (1, 99))
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = \
        (lidar_pcl_top[:, 3] - perc_1) / (perc_99 - perc_1)

    # visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    intensity_img = intensity_map * 255
    intensity_img = intensity_img.astype(np.uint8)
    cv2.imshow('Intensity layer', intensity_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

    ##################
    # Compute height layer of the BEV map
    ##################

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background

    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    lidar_pcl_top = []
    height_map = []
    intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


