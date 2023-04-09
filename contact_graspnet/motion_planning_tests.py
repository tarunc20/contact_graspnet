import sys
from os.path import abspath, dirname, join

sys.path.insert(0, "/home/tarunc/Desktop/research/contact_graspnet/ompl/py-bindings")
import ompl
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import robosuite as suite 
import numpy as np
import open3d as o3d
import robosuite.utils.camera_utils as CU
from robosuite.controllers import controller_factory
from robosuite.utils.transform_utils import (
    euler2mat,
    mat2pose,
    mat2quat,
    pose2mat,
    quat2mat,
    quat_conjugate,
    quat_multiply,
)
np.set_printoptions(suppress=True)

# imports for contact graspnet
import os
import argparse
import time
import glob
import cv2
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
#from visualization_utils import visualize_grasps, show_image
from mp_env import set_robot_based_on_ee_pos, update_controller_config, apply_controller, mp_to_point

def main(global_config, 
        checkpoint_dir, 
        input_paths, 
        K=None, 
        local_regions=True, 
        skip_border_objects=False, 
        filter_grasps=True, 
        segmap_id=None, 
        z_range=[0.2,1.8], 
        forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    np.random.seed(0)
    env = suite.make("Lift", "Panda", camera_names="frontview", camera_depths=True, camera_segmentations='element')
    o = env.reset()
    color_img = np.flipud(o['frontview_image'].copy())
    depth_img = np.flipud(o['frontview_depth'])
    #seg_map 
    cam_intrinsic_mat = CU.get_camera_intrinsic_matrix(env.sim, "frontview", 256, 256)
    depth_map = CU.get_real_depth_map(env.sim, depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_img.astype(np.uint8)), 
        o3d.geometry.Image((depth_map*1000).astype(np.float32)))
    # check depth map stuff
    print(f"Max min of original depth: {np.max(depth_map), np.min(depth_map)}")
    print(f"Max of new depth: {np.max(np.asarray(rgbd_image.depth)), np.min(np.asarray(rgbd_image.depth))}")
    new_depth = np.asarray(rgbd_image.depth).astype(np.float32)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(256, 256, cam_intrinsic_mat[0][0], #fx
                                                        cam_intrinsic_mat[1][1], #fy
                                                        cam_intrinsic_mat[0][2], #cx
                                                        cam_intrinsic_mat[1][2], #cy
                                                        ))
    d = {
        'rgb': color_img,
        'depth': depth_map[:, :, 0],
        'K': cam_intrinsic_mat,
        'seg': np.flipud(o['frontview_segmentation_element'].astype(np.float32))[:, :, 0],
        'xyz': np.asarray(pcd.points)
    }
    # set up code for grasp estimator
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    # load in data from what we previously computed
    segmap, rgb, depth, cam_K, pc_full, pc_colors = d['seg'], d['rgb'], d['depth'], d['K'], d['xyz'], None 
    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)
    print(pc_segments.keys())
    cube_pos = env.sim.data.get_body_xpos("cube_main")
    #assert False
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)
    # geom id 84 is the cube visual geom, so we need to consider that 
    cube_pred_grasps, cube_scores, cube_contacts = pred_grasps_cam[-1], scores[-1], contact_pts[-1]
    # get camera extrinsic matrix
    cam_extrinsic_mat = CU.get_camera_extrinsic_matrix(env.sim, "frontview")
    # convert all_pred_grasps to environment world frame
    for i in range(len(cube_pred_grasps)):
        cube_pred_grasps[i] = cam_extrinsic_mat @ cube_pred_grasps[i] 
    # get target pos + quat from SE(3) grasps
    target_poses = [(cube_pred_grasps[i][0:3, 3], cube_pred_grasps[i][0:3, 0:3]) for i in range(len(cube_pred_grasps))]
    # get cube position
    cube_pos = env.sim.data.get_body_xpos("cube_main")
    # try planning to target poses
    best_grasp = np.argmin([
        np.linalg.norm(target_poses[i][0] - cube_pos)
        for i in range(len(target_poses))
    ])
    target_pos, target_quat = target_poses[best_grasp][0], target_poses[best_grasp][1]
    # set up environment and ik controller
    qpos_curr = env.sim.data.qpos.copy()
    qvel_curr = env.sim.data.qvel.copy()
    ik_controller_config = {
        "type": "IK_POSE",
        "ik_pos_limit": 0.02,
        "ik_ori_limit": 0.05,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "converge_steps": 100,
    }
    osc_controller_config = {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "output_min": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2,
    }
    update_controller_config(env, ik_controller_config)
    ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
    ik_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)
    og_eef_xpos = env._eef_xpos.copy()
    og_eef_xquat = env._eef_xquat.copy()
    # set name since we need it
    env.name = "Lift"
    # set mp bounds
    env.mp_bounds_low=(-1.45, -1.25, 0.8)
    env.mp_bounds_high = (0.45, 0.85, 2.25)
    env.update_with_true_state = False
    o = mp_to_point(
            env,
            ik_controller_config,
            osc_controller_config,
            np.concatenate((cube_pos + np.array([0.05, 0., 0]), env._eef_xquat)).astype(np.float64),#np.concatenate((target_pos, mat2quat(target_quat))).astype(np.float64),
            qpos=qpos_curr,
            qvel=qvel_curr,
            grasp=False,
            ignore_object_collision=False,
            planning_time=1.0,
            # planning_time=env.planning_time,
            get_intermediate_frames=False,
        )
    print(f"Target pos: {target_pos}")
    print(f"Eef xpos: {env._eef_xpos}")
    print(f"Distance to cube: {np.linalg.norm(env._eef_xpos - cube_pos)}")
    o = env._get_observations()
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(color_img)
    plt.subplot(1, 2, 2)
    plt.title('After')
    plt.imshow(np.flipud(o['frontview_image']))
    plt.show()
    plt.savefig("res.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    main(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)