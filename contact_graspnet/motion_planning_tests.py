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
from collections import defaultdict 
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
from visualization_utils import visualize_grasps, show_image
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
    cube_pos = o["cube_pos"]
    # environment images
    color_img = np.flipud(o['frontview_image'].copy())
    depth_img = np.flipud(o['frontview_depth'])
    #seg_map and depth map
    depth_map = CU.get_real_depth_map(env.sim, depth_img)
    seg = np.flipud(o['frontview_segmentation_element'])
    # get camera matrix
    world_to_camera = CU.get_camera_transform_matrix(
        sim=env.sim,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
    )
    cam_K = CU.get_camera_intrinsic_matrix(env.sim, 'frontview', 256, 256)
    camera_to_world = np.linalg.inv(world_to_camera)
    # compute point cloud 
    pc_full = []
    pc_segments = defaultdict(list)
    for i in range(256):
        for j in range(256):
            estimated_obj_pos = CU.transform_from_pixels_to_world(
                pixels=np.array([i, j]),
                depth_map=depth_map,
                camera_to_world_transform=camera_to_world
            )
            pc_full.append(estimated_obj_pos.copy())
            pc_segments[seg[i, j].item()].append(estimated_obj_pos.copy())   
    # convert to array 
    pc_full = np.asarray(pc_full)
    for k in pc_segments.keys():
        print(k, np.asarray(pc_segments[k]).shape)
        pc_segments[k] = np.asarray(pc_segments[k])
    # check pc_segments k and make sure that this works 
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
    # try visualizing point cloud given just depth and color 
    print(depth_map.shape)
    print(seg.shape)
    # this is the correct transformation to get the visualization correct
    for i in range(len(pc_full)):
        point = pc_full[i].copy()
        point = np.array([-point[1], -point[2], -point[0]])
        pc_full[i] = point
    for k in pc_segments.keys():
        for i in range(len(pc_segments[k])):
            point = pc_segments[k][i].copy()
            point = np.array([-point[1], -point[2], -point[0]])
            pc_segments[k][i] = point
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)
    cube_pred_grasps, cube_scores, cube_contacts = pred_grasps_cam[84], scores[84], contact_pts[84]
    # show_image(color_img, seg)
    # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=None)
    np.savez(f'results/predictions_{3}.npz', 
                  pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)
    np.save('results/pcd_3.npy', pc_full)
    # check distance of pred grasps vector after transforming 
    extrinsic_mat = CU.get_camera_extrinsic_matrix(env.sim, "frontview")
    # contact distances -> if we perform reverse transformation we are good
    for i in range(len(cube_contacts)):
        cube_con = cube_contacts[i].copy()
        cube_con *= -1
        cube_con = np.array([cube_con[2], cube_con[0], cube_con[1]])
        print(np.linalg.norm(cube_con - cube_pos))
    # get target pos + quat from SE(3) grasps
    target_poses = [(cube_pred_grasps[i][:3, 3], cube_pred_grasps[i][:3, :3]) for i in range(len(cube_pred_grasps))]
    # get cube position
    cube_pos = env.sim.data.get_body_xpos("cube_main")
    # try planning to target poses
    best_grasp = np.argmax(cube_scores)
    print(f"Best cube score {np.max(cube_scores)}")
    target_pos, target_quat = target_poses[best_grasp][0], target_poses[best_grasp][1]
    print(f"Orig distance: {np.linalg.norm(target_pos - cube_pos)}")
    target_pos = -1*np.array([target_pos[2], target_pos[0], target_pos[1]])
    print(f"True distance: {np.linalg.norm(target_pos - cube_pos)}")
    assert False
    """
    Next steps
    1) find best grasp and visualize - see if it looks reasonable
    2) calculate inverse transformation - do i need to do it for just rotation matrix or both? (seems like only rotation matrix)
    3) check to see if visualized grasp matches that of 
    """
    # swap columns target_quat 
    # tmp = target_quat[:, 2].copy()
    # target_quat[:, 2] = target_quat[:, 0].copy()
    # target_quat[:, 0] = tmp
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
            np.concatenate((cube_contacts[best_grasp], og_eef_xquat)).astype(np.float64),
            qpos=qpos_curr,
            qvel=qvel_curr,
            grasp=False,
            ignore_object_collision=False,
            planning_time=1.0,
            # planning_time=env.planning_time,
            get_intermediate_frames=False,
        )
    #o = env.step(np.array([0.,0,0,0,0,0,0,5.0]))
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