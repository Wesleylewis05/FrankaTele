"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Operational Space Control
----------------
Operational Space Control of Franka robot to demonstrate Jacobian and Mass Matrix Tensor APIs
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import os
import argparse
import os.path as osp
from device.keyboard_interface import KeyboardInterface
from typing import List
import cv2
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


# Parse arguments
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
                                   {"name": "--num_frankas", "type": int, "default": 1, "help": "Number of frankas to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])


class TeleEnv:
    def __init__(self, args):
        self.args = args
        # Initialize gym
        self.gym = gymapi.acquire_gym()
        self.envs = []
        self.hand_idxs = []
        self.box_idxs = []
        self.init_pos_list = []
        self.init_orn_list = []
        self.all_cam_handles = []  # List to store camera handles for all environments.
        self.ee_handles = []
        self.ee_idxs = []
        self.camera_count = 1
        self.bodycam_count = 1
        self.num_cubes = 3
        self.cam_width = 720//2
        self.cam_height = 500
        # Number of frankas to be created in each environment
        self.num_frankas = args.num_frankas
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")

        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)

        if self.sim is None:
            raise Exception("Failed to create sim")


        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)


        # Set up the env grid
        self.num_envs = args.num_envs
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)


        self.setup_env()
        self.prepare_tensors()
        # Point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)





        
    def create_franka(self):
        self.frankas = []
        # Load franka asset
        asset_root = "../../assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
        
        
        for _ in range(self.num_frankas):
            self.franka_asset = self.gym.load_asset(
            self.sim, asset_root, franka_asset_file, asset_options)
            # get joint limits and ranges for Franka
            self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
            self.franka_lower_limits = self.franka_dof_props['lower']
            self.franka_upper_limits = self.franka_dof_props['upper']
            self.franka_ranges = self.franka_upper_limits - self.franka_lower_limits
            self.franka_mids = 0.5 * (self.franka_upper_limits + self.franka_lower_limits)
            self.franka_num_dofs = len(self.franka_dof_props)

            # set default DOF states
            self.default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
            self.default_dof_state["pos"][:7] = self.franka_mids[:7]

            # set DOF control properties (except grippers)
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.franka_dof_props["stiffness"][:7].fill(0.0)
            self.franka_dof_props["damping"][:7].fill(0.0)

            # set DOF control properties for grippers
            self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            self.franka_dof_props["stiffness"][7:].fill(800.0)
            self.franka_dof_props["damping"][7:].fill(40.0)

            self.franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
            self.franka_ee_index = self.franka_link_dict["k_ee_link"]
            self.franka_base_index = self.franka_link_dict["panda_link0"]

            franka_dict = {
            "asset": self.franka_asset,
            "dof_props": self.franka_dof_props,
            "lower_limits": self.franka_lower_limits,
            "upper_limits": self.franka_upper_limits,
            "ranges": self.franka_ranges,
            "mids": self.franka_mids,
            "num_dofs": self.franka_num_dofs,
            "default_dof_state": self.default_dof_state,
            "link_dict": self.franka_link_dict,
            "ee_index": self.franka_ee_index,
            "base_index": self.franka_base_index
            }

            self.frankas.append(franka_dict)

    def create_box(self, env_index, pose: List, env):
        """
        create box in given environment
        params:
            env_index: which environment the cubes will be in
            pose: list of position and rotation values
                [0]: x
                [1]: y
                [2]: z
                [3]: r
                [4]: p
                [5]: y
                [6]: w
        """
        # create box asset
        box_size = 0.045
        asset_options = gymapi.AssetOptions()
        box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        # box pose
        box_pose = gymapi.Transform()
        # add box
        box_pose.p = gymapi.Vec3(pose[0],pose[1],pose[2])
        box_pose.r = gymapi.Quat(pose[3],pose[4],pose[5],pose[6])
        box_handle = self.gym.create_actor(env, box_asset, box_pose, f"box_{len(self.box_idxs)}", env_index, 0)
        print(f"created box_{len(self.box_idxs)}")
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # get global index of box in rigid body state tensor
        box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
        self.box_idxs.append(box_idx)

    def display_camera_views(self, env, cam_handles):
        images = []
        for handle in cam_handles:
            # Get the RGBA image from the camera
            image = self.gym.get_camera_image(self.sim, env, handle, gymapi.IMAGE_COLOR)    
            image = image.reshape(image.shape[0], -1, 4)     
            images.append(image[:,:,:3])

        # Stack the images horizontally (side by side)
        combined_image = np.concatenate((images[0], images[1]), axis=1)

        # Display the combined image
        cv2.imshow('Camera Views', combined_image)
        cv2.waitKey(1)  # Refresh

    def setup_cam(self, env, cam_width, cam_height, cam_pos, cam_target):
        cam_props = gymapi.CameraProperties()
        cam_props.width = cam_width
        cam_props.height = cam_height    
        cam_handle = self.gym.create_camera_sensor(env, cam_props)
        self.gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
        return cam_handle, cam_props
    
    def setup_body_cam(self, env, cam_width, cam_height, transform, handle):
        """
        env: environment
        cam_width: width of camera
        cam_height: height of camera
        transform: gymapi.Transform() (position and rotation)
        handle: handle for part to follow
        """

        cam_props = gymapi.CameraProperties()
        cam_props.width = cam_width
        cam_props.height = cam_height    
        cam_handle = self.gym.create_camera_sensor(env, cam_props)
        self.gym.attach_camera_to_body(cam_handle, env, handle, transform, gymapi.FOLLOW_TRANSFORM)
        return cam_handle, cam_props
    
    def setup_env(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

        pose2 = gymapi.Transform()
        pose2.p = gymapi.Vec3(1.0, 1.0, 0.0)
        pose2.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

        self.pose_list = [pose, pose2]
        camera_poses = [gymapi.Vec3(0, 1, 0.5),gymapi.Vec3(1, 5, 2.5),] 
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        transform1 = gymapi.Transform(p=gymapi.Vec3(-0.04, 0, -0.05), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-70.0)))
        camera_transform = [transform1]
        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)
            self.create_box(i,[0.5,0.0,0.0,0.0,0.0,0.0,1], env)
            self.create_box(i,[0.5,0.25,0.0,0.0,0.0,0.0,1], env)
            self.create_box(i,[0.5,-0.25,0.0,0.0,0.0,0.0,1], env)
            # create frankas list
            self.create_franka()
            n = 0
            for franka in self.frankas:
                # Add franka
                franka_handle = self.gym.create_actor(env, franka["asset"], self.pose_list[n], "franka", i, 0)
                n += 1
                # Set initial DOF states
                self.gym.set_actor_dof_states(env, franka_handle, franka["default_dof_state"], gymapi.STATE_ALL)

                # Set DOF control properties
                self.gym.set_actor_dof_properties(env, franka_handle, franka["dof_props"])

                # Get inital hand pose
                hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
                hand_pose = self.gym.get_rigid_transform(env, hand_handle)
                self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
                self.init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
                self.ee_idxs.append(self.gym.get_actor_rigid_body_index(env, franka_handle, franka["ee_index"], gymapi.DOMAIN_SIM))
                self.ee_handles.append(self.gym.find_actor_rigid_body_handle(env, franka_handle, "k_ee_link") )
                # Get global index of hand in rigid body state tensor
                hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
                self.hand_idxs.append(hand_idx)
                env_cam_handles = []
                for j in range(self.camera_count):  # Set up cameras for the current environment.
                    cam_handle, _ = self.setup_cam(env, self.cam_width, self.cam_height, camera_poses[j], cam_target)
                    env_cam_handles.append(cam_handle)
                for j in range(self.bodycam_count):
                    cam_handle, _ = self.setup_body_cam(env, self.cam_width, self.cam_height, camera_transform[j], self.ee_handles[j])
                    env_cam_handles.append(cam_handle)

                self.all_cam_handles.append(env_cam_handles)


    def prepare_tensors(self):
        # ==== prepare tensors =====
        # from now on, we will use the tensor API to access and control the physics simulation
        self.gym.prepare_sim(self.sim)
        # Initialize lists to hold position and orientation for each Franka
        self.new_init_orn_list = []
        self.new_init_pos_list = []
        
        # Acquire and wrap the rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # Acquire and wrap the DOF state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        for i in range(self.num_frankas):
            print("This is i: ",i)
            # initial hand position and orientation tensors
            init_pos = torch.Tensor(self.init_pos_list[i]).view(self.num_envs, 3)
            init_orn = torch.Tensor(self.init_orn_list[i]).view(self.num_envs, 4)

            
            if args.use_gpu_pipeline:
                init_pos = init_pos.to('cuda:0')
                init_orn = init_orn.to('cuda:0')

            # desired hand positions and orientations
            self.pos_des = init_pos.clone()
            self.orn_des = init_orn.clone()
            
            # hand positions append to list
            self.new_init_orn_list.append(self.orn_des)
            self.new_init_pos_list.append(self.pos_des)

            # Prepare jacobian tensor
            # For franka, tensor shape is (num_envs, 10, 6, 9)
            _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
            self.jacobian = gymtorch.wrap_tensor(_jacobian)
            # Jacobian entries for end effector
            hand_index = self.gym.get_asset_rigid_body_dict(self.franka_asset)["panda_hand"]
            # jacobian entries corresponding to franka hand
            self.j_eef = self.jacobian[:, hand_index - 1, :, :7]
            # Prepare mass matrix tensor
            # For franka, tensor shape is (num_envs, 9, 9)
            _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
            self.mm = gymtorch.wrap_tensor(_massmatrix)
            self.mm = self.mm[:, :7, :7] # only need elements corresponding to the franka arm
            self.kp = 5
            self.kv = 2 * math.sqrt(self.kp)
            dof_vel = self.dof_states[:, 1].view(self.num_envs, 9*self.num_frankas, 1)
            self.trimmed_vel_list = []
            for j in range(self.num_frankas):
                # Extracting the DOF velocities for robot i and trimming to the first 7
                start_idx = j * 9  # Start index for robot i's DOFs
                end_idx = start_idx + 7  # End index for the arm DOFs, exclusive
                trimmed_vel = dof_vel[:, start_idx:end_idx]  # shape [num_envs, 7]
                self.trimmed_vel_list.append(trimmed_vel)



            dof_pos = self.dof_states[:, 0].view(self.num_envs, 9* self.num_frankas, 1)
            

            # Now, separate the positions for each robot within each environment
            self.dof_pos_list = []
            for k in range(self.num_frankas):
                start_idx = k * 9
                robot_dof_pos = dof_pos[:, start_idx:start_idx+9].view(self.num_envs, 9, 1)
                self.dof_pos_list.append(robot_dof_pos)

            if i == 0: # First Franka
                print("got here1")
                self.keyboard_interface = KeyboardInterface(init_pos, init_orn)
                self.pos_action = torch.zeros_like(self.dof_pos_list[i])
            elif i == 1: # Second Franka
                print("got here2")
                custom_pose_actions = ["1", "2", "3", "4", "5", "6"]
                custom_grip_actions = ["7"]
                custom_rot_actions = ["8", "9", "0", "b", "n", "m"]

                self.second_keyboard_interface = KeyboardInterface(
                    init_pos,
                    init_orn,
                    pose_actions=custom_pose_actions,
                    grip_actions=custom_grip_actions,
                    rot_actions=custom_rot_actions
                )
                self.second_pos_action = torch.zeros_like(self.dof_pos_list[i])
            
    def compute_hand_indices(self):
        # Base hand index for the first robot with no cubes
        base_idx = 8
        # For each robot, its hand index increments by 12
        # But each cube increases the hand index of each robot by 1
        return [base_idx + i*12 + self.num_cubes for i in range(self.num_frankas)]

    def get_hand_states(self, rb_states):
        hand_idxs = self.compute_hand_indices()
        pos_cur = rb_states[hand_idxs, :3]
        orn_cur = rb_states[hand_idxs, 3:7]
        return pos_cur, orn_cur


    def run_simulation(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Update jacobian and mass matrix
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
            for _ in self.frankas:

                # self.display_camera_views(self.envs[0], self.all_cam_handles[0])
                box_pos = self.rb_states[self.box_idxs, :3]
                box_rot = self.rb_states[self.box_idxs, 3:7]
                # Get hand poses
                pos_cur, orn_cur = self.get_hand_states(self.rb_states)
                # Store tensors in lists for easy access
                pos_cur_list = [pos_cur[i] for i in range(pos_cur.shape[0])]
                orn_cur_list = [orn_cur[i] for i in range(orn_cur.shape[0])]
                combined_actions = []
                for i in range(self.num_frankas):
                    if i == 0:
                        # check for movements
                        pos_des, orn_des, gripper = self.keyboard_interface.get_action() 
                        curr_pos = pos_cur_list[i].unsqueeze(0)
                        curr_orn = orn_cur_list[i].unsqueeze(0)
                        # Solve for control (Operational Space Control)
                        m_inv = torch.inverse(self.mm[i]) 
                        m_eef = torch.inverse(self.j_eef[i] @ m_inv @ torch.transpose(self.j_eef[i], 0, 1))
                        curr_orn /= torch.norm(curr_orn, dim=-1).unsqueeze(-1)
                        orn_err = orientation_error(orn_des, curr_orn)
                        pos_err = self.kp * (pos_des - curr_pos)
                        if not args.pos_control:
                            pos_err *= 0
                        dpose = torch.cat([pos_err, orn_err], -1)
                        # action Tensor
                        u = torch.transpose(self.j_eef[i], 0, 1) @ m_eef @ (self.kp * dpose).unsqueeze(-1) - self.kv * self.mm[i] @ self.trimmed_vel_list[i]
                        # Assign u to the correct slice of self.pos_action
                        self.pos_action[:, :7] = u
                        self.pos_action[:, 7:9] = torch.Tensor([[gripper,gripper]])
                        # Append the action for the first robot to the combined actions list
                        combined_actions.append(self.pos_action)
                    elif i == 1:
                        pos_des, orn_des, gripper = self.second_keyboard_interface.get_action()
                        curr_pos = pos_cur_list[i].unsqueeze(0)
                        curr_orn = orn_cur_list[i].unsqueeze(0)
                        # Solve for control (Operational Space Control)
                        m_inv = torch.inverse(self.mm[i]) 
                        m_eef = torch.inverse(self.j_eef[i] @ m_inv @ torch.transpose(self.j_eef[i], 0, 1))
                        curr_orn /= torch.norm(curr_orn, dim=-1).unsqueeze(-1)
                        orn_err = orientation_error(orn_des, curr_orn)
                        pos_err = self.kp * (pos_des - curr_pos)
                        if not args.pos_control:
                            pos_err *= 0
                        dpose = torch.cat([pos_err, orn_err], -1)
                        # action Tensor
                        u = torch.transpose(self.j_eef[i], 0, 1) @ m_eef @ (self.kp * dpose).unsqueeze(-1) - self.kv * self.mm[i] @ self.trimmed_vel_list[i]
                        # Assign u to the correct slice of self.pos_action
                        self.second_pos_action[:, :7] = u
                        self.second_pos_action[:, 7:9] = torch.Tensor([[gripper,gripper]])
                        combined_actions.append(self.second_pos_action)
                        
                    # Concatenate the actions for all robots
                    combined_pos_action = torch.cat(combined_actions, dim=1)
                    # Make sure the combined tensor is the correct shape: (num_envs, num_dofs * num_robots, 1)
                    combined_pos_action = combined_pos_action.view(self.num_envs, 9*self.num_frankas, 1)

                    # Set the combined tensor action for both robots
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(combined_pos_action))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            # gym.sync_frame_time(sim)

        print("Done")

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


        
t = TeleEnv(args=args)
t.run_simulation()