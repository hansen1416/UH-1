from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, MOTION_LIB_DIR
from legged_gym.envs.base.base_task import BaseTask
# from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.math import *
from legged_gym.utils import torch_utils

from .h1_2_legged_robot import LeggedRobot, euler_from_quaternion
from .h1_2_legged_robot_config import LeggedRobotCfg

import sys
import cv2
import pdb

sys.path.append(MOTION_LIB_DIR)
from motion_lib_retarget import MotionLibRetarget

class H1_2_Mimic(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        # Initialize the configs needed
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        
        # Initialize device information
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        # Initialize motion loading process 
        self.init_motions(cfg)
        if cfg.motion.num_envs_as_motions:
            self.cfg.env.num_envs = self._motion_lib.num_motions()
        
        # Initialize the base task for this class
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        # Set camera with viewer pose and lookat
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        # Initialize buffers and prepare reward function
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.init_motion_buffers(cfg)

        # Reset each environment for the first step
        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()
    
    def step(self, actions):
        # reindex and send the actions to device
        actions = self.reindex(actions)
        actions.to(self.device)

        # update self.action_history_buf
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        
        # add delay (self.delay won't change afterwards)
        # NOTE: why True? Overided in helpers.py
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            # self.delay = torch.randint(0, 3, (1,), device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        # Adding constraints to the actions in the ankle_pitch joints
        self.actions[:, [4, 10]] = torch.clamp(self.actions[:, [4, 10]], -0.5, 0.5)
        self.actions[:, [5, 11]] = torch.clamp(self.actions[:, [5, 11]], -0.2, 0.2)
        
        # NOTE: setting restraints for the dofs
        indexes = [5, 11, 17, 18, 19, 24, 25, 26]
        self.actions[:, indexes] = 0

        # Set actions, simulate, and fetch results
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for i in torch.topk(self.torques[self.lookat_id], 3).indices.tolist():
        #     print(self.dof_names[i], self.torques[self.lookat_id][i])

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        # self._motion_sync()
        super().post_physics_step()

        # step motion lib, add the timestep of one frame,
        # if one traj ends, reset the _motion_times to 0
        self._motion_times += self._motion_dt
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.
        self.update_demo_obs()
        # self.update_mimic_obs()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_rigid_bodies_demo()
            self.draw_rigid_bodies_actual()

        return

    ######### initialization #########
    # called at self.__init__()
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(
            self.cfg.domain_rand.gravity_rand_interval_s / self.dt
        )
        self.cfg.motion.resample_step_inplace_interval = np.ceil(
            self.cfg.motion.resample_step_inplace_interval_s / self.dt
        )

    # called at self.__init__()
    def init_motions(self, cfg):
        # Initialize the index tensors and initialize the MotionLib object

        '''
            NOTE: Used for the keypoint selection on the handless h1_2 robot

            The 27 dofs refer to the handless h1_2 robot dofs:
                [0: 3]      left_hip_yaw, left_hip_pitch, left_hip_roll
                [3]         left_knee
                [4: 6]      left_ankle_pitch, left_ankle_roll
                [6: 9]      right_hip_yaw, right_hip_pitch, right_hip_roll
                [9]         right_knee
                [10: 12]    right_ankle_pitch, right_ankle_roll
                [12]        torso
                [13: 16]    left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw
                [16: 18]    left_elbow_pitch, left_elbow_roll
                [18: 20]    left_wrist_pitch, left_wrist_yaw
                [20: 23]    right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw
                [23: 25]    right_elbow_pitch, right_elbow_roll
                [25: 27]    right_wrist_pitch, right_wrist_yaw

            The body ids = corresponding dof ids + 1,
            ranging from 1 to 27, where 0 refers to pelvis with a free joint.
        '''
        # Whole body keypoint joint ids, corresponding to _motion_lib.keypoint_trans
        self._key_body_ids_sim = torch.tensor(
            [ 3,  4,  5,    # left hip_roll, knee, ankle_pitch
              9,  10, 11,   # right hip_roll, knee, ankle_pitch
              15, 17, 19,   # left shoulder_roll, elbow_pitch, wrist_pitch
              22, 24, 26],  # right shoulder_roll, elbow_pitch, wrist_pitch
            device=self.device
        )
        # Upper body keypoint joint ids selected from the whole body keypoint joint ids
        self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        
        # Select the subset of 9 upper-body joint ids in the 27-dimensional dof space
        self._dof_ids_subset = torch.tensor([12, 13, 14, 15, 16, 20, 21, 22, 23], device=self.device, dtype=torch.long)
        self._n_demo_dof = self.cfg.env.num_actions             # 27
        self._n_demo_dof_subset = len(self._dof_ids_subset)     # 9

        self._load_motion()

    # called at self.init_motions()
    def _load_motion(self):
        motion_pkl_name = self.cfg.motion.motion_name

        motion_pkl_path = os.path.join(MOTION_LIB_DIR, f"motion_pkl/{motion_pkl_name}")
        motion_cfg_path = os.path.join(MOTION_LIB_DIR, "config/cfg_example.yaml")
        motion_folder_path = os.path.join(MOTION_LIB_DIR, "motion_item/")

        self._motion_lib = MotionLibRetarget(
            motion_pkl_path=motion_pkl_path,
            motion_cfg_path=motion_cfg_path,
            motion_folder_path=motion_folder_path,
            device=self.device,
            regen_pkl=False
        )

    # called at self.__init__(), the same as the one in legged_robot.py
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
  
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 2, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_actions, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)

    # called at self._init_buffers()
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    # called at self.__init__()
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    # called at self.__init__()
    def init_motion_buffers(self, cfg):
        num_motions = self._motion_lib.num_motions()
        # self._motion_ids is a tensor of shape (num_envs) containing the motion ids for each env,
        # each motion ids range from 0 to _motion_lib.num_motions() - 1
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)

        self._motion_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)

        self._motion_dt = self.dt
        self._motion_num_future_steps = self.cfg.env.n_demo_steps
        # shaped (n_demo_steps)
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]   # shaped (num_envs, n_demo)
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]
        # self._curr_mimic_obs_buf = torch.zeros_like(self._curr_demo_obs_buf, device=self.device)

        self._curr_demo_root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._curr_demo_root_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_keybody = torch.zeros((self.num_envs, self._num_key_bodies, 3), device=self.device)     # keep in track of the keybody positions
        self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)                  # flag for in-place motion

        self.dof_term_threshold = 3 * torch.ones(self.num_envs, device=self.device)
        self.keybody_term_threshold = 0.3 * torch.ones(self.num_envs, device=self.device)
        self.yaw_term_threshold = 0.5 * torch.ones(self.num_envs, device=self.device)
        self.height_term_threshold = 0.2 * torch.ones(self.num_envs, device=self.device)

        # self.step_inplace_ids = self.resample_step_inplace_ids()
    
    # called at self.create_sim(), adding video recording
    def _create_envs(self):
        super()._create_envs()

        if self.cfg.env.record_video or self.cfg.env.record_frame:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 640
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(min(self.num_envs, 2)): # NOTE brutely set the max number of cameras to 2
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                if camera_handle == -1:
                    print("Failed to create camera sensor")
                    return
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))

    ######### reset_idx() functions #########
    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
        # RSI
        if self.cfg.motion.motion_curriculum:
            # ep_length = self.episode_length_buf[env_ids] * self.dt
            completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lengths[env_ids]
            completion_rate_mean = completion_rate.mean()
            # if completion_rate_mean > 0.8:
            #     self._max_motion_difficulty = min(self._max_motion_difficulty + 1, 9)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            # elif completion_rate_mean < 0.4:
            #     self._max_motion_difficulty = max(self._max_motion_difficulty - 1, 0)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            relax_ids = completion_rate < 0.3
            strict_ids = completion_rate > 0.9
            # self.dof_term_threshold[env_ids[relax_ids]] += 0.05
            self.dof_term_threshold[env_ids[strict_ids]] -= 0.05
            self.dof_term_threshold.clamp_(1.5, 3)

            self.height_term_threshold[env_ids[relax_ids]] += 0.01
            self.height_term_threshold[env_ids[strict_ids]] -= 0.01
            self.height_term_threshold.clamp_(0.03, 0.1)

            relax_ids = completion_rate < 0.6
            strict_ids = completion_rate > 0.9
            self.keybody_term_threshold[env_ids[relax_ids]] -= 0.05
            self.keybody_term_threshold[env_ids[strict_ids]] += 0.05
            self.keybody_term_threshold.clamp_(0.1, 0.4)

            relax_ids = completion_rate < 0.4
            strict_ids = completion_rate > 0.8
            self.yaw_term_threshold[env_ids[relax_ids]] -= 0.05
            self.yaw_term_threshold[env_ids[strict_ids]] += 0.05
            self.yaw_term_threshold.clamp_(0.1, 0.6)

        self.update_motion_ids(env_ids)

        motion_ids = self._motion_ids[env_ids]
        motion_times = self._motion_times[env_ids]
        
        (
            root_pos,           # [num_envs, 3]
            root_rot,           # [num_envs, 4]
            dof_pos,            # [num_envs, 21]
            root_vel,           # [num_envs, 3]
            root_ang_vel,       # [num_envs, 3]
            dof_vel,            # [num_envs, 21]
            keypoint_trans      # [num_envs, 12, 3]
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids, dof_pos, dof_vel)
        # TODO: figure out whether needs root_ang_vel in resetting the root states
        self._reset_root_states(env_ids, root_vel, root_rot, root_pos[:, 2])

        # TODO: understand why store these tensors
        if init:
            # Initialize the initial value tensors, env_ids must contain all the envs
            self.init_root_pos_global = self.root_states[:, :3].clone()
            self.init_root_pos_global_demo = root_pos[:].clone()
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]
        else:
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()
            self.init_root_pos_global_demo[env_ids] = root_pos[:].clone()
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]

        self._resample_commands(env_ids)  # no resample commands
        # the three lines below represents how to reset the robot states in the simulator
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # print(env_ids)
        # print(self.root_states[env_ids, -3:])

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        self.extras["episode"]["curriculum_dof_term_thresh"] = self.dof_term_threshold.mean()
        self.extras["episode"]["curriculum_keybody_term_thresh"] = self.keybody_term_threshold.mean()
        self.extras["episode"]["curriculum_yaw_term_thresh"] = self.yaw_term_threshold.mean()
        self.extras["episode"]["curriculum_height_term_thresh"] = self.height_term_threshold.mean()
        
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return

    # called at self.reset_idx()
    def update_motion_ids(self, env_ids):
        self._motion_times[env_ids] = self._motion_lib.sample_time(self._motion_ids[env_ids])
        self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])

    # called at self.reset_idx()
    # self.reindex_dof_pos_vel() see the utils part below
    # self._update_terrain_curriculum() see legged_robot.py

    # called at self.reset_idx()
    def  _reset_dofs(self, env_ids, dof_pos, dof_vel):
        # already aligned, reset directly
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # called at self.reset_idx()
    def _reset_root_states(self, env_ids, root_vel=None, root_quat=None, root_height=None):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
            if root_vel is not None:
                self.root_states[env_ids, 7:10] = root_vel[:]
            if root_quat is not None:
                self.root_states[env_ids, 3:7] = root_quat[:]
            if root_height is not None:
                # root_height[root_height < self.base_init_state[2]] = self.base_init_state[2]
                self.root_states[env_ids, 2] = root_height[:] + 0.1
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # called at self.reset_idx()
    # self._resample_commands() see legged_robot.py, seems useless as we don't track 'reward'

    ######### pose_physics_step() functions #########
    # super().post_physics_step() see legged_robot.py

    # called at super().post_physics_step()
    def _update_goals(self):
        # calculate whether to reset the target pos: time up for resetting target pose
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0
        # keep the global x, y position
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]
        # estimate the global x, y position of one step later
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]

        # NOTE: calculate local xy angle due to the current yaw angle, which shows the relative 'forward' direction
        #       this process involves a coordinate transformation given the rotation angle by yaw
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])

        # calculate the target yaw angle
        r, p, y = euler_from_quaternion(self._curr_demo_quat)
        self.target_yaw = y.clone()

    # called at super().post_physics_step() 
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % self.cfg.motion.resample_step_inplace_interval == 0:
            self.resample_step_inplace_ids()
    
    # called at self._post_physics_step_callback()
    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    # called at self._post_physics_step_callback(), seems useless
    def resample_step_inplace_ids(self, ):
        self.step_inplace_ids = torch.rand(self.num_envs, device=self.device) < self.cfg.motion.step_inplace_prob

    # called at super().post_physics_step()
    def check_termination(self):
        """
            Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # reset if unable to track the dof_pos
        dof_dev = self._reward_tracking_demo_dof_pos() < 0.1
        self.reset_buf |= dof_dev

        # reset if tracking the current trajectory is finished
        motion_end = self.episode_length_buf * self.dt >= self._motion_lengths
        self.reset_buf |= motion_end

        # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.time_out_buf |= motion_end

        self.reset_buf |= self.time_out_buf

    # called at super().post_physics_step()
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew #if "demo" not in name else 0  # log demo rew but do not include in additative reward
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # called at super().post_physics_step()
    # self.reset_idx() redefined above
    # self._gather_cur_goals() see legged_robot.py
    # self.update_depth_buffer() see legged_robot.py

    # called at super().post_physics_step()
    def compute_observations(self):
        # motion_id_one_hot = torch.zeros((self.num_envs, self._motion_lib.num_motions()), device=self.device)
        # motion_id_one_hot[torch.arange(self.num_envs, device=self.device), self._motion_ids] = 1.
        
        obs_buf = self.compute_obs_buf()

        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale
        
        obs_demo = self.compute_obs_demo()

        # start from -4, the last 4 observation history
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)

        priv_explicit = torch.cat((
            0*self.base_lin_vel * self.obs_scales.lin_vel,
            # global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1),
        ), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)

        # default = False
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        # update self.obs_his_buf:
        # self.episode_length_buf <= 1 means the episode has just started, so init obs_history_buf with current obs_buf,
        # otherwise, concatenate the current obs_buf with the previous obs_history_buf (at the end)
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],        # keep shape and remove the earliest element
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        # similar to the above, update the contact_buf
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

    # called at self.compute_observations()
    # NOTE: robot proprioception, might need to be altered
    def compute_obs_buf(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        return torch.cat(
            (
                self.base_ang_vel  * self.obs_scales.ang_vel,               # [1,3]
                imu_obs,                                                    # [1,2]
                torch.sin(self.yaw - self.target_yaw)[:, None],             # [1,1]
                torch.cos(self.yaw - self.target_yaw)[:, None],             # [1,1] 
                self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.reindex(self.action_history_buf[:, -1]),
                self.reindex_feet(self.contact_filt.float()*0-0.5),
            ), dim=-1
        )
    
    # called at self.compute_observations()
    def compute_obs_demo(self):
        obs_demo = self._next_demo_obs_buf.clone()
        # FIXME: set the velocity of in-place motion to 0, the line below is not correct
        # obs_demo[self._in_place_flag, self._n_demo_dof:self._n_demo_dof+3] = 0
        return obs_demo

    # called at self.post_physics_step()
    def update_demo_obs(self):
        demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [1, n_demo_steps=2] + [num_envs, 1] -> [num_envs, n_demo_steps=2]

        (
            root_pos,           # [n_demo_steps * num_envs, 3]
            root_rot,           # [n_demo_steps * num_envs, 4]
            dof_pos,            # [n_demo_steps * num_envs, 21]
            root_vel,           # [n_demo_steps * num_envs, 3]
            root_ang_vel,       # [n_demo_steps * num_envs, 3]
            dof_vel,            # [n_demo_steps * num_envs, 21]
            keypoint_trans      # [n_demo_steps * num_envs, 12, 3]
        ) = self._motion_lib.get_motion_state(
            self._motion_ids.repeat_interleave(self._motion_num_future_steps),    # [id1, id1, id2, id2, ...] to be reviewed to [num_envs, n_demo_steps=2]
            demo_motion_times.flatten()
        )
        
        self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        self._curr_demo_quat[:]     = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]
        self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]

        self._curr_demo_keybody[:] = keypoint_trans[:, self._key_body_ids_sim_subset].view(
            self.num_envs, self._motion_num_future_steps, self._num_key_bodies, 3)[:, 0, :, :]
        self._in_place_flag = 0*(torch.norm(self._curr_demo_root_vel, dim=-1) < 0.2)

        demo_obs = build_demo_observations(
            root_pos, root_rot, root_vel, root_ang_vel,
            dof_pos, dof_vel, keypoint_trans
        )
        
        self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]

    # called at self.post_physics_step()
    # self.draw_rigid_bodies_demo() see the utils part below
    # self.draw_rigid_bodies_actual() see the utils part below
    # self._draw_goals() see the utils part below

    ######### step() functions #########
    # self.reindex() see legged_robot.py

    ######### utils #########
    def draw_rigid_bodies_demo(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            # curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
            curr_demo_xyz = self.root_states[:, :3]
        
        curr_quat = torch.zeros(self._curr_demo_quat.shape, device=self.device)
        curr_quat[:, -1] = 1
        # curr_quat = quat_inv_rotate(self._curr_demo_quat, self.base_quat)
        
        global_body_pos = local_to_global(curr_quat, local_body_pos, curr_demo_xyz)
        # global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)
        for i in range(global_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def draw_rigid_bodies_actual(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(1, 0, 0))
        rigid_body_pos = self.rigid_body_states[:, self._key_body_ids_sim, :3].clone()
        for i in range(rigid_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(rigid_body_pos[self.lookat_id, i, 0], rigid_body_pos[self.lookat_id, i, 1], rigid_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_goals(self, ):
        demo_geom = gymutil.WireframeSphereGeometry(0.2, 32, 32, None, color=(1, 0, 0))
        
        pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
        # print(self._curr_demo_obs_buf[self.lookat_id, 2*self.num_dof:2*self.num_dof+3])
        # demo_pos = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[self.lookat_id]
        # pose = gymapi.Transform(gymapi.Vec3(demo_pos[0], demo_pos[1], demo_pos[2]), r=None)
        # gymutil.draw_lines(demo_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            norm = torch.norm(self._curr_demo_root_vel[:, :2], dim=-1, keepdim=True)
            target_vec_norm = self._curr_demo_root_vel[:, :2] / (norm + 1e-5)
            for i in range(5):
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def render_record(self, mode="rgb_array"):
        if self.global_counter % 2 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            imgs = []
            for i in range(min(self.num_envs, 2)):
                cam = self._rendering_camera_handles[i]
                root_pos = self.root_states[i, :3].cpu().numpy()
                cam_pos = root_pos + np.array([0, -2, 0.3])
                self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
                
                img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
                w, h = img.shape
                imgs.append(img.reshape([w, h // 4, 4]))
            return imgs
        return None

    ######### rewards #########
    def _reward_tracking_demo_goal_vel(self):
        norm = torch.norm(self._curr_demo_root_vel[:, :3], dim=-1, keepdim=True)
        target_vec_norm = self._curr_demo_root_vel[:, :3] / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:10]
        norm_squeeze = norm.squeeze(-1)
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), norm_squeeze) / (norm_squeeze + 1e-5)

        rew_zeros = torch.exp(-4*torch.norm(cur_vel, dim=-1))
        small_cmd_ids = (norm<0.1).squeeze(-1)
        rew[small_cmd_ids] = rew_zeros[small_cmd_ids]
        # return torch.exp(-2 * torch.norm(cur_vel - self._curr_demo_root_vel[:, :2], dim=-1))
        return rew.squeeze(-1)

    # not altered, command based, seems useless
    def _reward_tracking_vx(self):
        rew = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        # print("vx rew", rew, self.base_lin_vel[:, 0], self.commands[:, 0])
        return rew
    
    # not altered, command based, seems useless
    def _reward_tracking_ang_vel(self):
        rew = torch.minimum(self.base_ang_vel[:, 2], self.commands[:, 2]) / (self.commands[:, 2] + 1e-5)
        return rew
    
    def _reward_tracking_demo_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        # print("yaw rew", rew, self.target_yaw, self.yaw)
        return rew

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        # print("lower dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 0])
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # print("upper dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 1])
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_demo_dof_pos(self):
        demo_dofs_all = self._curr_demo_obs_buf[:, :self._n_demo_dof]
        demo_dofs = demo_dofs_all[:, self._dof_ids_subset]
        dof_pos = self.dof_pos[:, self._dof_ids_subset]

        rew = torch.exp(-0.7 * torch.norm((dof_pos - demo_dofs), dim=1))
        # print(rew[self.lookat_id].cpu().numpy())

        return rew

    # def _reward_tracking_demo_dof_vel(self):
    #     demo_dof_vel = self._curr_demo_obs_buf[:, self.num_dof:self.num_dof*2]
    #     rew = torch.exp(- 0.01 * torch.norm(self.dof_vel - demo_dof_vel, dim=1))
    #     return rew
    
    def _reward_stand_still(self):
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :13], dim=1)
        dof_vel_error = torch.norm(self.dof_vel[:, :13], dim=1)
        rew = torch.exp(- 0.1 * dof_vel_error) * torch.exp(- dof_pos_error) 
        rew[~self._in_place_flag] = 0
        return rew
    
    def _reward_tracking_lin_vel(self):
        demo_vel = self._curr_demo_obs_buf[:, self._n_demo_dof:self._n_demo_dof+3]
        # demo_vel[self._in_place_flag] = 0
        rew = torch.exp(- 4 * torch.norm(self.base_lin_vel - demo_vel, dim=-1))
        return rew

    def _reward_tracking_demo_ang_vel(self):
        demo_ang_vel = self._curr_demo_obs_buf[:, self._n_demo_dof+3:self._n_demo_dof+6]
        rew = torch.exp(-torch.norm(self.base_ang_vel - demo_ang_vel, dim=1))
        return rew

    def _reward_tracking_demo_roll_pitch(self):
        demo_roll_pitch = self._curr_demo_obs_buf[:, self._n_demo_dof+6:self._n_demo_dof+8]
        cur_roll_pitch = torch.stack((self.roll, self.pitch), dim=1)
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
        return rew
    
    def _reward_tracking_demo_height(self):
        demo_height = self._curr_demo_obs_buf[:, self._n_demo_dof+8]
        cur_height = self.root_states[:, 2]
        rew = torch.exp(- 4 * torch.abs(cur_height - demo_height))
        return rew
    
    def _reward_tracking_demo_key_body(self):
        # demo_key_body_pos_local = self._curr_demo_obs_buf[:, self.num_dof*2+8:].view(self.num_envs, self._num_key_bodies, 3)[:,self._key_body_ids_sim_subset,:].view(self.num_envs, -1)
        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)
        
        # print(f"demo height: {self._curr_demo_root_pos[:, 2]}, cur height: {self.root_states[:, 2]}")
        
        # tracks the xyz position of the upper body to fit the motion data
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            # set the current global keypose with the current root xy pose and the demo root z pose
            # curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
            curr_demo_xyz = self.root_states[:, :3]
        
        # set the global demo keypose by transforming the local keypose,
        # which is based on the coordinate transformation by the quat rotation
        # the rotation is described by the demo quaternion

        # NOTE: demo_key_body_pos_local is already based on the demo quaternion, so no need to rotate
        curr_quat = torch.zeros(self._curr_demo_quat.shape, device=self.device)
        curr_quat[:, -1] = 1

        # rotate from target ori to current ori
        # curr_quat = quat_inv_rotate(self._curr_demo_quat, self.base_quat)

        demo_global_body_pos = local_to_global(curr_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        # demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        # sample the corresponding key body positions from the rigid body states: sampled index in dim 1 and sampled xyz pose in dim 2
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)

        rew = torch.exp(-torch.norm(cur_global_body_pos - demo_global_body_pos, dim=1))
        # print("key body rew", rew[self.lookat_id].cpu().numpy())
        return rew

    def _reward_tracking_mul(self):
        rew_key_body = self._reward_tracking_demo_key_body()
        rew_roll_pitch = self._reward_tracking_demo_roll_pitch()
        rew_ang_vel = self._reward_tracking_demo_yaw()
        # rew_dof_vel = self._reward_tracking_demo_dof_vel()
        rew_dof_pos = self._reward_tracking_demo_dof_pos()
        # rew_goal_vel = self._reward_tracking_lin_vel()#self._reward_tracking_demo_goal_vel()
        rew = rew_key_body * rew_roll_pitch * rew_ang_vel * rew_dof_pos# * rew_dof_vel
        # print(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+3][self.lookat_id], self.base_lin_vel[self.lookat_id])
        return rew

    def _reward_feet_drag(self):
        # print(contact_bool)
        # contact_forces = self.contact_forces[:, self.feet_indices, 2]
        # print(contact_forces[self.lookat_id], self.force_sensor_tensor[self.lookat_id, :, 2])
        # print(self.contact_filt[self.lookat_id])
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.contact_filt * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        # print(rew[self.lookat_id].cpu().numpy(), self.contact_filt[self.lookat_id].cpu().numpy(), feet_xy_vel[self.lookat_id].cpu().numpy())
        return rew
    
    def _reward_energy(self):
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        rew_airTime[self._in_place_flag] = 0
        return rew_airTime

    def _reward_feet_height(self):
        feet_height = self.rigid_body_states[:, self.feet_indices, 2]
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
        rew[self._in_place_flag] = 0
        # print("height: ", rew[self.lookat_id])
        return rew
    
    def _reward_feet_force(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < 500] = 0
        rew[rew > 500] -= 500
        rew[self._in_place_flag] = 0
        # print(rew[self.lookat_id])
        # print(self.dof_names)
        return rew

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :13], dim=1)
        return dof_error
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, keypoint_trans_select):
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    local_root_vel = quat_rotate_inverse(root_rot, root_vel)

    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos,              # 27
                      local_root_vel,       # 3
                      local_root_ang_vel,   # 3
                      roll[:, None],        # 1
                      pitch[:, None],       # 1
                      root_pos[:, 2:3],     # 1
                      keypoint_trans_select.view(keypoint_trans_select.shape[0], -1)), dim=-1) # 12*3
# return length 27 + 3 + 3 + 1 + 1 + 1 + 12*3

@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
    dof = dof.clone()
    # reindex in 21
    dof[:, indices_sim] = dof[:, indices_motion]
    # from 21 to 19
    return dof[:, valid_dof_body_ids]

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))
    return local_pos_delta.squeeze(-1)

@torch.jit.script
def quat_inv_rotate(quat_original, quat_target):
    quat_inverse = torch.cat([-quat_original[:, :3], quat_original[:, 3:4]], dim=-1)
    return quat_mul(quat_inverse, quat_target)

# NOTE: self._demo_obs_buf shaped (self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo),
#       where self.cfg.env.n_demo = 27 + 3 + 3 + 1 + 1 + 1 + 12*3,
#       but the last 12*3 representing keypoint_trans is not used in the reward functions,
#       keypoint_trans is tracked by self._curr_demo_keybody