import numpy as np
import torch

import engines.engine as engine
import envs.smp_env as smp_env
import envs.task_dodgeball_env as task_dodgeball_env
import util.torch_util as torch_util


class TaskSoccerEnv(task_dodgeball_env.TaskDodgeballEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._tar_speed_min = float(env_config.get("tar_speed_min", 1.0))
        self._tar_speed_max = float(env_config.get("tar_speed_max", 4.0))
        self._tar_change_time_min = float(env_config.get("tar_change_time_min", 4.0))
        self._tar_change_time_max = float(env_config.get("tar_change_time_max", 7.0))

        self._ball_spawn_dist_min = float(env_config.get("ball_spawn_dist_min", 1.0))
        self._ball_spawn_dist_max = float(env_config.get("ball_spawn_dist_max", 2.0))
        self._ball_spawn_speed_min = float(env_config.get("ball_spawn_speed_min", 0.0))
        self._ball_spawn_speed_max = float(env_config.get("ball_spawn_speed_max", 2.0))

        self._reward_ball_vel_w = float(env_config.get("reward_ball_vel_w", 0.9))
        self._reward_ball_dist_w = float(env_config.get("reward_ball_dist_w", 0.1))
        self._reward_ball_vel_scale = float(env_config.get("reward_ball_vel_scale", 0.5))
        self._reward_ball_dist_scale = float(env_config.get("reward_ball_dist_scale", 0.5))

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        num_envs = self.get_num_envs()
        self._tar_ball_dir = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._tar_ball_dir[..., 0] = 1.0
        self._tar_ball_speed = torch.ones([num_envs], device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (len(env_ids) > 0):
            self._reset_tar(env_ids)
            # Spawn the ball immediately at episode start (next _update_task tick).
            self._proj_trigger_times[env_ids] = self._time_buf[env_ids].unsqueeze(-1)
        return

    def _reset_tar(self, env_ids):
        n = env_ids.shape[0]
        rand_theta = 2 * np.pi * torch.rand(n, device=self._device) - np.pi
        self._tar_ball_dir[env_ids, 0] = torch.cos(rand_theta)
        self._tar_ball_dir[env_ids, 1] = torch.sin(rand_theta)
        self._tar_ball_speed[env_ids] = (
            (self._tar_speed_max - self._tar_speed_min)
            * torch.rand(n, device=self._device)
            + self._tar_speed_min
        )
        rand_dt = (
            (self._tar_change_time_max - self._tar_change_time_min)
            * torch.rand(n, device=self._device)
            + self._tar_change_time_min
        )
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt
        return

    def _update_task(self):
        reset_mask = self._time_buf >= self._tar_change_times
        reset_env_ids = reset_mask.nonzero(as_tuple=False).flatten()
        if (len(reset_env_ids) > 0):
            self._reset_tar(reset_env_ids)

        trigger_mask = self._time_buf.unsqueeze(-1) >= self._proj_trigger_times
        trigger_ids = trigger_mask.nonzero(as_tuple=False)
        if (trigger_ids.shape[0] > 0):
            self._launch_projectiles(env_ids=trigger_ids[:, 0], proj_ids=trigger_ids[:, 1])
        return

    def _launch_projectiles(self, env_ids, proj_ids):
        n = env_ids.shape[0]
        if (n == 0):
            return

        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)[env_ids]

        rand_theta = 2 * np.pi * torch.rand(n, device=self._device)
        rand_dist = (
            (self._ball_spawn_dist_max - self._ball_spawn_dist_min)
            * torch.rand(n, device=self._device)
            + self._ball_spawn_dist_min
        )
        spawn_pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        spawn_pos[:, 0] = char_root_pos[:, 0] + rand_dist * torch.cos(rand_theta)
        spawn_pos[:, 1] = char_root_pos[:, 1] + rand_dist * torch.sin(rand_theta)
        spawn_pos[:, 2] = self._proj_radius

        rand_vel_theta = 2 * np.pi * torch.rand(n, device=self._device)
        rand_vel_speed = (
            (self._ball_spawn_speed_max - self._ball_spawn_speed_min)
            * torch.rand(n, device=self._device)
            + self._ball_spawn_speed_min
        )
        spawn_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        spawn_vel[:, 0] = rand_vel_speed * torch.cos(rand_vel_theta)
        spawn_vel[:, 1] = rand_vel_speed * torch.sin(rand_vel_theta)

        spawn_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        spawn_rot[:, 3] = 1.0
        spawn_ang_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

        for local_proj_id, proj_obj_id in enumerate(self._proj_ids):
            curr_mask = proj_ids == local_proj_id
            if (curr_mask.any().item()):
                curr_env_ids = env_ids[curr_mask]
                self._engine.set_root_pos(curr_env_ids, proj_obj_id, spawn_pos[curr_mask])
                self._engine.set_root_rot(curr_env_ids, proj_obj_id, spawn_rot[curr_mask])
                self._engine.set_root_vel(curr_env_ids, proj_obj_id, spawn_vel[curr_mask])
                self._engine.set_root_ang_vel(curr_env_ids, proj_obj_id, spawn_ang_vel[curr_mask])
                self._prev_proj_vel[curr_env_ids, local_proj_id] = spawn_vel[curr_mask]

        # One spawn per episode: push next trigger past episode end so it never re-fires.
        self._proj_trigger_times[env_ids, proj_ids] = float("inf")
        return

    def _compute_obs(self, env_ids=None):
        obs = smp_env.SMPEnv._compute_obs(self, env_ids)

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        proj_pos, proj_vel = self._get_proj_states()

        tar_ball_dir = self._tar_ball_dir
        tar_ball_speed = self._tar_ball_speed

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            proj_pos = proj_pos[env_ids]
            proj_vel = proj_vel[env_ids]
            tar_ball_dir = tar_ball_dir[env_ids]
            tar_ball_speed = tar_ball_speed[env_ids]

        task_obs = compute_soccer_observations(root_pos=root_pos,
                                               root_rot=root_rot,
                                               proj_pos=proj_pos,
                                               proj_vel=proj_vel,
                                               tar_ball_dir=tar_ball_dir,
                                               tar_ball_speed=tar_ball_speed)
        obs = torch.cat([obs, task_obs], dim=-1)
        return obs

    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        proj_pos, proj_vel = self._get_proj_states()

        self._reward_buf[:] = compute_soccer_reward(
            proj_pos=proj_pos,
            proj_vel=proj_vel,
            root_pos=root_pos,
            tar_ball_dir=self._tar_ball_dir,
            tar_ball_speed=self._tar_ball_speed,
            ball_vel_w=self._reward_ball_vel_w,
            ball_dist_w=self._reward_ball_dist_w,
            ball_vel_scale=self._reward_ball_vel_scale,
            ball_dist_scale=self._reward_ball_dist_scale,
        )
        return

    def _update_done(self):
        smp_env.SMPEnv._update_done(self)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_soccer_observations(root_pos, root_rot, proj_pos, proj_vel,
                                tar_ball_dir, tar_ball_speed):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    num_proj = proj_pos.shape[1]
    heading_inv_rot_proj = heading_inv_rot.unsqueeze(-2).repeat((1, num_proj, 1))

    proj_relative_pos = proj_pos - root_pos.unsqueeze(-2)
    local_proj_pos = torch_util.quat_rotate(heading_inv_rot_proj, proj_relative_pos)
    local_proj_vel = torch_util.quat_rotate(heading_inv_rot_proj, proj_vel)
    proj_obs = torch.cat([local_proj_pos, local_proj_vel], dim=-1)
    proj_obs_flat = proj_obs.reshape(proj_obs.shape[0], -1)

    tar_dir3d = torch.cat([tar_ball_dir, torch.zeros_like(tar_ball_dir[..., 0:1])], dim=-1)
    local_tar_dir = torch_util.quat_rotate(heading_inv_rot, tar_dir3d)[..., 0:2]
    tar_speed = tar_ball_speed.unsqueeze(-1)

    obs = torch.cat([proj_obs_flat, local_tar_dir, tar_speed], dim=-1)
    return obs


@torch.jit.script
def compute_soccer_reward(proj_pos, proj_vel, root_pos, tar_ball_dir, tar_ball_speed,
                          ball_vel_w, ball_dist_w, ball_vel_scale, ball_dist_scale):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    tar_vel = tar_ball_speed.unsqueeze(-1) * tar_ball_dir

    ball_vel_xy = proj_vel[..., 0, 0:2]
    ball_vel_err = torch.sum(torch.square(tar_vel - ball_vel_xy), dim=-1)
    ball_vel_reward = torch.exp(-ball_vel_scale * ball_vel_err)

    ball_xy = proj_pos[..., 0, 0:2]
    dist_sq = torch.sum(torch.square(ball_xy - root_pos[..., 0:2]), dim=-1)
    ball_dist_reward = torch.exp(-ball_dist_scale * dist_sq)

    reward = ball_vel_w * ball_vel_reward + ball_dist_w * ball_dist_reward
    return reward
