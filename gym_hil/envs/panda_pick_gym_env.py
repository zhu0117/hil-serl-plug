#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickCubeGymEnv(FrankaGymEnv):
    """Environment for a Panda robot manipulating a plug-like object.

    Kept class name for compatibility with existing gym_hil registration.
    Success logic: top-down insertion success (not lift).
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
    ):
        self.reward_type = reward_type

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        # Task-specific setup
        self._block_z = float(self._model.geom("block").size[2])
        self._random_block_position = random_block_position

        # ---------- insertion success parameters ----------
        self._xy_tol = 0.0045                 # 4.5 mm XY center tolerance
        self._insert_depth_thresh = 0.010     # 10 mm insertion depth (based on PLUG BOTTOM)
        self._tilt_tol_deg = 20.0             # relaxed from 15 -> 20 deg
        self._tilt_tol_cos = np.cos(np.deg2rad(self._tilt_tol_deg))
        self._success_hold_steps = 3          # debounce steps
        self._success_counter = 0

        self._plug_half_h = self._block_z

        # Fallback socket entry if scene sensor is missing (match scene.xml site z=0.050)
        self._socket_entry_fallback = np.array([0.56, 0.0, 0.050], dtype=np.float64)

        # ---------- bounds / early termination controls ----------
        self._bounds_margin = 0.15
        self._enable_bounds_termination = True

        # ---------- debug controls ----------
        self._debug_every_step = False
        self._debug_print_period = 10
        self._step_id = 0

        # Observation space
        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def _get_socket_entry_pos(self) -> np.ndarray:
        """Read socket entry position from sensor; fallback if absent."""
        try:
            return self._data.sensor("socket_entry_pos").data.copy()
        except KeyError:
            return self._socket_entry_fallback.copy()

    def _insertion_metrics(self):
        """Compute insertion-related metrics."""
        block_pos = self._data.sensor("block_pos").data.copy()   # geom center
        block_quat = self._data.sensor("block_quat").data.copy() # (w, x, y, z)
        socket_entry = self._get_socket_entry_pos()

        # XY center alignment error
        xy_err = float(np.linalg.norm(block_pos[:2] - socket_entry[:2]))

        # Orientation alignment: plug local z-axis vs world z-axis
        R = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R.reshape(-1), block_quat)
        plug_axis_world = R[:, 2]
        cos_to_vertical = abs(float(np.dot(plug_axis_world, np.array([0.0, 0.0, 1.0]))))

        # Insertion depth (UPDATED):
        # use plug bottom face instead of top face.
        # plug bottom z ~= center_z - half_height
        plug_bottom_z = float(block_pos[2] - self._plug_half_h)
        socket_entry_z = float(socket_entry[2])
        insertion_depth = socket_entry_z - plug_bottom_z
        # > 0 means plug bottom is below entry plane

        return xy_err, cos_to_vertical, insertion_depth, block_pos, socket_entry

    def _bounds_check(self, block_pos: np.ndarray):
        """Return detailed bounds diagnostics."""
        x_min, y_min = (_SAMPLING_BOUNDS[0] - self._bounds_margin)
        x_max, y_max = (_SAMPLING_BOUNDS[1] + self._bounds_margin)

        out_x_low = bool(block_pos[0] < x_min)
        out_x_high = bool(block_pos[0] > x_max)
        out_y_low = bool(block_pos[1] < y_min)
        out_y_high = bool(block_pos[1] > y_max)

        exceeded_bounds = bool(out_x_low or out_x_high or out_y_low or out_y_high)

        return {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
            "out_x_low": out_x_low,
            "out_x_high": out_x_high,
            "out_y_low": out_y_low,
            "out_y_high": out_y_high,
            "exceeded_bounds": exceeded_bounds,
        }

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        self.reset_robot()

        # Sample plug XY
        if self._random_block_position:
            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        else:
            block_xy = np.asarray([0.5, 0.0])
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)

        mujoco.mj_forward(self._model, self._data)

        self._success_counter = 0
        self._step_id = 0
        self._z_init = float(self._data.sensor("block_pos").data[2])

        obs = self._compute_observation()
        info = {
            "reset_block_xy": block_xy.astype(float).tolist(),
            "socket_entry_pos": self._get_socket_entry_pos().astype(float).tolist(),
            "bounds_margin": float(self._bounds_margin),
            "bounds_termination_enabled": bool(self._enable_bounds_termination),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment with detailed debug info."""
        self.apply_action(action)
        self._step_id += 1

        obs = self._compute_observation()
        success = self._is_success()
        rew = self._compute_reward(success=success)

        # insertion diagnostics
        xy_err, cos_to_vertical, insertion_depth, block_pos, socket_entry = self._insertion_metrics()
        bounds_info = self._bounds_check(block_pos)
        exceeded_bounds = bounds_info["exceeded_bounds"]

        # termination logic
        if self._enable_bounds_termination:
            terminated = bool(success or exceeded_bounds)
        else:
            terminated = bool(success)

        terminated_reason = "none"
        if success:
            terminated_reason = "success"
        elif exceeded_bounds and self._enable_bounds_termination:
            terminated_reason = "exceeded_bounds"

        info: Dict[str, Any] = {
            "succeed": bool(success),
            "terminated_reason": terminated_reason,

            "xy_err": float(xy_err),
            "xy_tol": float(self._xy_tol),
            "cos_to_vertical": float(cos_to_vertical),
            "tilt_tol_cos": float(self._tilt_tol_cos),
            "insertion_depth": float(insertion_depth),
            "insert_depth_thresh": float(self._insert_depth_thresh),

            "block_pos_x": float(block_pos[0]),
            "block_pos_y": float(block_pos[1]),
            "block_pos_z": float(block_pos[2]),
            "socket_entry_x": float(socket_entry[0]),
            "socket_entry_y": float(socket_entry[1]),
            "socket_entry_z": float(socket_entry[2]),

            "success_counter": int(self._success_counter),
            "success_hold_steps": int(self._success_hold_steps),

            "bounds_margin": float(self._bounds_margin),
            "bounds_termination_enabled": bool(self._enable_bounds_termination),
            **bounds_info,
        }

        if self._debug_every_step and (self._step_id % self._debug_print_period == 0):
            print(
                f"[STEP {self._step_id}] "
                f"xy_err={xy_err:.4f}/{self._xy_tol:.4f}, "
                f"cos={cos_to_vertical:.3f}/{self._tilt_tol_cos:.3f}, "
                f"depth={insertion_depth:.4f}/{self._insert_depth_thresh:.4f}, "
                f"bounds={exceeded_bounds}, success={success}"
            )

        if terminated:
            print(
                f"[EP_END] step={self._step_id} reason={terminated_reason} "
                f"block=({block_pos[0]:.3f},{block_pos[1]:.3f},{block_pos[2]:.3f}) "
                f"socket=({socket_entry[0]:.3f},{socket_entry[1]:.3f},{socket_entry[2]:.3f}) "
                f"xy_err={xy_err:.4f} cos={cos_to_vertical:.3f} depth={insertion_depth:.4f} "
                f"bounds={exceeded_bounds}"
            )

        return obs, rew, terminated, False, info

    def _compute_observation(self) -> dict:
        """Compute current observation."""
        robot_state = self.get_robot_state().astype(np.float32)
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)

        if self.image_obs:
            front_view, wrist_view = self.render()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }

        return observation

    def _compute_reward(self, success: bool | None = None) -> float:
        """Insertion-oriented reward.

        For sparse reward, avoid calling _is_success() here to prevent
        double-counting success in the same environment step.
        """
        xy_err, cos_to_vertical, insertion_depth, block_pos, _ = self._insertion_metrics()

        if self.reward_type == "dense":
            r_xy = np.exp(-120.0 * xy_err)

            r_ori = np.clip(
                (cos_to_vertical - self._tilt_tol_cos) / (1.0 - self._tilt_tol_cos + 1e-6),
                0.0,
                1.0,
            )

            r_depth = np.clip(insertion_depth / self._insert_depth_thresh, 0.0, 1.0)

            z_pen = -0.1 if block_pos[2] < 0.01 else 0.0

            return float(0.45 * r_xy + 0.20 * r_ori + 0.35 * r_depth + z_pen)
        else:
            if success is None:
                success = self._success_counter >= self._success_hold_steps
            return float(success)

    def _is_success(self) -> bool:
        """Insertion success: aligned + near-vertical + deep enough, held for several steps."""
        xy_err, cos_to_vertical, insertion_depth, _, _ = self._insertion_metrics()

        cond_xy = bool(xy_err < self._xy_tol)
        cond_ori = bool(cos_to_vertical > self._tilt_tol_cos)
        cond_depth = bool(insertion_depth > self._insert_depth_thresh)

        success_now = cond_xy and cond_ori and cond_depth

        if success_now:
            self._success_counter += 1
        else:
            self._success_counter = 0

        return self._success_counter >= self._success_hold_steps


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper

    env = PandaPickCubeGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    obs, info = env.reset()
    print("[RESET]", info)
    for _ in range(200):
        action = np.random.uniform(-1, 1, 7)
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("[TERMINATED INFO]", info)
            env.reset()
    env.close()
