#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
Evaluation script for a trained SAC HIL policy in the gym_hil simulation environment.

This script loads a saved SAC policy checkpoint and evaluates it over multiple episodes,
reporting per-episode rewards and overall success rate.

Usage:
    conda activate lerobot
    cd /home/zhu/lerobot-main/lerobot-main
    python -m lerobot.rl.eval_hil_sim \\
        --config_path src/lerobot/rl/hil_sim_eval_config.json

You can override the number of evaluation episodes at runtime:
    python -m lerobot.rl.eval_hil_sim \\
        --config_path src/lerobot/rl/hil_sim_eval_config.json \\
        --n_eval_episodes 50

The pretrained checkpoint path is read from `policy.pretrained_path` in the config JSON.
"""

import logging

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.types import TransitionKey

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)

# ╔══════════════════════════════════════════════════════════════╗
# ║  【可修改】评估的 episode 总数。                               ║
# ║  也可直接修改 JSON 配置文件，在此处调整默认值即可。            ║
# ╚══════════════════════════════════════════════════════════════╝
N_EVAL_EPISODES = 20


def run_eval(cfg: TrainRLServerPipelineConfig, n_episodes: int = N_EVAL_EPISODES) -> None:
    """Run policy evaluation for *n_episodes* episodes and log summary statistics."""
    device = cfg.policy.device

    # ── 环境初始化 ────────────────────────────────────────────────────────────
    # 【通常不需要修改】环境和处理器流水线由 JSON 配置文件中的 "env" 节驱动，
    # 切换任务时只需更新 JSON，不用改这里。
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, device)

    # ── 加载策略 ──────────────────────────────────────────────────────────────
    # 【通常不需要修改】模型路径由 JSON 中 policy.pretrained_path 指定。
    # 切换模型时只需更新 JSON，不用改这里。
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy = policy.eval()

    print(f"\nRunning {n_episodes} evaluation episodes …\n")

    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    for ep in range(n_episodes):
        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()

        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)

        episode_reward = 0.0
        step_count = 0

        while True:
            observation = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            with torch.no_grad():
                action = policy.select_action(batch=observation)

            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            episode_reward += float(new_transition[TransitionKey.REWARD])
            step_count += 1

            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)
            if done or truncated:
                break

            transition = new_transition

        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)

    # ── 逐轮明细表 ────────────────────────────────────────────────────────────
    # 【可修改】判断"成功"的条件：当前为 reward > 0 即算成功。
    # 如果你的任务 reward 设计不同（例如 reward >= 0.5 才算成功），在下方两处同步修改。
    col_w = [8, 9, 8, 7]
    sep   = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    hdr   = "| {:^{}} | {:^{}} | {:^{}} | {:^{}} |".format(
        "Episode", col_w[0], "Status", col_w[1], "Reward", col_w[2], "Steps", col_w[3]
    )
    print("\n" + sep)
    print(hdr)
    print(sep)
    for i, (r, s) in enumerate(zip(episode_rewards, episode_steps)):
        status = "SUCCESS" if r > 0 else "fail"   # 【可修改】成功判断阈值
        print("| {:^{}} | {:^{}} | {:^{}.3f} | {:^{}} |".format(
            i + 1, col_w[0], status, col_w[1], r, col_w[2], s, col_w[3]
        ))
    print(sep)

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    n_success  = sum(r > 0 for r in episode_rewards)   # 【可修改】与上方阈值保持一致
    avg_reward = sum(episode_rewards) / n_episodes
    avg_steps  = sum(episode_steps) / n_episodes
    min_steps  = min(episode_steps)
    max_steps  = max(episode_steps)

    W = 62
    max_val = W - 23

    def row(label, value):
        value = str(value)
        if len(value) > max_val:
            value = "\u2026" + value[-(max_val - 1):]
        return f"| {label:<20} {value:<{W - 22}} |"

    print("\n+" + "=" * W + "+")
    print(f"| {'EVALUATION SUMMARY':^{W}} |")
    print("+" + "=" * W + "+")
    print(row("Model path",        str(cfg.policy.pretrained_path)))
    print(row("Environment",       f"{cfg.env.name} / {cfg.env.task}"))
    print("+" + "-" * W + "+")
    print(row("Total episodes",    str(n_episodes)))
    print(row("Successes",         f"{n_success}  ({100 * n_success / n_episodes:.1f}%)"))
    print(row("Avg reward",        f"{avg_reward:.4f}"))
    print(row("Avg steps/episode", f"{avg_steps:.1f}"))
    print(row("Min steps",         str(min_steps)))
    print(row("Max steps",         str(max_steps)))
    print("+" + "=" * W + "+")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    # 【可修改】如需在命令行直接改 episode 数而不动 JSON，可在此处硬编码，
    # 例如：run_eval(cfg, n_episodes=50)
    run_eval(cfg, n_episodes=N_EVAL_EPISODES)


if __name__ == "__main__":
    main()
