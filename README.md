# HIL-SERL 仿真实验使用指南

> 框架：`gym_hil` 仿真环境 + SAC 策略 + LeRobot 工具链

由于此实验环境在官方的基础上做了更改，在开始前我们需要替换用pip下载下来的gym_hil文件夹。详情请参见gym_hil文件夹下的readme.md文档
---

## 快速开始

```bash
conda activate lerobot
cd /home/zhu/lerobot-main/lerobot-main
```

---

## 配置文件总览

| 文件 | 用途 |
|------|------|
| `src/lerobot/rl/hil_sim_config.json` | 观察环境 / 录制演示数据 |
| `src/lerobot/rl/hil_sim_train_config.json` | SAC 强化学习训练 |
| `src/lerobot/rl/hil_sim_eval_config.json` | 评估已训练的模型 |

---

## 1. 基本使用模式

打开仿真窗口，用键盘手动控制机器人，或回放已录制的演示轨迹。

**配置文件：** `hil_sim_config.json`

| 字段 | 说明 |
|------|------|
| `mode` | `null`=仅观察，`"record"`=录制，`"replay"`=回放 |
| `env.task` | Gym 注册任务名，切换任务时修改 |
| `env.processor.reset.fixed_reset_joint_positions` | 复位关节角（7个弧度值） |
| `env.processor.reset.control_time_s` | 每个 episode 最大时长（秒） |
| `dataset.replay_episode` | 回放模式下指定回放的轨迹编号（从0计） |

```bash
python -m lerobot.rl.gym_manipulator \
    --config_path src/lerobot/rl/hil_sim_config.json
```

**键盘控制：** 方向键=XY移动，Shift=Z升降，Ctrl=夹爪，Enter=成功，Backspace=失败，Space=干预开关，ESC=退出

---

## 2. 录制模式

采集人工示教演示轨迹，用作 SAC 离线缓冲区的初始数据。建议录制 20-50 条。

**配置文件：** `hil_sim_config.json`

| 字段 | 说明 |
|------|------|
| `mode` | 改为 `"record"`（**必改**） |
| `dataset.repo_id` | 数据集名称，需与训练配置保持一致，如 `"franka_sim_task_v1"` |
| `dataset.root` | 本地保存路径，默认 `"data"` |
| `dataset.num_episodes_to_record` | 计划录制的 episode 总数 |

```bash
python -m lerobot.rl.gym_manipulator \
    --config_path src/lerobot/rl/hil_sim_config.json
```

录制完成后，读取统计数据（填写训练配置时需要）：

```bash
python -c "
import json
with open('data/meta/stats.json') as f:
    s = json.load(f)
print('action  min:', s['action']['min'])
print('action  max:', s['action']['max'])
print('state   min:', s['observation.state']['min'])
print('state   max:', s['observation.state']['max'])
"
```

---

## 3. 训练模式

SAC 训练需要两个进程：**Learner**（主进程，核心训练）和 **Actor**（运行环境，在线交互）。

**配置文件：** `hil_sim_train_config.json`

| 字段 | 说明 |
|------|------|
| `output_dir` / `job_name` | 输出目录与实验名，建议含任务名称 |
| `steps` / `save_freq` | 总训练步数 / checkpoint 保存间隔 |
| `wandb.enable` / `wandb.project` | 是否启用 wandb 及项目名 |
| `dataset.repo_id` | **必须**与录制时一致 |
| `env.task` | 与录制配置一致 |
| `env.processor.reset.*` | 与录制配置一致 |
| `policy.dataset_stats` | ⚠️ **最重要**：填入录制数据集的真实统计量（见录制章节）。填错会导致归一化偏移，训练无法收敛 |
| `policy.num_discrete_actions` | 离散夹爪档数，当前为 3；不用夹爪时设为 0 |
| `policy.device` | `"cuda"` 或 `"cpu"` |
| `policy.online_steps` / `online_buffer_capacity` / `offline_buffer_capacity` | 在线交互步数上限及缓冲区大小 |

### 3.1 子集训练（内存不够时优先用）

当演示数据较多、`learner` 在离线回放或训练中后期被系统 OOM 杀死时，建议先只使用前 N 条 episode 训练。

在 `src/lerobot/rl/hil_sim_train_config.json` 的 `dataset` 下增加 `episodes` 字段：

```json
"dataset": {
  "repo_id": "franka_sim_plug_v1",
  "root": "data",
  "episodes": [0, 1, 2, 3, 4],
  "use_imagenet_stats": false
}
```

常用照抄模板：
- 前 50 条 episode：`[0, 1, 2, ..., 49]`
- 前 60 条 episode：`[0, 1, 2, ..., 59]`

一键生成列表（避免手写长数组）：

```bash
# 生成前 60 条（0..59）
python -c "import json; print(json.dumps(list(range(60))))"

# 生成前 50 条（0..49）
python -c "import json; print(json.dumps(list(range(50))))"
```

把输出结果直接粘贴到 `dataset.episodes` 即可。

说明：
- `episodes` 是 episode 索引列表（从 0 开始）。
- 训练只会读取该列表里的轨迹，不会读取其余轨迹。
- 子集训练时，`offline_buffer_capacity` 需要不小于该子集对应的总帧数。

经验建议（16GB 内存机器，双相机视觉任务）：
- 先用前 50~60 条 episode 启动。
- `online_buffer_capacity` 先设 `2000~3000`。
- `offline_buffer_capacity` 先设 `4500~6000`，再按实际子集帧数上调。

### 第一次训练

**终端1 — 启动 Learner：**
```bash
cd /home/zhu/lerobot-main/lerobot-main
conda activate lerobot
RUN_DIR="output/franka_sim_$(date +%Y%m%d_%H%M%S)"
echo "$RUN_DIR" > .last_run_dir
PYTHONPATH=src python -m lerobot.rl.learner --config_path=src/lerobot/rl/hil_sim_train_config.json --output_dir="$RUN_DIR"
```

**终端2 — 启动 Actor（等 Learner 初始化完毕后再运行）：**
```bash
cd /home/zhu/lerobot-main/lerobot-main
conda activate lerobot
RUN_DIR="$(cat .last_run_dir)"
PYTHONPATH=src python -m lerobot.rl.actor --config_path=src/lerobot/rl/hil_sim_train_config.json --output_dir="${RUN_DIR}_actor"
```

**训练输出结构：**
```
output/<实验名>_<时间戳>/
├── checkpoints/
│   ├── last -> 0010000/            # 最新 checkpoint 的符号链接
│   ├── 0005000/pretrained_model/   # 可用于推理的模型文件
│   └── 0010000/pretrained_model/
├── dataset/                        # 在线缓冲区快照（评估不需要）
├── dataset_offline/                # 离线缓冲区快照（评估不需要）
└── logs/
```

### 断点续训

修改 `hil_sim_train_config.json`：
```json
{
  "resume": true,
  "output_dir": "output/franka_unplug_sac"
}
```

单一终端运行以启用自动 Actor：
```bash
python -m lerobot.rl.learner \
    --config_path src/lerobot/rl/hil_sim_train_config.json
```

---

## 4. 评估模式

加载训练好的 checkpoint，纯策略推理（无人工干预），输出成功率和步数统计表格。

**配置文件：** `hil_sim_eval_config.json`

| 字段 | 说明 |
|------|------|
| `policy.pretrained_path` | ⚠️ **必改**。指向 `pretrained_model/` 目录 |
| `env.task` / `env.processor.reset.*` | 与训练配置保持一致 |
| `policy.dataset_stats` | 与训练配置保持一致 |

**评估脚本：** `src/lerobot/rl/eval_hil_sim.py`

| 代码位置 | 说明 |
|----------|------|
| `N_EVAL_EPISODES = 20` | 评估 episode 总数，按需修改 |
| `"SUCCESS" if r > 0` | 成功判定阈值，与任务 reward 设计保持一致（共 2 处） |

```bash
python -m lerobot.rl.eval_hil_sim \
    --config_path src/lerobot/rl/hil_sim_eval_config.json
```

一次性依次评估所有模型用这个命令
```bash
cd /home/zhu/lerobot-main/lerobot-main
conda activate lerobot
python -m lerobot.rl.eval_all_checkpoints \
  --run_dir output/franka_sim_XXX（你自己的模型路径） \
  --eval_config src/lerobot/rl/hil_sim_eval_config.json
```
如果你想“遇到一个失败就立即停”，加这个参数：
--stop_on_error
---

## 5. 步长与步数限制调参（插插头必看）

插插头任务对精度敏感，建议把“控制步长”和“每回合步数上限”分开管理。

### 5.1 键盘/手柄步长在哪里改

- 文件：`gym_hil/wrappers/factory.py`
- 默认输入步长倍率：`DEFAULT_INPUTS_CONTROL_STEP_SIZE = {"x": 0.1, "y": 0.2, "z": 0.3}`
- 作用：缩放人工输入（键盘/手柄）位移增量。

说明：
- 倍率越小，单次按键移动越短，更适合精细插入。
- 倍率越大，移动更快，适合粗定位。

补充：
- 末端基础步长在 `gym_hil/wrappers/hil_wrappers.py` 的 `DEFAULT_EE_STEP_SIZE`（当前 0.025m）。
- 近似单次按键位移：`DEFAULT_EE_STEP_SIZE * DEFAULT_INPUTS_CONTROL_STEP_SIZE`。
  - 例如当前约为：
    - X 方向：`0.025 * 0.1 = 0.0025m`（2.5mm）
    - Y 方向：`0.025 * 0.2 = 0.0050m`（5.0mm）
    - Z 方向：`0.025 * 0.3 = 0.0075m`（7.5mm）

### 5.1.1 运行前自检：确认到底加载了哪个 `gym_hil`

很多“改了步长但体感没变化”的问题，根因是运行时加载了另一个环境里的 `gym_hil`。

请在训练/测试前执行：

```bash
conda run -n lerobot python -c "import gym_hil; from gym_hil.wrappers import factory; print('gym_hil file:', gym_hil.__file__); print('factory file:', factory.__file__); print('DEFAULT_INPUTS_CONTROL_STEP_SIZE =', factory.DEFAULT_INPUTS_CONTROL_STEP_SIZE); print('DEFAULT_EE_STEP_SIZE =', factory.DEFAULT_EE_STEP_SIZE)"
```

判定标准：
- `gym_hil file` 和 `factory file` 应该都指向当前工作区路径，例如 `/home/zhu/lerobot-main/lerobot-main/gym_hil/...`
- 输出的 `DEFAULT_INPUTS_CONTROL_STEP_SIZE` 必须与你刚修改的值一致

若路径不在工作区：
- 说明当前 Python 环境优先加载了 site-packages 的副本
- 需要先切到正确环境，或调整 `PYTHONPATH`/安装方式后再运行

### 5.2 每个 episode 最大步数在哪里改

`gym_manipulator` 实际生效上限是两层共同决定：

- 时间层（处理器）：`int(control_time_s * fps)`
  - 配置文件：
    - `src/lerobot/rl/hil_sim_config.json`
    - `src/lerobot/rl/hil_sim_train_config.json`
    - `src/lerobot/rl/hil_sim_eval_config.json`
  - 字段：`env.processor.reset.control_time_s` 和 `env.fps`

- Gym 注册层：`max_episode_steps`
  - 文件：`gym_hil/__init__.py`
  - 当前通过环境变量统一控制：
    - `MAX_EPISODE_STEPS = int(os.getenv("GYM_HIL_MAX_EPISODE_STEPS", "200"))`

最终有效步数：

```text
effective_episode_steps = min(max_episode_steps, int(control_time_s * fps))
```

示例（当前推荐）：
- `fps=10`
- `control_time_s=20`
- `GYM_HIL_MAX_EPISODE_STEPS=200`
- 有效上限 = `min(200, 200) = 200` 步。

临时改步数上限（不改代码）：

```bash
GYM_HIL_MAX_EPISODE_STEPS=240 python -m lerobot.rl.gym_manipulator \
  --config_path src/lerobot/rl/hil_sim_config.json
```

---

## 6. 新任务切换检查清单

### Step 1 — 修改仿真环境

- [ ] 修改 `scene.xml`：添加新任务物体模型和传感器
- [ ] 修改 `panda_pick_gym_env.py`：更新 `_is_success()`、`_compute_reward()`，确保 `_is_success()` 在 `step()` 中只调用一次
- [ ] 将修改后的文件复制到 site-packages：
  ```bash
  SITE=~/.../envs/lerobot/lib/python3.12/site-packages/gym_hil
  cp panda_pick_gym_env.py $SITE/envs/
  cp scene.xml $SITE/assets/
  ```

### Step 2 — 更新 `hil_sim_config.json`

- [ ] `mode` → `"record"`
- [ ] `env.task` → 新任务注册名
- [ ] `env.processor.reset.*` → 新复位姿态和时限（重点：`control_time_s`）
- [ ] `dataset.repo_id` → 新数据集名称
- [ ] 必要时同步调整 `env.fps`（会直接影响每回合最大步数）

### Step 2.5 — 调整控制步长与步数上限

- [ ] `gym_hil/wrappers/factory.py`：按任务精度调整 `DEFAULT_INPUTS_CONTROL_STEP_SIZE`
- [ ] `gym_hil/__init__.py`：确认 `GYM_HIL_MAX_EPISODE_STEPS` 默认值是否合适
- [ ] 三份配置文件中统一 `control_time_s`，避免训练/评估口径不一致

### Step 3 — 录制演示数据，记录 stats

- [ ] 运行录制命令
- [ ] 从 `data/meta/stats.json` 记录 action / state 的 min / max

### Step 4 — 更新 `hil_sim_train_config.json`

- [ ] `output_dir` / `job_name` → 新实验名
- [ ] `dataset.repo_id` → 与 Step 2 一致
- [ ] `env.task` / `env.processor.reset.*` → 同上
- [ ] `policy.dataset_stats` → 填入 Step 3 的真实统计量
- [ ] `policy.input_features` / `output_features` → 如特征维度有变化需修改

### Step 5 — 训练

- [ ] 运行训练命令，记录 checkpoint 路径

### Step 6 — 更新 `hil_sim_eval_config.json`

- [ ] `policy.pretrained_path` → Step 5 的 checkpoint 路径
- [ ] `env.task` / `policy.dataset_stats` → 与训练配置一致

### Step 7 — 评估

- [ ] 运行评估命令

---

## 附录：关键文件路径

| 类型 | 路径 |
|------|------|
| 录制配置 | `src/lerobot/rl/hil_sim_config.json` |
| 训练配置 | `src/lerobot/rl/hil_sim_train_config.json` |
| 评估配置 | `src/lerobot/rl/hil_sim_eval_config.json` |
| 评估脚本 | `src/lerobot/rl/eval_hil_sim.py` |
| 视觉编码器 | `src/lerobot/model/resnet10/` |
| 演示数据集 | `data/` |
| 训练输出 | `output/` |
| gym_hil 环境类 | `~/.../site-packages/gym_hil/envs/panda_pick_gym_env.py` |
| gym_hil 场景文件 | `~/.../site-packages/gym_hil/assets/scene.xml` |
| 步长配置（输入控制） | `gym_hil/wrappers/factory.py` |
| 步长配置（EE基础步长） | `gym_hil/wrappers/hil_wrappers.py` |
| Gym 步数上限配置 | `gym_hil/__init__.py` |