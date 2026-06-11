# MARL MPE Learning

这是一个用于学习多智能体强化学习的实践仓库，重点围绕 MPE 环境以及 PPO、QMIX、MADDPG 三类算法展开。当前阶段只搭建项目结构，不实现复杂算法。

## 项目目标

- 熟悉 PettingZoo / MPE 风格的多智能体环境接口。
- 理解多智能体任务中的观测、动作、奖励、终止条件和协作/竞争设定。
- 逐步学习并复现 PPO、QMIX、MADDPG 的核心思想。
- 建立可重复的实验记录方式，方便比较不同算法和配置。

## 学习路线

1. 环境基础
   - 了解 MPE 场景的智能体、地标、通信和奖励设计。
   - 在 `envs/` 中整理环境封装、观察空间和动作空间说明。

2. PPO
   - 从单智能体 PPO 的 clipped objective、GAE、actor-critic 开始。
   - 在 `algorithms/ppo/` 中逐步整理实现。
   - 在 `experiments/ppo_mpe/` 中保存 MPE 实验入口和配置。

3. QMIX
   - 学习集中训练、分散执行、mixing network 和 monotonic constraint。
   - 在 `algorithms/qmix/` 中逐步整理实现。
   - 在 `experiments/qmix_mpe/` 中保存 MPE 实验入口和配置。

4. MADDPG
   - 学习多智能体 actor-critic、集中式 critic、经验回放和连续动作控制。
   - 在 `algorithms/maddpg/` 中逐步整理实现。
   - 在 `experiments/maddpg_mpe/` 中保存 MPE 实验入口和配置。

5. 对比与复盘
   - 将实验日志、曲线、关键结论记录在 `notes/`。
   - 将模型、TensorBoard 日志和输出结果放在 `outputs/` 的约定子目录中。

## 环境安装

建议使用 Python 虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果安装 `torch` 时需要选择 CUDA 版本，请参考 PyTorch 官方安装命令，再安装其余依赖。

## 目录说明

- `notes/`: 学习笔记、论文总结、实验复盘。
- `envs/`: MPE 环境封装和环境相关工具。
- `algorithms/ppo/`: PPO 相关代码。
- `algorithms/qmix/`: QMIX 相关代码。
- `algorithms/maddpg/`: MADDPG 相关代码。
- `experiments/ppo_mpe/`: PPO + MPE 实验入口与配置。
- `experiments/qmix_mpe/`: QMIX + MPE 实验入口与配置。
- `experiments/maddpg_mpe/`: MADDPG + MPE 实验入口与配置。
- `outputs/`: 实验输出、日志、模型和图表。
- `scripts/`: 辅助脚本。

## 实验记录方式

每次实验建议记录以下信息：

- 日期和实验名称。
- 算法、环境、随机种子。
- 关键超参数。
- 代码版本或提交号。
- 训练曲线位置。
- checkpoint 位置。
- 观察到的问题和下一步改进。

推荐在 `notes/` 中按日期创建记录，例如：

```text
notes/2026-06-10_ppo_simple_spread.md
```

推荐在 `outputs/` 中按算法和实验名保存结果，例如：

```text
outputs/
  tensorboard/
  checkpoints/
  figures/
  logs/
```

注意：`outputs/checkpoints` 和 `outputs/tensorboard` 默认不纳入 Git 跟踪，避免提交大型训练产物。

## MPE 环境测试

运行 simple_spread 随机策略测试：

```bash
python envs/run_mpe_simple_spread.py
```

如果想打开可视化窗口：

```bash
python envs/run_mpe_simple_spread.py --render-mode human
```

## 阶段 1：MPE 环境学习

详细检查 `simple_spread` 的 agent、空间、reset 返回值和 step 返回值：

```bash
python envs/inspect_mpe_simple_spread.py
```

对比几个常见 MPE 环境的 agent 数量、观察空间和动作空间：

```bash
python envs/compare_mpe_envs.py
```

阅读学习笔记：

```text
notes/mpe.md
```

## MPE 深入学习脚本

对比多个 MPE 环境的 agent、观察空间和动作空间：

```bash
python envs/compare_mpe_envs.py
```

图形化观察不同 MPE 环境：

```bash
python envs/render_mpe_env.py --env simple --steps 100 --sleep 0.1
python envs/render_mpe_env.py --env simple_spread --steps 300 --sleep 0.05
python envs/render_mpe_env.py --env simple_adversary --steps 300 --sleep 0.05
python envs/render_mpe_env.py --env simple_tag --steps 300 --sleep 0.05
python envs/render_mpe_env.py --env simple_push --steps 300 --sleep 0.05
```

运行 `simple_spread_v3` 随机策略 baseline，并保存 CSV 与曲线：

```bash
python envs/run_mpe_random_episodes.py
```
