# envs 脚本说明

这个目录用于学习和观察 MPE 环境本身。这里的脚本只做环境检查、图形化观察、实体检查和随机策略 baseline，不实现 PPO、QMIX、MADDPG。

## compare_mpe_envs.py

功能：
对比多个 MPE 环境的基础接口信息，包括 `possible_agents`、agent 数量、每个 agent 的 `observation_space` 和 `action_space`。脚本会优先导入 `mpe2.<env_name>`，失败后再尝试 `pettingzoo.mpe.<env_name>`。如果某个环境不可用，会打印错误并继续检查下一个环境。

运行命令：

```bash
python envs/compare_mpe_envs.py
```

适合什么时候使用：
适合在学习某个环境前先扫一遍全局，快速知道哪些 MPE 环境能导入、每个环境有多少 agent、观察空间和动作空间是什么。

是否和其他文件有重复：
和 `inspect_mpe_simple_spread.py` 都会打印空间信息，但本脚本是横向对比多个环境；`inspect_mpe_simple_spread.py` 是深入检查一个环境的 reset/step 数据流。

## inspect_mpe_entities.py

功能：
检查图形界面里的实体对象，支持 `simple`、`simple_spread`、`simple_tag`、`simple_adversary`、`simple_push`。reset 后会尽量访问底层 `world`，打印 agents 和 landmarks 的名称、角色属性、是否可移动、是否碰撞、颜色和位置。

运行命令：

```bash
python envs/inspect_mpe_entities.py --env simple_spread
python envs/inspect_mpe_entities.py --env simple_tag
```

适合什么时候使用：
适合想弄清楚“画面里哪个点是 agent、哪个点是 landmark、哪个是 adversary”的时候使用。

是否和其他文件有重复：
和一些调试型实体脚本思路相近，但它是当前保留的通用实体检查入口，适合优先使用。

## inspect_mpe_simple_spread.py

功能：
详细检查 `simple_spread_v3` 的 Parallel API。它会打印 `possible_agents`、每个 agent 的 observation/action space、reset 后 observations 的 keys 和 shape，并随机执行 3 step，逐步打印 actions、rewards、terminations、truncations、infos 和新的 observations。

运行命令：

```bash
python envs/inspect_mpe_simple_spread.py
```

适合什么时候使用：
适合第一次认真学习 `simple_spread_v3` 的数据流：reset 返回什么、step 需要什么、step 又返回什么。

是否和其他文件有重复：
它是当前保留的 simple_spread 接口拆解脚本，和 `run_mpe_random_episodes.py` 的统计 baseline 目标不同。

## render_mpe_env.py

功能：
图形化观察多个 MPE 环境，支持 `--env simple/simple_spread/simple_adversary/simple_tag/simple_push`，并支持 `--mode random/stay/right/left/down/up/cycle`。可以用固定动作观察 agent 在图形界面里的运动方向。

运行命令：

```bash
python envs/render_mpe_env.py --env simple_spread --mode random --steps 300 --sleep 0.05
python envs/render_mpe_env.py --env simple_spread --mode cycle --steps 300 --sleep 0.05
python envs/render_mpe_env.py --env simple --mode right --steps 100 --sleep 0.1
```

适合什么时候使用：
适合观察 MPE 图形界面、理解动作编号和移动方向、比较不同环境的画面差异。

是否和其他文件有重复：
它是当前保留的图形化观察主脚本，覆盖 simple_spread 随机渲染和多个环境的固定动作观察。

## run_mpe_random_episodes.py

功能：
不打开图形界面，统计 `simple_spread_v3` 随机策略的表现。默认运行 20 个 episode，每个 episode 最多 100 step，用每步 `mean(rewards.values())` 作为 team reward，最后打印 average/best/worst return，并保存 CSV 和曲线到 `outputs/mpe/`。

运行命令：

```bash
python envs/run_mpe_random_episodes.py
```

适合什么时候使用：
适合建立 random baseline，观察随机策略在 simple_spread 上的回报范围，为后续学习算法前的对照做准备。

是否和其他文件有重复：
它是当前保留的 simple_spread 随机 baseline 统计脚本，和 `inspect_mpe_simple_spread.py` 的接口学习目标不同。

## 推荐使用顺序

1. `compare_mpe_envs.py`
   先确认本机能导入哪些 MPE 环境，并查看 agent 数量和空间信息。

2. `inspect_mpe_simple_spread.py`
   重点理解 `simple_spread_v3` 的 reset/step 数据流。

3. `render_mpe_env.py --env simple --mode stay/right/up`
   先用最简单环境观察动作编号和移动方向。

4. `inspect_mpe_entities.py --env simple_spread`
   对照图形界面，确认 agent、landmark、adversary 等实体含义。

5. `render_mpe_env.py --env simple_spread --mode cycle`
   观察多个 agent 在 simple_spread 里的运动和 landmark 关系。

6. `run_mpe_random_episodes.py`
   建立 simple_spread 随机策略 baseline，记录 average/best/worst return。

7. `render_mpe_env.py --env simple_adversary/simple_tag/simple_push`
   再进入更复杂的对抗、追逐、推动环境。
