# QMIX simple_spread

## 1. QMIX 一句话解释

QMIX 是一种多智能体价值分解方法：每个 agent 只根据自己的 observation 估计动作价值，训练时再用一个单调 mixer 把各 agent 的 Q 值合成为团队 Q_tot。

## 2. 为什么适合 simple_spread

simple_spread_v3 是协作任务，多个 agent 需要覆盖 landmark，奖励通常可以看作团队共享目标。QMIX 的 decentralized execution / centralized training 适合这个设置：执行时每个 agent 独立选动作，训练时用全局信息学习团队价值。

当前项目里的版本是教学用最小实现：

- 不使用 RNN。
- 不使用 episode buffer。
- 使用 transition replay buffer。
- global state 暂时由所有 agent observation 拼接得到。
- 使用 target networks 和 epsilon-greedy 跑通核心训练逻辑。

## 3. 当前项目目录结构

- `algorithms/qmix/`：QMIX 算法本体，包括 individual Q network、mixer、replay buffer。
- `envs/`：MPE 环境创建和检查脚本，统一入口是 `envs/mpe_env_factory.py`。
- `experiments/qmix_mpe/`：simple_spread 的训练、评估、画图、动作分布检查、debug 和 random baseline 脚本。
- `outputs/`：训练日志、配置、summary、checkpoint 和图表等实验结果。

## 4. run001 smoke test 结果解释

已跑通的 run001：

```text
run directory: D:\fishstar\software\PyCharm 2024.3.5\projects\marl-mpe-learning\outputs\qmix_mpe\simple_spread\run001
total episodes: 30
total steps: 1500
final eval_team_return: -108.32358552219424
final td_loss: 0.694863498210907
final epsilon: 0.55
policy collapse risk: no obvious collapse
training health: OK
```

这说明最小 QMIX 数据流已经闭环：采样、replay buffer、TD target、mixer、target network、评估和 checkpoint 都能正常工作。30 episodes 只是 smoke test，不代表算法已经学到稳定策略；`final eval_team_return` 主要用于确认流程可运行，后续需要 random baseline 和更长训练来判断学习效果。

`policy collapse risk: no obvious collapse` 表示检查时没有发现某个 agent 超过 95% 时间都选同一个动作。`training health: OK` 表示最后的 TD loss 是有限值，且没有出现明显爆炸。

## 5. 下一步

下一步先跑 random baseline，确认 simple_spread 在当前 `max_cycles` 下的随机回报范围；再增加正式训练 episode 数，观察 `eval_team_return` 是否稳定超过 random baseline，同时继续关注 `td_loss`、`q_tot_mean`、`target_q_tot_mean` 和动作分布。

## 6. Random baseline 管理

Random baseline 不再用单个 `random_baseline.json` 覆盖保存。不同 `max_cycles` 和 `team_reward_mode` 会保存为不同文件：

```text
outputs/qmix_mpe/simple_spread/random_baselines/random_baseline_cycles50_mode_mean.json
outputs/qmix_mpe/simple_spread/random_baselines/random_baseline_cycles100_mode_mean.json
```

CSV 也使用相同命名规则。训练、绘图和 debug 都会根据当前 run 的 `max_cycles` 和 `team_reward_mode` 自动匹配对应 baseline。比较 `qmix better than random` 时必须保证 `max_cycles` 和 `team_reward_mode` 一致；如果没有匹配 baseline，训练 summary 会写 `not checked`，不会读取不匹配的旧 baseline。
