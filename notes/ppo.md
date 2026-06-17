# PPO 学习笔记

## PPO 是什么

PPO 是 Proximal Policy Optimization，常译为近端策略优化。它是一种 policy gradient 算法，核心目标是让策略朝着更高 reward 的方向更新，同时限制每次更新不要离旧策略太远。

直观理解：PPO 不希望模型因为一次 batch 的数据就大幅改变行为。它会比较“新策略”和“采样数据时的旧策略”，如果变化太大，就用 clipped objective 把更新幅度压住。

*怎么判断策略变化大小 ratio = 新策略选择这个 action 的概率 / 旧策略选择这个 action 的概率
*clip_eps = 0.2 ratio[0.8,1.2]属于正常 过高过低会限制在这个范围内

## Actor 是什么

actor 是负责选择动作的策略网络。在当前 simple_spread 教学实现里：

- 输入：某个 agent 自己的 18 维 observation。
- 输出：5 个离散动作对应的 logits。
- 动作分布：用 `Categorical(logits=logits)` 得到。
- 执行：从分布里采样一个动作，例如 0 到 4 中的某一个。

actor 学到的是“看到这个 observation 时，应该更倾向于选择哪个动作”。

## Critic 是什么

critic 是负责估计价值的网络。在当前实现里，critic 和 actor 共用前面的特征层，但有自己的 value head。

- 输入：同样是 18 维 observation。
- 输出：1 个 value 标量。

value 可以理解为 critic 对“从当前 observation 出发，未来大概能拿到多少累计 reward”的估计。PPO 用 critic 来降低策略梯度的方差，也用它来计算 advantage。

## Advantage 是什么

advantage 表示某个动作比 critic 原本预期的表现好多少。

如果 advantage 是正数，说明这个动作结果比预期好，PPO 会提高这个动作在类似 observation 下被选中的概率。

如果 advantage 是负数，说明这个动作结果比预期差，PPO 会降低这个动作在类似 observation 下被选中的概率。

当前实现使用 GAE，也就是 Generalized Advantage Estimation。它会结合多步 reward 和 critic 的 value 估计，在“偏差”和“方差”之间做一个折中。`gamma` 控制未来 reward 的折扣，`gae_lambda` 控制多步估计的平滑程度。

*广义优势估计 直接用累计回报，变量多方差大，不稳定 buffer.py

## old_log_prob 为什么要保存

PPO 使用 rollout 数据训练时，这些数据是由“采样当时的旧策略”产生的。训练时模型参数已经在变化，所以我们必须知道旧策略当时对这个动作给出的概率。

代码里保存的是 `old_log_prob`，也就是旧策略下动作的 log probability。更新时会重新用新策略计算同一个动作的 `new_log_prob`，然后得到：

```text
ratio = exp(new_log_prob - old_log_prob)
```

这个 ratio 表示新策略相对于旧策略有多偏向这个动作。没有 `old_log_prob`，就无法计算 PPO clipped objective 里的 ratio。

## Clipped Objective 是什么

PPO 的 clipped objective 用来限制策略更新幅度。它会同时看两个目标：

```text
ratio * advantage
clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
```

然后取较保守的那个。这样做的含义是：

- 如果新策略只比旧策略变化一点点，可以正常更新。
- 如果新策略对某个动作的概率变化太大，就把 ratio 截断。
- 截断后，策略不容易因为一次训练就跳得太远。

这就是 PPO 名字里 Proximal 的含义：每次更新尽量保持在旧策略附近。

## simple_spread 中 shared PPO 如何处理 3 个 agent

`simple_spread_v3` 默认有 3 个 agent，且它们的 observation space 都是 18 维，action space 都是 `Discrete(5)`。因此教学版 shared PPO 让 3 个 agent 共用同一个 ActorCritic 网络。

每一步环境交互时：

1. 读取 `{agent: observation}` 字典。
2. 按固定 agent 顺序堆成形状为 `[3, 18]` 的数组。
3. 同一个 ActorCritic 对 3 条 observation 同时输出动作分布和 value。
4. 采样得到 3 个动作，再组装回 `{agent: action}` 字典传给环境。
5. buffer 保存每个 agent 的 `obs、action、log_prob、value、reward、done`。

训练时，buffer 先按时间和 agent 计算 GAE，再把 `[rollout_steps, 3]` 的数据展平成一个 batch。因为网络参数共享，所以来自 3 个 agent 的样本都会一起更新同一个策略网络。

## 当前实现和标准 MAPPO 有什么区别

当前实现是教学版 shared PPO，不是完整标准 MAPPO。主要区别包括：

- 当前 critic 只看单个 agent 的局部 observation；标准 MAPPO 通常使用 centralized critic，可以看到全局状态或多个 agent 的联合信息。
- 当前没有显式区分 agent id；标准 MAPPO 常会加入 agent id 或其他角色信息，让共享策略知道“自己是谁”。
- 当前直接把 3 个 agent 的样本展平成 PPO batch；标准 MAPPO 对多智能体 episode、mask、active agent、集中式 value target 的处理更完整。
- 当前只支持 simple_spread 的离散动作入门设置；标准 MAPPO 通常支持更多环境、更多动作空间和更复杂的训练配置。
- 当前日志和模型保存较简单，目标是帮助理解 PPO 数据流，而不是追求最强性能。

所以，这份代码适合作为“从单智能体 PPO 走向多智能体 PPO/MAPPO”的第一步：先理解 shared actor、critic、GAE、old_log_prob 和 clipped objective，再进一步学习 centralized critic 和更完整的 MAPPO 训练细节。
