# MPE 学习笔记

## MPE 是什么

MPE 是 Multi-Agent Particle Environment 的缩写，常译为多智能体粒子环境。它是一组轻量级的二维多智能体任务，环境里通常有多个 agent、landmark 和简单物理运动规则。

MPE 的重点不是复杂画面，而是帮助我们观察多智能体强化学习中的核心问题：多个智能体如何同时行动、如何共享或竞争奖励、如何在局部观测下学到协作策略。

## simple_spread 是什么

`simple_spread` 是 MPE 里最经典的协作任务之一。环境中有多个 agent 和多个 landmark，目标通常是让 agent 分散覆盖 landmark，同时避免彼此碰撞。

直觉上可以把它理解为“几个机器人要各自去占不同的位置”。如果所有 agent 都挤向同一个 landmark，整体效果会变差；如果它们能自动分工，奖励会更好。

## agent 是什么

agent 是环境中会做决策的智能体。每个 agent 在每一步都会收到自己的 observation，并选择一个 action。

在多智能体强化学习中，我们通常关心 agent 之间的关系：

- 协作：大家共享目标，例如一起覆盖 landmark。
- 竞争：一部分 agent 和另一部分 agent 目标相反。
- 混合：既有合作，也有竞争。

## landmark 是什么

landmark 是环境中的目标点或参考点。它通常不会主动行动，但会影响 reward 和 observation。

在 `simple_spread` 中，landmark 是 agent 要尽量覆盖的位置。agent 离 landmark 越近，通常任务完成得越好。

## observation 是什么

observation 是 agent 在某一步看到的信息。它通常是一个向量，可能包含：

- 自己的位置和速度。
- landmark 的相对位置。
- 其他 agent 的相对位置。
- 某些任务里的通信信息或角色信息。

注意，observation 不一定等于完整环境状态。很多多智能体任务是局部观测的，也就是说单个 agent 只能看到一部分信息。

## action 是什么

action 是 agent 在当前 step 做出的动作。MPE 常见动作包括：

- 不动。
- 向左、向右、向上、向下移动。
- 某些场景中的通信动作。

在 PettingZoo / mpe2 的 MPE 环境中，动作空间通常可以通过 `env.action_space(agent)` 查看。

## reward 是什么

reward 是环境在一步交互后给 agent 的反馈。强化学习算法会尝试最大化长期累计 reward。

在 `simple_spread` 中，reward 通常鼓励 agent 靠近 landmark，并惩罚碰撞。这个任务常用于学习“协作分工”：不是每个 agent 都追同一个目标，而是整体覆盖效果更重要。

## terminated 和 truncated 的区别

`terminated` 表示 episode 因为任务本身的终止条件而结束。例如游戏胜利、失败、agent 死亡等自然终点。

`truncated` 表示 episode 因为外部限制而被截断，最常见的是达到最大步数上限。此时不一定代表任务自然完成，只是环境规定“这局到时间了”。

学习时可以先记住：

- `terminated`: 环境逻辑上的结束。
- `truncated`: 时间限制或外部限制导致的结束。

## Parallel API 和 AEC API 区别

PettingZoo 常见两种多智能体接口：Parallel API 和 AEC API。

Parallel API 的思路是“所有当前活跃 agent 同时行动”。每次 `env.step(actions)` 接收一个 action 字典，返回 observations、rewards、terminations、truncations、infos 这些字典。它很适合理解同步多智能体任务，也更接近很多 MARL 训练代码的写法。

AEC API 的思路是 Agent Environment Cycle，也就是 agent 轮流行动。你会看到环境按顺序切换当前 agent，每次只给当前 agent 一个动作。它能表达更一般的多智能体交互，但初学时会比 Parallel API 多一层循环概念。

入门建议先看 Parallel API，再回头理解 AEC API。

## 为什么 MPE 适合 PPO、MADDPG、QMIX 入门

MPE 适合入门，是因为它足够简单，又保留了多智能体学习的关键难点。

对 PPO 来说，可以先把每个 agent 看成一个共享策略或独立策略的 actor，练习多 agent 轨迹收集、优势估计和策略更新。

对 MADDPG 来说，MPE 很适合理解“集中训练、分散执行”。训练时 critic 可以看到更多全局信息，执行时每个 actor 只根据自己的 observation 行动。

对 QMIX 来说，MPE 可以帮助理解多 agent reward、局部观测和集中式 value mixing 的概念。不过 QMIX 更常用于离散动作和团队共享奖励任务，所以使用 MPE 时要特别注意环境动作空间和奖励设定是否适配。

总之，MPE 的价值在于：环境小、反馈快、概念集中，适合先把多智能体接口和训练数据流弄清楚，再进入更复杂的算法实现。

## 本次 inspect_mpe_simple_spread.py 输出记录

这次运行 `envs/inspect_mpe_simple_spread.py` 时，`simple_spread_v3` 使用 `mpe2` 加载成功。这说明当前项目优先使用新版 MPE 包的路径是可行的，后续学习 simple_spread 时可以先以 `mpe2` 的接口表现为准。

环境中的 `possible_agents` 是：

```text
agent_0
agent_1
agent_2
```

这说明当前 `simple_spread_v3` 配置下有 3 个智能体。每个智能体都需要在同一个环境里根据自己的 observation 选择 action。

三个 agent 的 observation space 都是：

```text
Box(-inf, inf, (18,), float32)
```

这表示每个 agent 每一步收到的是一个 18 维的 float32 向量，数值范围理论上没有固定上下界。对神经网络来说，这个 observation 可以直接看作长度为 18 的输入向量。

三个 agent 的 action space 都是：

```text
Discrete(5)
```

这表示每个 agent 每一步只能从 5 个离散动作中选一个，动作值通常是 `0, 1, 2, 3, 4`。可以先粗略理解为不动、向不同方向移动等离散控制动作。

reset 之后，`observations` 是一个字典：

```text
{
  "agent_0": 18 维 observation,
  "agent_1": 18 维 observation,
  "agent_2": 18 维 observation
}
```

也就是说，字典的 key 是 agent 名称，value 是该 agent 当前看到的 18 维 observation。学习 Parallel API 时，要记住多智能体数据通常不是一个单独数组，而是“按 agent 名称组织的字典”。

step 时，`actions` 也是一个字典：

```text
{
  "agent_0": 0~4 中的一个动作,
  "agent_1": 0~4 中的一个动作,
  "agent_2": 0~4 中的一个动作
}
```

这说明环境希望我们给每个当前活跃 agent 都提供动作。后续写训练代码时，策略网络输出动作之后，也要重新组装成这种 `{agent: action}` 的形式再传给 `env.step(actions)`。

每次 `step` 返回：

- `observations`: 下一步每个 agent 的 observation。
- `rewards`: 每个 agent 当前 step 得到的 reward。
- `terminations`: 每个 agent 是否因为任务自然结束而结束。
- `truncations`: 每个 agent 是否因为时间上限等外部限制而结束。
- `infos`: 环境额外信息，常用于调试或记录。

本次随机执行了 3 step，观察到三个 agent 的 reward 相同。这说明当前 `simple_spread_v3` 配置下体现了团队共享奖励：虽然每个 agent 都有自己的 observation 和 action，但 reward 反馈是团队层面的。这对多智能体协作任务非常重要，因为它鼓励 agent 学会分工，而不是只优化自己的局部行为。

还观察到 observation 的前两个值会随着移动动作变化。可以先把这两个值初步理解为速度相关信息。后续如果要更精确理解 18 维 observation 的每一段含义，需要结合 MPE simple_spread 的源码或文档继续拆解。

## 对算法输入输出设计的意义

对 PPO 来说，单个 agent 的 actor 网络输入可以设计为 18 维 observation，输出为 5 个离散动作的概率分布。因为三个 agent 的 observation/action space 一样，入门时可以考虑共享同一个策略网络：每个 agent 都用同一套 actor 参数，只是输入不同 observation。

对 MADDPG 来说，每个 actor 仍然可以接收自己的 18 维 observation，并输出动作选择或动作分布。训练 critic 时，则可以考虑把多个 agent 的 observation 和 action 组合起来，让 critic 看到更完整的联合信息。这正好对应 MADDPG 的“集中训练、分散执行”思想。

对 QMIX 来说，`Discrete(5)` 的动作空间比较适合 value-based 方法。每个 agent 的局部 Q 网络可以输入 18 维 observation，输出 5 个动作对应的 Q 值。因为本次观察到 reward 是团队共享的，QMIX 可以把多个 agent 的局部 Q 值通过 mixing network 合成为团队 Q 值，用团队 reward 来训练。

这次 inspect 的核心收获是：`simple_spread_v3` 在当前配置下是 3 个 agent、每个 agent 18 维 observation、5 个离散动作、团队共享 reward。后续不管学习 PPO、MADDPG 还是 QMIX，都要先围绕这些接口事实设计数据收集、网络输入、网络输出和训练日志。
