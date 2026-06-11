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

## MPE 深入学习笔记

### MPE 是什么

MPE 是 Multi-Agent Particle Environment，也就是多智能体粒子环境。它把多智能体问题简化到二维平面里：agent 可以移动，环境中可能有 landmark、目标、障碍、通信或对抗角色。它的价值在于足够小，能让我们专心学习多智能体接口、观测、动作和奖励，而不是被复杂模拟器淹没。

### MPE2 和 PettingZoo 的关系

PettingZoo 是多智能体环境接口库，早期 MPE 环境常从 `pettingzoo.mpe` 导入。MPE2 可以理解为更新的 MPE 环境包，很多环境仍然保持类似的 `simple_spread_v3.parallel_env(...)` 用法。

本仓库脚本的导入策略是：优先使用 `mpe2`，如果本机不可用，再回退到 `pettingzoo.mpe`。这样既方便使用新版环境，也兼容旧教程和旧机器。

### AEC API 和 Parallel API

AEC API 是 Agent Environment Cycle，强调 agent 轮流行动。它表达能力强，但初学时循环结构会更绕。

Parallel API 是多个 agent 同时行动。一次 `env.step(actions)` 接收一个动作字典，并返回 `observations, rewards, terminations, truncations, infos`。这些返回值也都是按 agent 名称组织的字典。

### 为什么现在优先使用 parallel_env

我现在优先使用 `parallel_env`，因为它更接近很多多智能体训练代码的数据流：

- `observations` 是 `{agent: observation}`。
- `actions` 是 `{agent: action}`。
- `rewards` 是 `{agent: reward}`。
- 所有当前活跃 agent 在同一个 step 里一起行动。

先掌握这个同步交互过程，再学习 AEC API 会轻松很多。

### 常见 MPE 环境

`simple_v3` 是单智能体调试环境。它适合先确认 observation、action、reward 的基本含义，也适合检查渲染窗口是否正常。

`simple_spread_v3` 是协作覆盖 landmark 的环境，是本阶段重点学习环境。多个 agent 需要分散到不同 landmark 附近，并避免互相碰撞。它很适合理解团队共享奖励和协作分工。

`simple_adversary_v3` 是混合对抗环境。环境里通常既有合作方，也有 adversary。它适合理解“不是所有 agent 的目标都一样”。

`simple_tag_v3` 是追逐逃跑对抗环境。通常 adversary 追逐 good agents，good agents 尝试逃跑。它适合观察多智能体竞争和角色差异。

`simple_push_v3` 涉及推动和干扰。它能帮助理解物理交互、目标争夺和对抗行为如何影响 reward。

### 核心接口概念

`observation_space(agent)` 表示某个 agent 的观测空间。它决定了后续神经网络输入的形状。

`action_space(agent)` 表示某个 agent 的动作空间。它决定了策略输出要表示什么，例如离散动作编号或连续动作向量。

`reward` 是环境给 agent 的反馈。在协作任务中，多个 agent 的 reward 可能相同；在对抗任务中，不同角色的 reward 可能方向相反。

`termination` 表示任务自然结束，例如胜利、失败或环境逻辑上的终点。

`truncation` 表示外部限制导致 episode 结束，最常见的是达到最大步数。

### 图形化观察记录模板

```text
日期：
环境：simple
source：mpe2 / pettingzoo.mpe
运行命令：python envs/render_mpe_env.py --env simple --steps 100 --sleep 0.1
possible_agents：agent0
observation_space：
action_space：
画面观察：
agent 行为：深灰色
landmark：深红色
reward 变化：
terminated / truncated 情况：
```

*留个指令方便回头跑跑跑

 * python envs/render_mpe_env.py --env simple --steps 100 --sleep 0.1
1agent深灰色 1landmark暗红色
单智能体环境
 * python envs/render_mpe_env.py --env simple_spread --steps 300 --sleep 0.05
3agent蓝紫色 3landmark深灰色 
 * python envs/render_mpe_env.py --env simple_adversary --steps 300 --sleep 0.05
1adversary蓝紫色 2agent暗红色 2landmark 黑+绿
混合对抗环境
good agents 知道真正目标是哪一个，所以它们 observation 里包含目标 landmark 信息。
adversary 不知道目标是哪一个，所以它 observation 里少了这部分信息。
 * python envs/render_mpe_env.py --env simple_tag --steps 300 --sleep 0.05
3adversary暗红 1agent绿 2landmark（obstacle）深灰
追逐逃跑环境 他追他逃
 * python envs/render_mpe_env.py --env simple_push --steps 300 --sleep 0.05
1adversary暗红 1agent蓝紫色 2landmark绿+蓝紫

### random baseline 记录模板

```text
日期：
环境：simple_spread_v3
episode 数：
max_cycles：
average return：
best return：
worst return：
CSV 路径：
曲线路径：
观察结论：
下一步：
```

### 为什么进入算法前要先理解 MPE

在进入 PPO、QMIX、MADDPG 之前，必须先理解 MPE，因为算法实现本质上是在处理环境返回的数据。如果还没有搞清楚 `observations`、`actions`、`rewards` 这些字典如何流动，就很容易把算法 bug 和环境接口误解混在一起。

先把 MPE 学清楚，可以明确三件事：

- 网络输入是什么：来自每个 agent 的 observation。
- 网络输出是什么：要变成每个 agent 的 action。
- 训练目标是什么：来自每一步 reward 和 episode return。

等这些接口事实稳定后，再进入 PPO、QMIX、MADDPG，会更容易判断每个算法到底解决了什么问题。

## MPE 图形化界面学习记录

### 图形界面里的对象含义

在 MPE 的图形化窗口中，常见对象包括 agent、landmark 和 adversary。

agent 是会行动的智能体。它会根据当前 observation 选择 action，然后在二维平面里移动。不同环境中 agent 的目标可能不同：有的要靠近 landmark，有的要逃跑，有的要合作完成任务。

landmark 是环境里的目标点、参考点或障碍点。它通常不会主动行动，但会影响 observation 和 reward。在 simple_spread 里，landmark 是需要被多个 agent 分散覆盖的位置。

adversary 是对抗方智能体。它也是 agent 的一种，只是角色和奖励目标不同。比如在 simple_tag 中，adversary 往往负责追逐 good agent；在 simple_adversary 中，adversary 会和合作方目标冲突。

### simple 环境观察记录

`simple_v3` 可以看作最小的 MPE 调试环境。图形界面中通常只有一个可行动 agent 和一个目标 landmark。

这个环境适合先观察最基础的问题：

- agent 采取动作后，位置如何变化。
- observation 中哪些数值可能对应速度或相对位置。
- reward 是否随着 agent 接近 landmark 而变化。
- 固定动作模式下，例如一直 right 或一直 up，图形界面里的运动方向是否符合预期。

因为对象少、干扰少，`simple_v3` 很适合用来区分 agent 和 landmark。

### simple_spread 环境观察记录

`simple_spread_v3` 是当前重点学习环境。图形界面中有多个 agent 和多个 landmark，任务目标是让 agent 分散覆盖 landmark，并尽量避免互相碰撞。

观察这个环境时，可以重点看：

- 多个 agent 如果都随机运动，会不会聚在一起或错过 landmark。
- 如果所有 agent 都执行相同方向动作，它们会不会整体一起移动。
- reward 通常体现团队效果，而不是某个 agent 单独表现。
- 最好的策略不应该是所有 agent 追同一个 landmark，而是自动分工。

这个环境很好地展示了多智能体协作的核心：每个 agent 只控制自己，但奖励鼓励团队整体覆盖。

### simple_adversary 环境观察记录

`simple_adversary_v3` 是混合对抗环境。图形界面里通常能看到合作方 agent、adversary 和 landmark。

观察重点是角色不同导致行为目标不同：合作方希望完成自己的目标，adversary 则会干扰或追求相反目标。这个环境比 simple_spread 更复杂，因为 reward 不再只是单纯团队协作，还包含对抗关系。

初学时可以先观察对象颜色和运动方式，再结合 rewards 判断哪些 agent 属于同一阵营。

### simple_tag 环境观察记录

`simple_tag_v3` 是追逐逃跑类环境。图形界面中通常有 adversary 和 good agents。

直观理解：

- adversary 像追捕者，目标是接近或抓到 good agents。
- good agents 像逃跑者，目标是远离 adversary，避免被追到。

random policy 下，这个环境会显得特别乱，因为追捕者和逃跑者都没有形成稳定策略，只是在随机移动。后续如果训练成功，应该能看到 adversary 更会包围或追逐，good agents 更会躲避。

### simple_push 环境观察记录

`simple_push_v3` 涉及推动、目标争夺或干扰。图形界面中除了 agent 和 landmark，还要注意物理接触和位置阻挡。

这个环境的难点在于：agent 的动作不只是改变自己的位置，还可能通过碰撞或推动影响其他对象。它适合在理解 simple 和 simple_spread 之后再看，因为它多了一层物理交互含义。

### 为什么 random policy 看起来很乱

random policy 每一步都是从 action space 里随机选动作。它没有记忆，也没有目标，更不会根据 landmark 或 adversary 的位置做计划。

所以图形界面中会看到：

- agent 一会儿向左，一会儿向右，轨迹抖动。
- 多个 agent 之间没有分工。
- adversary 不会持续追逐目标。
- good agent 不会稳定逃跑。
- reward 可能波动很大，episode return 通常不稳定。

这正是 random baseline 的意义：它不是为了表现好，而是给后续学习算法一个最低参照。

### 为什么 simple_spread 适合作为第一个训练环境

`simple_spread_v3` 很适合作为后续 PPO、QMIX、MADDPG 的第一个训练环境，因为它同时满足几个条件：

- 环境不复杂，画面容易理解。
- 多个 agent 的 observation/action space 通常一致。
- 动作空间是离散的，便于先做策略输出或 Q 值输出。
- reward 体现团队协作，适合学习多智能体信用分配问题。
- 图形界面能直观看到策略有没有从随机乱动变成分散覆盖 landmark。

在进入算法之前，先把 simple_spread 的图形表现、字典接口、reward 结构搞清楚，可以减少后续调试难度。

### MPE 环境对比表

| 环境 | 常见 agent 数量 | 任务类型 | 是否适合入门 |
| --- | ---: | --- | --- |
| `simple_v3` | 1 | 单智能体接近 landmark | 非常适合，用于理解基础接口和图形对象 |
| `simple_spread_v3` | 3 | 多智能体协作覆盖 landmark | 非常适合，建议作为第一个 MARL 训练环境 |
| `simple_adversary_v3` | 多个，含 adversary | 混合协作与对抗 | 适合进阶，先理解角色差异 |
| `simple_tag_v3` | 多个，含 adversary 和 good agents | 追逐逃跑对抗 | 适合进阶，行为更复杂 |
| `simple_push_v3` | 多个，含对抗或干扰角色 | 推动、目标争夺、物理干扰 | 适合进阶，需要先理解物理交互 |

这张表只是学习阶段的粗略整理。不同版本的 MPE2 或 PettingZoo 可能在默认 agent 数量、颜色和细节上略有差异，最终应以 `envs/compare_mpe_envs.py` 的实际输出为准。
