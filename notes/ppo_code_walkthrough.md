# PPO 代码阅读笔记

这份笔记用于学习当前项目里的 shared PPO 代码。它只解释现有代码，不修改训练逻辑，也不涉及 QMIX/MADDPG。

需要先注意一个命名差异：很多教程会把“选动作”的函数叫 `select_action`，但当前项目代码里没有这个函数名。当前代码中对应“选动作”的是 `PPOAgent.act()`，它内部调用 `ActorCritic.get_action_and_value()`。

## 一、PPO 总流程图

当前 PPO 数据流可以用下面的文字流程图理解：

```text
env.reset
  ↓
得到 observations 字典
  ↓
把 observations 转成 obs_array
  ↓
PPOAgent.act
  ↓
ActorCritic.get_action_and_value
  ↓
得到 actions / log_probs / values
  ↓
把 actions 数组转回 {agent: action} 字典
  ↓
env.step(actions)
  ↓
得到 next_observations / rewards / terminations / truncations
  ↓
整理 rewards 和 dones
  ↓
buffer.add
  ↓
rollout_steps 步收集完成
  ↓
RolloutBuffer.compute_gae
  ↓
得到 advantages / returns / 展平后的 batch
  ↓
PPOAgent.update
  ↓
计算 ratio / policy_loss / value_loss / entropy / total_loss
  ↓
反向传播更新 ActorCritic
```

在本项目实际代码中，这条链对应为：

```text
reset_env(env)
→ PPOAgent.act()
→ ActorCritic.get_action_and_value()
→ env.step(action_dict)
→ RolloutBuffer.add()
→ RolloutBuffer.compute_gae()
→ PPOAgent.update()
```

## 二、四个代码文件分别负责什么

### networks.py：神经网络

文件位置：`algorithms/ppo/networks.py`

它定义 `ActorCritic`。这是 PPO 的核心神经网络，里面同时包含 actor 和 critic。

- actor：负责根据 observation 产生动作分布。
- critic：负责根据 observation 估计 value。
- 当前 simple_spread 设置中，输入是 18 维 observation，输出是 5 个离散动作的 logits 和 1 个 value。

简单理解：

```text
obs
→ ActorCritic
→ actor logits
→ Categorical 分布
→ action 和 log_prob

obs
→ ActorCritic
→ critic value
```

### buffer.py：存 rollout 数据和算 GAE

文件位置：`algorithms/ppo/buffer.py`

它定义 `RolloutBuffer` 和 `PPOBatch`。

`RolloutBuffer` 负责保存 PPO 一次 rollout 里的数据：

- `obs`
- `actions`
- `log_probs`
- `values`
- `rewards`
- `dones`

然后它用 `compute_gae()` 计算：

- `advantages`
- `returns`

最后，它把原本按 `[rollout_steps, num_agents, ...]` 存储的数据展平成 PPO 更新需要的 batch。

### ppo_agent.py：选动作和更新 PPO

文件位置：`algorithms/ppo/ppo_agent.py`

它定义 `PPOAgent`。这个类是对 PPO 算法的封装。

它主要做三件事：

- 创建共享的 `ActorCritic` 网络。
- 用 `act()` 根据 observation 采样动作。
- 用 `update()` 执行 PPO clipped objective 更新。

这里的 `PPOAgent` 不是 MPE 环境里的某一个 agent，而是“算法对象”。MPE 里有 3 个 agent，但它们共享同一个 `PPOAgent.model`。

### train_ppo_simple_spread.py：训练主循环

文件位置：`experiments/ppo_mpe/train_ppo_simple_spread.py`

它负责把环境和 PPO 算法串起来。

它做的事情包括：

- 导入 `simple_spread_v3`。
- 创建 `parallel_env`。
- reset 环境。
- 反复收集 rollout。
- 调用 `buffer.compute_gae()`。
- 调用 `agent.update()`。
- 打印日志。
- 保存 CSV、reward 曲线和 checkpoint。

## 三、PPO 伪代码和本项目代码对应

```text
1. 导入 simple_spread_v3 环境
   对应：train_ppo_simple_spread.py / load_simple_spread()

2. 创建 parallel_env(render_mode=None, max_cycles=max_cycles)
   对应：train_ppo_simple_spread.py / main()

3. reset 环境，得到 observations
   对应：train_ppo_simple_spread.py / reset_env()

4. 创建 PPOAgent
   对应：ppo_agent.py / PPOAgent.__init__()

5. PPOAgent 内部创建 ActorCritic 网络
   对应：ppo_agent.py / PPOAgent.__init__()
   对应：networks.py / ActorCritic.__init__()

6. 创建 RolloutBuffer
   对应：buffer.py / RolloutBuffer.__init__()

7. for update in total_updates:
   对应：train_ppo_simple_spread.py / main() 里的 update 循环

8. 清空 buffer
   对应：buffer.py / RolloutBuffer.reset()

9. for step in rollout_steps:
   对应：train_ppo_simple_spread.py / main() 里的 rollout 循环

10. 把 observations 字典转成 obs_array
    对应：train_ppo_simple_spread.py / obs_to_array()

11. 用共享 ActorCritic 根据 obs_array 采样动作
    对应：ppo_agent.py / PPOAgent.act()
    对应：networks.py / ActorCritic.get_action_and_value()

12. 得到 actions、log_probs、values
    对应：ppo_agent.py / PPOAgent.act()

13. 把 actions 数组转成 MPE 需要的 action_dict
    对应：train_ppo_simple_spread.py / main() 里的 action_dict

14. 调用 env.step(action_dict)
    对应：train_ppo_simple_spread.py / main()

15. 得到 rewards、terminations、truncations
    对应：train_ppo_simple_spread.py / main()

16. 把 rewards 转成 reward_array，把 terminations/truncations 合成 done_array
    对应：train_ppo_simple_spread.py / dict_values_to_array()
    对应：train_ppo_simple_spread.py / main() 里的 done_array

17. 保存 obs/action/log_prob/value/reward/done
    对应：buffer.py / RolloutBuffer.add()

18. rollout 结束后，用最后的 observation 计算 last_values
    对应：ppo_agent.py / PPOAgent.value()

19. 用 GAE 计算 advantages 和 returns
    对应：buffer.py / RolloutBuffer.compute_gae()

20. 把 rollout 数据展平成 PPO batch
    对应：buffer.py / RolloutBuffer.compute_gae()

21. 执行 PPO 多轮 epoch 更新
    对应：ppo_agent.py / PPOAgent.update()

22. 用新策略重新计算旧动作的 new_log_probs
    对应：ppo_agent.py / PPOAgent.update()
    对应：networks.py / ActorCritic.get_action_and_value()

23. 计算 ratio = exp(new_log_probs - old_log_probs)
    对应：ppo_agent.py / PPOAgent.update()

24. 计算 clipped policy loss
    对应：ppo_agent.py / PPOAgent.update()

25. 计算 value loss
    对应：ppo_agent.py / PPOAgent.update()

26. 计算 entropy
    对应：ppo_agent.py / PPOAgent.update()

27. 组合 total_loss 并反向传播
    对应：ppo_agent.py / PPOAgent.update()

28. 打印日志，保存曲线和模型
    对应：train_ppo_simple_spread.py / main()
```

## 四、逐函数解释

### ActorCritic.forward

位置：`algorithms/ppo/networks.py`

```python
def forward(self, obs):
    features = self.backbone(obs)
    logits = self.actor(features)
    values = self.critic(features).squeeze(-1)
    return logits, values
```

这个函数做一次神经网络前向传播。

输入：

```text
obs
```

如果是 3 个 agent 一起输入，形状通常是：

```text
[3, 18]
```

处理过程：

```text
obs
→ backbone
→ features
→ actor head
→ logits

features
→ critic head
→ values
```

输出：

- `logits`：动作 logits，形状大致是 `[3, 5]`。
- `values`：critic value，形状大致是 `[3]`。

这里 `logits` 还不是动作本身。它只是 5 个动作的未归一化分数，后面会交给 `Categorical` 分布。

### ActorCritic.get_action_and_value

位置：`algorithms/ppo/networks.py`

```python
logits, values = self.forward(obs)
dist = Categorical(logits=logits)

if action is None:
    action = dist.sample()

log_prob = dist.log_prob(action)
entropy = dist.entropy()
return action, log_prob, entropy, values
```

这个函数是 actor 和 critic 的综合出口。

它有两种用途。

第一种：rollout 时采样动作。

```text
action=None
→ 根据 logits 建立 Categorical 分布
→ dist.sample()
→ 得到 action
→ 计算 action 的 log_prob
→ 同时返回 value
```

第二种：PPO 更新时重新计算旧动作概率。

```text
传入 action=batch.actions
→ 不重新采样
→ 只计算新策略下旧 action 的 new_log_prob
```

PPO 必须这样做，因为它要比较“旧策略采样动作时的概率”和“新策略现在对同一个动作的概率”。

### RolloutBuffer.add

位置：`algorithms/ppo/buffer.py`

```python
self.obs[self.step] = obs
self.actions[self.step] = actions
self.log_probs[self.step] = log_probs
self.values[self.step] = values
self.rewards[self.step] = rewards
self.dones[self.step] = dones
self.step += 1
```

这个函数保存一个环境 step 中所有 agent 的数据。

在 simple_spread 中，一个 step 通常包含 3 个 agent 的数据：

```text
agent_0 的 obs/action/log_prob/value/reward/done
agent_1 的 obs/action/log_prob/value/reward/done
agent_2 的 obs/action/log_prob/value/reward/done
```

所以每调用一次 `add()`，buffer 不是只保存一条 transition，而是保存一整个时间步上的 3 条 agent transition。

### RolloutBuffer.compute_gae

位置：`algorithms/ppo/buffer.py`

这个函数负责把 rollout 中的 reward 和 value 变成 PPO 更新需要的 `advantages` 和 `returns`。

核心代码：

```python
for t in reversed(range(self.rollout_steps)):
    if t == self.rollout_steps - 1:
        next_values = last_values
    else:
        next_values = self.values[t + 1]

    next_non_terminal = 1.0 - self.dones[t]
    delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    advantages[t] = last_gae
```

它是从后往前算的。原因是当前时刻的 advantage 会参考未来若干步的 TD 误差。

几个关键公式：

```text
delta = reward + gamma * next_value * (1 - done) - value
advantage = delta + gamma * gae_lambda * (1 - done) * 后一时刻的 advantage
return = advantage + value
```

`done` 的作用是切断 episode：

- `done=0`：可以继续使用下一步 value。
- `done=1`：episode 已结束，不再 bootstrap 下一步 value。

最后它会展平数据：

```python
flat_obs = self.obs.reshape(-1, self.obs_dim)
flat_actions = self.actions.reshape(-1)
flat_advantages = advantages.reshape(-1)
flat_returns = returns.reshape(-1)
```

也就是从：

```text
[rollout_steps, num_agents, ...]
```

变成：

```text
[rollout_steps * num_agents, ...]
```

### PPOAgent.select_action

当前代码没有 `select_action` 这个函数名。对应功能是：

```text
PPOAgent.act()
```

位置：`algorithms/ppo/ppo_agent.py`

```python
obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
actions, log_probs, _, values = self.model.get_action_and_value(obs_tensor)
```

它做的事情是：

```text
obs_batch
→ 转成 torch tensor
→ 输入共享 ActorCritic
→ 采样 actions
→ 得到 log_probs
→ 得到 values
→ 转回 numpy
```

如果 `obs_batch` 是 `[3, 18]`，输出通常是：

```text
actions   [3]
log_probs [3]
values    [3]
```

这 3 个位置分别对应 3 个 MPE agent。

### PPOAgent.update

位置：`algorithms/ppo/ppo_agent.py`

这个函数是真正更新 PPO 网络的地方。

第一步：标准化 advantage。

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

这样做可以让 advantage 的尺度更稳定，避免 policy loss 过大或过小。

第二步：多轮 epoch 和 minibatch。

```python
for _ in range(ppo_epochs):
    indices = torch.randperm(batch_size, device=self.device)

    for start in range(0, batch_size, minibatch_size):
        mb_idx = indices[start : start + minibatch_size]
```

PPO 会对同一批 rollout 数据重复训练几轮，每轮打乱顺序，再切成 minibatch。

第三步：重新计算旧动作在新策略下的概率。

```python
_, new_log_probs, entropy, new_values = self.model.get_action_and_value(
    batch.obs[mb_idx],
    batch.actions[mb_idx],
)
```

注意，这里传入的是旧动作 `batch.actions[mb_idx]`，不是重新采样动作。

第四步：计算 ratio。

```python
log_ratio = new_log_probs - batch.old_log_probs[mb_idx]
ratio = log_ratio.exp()
```

ratio 表示：

```text
新策略对这个动作的概率 / 旧策略对这个动作的概率
```

第五步：计算 PPO clipped policy loss。

```python
unclipped = ratio * mb_advantages
clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
clipped = clipped * mb_advantages
policy_loss = -torch.min(unclipped, clipped).mean()
```

这里 `clip_eps` 默认是 0.2，也就是 ratio 通常被限制在：

```text
[0.8, 1.2]
```

这能防止新策略相对旧策略变化太大。

第六步：计算 value loss。

```python
value_loss = nn.functional.mse_loss(new_values, batch.returns[mb_idx])
```

critic 的目标是让 `new_values` 接近 `returns`。

第七步：计算 entropy。

```python
entropy_loss = entropy.mean()
```

entropy 越大，动作分布越随机。PPO 通常会加入 entropy bonus，鼓励策略保留一定探索。

第八步：组合 total loss 并更新网络。

```python
loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
self.optimizer.step()
```

代码里的 `loss` 就是这里说的 `total_loss`。

### train_ppo_simple_spread.py 里的训练循环

位置：`experiments/ppo_mpe/train_ppo_simple_spread.py`

主循环结构是：

```text
for update in total_updates:
    buffer.reset()

    for step in rollout_steps:
        obs_array = obs_to_array(observations, agents)
        actions, log_probs, values = agent.act(obs_array)
        action_dict = {agent: action}
        next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
        reward_array = ...
        done_array = ...
        buffer.add(...)
        observations = next_observations

    last_values = agent.value(last_obs_array)
    batch = buffer.compute_gae(...)
    info = agent.update(...)
```

这段代码把 MPE 环境和 PPO 算法连接起来。

最关键的是两个格式转换：

第一，MPE 给的是字典：

```text
observations = {
  "agent_0": obs0,
  "agent_1": obs1,
  "agent_2": obs2
}
```

代码用 `obs_to_array()` 变成：

```text
obs_array = [obs0, obs1, obs2]
```

第二，PPO 输出的是动作数组：

```text
actions = [action0, action1, action2]
```

代码再变回 MPE 需要的动作字典：

```text
action_dict = {
  "agent_0": action0,
  "agent_1": action1,
  "agent_2": action2
}
```

## 五、关键变量表

| 变量 | 在代码中的含义 | 初学者理解 |
| --- | --- | --- |
| `obs` | 单个 agent 的 observation，或多个 agent 组成的 obs batch | agent 当前看到的信息 |
| `actions` | actor 采样出的离散动作 | agent 要执行的动作编号，范围通常是 0 到 4 |
| `log_probs` | 采样动作时，该动作在旧策略下的 log probability | PPO 以后要用它比较新旧策略变化 |
| `values` | critic 对 obs 的 value 估计 | critic 认为当前局面未来大概值多少 |
| `rewards` | 环境 step 后返回的奖励 | 环境给 agent 的反馈 |
| `dones` | episode 是否结束 | 结束后 GAE 不再接下一步 value |
| `advantages` | GAE 算出的优势值 | 这个动作比 critic 预期好多少 |
| `returns` | critic 学习的目标 | value 应该拟合的目标值 |
| `old_log_probs` | rollout 时保存下来的旧策略 log probability | PPO ratio 的分母信息 |
| `ratio` | `exp(new_log_probs - old_log_probs)` | 新策略相对旧策略有多偏向这个动作 |
| `policy_loss` | actor 的 PPO clipped loss | 控制策略怎么改动作概率 |
| `value_loss` | critic 的 MSE loss | 控制 value 怎么接近 returns |
| `entropy` | 动作分布的熵 | 衡量策略随机性和探索程度 |
| `total_loss` | 代码中叫 `loss` | `policy_loss + value_coef * value_loss - entropy_coef * entropy` |

## 六、用 rollout_steps=5、agent_num=3 举一个小例子

假设：

```text
rollout_steps = 5
agent_num = 3
obs_dim = 18
```

每一步环境交互时，simple_spread 有 3 个 agent：

```text
step 0: agent_0, agent_1, agent_2
step 1: agent_0, agent_1, agent_2
step 2: agent_0, agent_1, agent_2
step 3: agent_0, agent_1, agent_2
step 4: agent_0, agent_1, agent_2
```

每个 agent 的 obs 都是 18 维，所以 buffer 要保存：

```text
5 个时间步
每个时间步 3 个 agent
每个 agent 18 个 obs 数值
```

因此：

```text
buffer.obs shape = [5, 3, 18]
```

`buffer.actions` 不需要保存 18 维 observation，只需要保存动作编号，所以：

```text
buffer.actions shape = [5, 3]
```

`buffer.rewards`、`buffer.values`、`buffer.dones` 也是每个 step 每个 agent 一个数，所以：

```text
buffer.rewards shape = [5, 3]
buffer.values shape  = [5, 3]
buffer.dones shape   = [5, 3]
```

PPO 更新时，这 5 步乘以 3 个 agent，可以看作 15 条 transition：

```text
5 * 3 = 15
```

所以展平后：

```text
batch.obs shape = [15, 18]
batch.actions shape = [15]
batch.advantages shape = [15]
batch.returns shape = [15]
```

为什么可以展平？因为这是 shared PPO。3 个 agent 共用同一个 `ActorCritic`，所以来自不同 agent 的样本都可以作为同一个网络的训练数据。

## 七、当前 PPO 为什么可能不收敛

这里只分析可能原因，不修改代码。

### 1. 当前是教学版 shared PPO，不是完整 MAPPO

当前 critic 只看单个 agent 自己的 18 维 observation。标准 MAPPO 常见做法是使用 centralized critic，让 critic 看到更多全局信息，例如所有 agent 的状态或全局 state。

simple_spread 是多智能体协作任务。只看局部 observation 的 critic 可能估计不准，从而让 advantage 噪声变大。

### 2. 没有 agent id

3 个 agent 共用同一个网络，但输入中没有显式加入 agent id。

共享网络的好处是样本更多、结构简单；风险是网络不一定能清楚区分“我现在是哪个 agent”。如果任务需要角色差异，缺少 agent id 可能影响学习。

### 3. reward 没有归一化或缩放

当前代码直接使用环境 reward：

```text
reward_array -> buffer -> compute_gae
```

没有 reward normalization，也没有 reward scaling。如果 reward 数值变化较大，critic 的 `value_loss` 可能变大，进而影响 PPO 更新稳定性。

### 4. value loss 没有 clipping

当前 `value_loss` 是普通 MSE：

```python
value_loss = mse(new_values, returns)
```

一些 PPO 实现会对 value 更新也做 clipping，防止 critic 一次变化太猛。当前教学版没有做这一步，所以 critic 可能不稳定。

### 5. episode return 统计比较简单

训练脚本用每一步 3 个 agent reward 的均值累计 episode return：

```python
current_episode_return += float(np.mean(reward_array))
```

对于 simple_spread 这种团队 reward 环境，这通常可以接受。但如果不同 agent reward 不完全一致，均值会隐藏个体差异。

另外，如果一个 update 内没有完整 episode 结束，日志会用当前未结束 episode 的累计 return，这可能让早期日志波动较大。

### 6. 没有单独 eval

当前日志来自训练过程中的采样行为。训练时 action 是从 `Categorical` 中采样的，所以带有随机性。

没有单独 evaluation 时，`mean_episode_return` 不一定稳定反映当前策略真正表现。

### 7. entropy 可能过高或过低

动作空间是 `Discrete(5)`。如果策略接近均匀随机，entropy 大约接近：

```text
log(5) ≈ 1.609
```

如果 entropy 长期接近 1.609，说明策略可能还很随机；如果很快接近 0，说明策略可能过早变得确定，探索不足。

### 8. rollout 较短时 GAE 估计可能噪声大

默认 `rollout_steps=256` 不算特别短，但多智能体任务本身噪声较大。如果 rollout 太短，advantage 估计可能波动明显。

### 9. 超参数可能需要调

PPO 对这些超参数比较敏感：

- `lr`
- `clip_eps`
- `gae_lambda`
- `ppo_epochs`
- `minibatch_size`
- `entropy_coef`
- `value_coef`

当前参数适合教学入门，但不保证在所有机器、所有环境版本上都稳定收敛。

总之，当前代码的价值是帮助理解 PPO 数据流：从 MPE 的 observations 字典，到 shared ActorCritic，再到 buffer、GAE 和 PPO update。它是一个学习 baseline，不是追求最强性能的完整 MAPPO 实现。

# 啊啊啊原论文是连续动作 现在是离散动作！argmax求导之后变成 0了啊
# actor 出现 policy collapse，所有 agent 固定输出单一动作
# 加入 entropy regularization 和 reward scaling