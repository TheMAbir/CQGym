import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchstat import stat


#   Actor目标是最大化累积奖励函数  Actor网络输出的是当前状态下每个可行动作的概率分布
class ActorNet(nn.Module):

    def __init__(self, num_inputs, hidden1_size, hidden2_size, alloc_outputs,device):
        super(ActorNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden2_size, alloc_outputs),
            nn.Softmax(dim=0)
        )
        self.actor = self.actor.to(device)
    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        probs = self.actor(x)
        return probs

#   Critic目标是最小化价值函数的预测误差  Critic网络输出的是当前状态的状态价值函数
class CriticNet(nn.Module):

    def __init__(self, num_inputs, hidden1_size, hidden2_size,device):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden2_size, 1)
        )
        self.critic = self.critic.to(device)

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        value = self.critic(x)
        return value


class PPO():
    def __init__(self, env, num_inputs, num_outputs, std=0.0, window_size=50,
                 learning_rate=1e-2, gamma=0.99, batch_size=10, layer_size=[],device = None):
        super(PPO, self).__init__()
        self.hidden1_size = layer_size[0]
        self.hidden2_size = layer_size[1]

        self.actor_net = ActorNet(
            num_inputs, self.hidden1_size, self.hidden2_size, num_outputs,device)
        self.critic_net = CriticNet(
            num_inputs, self.hidden1_size, self.hidden2_size,device)

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.window_size = window_size
        self.rewards = []
        self.states = []
        self.action_probs = []
        self.ppo_update_time = 10
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.training_step = 0
        self.rewards_seq = []
        self.num_inputs = num_inputs

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=self.lr)

        self.device = device
        #输出网络参数信息
        Actor_Params = sum(p.numel() for p in self.actor_net.parameters())
        print(f"Total number of parameters: {Actor_Params}")
    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        value = self.critic_net(x)
        probs = self.actor_net(x)
        return probs, value

    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor_net(state)
            value = self.critic_net(state)
        return probs, value

    def remember(self, probs, value, reward, done, device, action, state, next_state, action_p, obs):
        dist = Categorical(torch.softmax(probs, dim=-1))

        self.rewards.append(torch.FloatTensor(
            [reward]).unsqueeze(-1).to(device))
        self.rewards_seq.append(reward)

        state = state.detach().cpu()
        self.states.append(state.numpy())
        self.action_probs.append(action_p)

    def train(self):
        if len(self.states) < self.batch_size:
            return
        old_action_log_prob = torch.stack(self.action_probs)

        R = 0
        Gt = []
        # 倒序循环,遍历存储的奖励列表
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            # 计算得到的折扣累积奖励插入到列表的开头.由于是倒序循环,插入到开头保证了列表中的顺序与时间步一致.
            Gt.insert(0, R)
        # 计算每个时间步的折扣累积奖励,这些折扣累积奖励将用于计算优势以及在PPO算法中进行优化
        Gt = torch.tensor(Gt, dtype=torch.float)
        self.states = torch.tensor(np.array(self.states), dtype=torch.float)

        for i in range(self.ppo_update_time):
            # 随机采样,从已收集的状态序列中选择一个索引(代表一个时间步)
            for index in BatchSampler(SubsetRandomSampler(range(len(self.states))), 1, False):
                # 选择对应索引的折扣累积奖励,将其调整为一个列向量
                Gt_index = Gt[index].view(-1, 1)
                # 选择对应索引的状态序列,将其调整为神经网络可以处理的形状
                sampled_states = self.states[index].view(
                    1, 1, self.num_inputs, 2)

                sampled_states = sampled_states.to(self.device)
                Gt_index = Gt_index.to(self.device)

                # 通过Critic网络计算状态的值函数估计
                V = self.critic_net(sampled_states)
                # 折扣累积奖励与值函数估计之间的差,使用 .detach() 方法防止梯度反向传播到Critic网络
                advantage = (Gt_index - V).detach()

                # 使用Actor网络计算状态的动作策略
                action_prob = self.actor_net(sampled_states)
                # 比较新策略和旧策略在某个状态下选择某个动作的概率的变化;exp函数用于处理概率值的比例,以将其转化为线性形式;可以将结果映射到正数范围,以确保在策略更新时得到稳定的结果.
                ratio = torch.nan_to_num(
                    torch.exp(action_prob - old_action_log_prob[index]))
                # adv:优势函数,衡量某个状态下采用当前策略相对于基准策略的性能优势 ratio:新策略相对于旧策略的重要性采样比率
                surr1 = ratio * advantage   # 未经剪切的策略损失,表示表示当前策略更新对于优势函数的贡献
                surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                    1 + self.clip_param) * advantage  # 通过剪切函数限制ratio的取值范围,确保它在一定区间内,防止过大的策略更新,提高训练稳定性

                # 要采用梯度上升来更新策略,即最大化策略损失;优化器通常用于最小化损失函数,为了与优化器的期望行为保持一致,通常给策略损失加负号,将最大化问题转换为最小化问题;
                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = -F.mse_loss(Gt_index[0], V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        self.rewards = []
        self.states = []
        self.action_probs = []

    def save_using_model_name(self, model_name_path):
        torch.save(self.actor_net.state_dict(), model_name_path + "_actor.pkl")
        torch.save(self.critic_net.state_dict(),
                   model_name_path + "_critic.pkl")

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(
            torch.load(model_name_path + "_actor.pkl"))
        self.critic_net.load_state_dict(
            torch.load(model_name_path + "_critic.pkl"))
