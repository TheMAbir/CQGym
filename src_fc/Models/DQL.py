import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQN(nn.Module):
    def __init__(self, num_inputs, hidden1_size, hidden2_size, alloc_outputs,device):
        super(DQN, self).__init__()
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


class DQL:
    def __init__(self, env,num_inputs, num_actions, learning_rate=1e-2, gamma=0.99, batch_size=10, layer_size=[],device=None):

        self.hidden1_size = layer_size[0]
        self.hidden2_size = layer_size[1]
        self.batch_size = batch_size
        self.lr = learning_rate
        self.gamma = gamma
        self.memory = deque(maxlen=100)
        self.reward_seq = []
        self.env= env

        self.policy = DQN(num_inputs,self.hidden1_size,self.hidden2_size,num_actions,device)
        self.predict = DQN(num_inputs,self.hidden1_size,self.hidden2_size,num_actions,device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, obs):
        # obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            return self.predict(obs)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states = torch.zeros((self.batch_size, self.env.observation_space.shape[0]))
        actions = torch.zeros((self.batch_size, 1))
        G = torch.zeros(self.batch_size)

        for i in range(self.batch_size):
            reward_sum = 0
            discount = 1
            for j in range(i, self.batch_size):
                _, _, reward, _ = self.memory[j]
                reward_sum += reward * discount
                discount *= self.gamma
            G[i] = reward_sum

            state, actions[i, :], _, _ = self.memory[i]
            states[i,:] = torch.from_numpy(state)
        mean = G.mean()
        std = G.std() if G.std() > 0 else 1
        G = (G - mean) / std

        self.optimizer.zero_grad()
        output = self.policy(states)
        loss = self.loss_fn(output, G)
        loss.backward()
        self.optimizer.step()

        self.memory = deque(maxlen=self.batch_size)

    def remember(self, obs, action, reward, new_obs):
        self.memory.append([obs, action, reward, new_obs])
        self.reward_seq.append(reward)

    def save(self, policy_fp, predict_fp):
        torch.save(self.policy.state_dict(), policy_fp)
        torch.save(self.predict.state_dict(), predict_fp)

    def load(self, policy_fp, predict_fp):
        self.policy.load_state_dict(torch.load(policy_fp))
        self.predict.load_state_dict(torch.load(predict_fp))

    def save_using_model_name(self, model_name_path):
        self.save(model_name_path + "_policy_.pth", model_name_path + "_predict_.pth")

    def load_using_model_name(self, model_name_path):
        self.load(model_name_path + "_policy_.pth", model_name_path + "_predict_.pth")
