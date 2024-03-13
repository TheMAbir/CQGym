import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, hidden1_size, hidden2_size, alloc_outputs,device):
        super(PolicyNetwork, self).__init__()
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

class PG:
    def __init__(self, env,num_inputs, num_actions, learning_rate=1e-2, gamma=0.99, batch_size=10, layer_size=[],device=None):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.gamma = gamma
        self.num_input = num_inputs
        self.num_outputs = num_actions
        self.memory = []
        self.hidden1_size = layer_size[0]
        self.hidden2_size = layer_size[1]
        self.rewards_seq = []

        self.policy = PolicyNetwork(num_inputs, self.hidden1_size,self.hidden2_size,num_actions,device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.device = device
    def act(self, obs):
        # obs = torch.from_numpy(obs).float()
        # probs = self.policy(obs)
        # m = torch.distributions.Categorical(probs)
        # action = m.sample()
        # return action.item(), m.log_prob(action)
        with torch.no_grad():
            probs = self.policy(obs)
        return probs

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states = torch.Tensor([x[0] for x in self.memory]).to(self.device)
        actions = torch.LongTensor([x[1] for x in self.memory]).to(self.device)
        rewards = [x[2] for x in self.memory]

        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        action_probs = self.policy(states)
        action_probs = action_probs[range(len(actions)), actions]

        loss = -torch.sum(torch.log(action_probs) * discounted_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []

    def remember(self, obs, action, reward, new_obs):
        self.memory.append((obs, action, reward, new_obs))
        self.rewards_seq.append(reward)

    def save(self, policy_fp):
        torch.save(self.policy.state_dict(), policy_fp)

    def load(self, policy_fp):
        self.policy.load_state_dict(torch.load(policy_fp))

    def save_using_model_name(self, model_name_path):
        self.save(model_name_path + "_policy_.pt")

    def load_using_model_name(self, model_name_path):
        self.load(model_name_path + "_policy_.pt")
