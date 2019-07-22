import copy
import random
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
n_agents = len(env_info.agents)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state for one agent
            action_size (int): Dimension of each action for one agent
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (for all agents)
            action_size (int): Dimension of each action (for all agents)
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """ Agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents, random_seed, agent_idx=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int) : number of agents in the environment
            random_seed (int): random seed
        """
        # Parameters of the training
        self.BUFFER_SIZE = int(1e6)   # replay buffer size
        self.BATCH_SIZE = 128         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 0.2               # for soft update of target parameters
        self.LR_ACTOR = 1e-4  # learning rate of the actor
        self.LR_CRITIC = 1e-3       # learning rate of the critic
        self.WEIGHT_DECAY = 0        # L2 weight decay
        # no of agents
        self.n_agents = n_agents

        # no of updates per step
        self.n_updates = 5
        # track no of steps before update
        self.step_no = 0
        # updating after how many steps
        self.update_rate = 1
        # scaling noise to increase exploitation at later stages
        self.noise_scale = 1
        # decay rate of noise scale
        self.noise_decay = 0.995
        # minimum noise scale
        self.min_noise_scale = 0.01
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_agents = n_agents
        self.agent_idx = torch.cuda.LongTensor([agent_idx])
        # Actor Network (w/ Target Network)
        # The actor only sees its own state
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        # The critic sees the whole state as well as the actions taken from the other agents
        self.critic_local = Critic(
            state_size*n_agents, action_size*n_agents, random_seed).to(device)
        self.critic_target = Critic(
            state_size*n_agents, action_size*n_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Params:
        =======
            state : state seen by the current agent
            add_noise (bool) : True if we add the UO noise
        """
        if(not torch.is_tensor(state)):
            state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = (self.actor_local(state).cpu().data.numpy())
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, action_proposal, action_proposal_next):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        next_actions = torch.cat(action_proposal_next, dim=1).to(device)
        # ---------------------------- Update Critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards.index_select(
            1, self.agent_idx) + (gamma * Q_targets_next * (1 - dones.index_select(1, self.agent_idx)))
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #
        self.actor_optimizer.zero_grad()

        actions_agent = [p_a if i == self.agent_idx else p_a.detach()
                         for i, p_a in enumerate(action_proposal)]
        actions_agent = torch.cat(actions_agent, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_agent).mean()

        # Minimize the loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self):
        torch.save(self.actor_local.state_dict(),
                   'checkpoint_actor'+str(self.agent_idx.item())+'.pth')
        torch.save(self.critic_local.state_dict(),
                   'checkpoint_critic'+str(self.agent_idx.item())+'.pth')

    def load(self):
        self.critic_local.load_state_dict(torch.load(
            'checkpoint_critic'+str(self.agent_idx.item())+'.pth'))
        self.actor_local.load_state_dict(torch.load(
            'checkpoint_actor'+str(self.agent_idx.item())+'.pth'))
        self.critic_target.load_state_dict(torch.load(
            'checkpoint_critic'+str(self.agent_idx.item())+'.pth'))
        self.actor_target.load_state_dict(torch.load(
            'checkpoint_actor'+str(self.agent_idx.item())+'.pth'))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.normal(0, 1, self.size)
        self.state = x + dx
        return self.state


class MADDPG():
    """  Deep Deterministic Policy Gradient (MADDPG) class """

    def __init__(self, state_size, action_size, n_agents):
        self.BUFFER_SIZE = int(1e6)   # replay buffer size
        self.BATCH_SIZE = 128         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 0.2               # for soft update of target parameters
        self.LR_ACTOR = 1e-4  # learning rate of the actor
        self.LR_CRITIC = 1e-3       # learning rate of the critic
        self.WEIGHT_DECAY = 0        # L2 weight decay
        # no of agents
        self.n_agents = n_agents

        # no of updates per step
        self.n_updates = 5
        # track no of steps before update
        self.step_no = 0
        # updating after how many steps
        self.update_rate = 1
        # scaling noise to increase exploitation at later stages
        self.noise_scale = 1
        # decay rate of noise scale
        self.noise_decay = 0.995
        # minimum noise scale
        self.min_noise_scale = 0.01
        random_seed = 1
        # Environment parameters
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        # Initialize the Agents. Each agent must know the dimension of its own state_size and action_size, as well as the number of other agents in the envorionment (as the critic depends on it) and its agent_idx in the agents list.
        self.agents = [Agent(
            state_size, action_size, n_agents, random_seed, i) for i in range(0, n_agents)]
        # Replay memory
        self.memory = ReplayBuffer(
            action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        # Parameters of the training

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state.reshape(1, -1), action, reward,
                        next_state.reshape(1, -1), done)
        self.step_no = (self.step_no + 1) % self.update_rate
        if self.step_no == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(0, self.n_updates):
                    experiences = [self.memory.sample()
                                   for _ in range(0, self.n_agents)]
                    self.learn(experiences)

    def learn(self, experiences):
        # Use actor target network to guess the optimal next action
        # Use actor local network to compute the optimal action from the current sate
        for i, agent_to_train in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            next_actions = []
            actions = []
            for j, current_agent in enumerate(self.agents):
                agent_id = torch.tensor([j]).to(device)
                # Get the state and the next state seen by the current agent
                experience_state = states.reshape(
                    -1, self.n_agents, self.state_size).index_select(1, agent_id).squeeze(1)
                experience_next_state = next_states.reshape(
                    -1, self.n_agents, self.state_size).index_select(1, agent_id).squeeze(1)
                # Get the action with target network and current network
                agent_next_action = current_agent.actor_target(
                    experience_next_state)
                agent_action = current_agent.actor_local(experience_state)
                actions.append(agent_action)
                next_actions.append(agent_next_action)
            agent_to_train.learn(
                experiences[i], self.GAMMA, actions, next_actions)

    def act(self, states):
        actions = [current_agent.act(states[i])
                   for i, current_agent in enumerate(self.agents)]
        return np.array(actions).reshape(1, -1)

    def reset(self):
        for a in self.agents:
            a.reset()

    def save(self):
        for agent in self.agents:
            agent.save()

    def load(self):
        for agent in self.agents:
            agent.load()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


agent = MADDPG(24, 2, 2)


def train_agent(n_episodes=10000, print_every=25):
    scores_deque = deque(maxlen=100)
    scores = list()
    for i in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[
            brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
            score += rewards
            agent.step(states, actions, rewards, next_states, dones)
            if np.any(dones):
                break
            states = next_states

        scores.append(np.max(score))
        scores_deque.append(np.max(score))
        agent.save()
        if(np.mean(scores_deque) > 0.5):
            print('Environment Solved in {:d} episodes!'.format(i-100))
            break
        if (i % print_every == 0):
            print('Ep. {:d} : Average maximum score over 100 last episodes {:.6f}'.format(
                i, np.mean(scores_deque)))

    return scores


fig = plt.figure()

scores = train_agent()

ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# # get the default brain
# brain_name = env.brain_names[0]
# brain = env.brains[brain_name]

# # Load the agent
# agent.load()

# # Watch it
# scores_agent = []
# my_scores = []
# print_every = 1
# for i in range(1, 101):
# env_info = env.reset(train_mode=False)[
# brain_name]     # reset the environment
# # get the current state (for each agent)
# states = env_info.vector_observations
# agent.reset()
# # initialize the score (for each agent)
# scores = np.zeros(2)
# while True:
# actions = agent.act(states)
# # send all actions to the environment
# env_info = env.step(actions)[brain_name]
# # get next state (for each agent)
# next_states = env_info.vector_observations
# # get reward (for each agent)
# rewards = env_info.rewards
# dones = env_info.local_done                        # see if episode finished
# # update the score (for each agent)
# scores += env_info.rewards
# if np.any(dones):                                  # exit loop if episode finished
# break
# states = next_states

# my_scores.append(np.max(scores))
# scores_agent.append(np.max(scores))

# if (i % print_every == 0):
# print('Ep. {} : Av score (max over agents) over 100 last episodes {}'.format(
# i, np.mean(scores_agent)))

env.close()
