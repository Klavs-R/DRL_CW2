import copy
import numpy as np
import random
import torch
import torch.nn.functional as f
import torch.optim as optim

from collections import namedtuple, deque
from nn_models import Actor, Critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_A = 1e-4
LR_C = 1e-3
W_DECAY = 0
REPLAY_EVERY = 20

# Memory Buffer for Prioritized Experience Replay
class MemoryBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Noise:
    """
    Implement Ornstein-Uhlenbeck noise
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters
        """

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.state = copy.copy(self.mu)

    def reset(self):
        """
        Reset the noise to mean
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state with noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class Agent:

    def __init__(self, state_n, action_n, seed, action_layers=None, critic_layers=None, multi=False):
        """
        Initialise agent to interact with and learn from the env

        :param state_n (Int): Dimensions of state space
        :param action_n (Int): Dimensions of the action space
        :param layer_nodes (List[Int]): List of number of nodes in each hidden layer
        """
        if action_layers is None:
            action_layers = [256, 256, 128]

        if critic_layers is None:
            critic_layers = [256, 128, 128]

        self.state_n = state_n
        self.action_n = action_n
        self.noise = Noise(action_n, seed)
        self.multi = multi
        random.seed(seed)

        # Replay memory
        self.memory = MemoryBuffer(BUFFER_SIZE, seed)

        # Actor Networks
        self.local_actor = Actor(state_n, action_n, action_layers, seed).to(device)
        self.target_actor = Actor(state_n, action_n, action_layers, seed).to(device)
        self.optimiser_a = optim.Adam(self.local_actor.parameters(), lr=LR_A)

        # Critic Networks
        self.local_critic = Critic(state_n, action_n, critic_layers, seed).to(device)
        self.target_critic = Critic(state_n, action_n, critic_layers, seed).to(device)
        self.optimiser_c = optim.Adam(self.local_critic.parameters(), lr=LR_C, weight_decay=W_DECAY)

        # Initial time step
        self.time_step = 0

    def act(self, state, inc_noise=True):
        """
        Return actions for given states with current policy

        :param state: Current state
        :param inc_noise: Include random noise to action
        :return: Action according to current policy
        """
        state = torch.from_numpy(state).float().to(device)

        self.local_actor.eval()
        with torch.no_grad():
            action_values = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()

        if inc_noise:
            action_values += self.noise.sample()

        return np.clip(action_values, -1, 1)

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        """
        Complete updates at each time step
        """
        if self.multi:
            for i in range(len(state)):
                self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])

            if len(self.memory) > BATCH_SIZE:
                self.replay()

        else:

            # First, save the last experience in memory
            self.memory.add(state, action, reward, next_state, done)

            # Learn if available
            if self.time_step >= REPLAY_EVERY and len(self.memory) > BATCH_SIZE:
                self.replay()
                self.time_step = 0

            self.time_step += 1

    def replay(self):
        """
        Update parameters from batch of experiences.
        """

        # Sample a batch of experiences from the memory buffer
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Critic ------------------------------------------------------------ #
        next_actions = self.target_actor(next_states)
        q_targets = self.target_critic(next_states, next_actions)
        q_targets = rewards + (GAMMA * q_targets * (1 - dones))

        q_expected = self.local_critic(states, actions)

        # Minimise loss
        c_loss = f.mse_loss(q_expected, q_targets)#.detach())
        self.optimiser_c.zero_grad()
        c_loss.backward()
        self.optimiser_c.step()

        # Actor ------------------------------------------------------------ #
        pred_actions = self.local_actor(states)

        # Minimise loss
        a_loss = -self.local_critic(states, pred_actions).mean()
        self.optimiser_a.zero_grad()
        a_loss.backward()
        self.optimiser_a.step()

        # Update Targets -------------------------------------------------- #
        self.soft_update(self.local_critic, self.target_critic, TAU)
        self.soft_update(self.local_actor, self.target_actor, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Update target model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
