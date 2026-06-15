import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class Network(nn.Module):
    """
    Standard Q-Network: Take a state, pass it through two hidden layers and output the action values.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DuelingNetwork(nn.Module):
    """
    Dueling Q-Network architecture separating state-value V(s) and advantage functions A(s, a).
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared feature representation
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.feature_network(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Combine V(s) and A(s, a) with mean subtraction for identifiability
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

class ReplayBuffer:
    """
    Implementation of a Replay Buffer via a double-ended queue.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        return (
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int64), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states, dtype=np.float32), 
            np.array(done, dtype=np.float32)
        )
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Implementation for DQN Agent supporting Vanilla, Double, and Dueling extensions.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=5e-4,
        gamma=0.99,
        buffer_size=15000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        tau=0.005,
        double_dqn=False,
        dueling=False,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.tau = tau
        self.double_dqn = double_dqn
        self.dueling = dueling

        import os
        # Device: MPS (Mac Apple Silicon), CUDA (NVIDIA), or CPU
        if os.environ.get("FORCE_CPU") == "1":
            self.device = torch.device("cpu")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Network selection based on dueling flag
        if dueling:
            self.network = DuelingNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DuelingNetwork(state_dim, action_dim).to(self.device)
        else:
            self.network = Network(state_dim, action_dim).to(self.device)
            self.target_network = Network(state_dim, action_dim).to(self.device)
            
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        # Optimizer & loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
    
        # For logging
        self.training_losses = []

    def epsilon(self):
        """Current epsilon (linear decay)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.steps_done / self.epsilon_decay)

    # ── Action Selection ─────────────────────────────────────────────────
    def select_action(self, state, evaluate=False):
        """
        Epsilon-greedy action selection.
        In evaluate mode: always greedy (no exploration).
        """
        if not evaluate and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        # Convert state to tensor, add batch dimension
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.network(state_t)
        return q_values.argmax(dim=1).item()

    # ── Store Transition ─────────────────────────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training Step ────────────────────────────────────────────────────
    def train_step(self):
        """
        Sample a batch, compute TD loss, backprop.
        Returns loss value or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None
        self.steps_done += 1
        # 1. Sample batch from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        
        # 2. Q(s, a) — Q-value for the action we actually took
        q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 3. Target: r + gamma * max_a' Q_target(s', a') * (1 - done) (or Double DQN)
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action selected by online network, value evaluated by target network
                next_actions = self.network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
            
        # 4. Loss & backprop
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 5. Soft update target network
        self._soft_update()
        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    # ── Soft Update ──────────────────────────────────────────────────────
    def _soft_update(self):
        """θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )
            
    # ── Save / Load ──────────────────────────────────────────────────────
    def save(self, path):
        torch.save({
            "q_network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "training_losses": self.training_losses,
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.training_losses = checkpoint["training_losses"]
        self.network.eval()
        print(f"Model loaded from {path}")
