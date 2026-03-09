"""
Deep Q-Network (DQN) Agent for Multi-Route Multi-Class Airline Revenue Management
File: agents/model.py

FIX applied:
  - get_action_distribution() and get_best_action() now switch to eval() mode
    before inference and restore train() mode after, so Dropout is correctly
    disabled during prediction (was causing stochastic Q-values).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


class DQNNetwork(nn.Module):
    """Enhanced Deep Q-Network for multi-route, multi-class pricing"""

    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, action_size)

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout    = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln4(self.fc4(x)))
        x = self.fc5(x)
        return x


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity       = capacity
        self.alpha          = alpha
        self.beta           = beta
        self.beta_increment = beta_increment

        self.buffer     = []
        self.priorities = []
        self.position   = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0

        state      = np.array(state,      dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position]     = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        priorities    = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        total   = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.stack(states,      axis=0),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states, axis=0),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    """Standard Experience Replay Buffer"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.array(state,      dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states,      axis=0),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states, axis=0),
            np.array(dones),
            None,
            None,
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Enhanced Deep Q-Learning Agent for Multi-Route Multi-Class Airline RM"""

    def __init__(self, state_size, action_size,
                 learning_rate=0.0005,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.9998,   # FIX: was 0.995
                 epsilon_min=0.05,       # FIX: was 0.01
                 batch_size=64,
                 hidden_size=256,
                 replay_buffer_size=50000,
                 use_prioritized_replay=True,
                 priority_alpha=0.6,
                 priority_beta=0.4,
                 priority_beta_increment=0.001,
                 gradient_clip=1.0,
                 learning_rate_decay=0.9,
                 lr_decay_step=200,
                 device=None):

        self.state_size            = state_size
        self.action_size           = action_size
        self.gamma                 = gamma
        self.epsilon               = epsilon
        self.epsilon_decay         = epsilon_decay
        self.epsilon_min           = epsilon_min
        self.batch_size            = batch_size
        self.hidden_size           = hidden_size
        self.use_prioritized_replay = use_prioritized_replay
        self.gradient_clip         = gradient_clip
        self.lr_decay_step         = lr_decay_step

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"✓ DQN Agent initialized")
        print(f"  Device:           {self.device}")
        print(f"  State size:       {state_size}")
        print(f"  Action size:      {action_size}")
        print(f"  Hidden size:      {hidden_size}")
        print(f"  Epsilon decay:    {epsilon_decay}  (min={epsilon_min})")
        print(f"  Prioritized:      {use_prioritized_replay}")

        self.policy_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler  = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=learning_rate_decay
        )
        self.criterion  = nn.SmoothL1Loss()

        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                capacity       = replay_buffer_size,
                alpha          = priority_alpha,
                beta           = priority_beta,
                beta_increment = priority_beta_increment,
            )
        else:
            self.memory = ReplayBuffer(capacity=replay_buffer_size)

        self.training_steps   = 0
        self.episode_count    = 0
        self.training_rewards = []
        self.losses           = []

        self.action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→',   2: 'E↓10% B↑10%',
            3: 'E→ B↓10%',    4: 'E→ B→',        5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→',    8: 'E↑10% B↑10%',
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION SELECTION
    # ─────────────────────────────────────────────────────────────────────────
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # FIX: switch to eval mode so Dropout is disabled during inference
        self.policy_net.eval()
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values     = self.policy_net(state_tensor)
            action       = q_values.argmax().item()
        self.policy_net.train()  # restore training mode
        return action

    def get_action_distribution(self, state):
        """Get Q-values for all actions (used by app.py for recommendations)."""
        # FIX: eval mode ensures Dropout is off → deterministic Q-values
        self.policy_net.eval()
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values     = self.policy_net(state_tensor)
        self.policy_net.train()
        return q_values.cpu().numpy()[0]

    def get_best_action(self, state):
        """Get best action without exploration."""
        # FIX: eval mode for deterministic inference
        self.policy_net.eval()
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values     = self.policy_net(state_tensor)
            action       = q_values.argmax().item()
            q_value      = q_values.max().item()
        self.policy_net.train()
        action_name = self.action_names.get(action, f"Action {action}")
        return action, q_value, action_name

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY
    # ─────────────────────────────────────────────────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING STEP
    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, indices, weights = batch

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        if weights is not None:
            weights_t = torch.FloatTensor(weights).to(self.device)
        else:
            weights_t = torch.ones(self.batch_size).to(self.device)

        # Current Q-values
        self.policy_net.train()
        current_q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            self.policy_net.eval()
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            self.policy_net.train()

            self.target_net.eval()
            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values

        td_errors = current_q_values.squeeze() - target_q_values
        loss = (weights_t * self.criterion(
            current_q_values.squeeze(),
            target_q_values
        )).mean()

        if self.use_prioritized_replay and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.training_steps += 1
        return loss.item()

    # ─────────────────────────────────────────────────────────────────────────
    # NETWORK MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────
    def save_model(self, filepath, include_optimizer=True):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'policy_net':      self.policy_net.state_dict(),
            'target_net':      self.target_net.state_dict(),
            'epsilon':         self.epsilon,
            'episode_count':   self.episode_count,
            'training_steps':  self.training_steps,
            'training_rewards': self.training_rewards,
            'losses':          self.losses,
            'state_size':      self.state_size,
            'action_size':     self.action_size,
            'hidden_size':     self.hidden_size,
        }

        if include_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath, load_optimizer=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon        = checkpoint.get('epsilon',        self.epsilon_min)
        self.episode_count  = checkpoint.get('episode_count',  0)
        self.training_steps = checkpoint.get('training_steps', 0)
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.losses         = checkpoint.get('losses', [])

        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_optimizer and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        # Ensure both nets are in the right mode after loading
        self.policy_net.train()
        self.target_net.eval()

        print(f"✓ Model loaded from {filepath}")
        print(f"  Episodes trained: {self.episode_count}")
        print(f"  Training steps:   {self.training_steps}")
        print(f"  Current epsilon:  {self.epsilon:.4f}")


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MULTI-ROUTE MULTI-CLASS DQN AGENT - MODEL TESTING")
    print("=" * 70)

    state_size  = 17   # e.g. 7 base + 5 routes + 5 extra
    action_size = 9

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        use_prioritized_replay=True,
    )

    test_state = np.random.rand(state_size).astype(np.float32)

    print("\n1. Testing Action Selection:")
    action = agent.select_action(test_state, training=True)
    print(f"   Selected action (with exploration): {action} - {agent.action_names[action]}")

    best_action, q_value, action_name = agent.get_best_action(test_state)
    print(f"   Best action (greedy): {best_action} - {action_name} (Q={q_value:.2f})")

    print("\n2. Q-values for all actions:")
    q_values = agent.get_action_distribution(test_state)
    for i, (name, q) in enumerate(zip(agent.action_names.values(), q_values)):
        print(f"   Action {i} ({name:>15}): Q = {q:>8.4f}")

    print("\n3. Testing Prioritized Replay Buffer:")
    for i in range(100):
        s  = np.random.rand(state_size).astype(np.float32)
        a  = np.random.randint(action_size)
        r  = np.random.rand()
        ns = np.random.rand(state_size).astype(np.float32)
        d  = (i % 20 == 0)
        agent.store_transition(s, a, r, ns, d)
    print(f"   Buffer size: {len(agent.memory)}/50000")

    print("\n4. Testing Training Step:")
    loss = agent.train_step()
    if loss:
        print(f"   Training loss: {loss:.4f}")

    print("\n5. Testing Save/Load:")
    test_path = "models/test_multiclass_model.pth"
    agent.save_model(test_path)

    new_agent = DQNAgent(state_size, action_size)
    new_agent.load_model(test_path)

    if os.path.exists(test_path):
        os.remove(test_path)
        print("   Test model file cleaned up")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)