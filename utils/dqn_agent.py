"""
Double DQN Agent built on the pure NumPy MLP + Priority Experience Replay.

Double DQN update rule:
  target = r + γ * Q_target(s', argmax_a Q_online(s', a))

Exploration: ε-greedy with exponential decay.
Target-net sync: every SYNC_STEPS training steps (hard copy).
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

from utils.numpy_dqn import NumpyMLP, PriorityReplayBuffer, softmax

logger = logging.getLogger(__name__)

MODEL_PATH      = Path("dqn_sports_model.npy")
CHECKPOINT_PATH = Path("dqn_checkpoint_latest.npy")
BEST_PATH       = Path("dqn_best_optimizer.npy")


class DQNAgent:
    """
    Double DQN agent for NBA game prediction.

    State  : 12-D normalized float32 vector
    Actions: 0 = away wins, 1 = home wins
    Reward : +[1.0, 1.5] correct, -1.0 incorrect
    """

    SYNC_STEPS  = 50
    BETA_START  = 0.4
    BETA_END    = 1.0
    BETA_FRAMES = 5000

    def __init__(
        self,
        state_size:    int   = 12,
        action_size:   int   = 2,
        lr:            float = 0.001,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.997,
        memory_size:   int   = 10_000,
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps         = 0
        self.frame         = 0

        self.online = NumpyMLP(lr=lr)
        self.target = copy.deepcopy(self.online)
        self.replay_buffer = PriorityReplayBuffer(maxlen=memory_size)

    @property
    def lr(self) -> float:
        return self.online.lr

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        priority:   float | None = None,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done, priority=priority)

    def act(self, state: np.ndarray) -> int:
        """ε-greedy action."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.online.predict(state[np.newaxis, :])
        return int(np.argmax(q[0]))

    def act_greedy(self, state: np.ndarray) -> tuple[int, float]:
        """Pure greedy with softmax confidence (no exploration)."""
        q      = self.online.predict(state[np.newaxis, :])[0]
        action = int(np.argmax(q))
        probs  = softmax(q)
        conf   = float(np.max(probs)) * 100.0
        return action, conf

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.online.predict(state[np.newaxis, :])[0]

    def replay(self, batch_size: int = 64) -> Optional[float]:
        """
        Double DQN experience replay with PER and importance-sampling.
        Returns Huber loss or None if buffer is below batch_size.
        """
        result = self.replay_buffer.sample(batch_size, beta=self._beta())
        if result is None:
            return None

        batch, indices, is_weights = result
        states      = np.array([b[0] for b in batch], dtype=np.float32)
        actions     = np.array([b[1] for b in batch], dtype=np.int32)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones       = np.array([b[4] for b in batch], dtype=np.float32)

        # Double DQN targets
        q_online_next  = self.online.predict(next_states)
        q_target_next  = self.target.predict(next_states)
        best_actions   = np.argmax(q_online_next, axis=1)
        max_q_next     = q_target_next[np.arange(batch_size), best_actions]

        q_current      = self.online.predict(states)
        targets        = q_current.copy()
        for i in range(batch_size):
            t               = rewards[i] if dones[i] else rewards[i] + self.gamma * max_q_next[i]
            targets[i, actions[i]] = t

        loss, td_errors = self.online.train_step(states, targets, is_weights)
        self.replay_buffer.update_priorities(indices, td_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps  += 1
        self.frame  += batch_size
        if self.steps % self.SYNC_STEPS == 0:
            self.target.set_weights(self.online.get_weights())

        return loss

    def _beta(self) -> float:
        """Linearly anneal IS weight exponent from BETA_START → BETA_END."""
        frac = min(1.0, self.frame / self.BETA_FRAMES)
        return self.BETA_START + frac * (self.BETA_END - self.BETA_START)

    def calculate_reward(
        self,
        prediction:    int,
        actual:        int,
        pred_margin:   float,
        actual_margin: float,
    ) -> float:
        """
        Shaped reward:
          correct   → +1.0 + ≤0.5 margin bonus (max when margin error = 0)
          incorrect → -1.0
        """
        if prediction == actual:
            err   = abs(pred_margin - actual_margin)
            bonus = max(0.0, 0.5 - err / 40.0)
            return round(1.0 + bonus, 4)
        return -1.0

    def update_hyperparams(
        self,
        lr:            float | None = None,
        gamma:         float | None = None,
        epsilon_decay: float | None = None,
        epsilon_min:   float | None = None,
    ) -> None:
        if lr            is not None:
            self.online.lr = lr
            self.target.lr = lr
        if gamma         is not None: self.gamma         = gamma
        if epsilon_decay is not None: self.epsilon_decay = epsilon_decay
        if epsilon_min   is not None: self.epsilon_min   = epsilon_min

    def save(self, path: Path | str | None = None) -> None:
        p = Path(path) if path else MODEL_PATH
        self.online.save(p)

    def load(self, path: Path | str | None = None) -> bool:
        p = Path(path) if path else MODEL_PATH
        ok = self.online.load(p)
        if ok:
            self.target.set_weights(self.online.get_weights())
        return ok

    def get_state_dict(self) -> dict:
        return {
            "epsilon":       round(self.epsilon,       4),
            "epsilon_min":   round(self.epsilon_min,   4),
            "epsilon_decay": round(self.epsilon_decay, 4),
            "gamma":         round(self.gamma,         4),
            "lr":            self.online.lr,
            "steps":         self.steps,
            "buffer_size":   len(self.replay_buffer),
            "adam_t":        self.online.t,
        }
