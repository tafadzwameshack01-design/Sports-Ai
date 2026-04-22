import random
import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH      = Path("dqn_sports_model.keras")
CHECKPOINT_PATH = Path("dqn_checkpoint_latest.keras")


class DQNAgent:
    """
    Double Deep Q-Network agent for NBA win/loss prediction.

    Architecture: 128 → BN → Dropout(0.2) → 64 → 32 → 2
    Loss: Huber  |  Optimizer: Adam
    Update rule: Double DQN (online net selects action, target net evaluates)
    Target sync: every 50 training steps

    State space  : 12 normalized features [0, 1]
    Action space : 0 = away team wins, 1 = home team wins
    Reward       : +[1.0, 1.5] correct, -1.0 incorrect
    """

    def __init__(
        self,
        state_size: int = 12,
        action_size: int = 2,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        memory_size: int = 5000,
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=memory_size)
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr            = lr
        self.steps         = 0
        self._tf_loaded    = False
        self._build_networks()

    def _build_networks(self) -> None:
        import tensorflow as tf
        self._tf = tf
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.sync_target()
        self._tf_loaded = True

    def _build_model(self):
        tf = self._tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(
            loss="huber",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
        )
        return model

    def sync_target(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state[np.newaxis, :], verbose=0)
        return int(np.argmax(q[0]))

    def act_greedy(self, state: np.ndarray) -> tuple[int, float]:
        """Pure greedy action with softmax confidence (no exploration noise)."""
        tf = self._tf
        q_vals     = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        action     = int(np.argmax(q_vals))
        probs      = tf.nn.softmax(q_vals).numpy()
        confidence = float(np.max(probs)) * 100.0
        return action, confidence

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(state[np.newaxis, :], verbose=0)[0]

    def replay(self, batch_size: int = 64) -> Optional[float]:
        """
        Double DQN experience replay.
        Online network selects best next action; target network evaluates its value.
        Returns Huber training loss, or None if buffer is below batch_size.
        """
        if len(self.memory) < batch_size:
            return None

        batch       = random.sample(self.memory, batch_size)
        states      = np.array([b[0] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)

        q_now    = self.model.predict(states,      verbose=0)
        q_next_t = self.target_model.predict(next_states, verbose=0)
        q_next_o = self.model.predict(next_states, verbose=0)

        for i, (_, a, r, _, done) in enumerate(batch):
            if done:
                target = r
            else:
                best_a = int(np.argmax(q_next_o[i]))
                target = r + self.gamma * q_next_t[i][best_a]
            q_now[i][a] = target

        history = self.model.fit(states, q_now, epochs=1, verbose=0)
        loss    = float(history.history["loss"][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % 50 == 0:
            self.sync_target()

        return loss

    def calculate_reward(
        self,
        prediction: int,
        actual: int,
        pred_margin: float,
        actual_margin: float,
    ) -> float:
        """
        Shaped reward:
          correct prediction  → +1.0 base + ≤0.5 margin bonus
          incorrect           → -1.0
        Margin bonus decays linearly: full at 0 error, zero at ≥20 points error.
        """
        if prediction == actual:
            margin_error = abs(pred_margin - actual_margin)
            margin_bonus = max(0.0, 0.5 - margin_error / 40.0)
            return 1.0 + margin_bonus
        return -1.0

    def save(self, path: Optional[Path | str] = None) -> None:
        target = Path(path) if path else MODEL_PATH
        self.model.save(str(target))
        logger.info(f"Model saved → {target}")

    def load(self, path: Optional[Path | str] = None) -> bool:
        target = Path(path) if path else MODEL_PATH
        if target.exists():
            try:
                tf = self._tf
                self.model = tf.keras.models.load_model(str(target))
                self.sync_target()
                logger.info(f"Model loaded ← {target}")
                return True
            except Exception as exc:
                logger.warning(f"Could not load model from {target}: {exc}")
        return False

    def update_hyperparams(
        self,
        lr: Optional[float] = None,
        gamma: Optional[float] = None,
        epsilon_decay: Optional[float] = None,
        epsilon_min: Optional[float] = None,
    ) -> None:
        """Hot-update hyperparameters without rebuilding the network."""
        if lr is not None:
            self.lr = lr
            self.model.optimizer.learning_rate.assign(lr)
            self.target_model.optimizer.learning_rate.assign(lr)
        if gamma         is not None: self.gamma         = gamma
        if epsilon_decay is not None: self.epsilon_decay = epsilon_decay
        if epsilon_min   is not None: self.epsilon_min   = epsilon_min

    def get_state_dict(self) -> dict:
        return {
            "epsilon":       round(self.epsilon, 4),
            "steps":         self.steps,
            "memory_size":   len(self.memory),
            "gamma":         self.gamma,
            "lr":            self.lr,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
