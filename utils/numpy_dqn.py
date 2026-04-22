"""
Pure NumPy Deep Q-Network with Priority Experience Replay.
No TensorFlow, no PyTorch. Works on any Python version.

Architecture: 12 → 256 → 128 → 64 → 2
Activation  : LeakyReLU (hidden), Linear (output)
Loss        : Huber (δ=1.0)
Optimizer   : Adam (β1=0.9, β2=0.999)
Init        : He normal
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Leaky ReLU helpers ─────────────────────────────────────────────────────────
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def huber_loss(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> float:
    r = y_pred - y_true
    return float(np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta)).mean())

def huber_grad(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> np.ndarray:
    r = y_pred - y_true
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))


# ══════════════════════════════════════════════════════════════════════════════
#  Priority Experience Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════
class PriorityReplayBuffer:
    """
    Prioritized Experience Replay (PER).

    Experiences with higher |TD error| are sampled more frequently,
    ensuring the agent learns disproportionately from its mistakes.

    p_i = (|δ_i| + ε)^α
    P(i) = p_i / Σ p_j
    IS weight w_i = (N · P(i))^(−β)
    """

    def __init__(self, maxlen: int = 10_000, alpha: float = 0.6):
        self.maxlen   = maxlen
        self.alpha    = alpha
        self.buffer:    list = []
        self.priorities: list[float] = []
        self.pos      = 0
        self._max_p   = 1.0

    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        priority:   float | None = None,
    ) -> None:
        p = (abs(priority) + 1e-6) ** self.alpha if priority is not None else self._max_p
        experience = (state.copy(), action, reward, next_state.copy(), done)

        if len(self.buffer) < self.maxlen:
            self.buffer.append(experience)
            self.priorities.append(p)
        else:
            self.buffer[self.pos]    = experience
            self.priorities[self.pos] = p
        self.pos     = (self.pos + 1) % self.maxlen
        self._max_p  = max(self._max_p, p)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[list, list[int], np.ndarray] | None:
        n = len(self.buffer)
        if n < batch_size:
            return None
        probs  = np.array(self.priorities[:n], dtype=np.float64)
        probs /= probs.sum()
        idx    = np.random.choice(n, batch_size, p=probs, replace=False).tolist()
        batch  = [self.buffer[i] for i in idx]
        w      = (n * probs[idx]) ** (-beta)
        w     /= w.max()
        return batch, idx, w.astype(np.float32)

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        for i, err in zip(indices, td_errors):
            if 0 <= i < len(self.priorities):
                self.priorities[i] = (abs(float(err)) + 1e-6) ** self.alpha
                self._max_p = max(self._max_p, self.priorities[i])

    def __len__(self) -> int:
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
#  Pure NumPy MLP
# ══════════════════════════════════════════════════════════════════════════════
class NumpyMLP:
    """
    Fully-connected MLP: 12 → 256 → 128 → 64 → 2
    All in NumPy. Adam optimizer with He init.
    """

    LAYERS = [12, 256, 128, 64, 2]

    def __init__(self, lr: float = 0.001):
        self.lr = lr
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        # Adam first/second moments
        self.mW: list[np.ndarray] = []
        self.vW: list[np.ndarray] = []
        self.mb: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []
        self.t  = 0
        self._init()

    def _init(self) -> None:
        for i in range(len(self.LAYERS) - 1):
            fan_in, fan_out = self.LAYERS[i], self.LAYERS[i + 1]
            std = np.sqrt(2.0 / fan_in)
            W   = (np.random.randn(fan_in, fan_out) * std).astype(np.float32)
            b   = np.zeros(fan_out, dtype=np.float32)
            self.W.append(W);    self.b.append(b)
            self.mW.append(np.zeros_like(W)); self.vW.append(np.zeros_like(W))
            self.mb.append(np.zeros_like(b)); self.vb.append(np.zeros_like(b))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. X: (batch, 12) → (batch, 2)."""
        a = X.astype(np.float32)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = leaky_relu(z) if i < len(self.W) - 1 else z
        return a

    def train_step(
        self,
        X: np.ndarray,
        y_target: np.ndarray,
        is_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        One forward + backprop step.
        Returns (huber_loss, td_errors).
        """
        X  = X.astype(np.float32)
        yt = y_target.astype(np.float32)

        # ── Forward pass (cache activations and pre-activations) ──────────────
        acts = [X]
        zs   = []
        a    = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            a = leaky_relu(z) if i < len(self.W) - 1 else z
            acts.append(a)

        output   = acts[-1]
        td_errors = huber_grad(output, yt)

        # Apply importance-sampling weights
        if is_weights is not None:
            td_errors *= is_weights[:, np.newaxis]

        loss  = huber_loss(output, yt)
        # raw per-sample TD error for PER priority update
        raw_td = np.abs(output - yt).max(axis=1)

        # ── Backward pass ─────────────────────────────────────────────────────
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grad = td_errors / X.shape[0]

        for i in reversed(range(len(self.W))):
            dW = acts[i].T @ grad
            db = grad.sum(axis=0)

            # Adam
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * dW**2
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * db**2

            mW_hat = self.mW[i] / (1 - beta1**self.t)
            vW_hat = self.vW[i] / (1 - beta2**self.t)
            mb_hat = self.mb[i] / (1 - beta1**self.t)
            vb_hat = self.vb[i] / (1 - beta2**self.t)

            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i]  -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if i > 0:
                grad = (grad @ self.W[i].T) * leaky_relu_grad(zs[i - 1])

        return loss, raw_td

    def get_weights(self) -> list[tuple]:
        return [(W.copy(), b.copy()) for W, b in zip(self.W, self.b)]

    def set_weights(self, weights: list[tuple]) -> None:
        for i, (W, b) in enumerate(weights):
            self.W[i] = W.copy().astype(np.float32)
            self.b[i] = b.copy().astype(np.float32)

    def save(self, path: Path) -> None:
        np.save(str(path), {
            "W":  [W.tolist() for W in self.W],
            "b":  [b.tolist() for b in self.b],
            "mW": [m.tolist() for m in self.mW],
            "vW": [v.tolist() for v in self.vW],
            "mb": [m.tolist() for m in self.mb],
            "vb": [v.tolist() for v in self.vb],
            "t":  self.t,
            "lr": self.lr,
        }, allow_pickle=True)
        logger.info(f"Model saved → {path}")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            d = np.load(str(path), allow_pickle=True).item()
            self.W   = [np.array(w, dtype=np.float32) for w in d["W"]]
            self.b   = [np.array(b, dtype=np.float32) for b in d["b"]]
            self.mW  = [np.array(m, dtype=np.float32) for m in d["mW"]]
            self.vW  = [np.array(v, dtype=np.float32) for v in d["vW"]]
            self.mb  = [np.array(m, dtype=np.float32) for m in d["mb"]]
            self.vb  = [np.array(v, dtype=np.float32) for v in d["vb"]]
            self.t   = int(d["t"])
            self.lr  = float(d["lr"])
            logger.info(f"Model loaded ← {path}")
            return True
        except Exception as exc:
            logger.warning(f"Model load failed: {exc}")
            return False
