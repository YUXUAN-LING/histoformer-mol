from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    an = a / (np.linalg.norm(a) + eps)
    bn = b / (np.linalg.norm(b) + eps)
    return float(np.dot(an, bn))


def proximity_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    dist = np.linalg.norm(a - b)
    return float(1.0 / max(float(dist), eps))


def softmax_temperature(x: np.ndarray, temp: float) -> np.ndarray:
    t = max(float(temp), 1e-6)
    z = (x / t) - np.max(x / t)
    e = np.exp(z)
    return e / max(float(e.sum()), 1e-12)
