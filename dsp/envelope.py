import numpy as np
from typing import Optional
from config import NUM_BANDS, ATTACK_MS_LOW, DECAY_MS_LOW, ATTACK_MS_HIGH, DECAY_MS_HIGH, FFT_SIZE, SAMPLE_RATE
import math

def compute_alpha(time_ms: float) -> float:
    """
    Compute exponential smoothing coefficient (alpha) for a given time constant in ms.
    Args:
        time_ms (float): Smoothing time in milliseconds.
    Returns:
        float: Smoothing coefficient alpha.
    """
    tau = time_ms / 1000.0
    H = FFT_SIZE // 2  # hop size
    return math.exp(- H / (SAMPLE_RATE * tau))

class EnvelopeSmoother:
    """
    Maintains smoothed per-band envelopes and dynamic gain normalization.
    Attributes:
        num_bands (int): Number of frequency bands.
        prev (np.ndarray): Previous smoothed envelope values.
        long_term_avg (np.ndarray): Long-term average for dynamic gain.
        alpha_attack (np.ndarray): Attack smoothing coefficients per band.
        alpha_decay (np.ndarray): Decay smoothing coefficients per band.
    """
    def __init__(self, num_bands: int = NUM_BANDS) -> None:
        self.num_bands: int = num_bands
        self.prev: np.ndarray = np.zeros(num_bands)
        self.long_term_avg: np.ndarray = np.ones(num_bands) * 1e-6
        self.alpha_attack: np.ndarray = np.zeros(num_bands)
        self.alpha_decay: np.ndarray = np.zeros(num_bands)
        for i in range(num_bands):
            if i < num_bands // 2:
                self.alpha_attack[i] = compute_alpha(ATTACK_MS_LOW)
                self.alpha_decay[i] = compute_alpha(DECAY_MS_LOW)
            else:
                self.alpha_attack[i] = compute_alpha(ATTACK_MS_HIGH)
                self.alpha_decay[i] = compute_alpha(DECAY_MS_HIGH)

    def process(
        self,
        raw_env: np.ndarray,
        smoothing: bool = True,
        apply_gain: bool = True,
        gain_factor: float = 1.0
    ) -> np.ndarray:
        sm = np.zeros(self.num_bands)
        for i in range(self.num_bands):
            prev = self.prev[i]
            x = raw_env[i]
            if smoothing:
                alpha = self.alpha_attack[i] if x >= prev else self.alpha_decay[i]
                sm_i = alpha * prev + (1 - alpha) * x
            else:
                sm_i = x
            sm[i] = sm_i
            self.long_term_avg[i] = 0.99 * self.long_term_avg[i] + 0.01 * sm_i
        self.prev = sm
        if apply_gain:
            gains = gain_factor / (self.long_term_avg + 1e-6)
            sm = sm * gains
            sm = np.clip(sm, 0, 1)
        return sm
