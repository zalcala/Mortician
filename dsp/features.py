import numpy as np
from typing import Tuple
from config import FFT_SIZE, NUM_BANDS, SAMPLE_RATE
from dsp.windowing import hann_window
from dsp.bands import get_band_edges

def stft_band_energy(frame: np.ndarray) -> np.ndarray:
    """
    Compute per-band energy from a mono audio frame.
    Args:
        frame (np.ndarray): 1D mono audio buffer of length FFT_SIZE.
    Returns:
        np.ndarray: Normalized energy (0–1) per frequency band, shape (NUM_BANDS,).
    """
    window = hann_window()
    spectrum = np.fft.rfft(frame * window)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    band_edges = get_band_edges()
    band_env = []
    for i in range(NUM_BANDS):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        if np.any(mask):
            band_env.append(np.mean(mag[mask]))
        else:
            band_env.append(0.0)
    band_env = np.array(band_env)
    maxv = np.max(band_env)
    if maxv > 0:
        band_env = band_env / maxv
    return band_env

def compute_global_features(band_env: np.ndarray, prev_env: np.ndarray) -> Tuple[float, float, float, int]:
    """
    Compute global features from per-band envelope values.
    Args:
        band_env (np.ndarray): Current per-band envelope values, shape (NUM_BANDS,).
        prev_env (np.ndarray): Previous per-band envelope values, shape (NUM_BANDS,).
    Returns:
        Tuple[float, float, float, int]:
            - centroid (float): Spectral centroid (band-weighted mean index).
            - flux (float): Mean absolute difference (spectral flux).
            - loudness (float): Mean squared envelope (proxy for loudness).
            - transient (int): 1 if a transient is detected, else 0.
    """
    centroid = float(np.sum(np.arange(NUM_BANDS) * band_env) / (np.sum(band_env) + 1e-6))
    flux = float(np.mean(np.abs(band_env - prev_env)))
    loudness = float(np.mean(band_env ** 2))
    transient = int(np.max(band_env) > 0.8)
    return centroid, flux, loudness, transient
