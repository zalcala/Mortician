import numpy as np
from config import SAMPLE_RATE, NUM_BANDS

def get_band_edges() -> np.ndarray:
    """
    Compute perceptual/logarithmic band edges.
    Returns:
        np.ndarray: Array of band edge frequencies.
    """
    return np.geomspace(20.0, SAMPLE_RATE / 2.0, NUM_BANDS + 1)
