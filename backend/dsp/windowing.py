import numpy as np
from config import FFT_SIZE

def hann_window(size: int = FFT_SIZE) -> np.ndarray:
    """
    Generate a Hann window of the given size.
    Args:
        size (int): Length of the window.
    Returns:
        np.ndarray: The Hann window.
    """
    return np.hanning(size)
