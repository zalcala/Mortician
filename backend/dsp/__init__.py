"""
DSP package API for Mortician project.
Provides modular access to envelope smoothing, feature extraction, windowing, and band utilities.
"""
from dsp.envelope import EnvelopeSmoother
from dsp.features import stft_band_energy, compute_global_features
from dsp.windowing import hann_window
from dsp.bands import get_band_edges
