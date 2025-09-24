import os
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "48000"))
FFT_SIZE: int = int(os.getenv("FFT_SIZE", "1024"))
NUM_BANDS: int = int(os.getenv("NUM_BANDS", "32"))
ATTACK_MS_LOW: float = float(os.getenv("ATTACK_MS_LOW", "10.0"))
DECAY_MS_LOW: float = float(os.getenv("DECAY_MS_LOW", "200.0"))
ATTACK_MS_HIGH: float = float(os.getenv("ATTACK_MS_HIGH", "5.0"))
DECAY_MS_HIGH: float = float(os.getenv("DECAY_MS_HIGH", "80.0"))
