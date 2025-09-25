from audio_stream import start_stream
from config import DEVICE_INDEX

if __name__ == "__main__":
    start_stream(device=DEVICE_INDEX)  # Set from config.py/.env