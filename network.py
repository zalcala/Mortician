import os
from dotenv import load_dotenv
import socket
import struct

load_dotenv()

UDP_IP = os.getenv('UDP_IP', '127.0.0.1')
UDP_PORT = int(os.getenv('UDP_PORT', 5005))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_frame(frame_data):
    """Pack floats to binary and send via UDP"""
    sock.sendto(struct.pack(f'{len(frame_data)}f', *frame_data), (UDP_IP, UDP_PORT))
