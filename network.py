import socket
import struct
from config import UDP_IP, UDP_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_frame(frame_data):
    """Pack floats to binary and send via UDP"""
    sock.sendto(struct.pack(f'{len(frame_data)}f', *frame_data), (UDP_IP, UDP_PORT))
