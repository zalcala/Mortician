import socket
import struct

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Set up UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP frames on {UDP_IP}:{UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(4096)  # buffer size in bytes
    num_floats = len(data) // 4  # float32 = 4 bytes
    frame = struct.unpack(f'{num_floats}f', data)

    # Print first few values to confirm streaming
    print("Frame received:", frame[:10], "...")  # only first 10 floats
    print("Total floats in frame:", num_floats)
