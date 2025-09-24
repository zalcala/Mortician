import sounddevice as sd

print("Available audio devices:")
for idx, device in enumerate(sd.query_devices()):
    print(f"{idx}: {device['name']} (Input channels: {device['max_input_channels']}, Output channels: {device['max_output_channels']})")
