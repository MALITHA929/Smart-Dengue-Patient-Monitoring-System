import socket
import struct
import wave
import numpy as np
import time
import os
from datetime import datetime

# ==== CONFIGURATION ====
UDP_IP = "0.0.0.0"
UDP_PORT = 4444
FS = 8000
RECORD_SECONDS = 10
OUTPUT_DIR = "dataset"

def record_udp_clip(label):
    print(f"\n[Recording] Ready to intercept ESP32 Audio Stream for '{label}'...")
    print("Ensure the ESP32 is powered on and streaming.")
    print("Press ENTER to start recording exactly 10 seconds of data...")
    input()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # We expect 10 seconds at 8000 Hz = 80000 samples
    target_samples = FS * RECORD_SECONDS
    samples_received = []
    
    print("Recording started. Breathe/Rest normally...")
    
    try:
        sock.settimeout(5.0) # 5 second timeout if ESP32 fails to send data
        while len(samples_received) < target_samples:
            data, addr = sock.recvfrom(8192)
            num_samples = len(data) // 2
            incoming_samples = struct.unpack(f'<{num_samples}h', data)
            samples_received.extend(incoming_samples)
            
            # Print progress dynamically
            percent = int((len(samples_received) / target_samples) * 100)
            print(f"\rProgress: {min(percent, 100)}% [{len(samples_received)} / {target_samples} samples]", end="")
            
    except socket.timeout:
        print("\n[!] Time out waiting for ESP32. Is the ESP32 running and your firewall allowing it?")
    finally:
        sock.close()
        
    print("\nRecording stopped.")
    
    if len(samples_received) == 0:
        print("Failed to record any data.")
        return
        
    audio_data_int16 = np.array(samples_received[:target_samples], dtype=np.int16)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    filename = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.wav")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16 bit
        wf.setframerate(FS)
        wf.writeframes(audio_data_int16.tobytes())
        
    print(f"Saved recording to: {filename}")

if __name__ == "__main__":
    print("=== ESP32 Wi-Fi Audio Data Collector ===")
    print("Use this to collect training `.wav` files directly from your ESP32 INMP441 stream.\n")
    
    while True:
        try:
            print("\nOptions:")
            print("1. Record 'resting' (Normal heart/breathing)")
            print("2. Record 'active' (Elevated heart rate/breathing)")
            print("3. Record 'noise' (Background noise, talking, moving)")
            print("4. Exit")
            
            choice = input("Select an option (1-4): ")
            
            if choice == '1':   record_udp_clip('resting')
            elif choice == '2': record_udp_clip('active')
            elif choice == '3': record_udp_clip('noise')
            elif choice == '4':
                print("Exiting...")
                break
            else: print("Invalid choice.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
