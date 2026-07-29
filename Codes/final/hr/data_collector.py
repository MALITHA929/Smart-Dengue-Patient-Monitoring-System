import time
import wave
import numpy as np
import sounddevice as sd
from datetime import datetime
import os

# ==== CONFIGURATION ====
FS = 4000              # Sampling frequency (Must match main.py later)
RECORD_SECONDS = 10    # How long each audio clip should be
DEVICE_ID = None       # Set to I2S device index once configured
OUTPUT_DIR = "dataset" # Directory where WAV files will be saved

def record_audio_clip(label):
    print(f"\n[Recording] Get ready. Recording {RECORD_SECONDS}s for label: '{label}'...")
    time.sleep(1) # Give user a second to get ready
    print("Recording started. Breathe/Rest normally.")
    
    # Record audio synchronously (blocking)
    audio_data = sd.rec(int(RECORD_SECONDS * FS), samplerate=FS, channels=1, dtype='float32', device=DEVICE_ID)
    sd.wait() # Wait until recording is finished
    print("Recording stopped.")
    
    # Scale to 16-bit PCM for WAV saving
    audio_data_int16 = np.int16(audio_data * 32767)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.wav")
    
    # Save to WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 2 bytes = 16 bit
        wf.setframerate(FS)
        wf.writeframes(audio_data_int16.tobytes())
        
    print(f"Saved: {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("=== Biosignal ML Data Collector ===")
    print("Use this script to collect clean 10-second clips of your heart/breathing.")
    print("These WAV files will be used to train your TensorFlow model later.\n")
    
    print("Available Audio Devices:")
    print(sd.query_devices())
    print("\nTip: Look for your I2S mic (usually starts with 'snd_rpi_i2s_card' or similar) and set DEVICE_ID in this script if needed.\n")
    
    while True:
        try:
            print("\nOptions:")
            print("1. Record 'resting' (Normal heart/breathing)")
            print("2. Record 'active' (Elevated heart rate/breathing)")
            print("3. Record 'noise' (Background noise, talking, moving)")
            print("4. Exit")
            
            choice = input("Select an option (1-4): ")
            
            if choice == '1':
                record_audio_clip('resting')
            elif choice == '2':
                record_audio_clip('active')
            elif choice == '3':
                record_audio_clip('noise')
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
