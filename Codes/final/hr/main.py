import time
import numpy as np
import sounddevice as sd
from dsp_utils import calculate_rates_dsp
from ml_inference import BiosignalPredictor
import sys

# ==== CONFIGURATION ====
FS = 4000              # Sampling frequency (4kHz is enough for <2kHz frequencies of heart/lung)
BUFFER_DURATION = 10   # Window length in seconds to analyze at a time (e.g., 10 seconds)
DEVICE_ID = None       # Index of MEMS mic. Set to None to use default system microphone.
                       # Use `python -m sounddevice` or `print(sd.query_devices())` to find the ID.

def list_audio_devices():
    print("Available Audio Devices:")
    print(sd.query_devices())

def audio_callback(indata, frames, time_info, status):
    """
    This callback is called by sounddevice for each audio block.
    We append it to our global buffer.
    """
    if status:
        print(f"Status: {status}", file=sys.stderr)
        
    global audio_buffer
    audio_buffer = np.append(audio_buffer, indata[:, 0])

if __name__ == "__main__":
    print("Starting Biosignal Audio Monitor...")
    list_audio_devices()
    
    # Buffer to hold the audio data
    buffer_size = FS * BUFFER_DURATION
    audio_buffer = np.zeros(0)
    
    # Initialize the ML Predictor (it will gracefully fail if no model.tflite exists yet)
    # The default path is 'model.tflite', you must provide a trained model later.
    ml_predictor = BiosignalPredictor(model_path="model.tflite")
    
    print(f"Initializing audio stream (Sample Rate: {FS}Hz, Window: {BUFFER_DURATION}s)...")
    
    try:
        # Start the non-blocking audio stream
        stream = sd.InputStream(
            device=DEVICE_ID,
            channels=1,
            samplerate=FS,
            callback=audio_callback
        )
        
        with stream:
            print("Listening to microphone... Let the buffer fill up.")
            while True:
                time.sleep(1.0) # Check every second
                
                # Check if we have enough data (e.g., 10 seconds of audio)
                if len(audio_buffer) >= buffer_size:
                    # Take the latest 'buffer_size' samples
                    current_window = audio_buffer[-buffer_size:]
                    
                    # 1. Classical DSP Calculation
                    hr_dsp, rr_dsp = calculate_rates_dsp(current_window, FS)
                    
                    # 2. Machine Learning Prediction
                    hr_ml, rr_ml = ml_predictor.predict(current_window, FS)
                    
                    # Print the Results
                    print("\n--- Rates Calculated ---")
                    
                    if hr_dsp > 0:
                        print(f"DSP - Heart Rate:       {hr_dsp:.1f} BPM")
                    else:
                        print("DSP - Heart Rate:       [Not detected]")
                        
                    if rr_dsp > 0:
                        print(f"DSP - Respiratory Rate: {rr_dsp:.1f} BPM")
                    else:
                        print("DSP - Respiratory Rate: [Not detected]")
                        
                    if hr_ml is not None:
                        print(f"ML  - Heart Rate:       {hr_ml:.1f} BPM")
                        print(f"ML  - Respiratory Rate: {rr_ml:.1f} BPM")
                    else:
                        print("ML  - No model loaded. Standard DSP rates used above.")
                    
                    print("------------------------")
                    
                    # Keep only the last 30% of the buffer to allow overlapping windows,
                    # or clear it completely to wait a full 10 seconds again.
                    # Here we wait a full 10 seconds for the next brand new window:
                    audio_buffer = np.zeros(0) 

    except KeyboardInterrupt:
        print("\nExiting monitor...")
    except Exception as e:
        print(f"\nError opening audio stream: {e}")
        print("Tip: If you are on Raspberry Pi, ensure your microphone is selected properly.")
        print("Run `python -c 'import sounddevice as sd; print(sd.query_devices())'` to find device index,")
        print("then set DEVICE_ID in main.py to that number.")
