import socket
import numpy as np
import time
import struct
from dsp_utils import calculate_rates_dsp
from ml_inference import BiosignalPredictor
import datetime
from supabase import create_client, Client

# ==== CONFIGURATION ====
UDP_IP = "0.0.0.0"     # Listen on all network interfaces
UDP_PORT = 4444        # Port must match targetPort in ESP32 code
FS = 8000              # Must match ESP32 sampleRate
BUFFER_DURATION = 10   # Window length in seconds to analyze

# ==== SUPABASE DB CONFIGURATION ====
SUPABASE_URL = "https://bmvypwtaxhtxrozypdbo.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJtdnlwd3RheGh0eHJvenlwZGJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyMzc5ODYsImV4cCI6MjA4MzgxMzk4Nn0.NAX67YKRr5Mev0hzEGj_cXXLznBEUzu2ClHtr_OUYO0"

HOSPITAL_ID = 6
WARD_ID = 1
PATIENT_ID = 2

if __name__ == "__main__":
    print("Starting Raspberry Pi UDP Audio Streamer...")
    print(f"Waiting to receive Wi-Fi audio from ESP32 on port {UDP_PORT}...")
    
    # Open the network socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Initialize buffers and ML Predictor
    buffer_size = FS * BUFFER_DURATION
    audio_buffer = np.zeros(0)
    ml_predictor = BiosignalPredictor(model_path="model.tflite")
    
    # Initialize Supabase client
    try:
        print("Connecting to Supabase Cloud Database...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        print(f"Error initializing Supabase: {e}")
    
    last_process_time = time.time()
    last_db_push_time = time.time()
    
    # Buffers to store values for hourly averaging
    hr_history = []
    rr_history = []
    packet_count = 0
    
    while True:
        try:
            # Receive UDP packet (max size is usually 8KB or less depending on network)
            data, addr = sock.recvfrom(8192)
            
            # Provide initial feedback when connection is successful
            if packet_count == 0:
                print(f"Stream connected! Receiving data from {addr[0]}")
            packet_count += 1
            
            # Unpack the 16-bit integer array sent from ESP32
            num_samples = len(data) // 2
            samples = struct.unpack(f'<{num_samples}h', data)
            
            # Convert to float32 between -1.0 and 1.0
            float_samples = np.array(samples, dtype=np.float32) / 32768.0
            audio_buffer = np.append(audio_buffer, float_samples)
            
            # Restrict buffer from growing infinitely
            if len(audio_buffer) > buffer_size * 2:
                 audio_buffer = audio_buffer[-buffer_size:]
            
            # Process data when we have a full 10-second window
            current_time = time.time()
            if current_time - last_process_time >= 1.0 and len(audio_buffer) >= buffer_size:
                
                # Fetch the latest 10 seconds (buffer_size)
                current_window = audio_buffer[-buffer_size:]
                
                # DSP & ML Processing
                hr_dsp, rr_dsp = calculate_rates_dsp(current_window, FS)
                hr_ml, rr_ml = ml_predictor.predict(current_window, FS)
                
                # Print output display
                print("\n" + "="*40)
                print(f"Rates Calculated at {time.strftime('%H:%M:%S')}")
                print("="*40)
                
                if hr_dsp > 0: print(f" DSP  | Heart Rate:       {hr_dsp:.1f} BPM")
                else:          print(" DSP  | Heart Rate:       [Not detected]")
                    
                if rr_dsp > 0: print(f" DSP  | Respiratory Rate: {rr_dsp:.1f} BPM")
                else:          print(" DSP  | Respiratory Rate: [Not detected]")
                
                print("-" * 40)
                    
                if hr_ml is not None:
                    print(f" ML   | Heart Rate:       {hr_ml:.1f} BPM")
                    print(f" ML   | Respiratory Rate: {rr_ml:.1f} BPM")
                else:
                    print(" ML   | Error: TFLite model not provided.")
                
                print("="*40)
                
                # --- SUPABASE DATABASE PUSH (1-MINUTE AVERAGING) ---
                # Prefer ML, fallback to DSP if ML failed/missing
                final_hr = hr_ml if hr_ml is not None and hr_ml > 0 else hr_dsp
                final_rr = rr_ml if rr_ml is not None and rr_ml > 0 else rr_dsp
                
                if final_hr > 0 and final_rr > 0:
                    hr_history.append(final_hr)
                    rr_history.append(final_rr)
                    secs_left = 60.0 - (current_time - last_db_push_time)
                    print(f" [DB Buffer] Value cached. Next automatic database push in {max(0, secs_left):.1f} seconds.")
                
                # Check if 1 minute (60 seconds) has elapsed
                if current_time - last_db_push_time >= 60.0:
                    if len(hr_history) > 0 and len(rr_history) > 0:
                        # Calculate the average of all tracked values over the past 1 minute
                        avg_hr = int(np.mean(hr_history))
                        avg_rr = int(np.mean(rr_history))
                        current_dt = datetime.datetime.now()
                        
                        try:
                            record = {
                                "hospital_id": HOSPITAL_ID,
                                "ward_id": WARD_ID,
                                "patient_id": PATIENT_ID,
                                "pr_min": avg_hr,
                                "rr_min": avg_rr,
                                "chart_date": current_dt.date().isoformat(),
                                "chart_time": current_dt.time().isoformat()
                            }
                            # Insert into chart_vitals
                            supabase.table("chart_vitals").insert(record).execute()
                            print(f" [DB] ✅ SUCCESS: Pushed 1-MINUTE AVERAGE to Supabase (HR {avg_hr}, RR {avg_rr})")
                        except Exception as db_err:
                            print(f" [DB] ❌ Database push failed: {db_err}")
                    else:
                        print(" [DB] ⚠️ No stable vitals recorded over the past 1 minute. Skipping DB push.")
                    
                    # Reset memory completely for the next 1 minute
                    last_db_push_time = current_time
                    hr_history = []
                    rr_history = []
                
                last_process_time = current_time
                # Clear buffer waiting for next full refresh
                audio_buffer = np.zeros(0)
                
        except KeyboardInterrupt:
            print("\nExiting monitor...")
            break
        except Exception as e:
            print(f"Error during receiving: {e}")
