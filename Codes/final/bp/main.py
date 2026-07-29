import socket
import numpy as np
import time
from datetime import datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import requests

# Clear the screen and suppress heavy TF logging
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ==========================================
# 1. CONFIGURATION
# ==========================================
HOST = '0.0.0.0' # Listens on all available IP addresses
PORT = 8080      # Must match the ESP32 port
WINDOW_SIZE = 3 
CALIBRATION_TARGET = 150 

SUPABASE_URL = "https://bmvypwtaxhtxrozypdbo.supabase.co"
SUPABASE_ANON_KEY ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJtdnlwd3RheGh0eHJvenlwZGJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyMzc5ODYsImV4cCI6MjA4MzgxMzk4Nn0.NAX67YKRr5Mev0hzEGj_cXXLznBEUzu2ClHtr_OUYO0" 

HOSPITAL_ID = 6
WARD_ID = 1
PATIENT_ID = 2
current_age = 45.0
current_sex = 1 

# ==========================================
# 2. LOAD THE AI BRAIN
# ==========================================
print("Initializing Deep Learning Model...")
try:
    model = load_model('medical_bp_lstm.h5')
    scaler = joblib.load('tabular_scaler.pkl')
    tab_data = scaler.transform([[current_age, current_sex]])
    tab_input_tf = tf.convert_to_tensor(tab_data, dtype=tf.float32)
except Exception as e:
    print(f"Load Error: {e}")
    raise

def apply_ema_filter(data, alpha=0.6):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

# ==========================================
# 3. WIFI SERVER MONITORING LOOP
# ==========================================
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow immediate port reuse if the script restarts
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"Server listening on Port {PORT}... Waiting for ESP32 to wake up.")
    print("-" * 50)

    while True:
        # Script pauses here until the ESP32 connects
        conn, addr = server_socket.accept() 
        print(f"\n[ESP32 WOKE UP] Connection established from {addr}")
        
        # ---> THE FIX: 70-second timeout to allow the 60s recording to finish <---
        conn.settimeout(70.0) 
        
        # Reset variables for the new session
        live_buffer = []
        calibration_buffer = []
        is_calibrated = False
        global_min, global_max = 0, 1
        sbp_history = []
        dbp_history = []
        remainder = "" # Handles split TCP packets
        
        try:
            while True:
                # ---> THE FIX: Catch the timeout gracefully <---
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    print("\n[TIMEOUT] Silence detected for 70s. Assuming ESP32 is asleep.")
                    break # Break out of the receiving loop cleanly!

                if not data:
                    break # ESP32 cleanly closed the connection
                
                # Combine remainder from last packet with new data
                chunk = remainder + data.decode('utf-8', errors='ignore')
                lines = chunk.split('\n')
                
                # The last item might be an incomplete line, save it for the next loop
                remainder = lines.pop() 
                
                for raw_line in lines:
                    raw_line = raw_line.strip()
                    if not raw_line: continue
                    
                    try:
                        ppg_val = float(raw_line)
                    except ValueError:
                        continue # Ignores "NO_FINGER", "SATURATED", or garbage text
                        
                    # --- PHASE 1: CALIBRATION ---
                    if not is_calibrated:
                        calibration_buffer.append(ppg_val)
                        if len(calibration_buffer) % 50 == 0:
                            prog = int((len(calibration_buffer)/CALIBRATION_TARGET)*100)
                            # ---> THE FIX: flush=True makes Jupyter print immediately <---
                            print(f"Syncing Heartbeat... {prog}%", end='\r', flush=True)
                        
                        if len(calibration_buffer) >= CALIBRATION_TARGET:
                            clean_data = calibration_buffer[30:]
                            global_min, global_max = np.min(clean_data), np.max(clean_data)
                            
                            if (global_max - global_min) < 30:
                                calibration_buffer.clear()
                                continue
                            
                            is_calibrated = True
                            print("\nSIGNAL LOCKED. Collecting Data...")
                        continue

                    # --- PHASE 2: REAL-TIME INFERENCE ---
                    live_buffer.append(ppg_val)
                    if len(live_buffer) >= WINDOW_SIZE:
                        raw_sig = np.array(live_buffer, dtype=float)
                        scaled_sig = np.clip((raw_sig - global_min) / (global_max - global_min), 0, 1)
                        filtered_sig = apply_ema_filter(scaled_sig)
                        vpg = np.gradient(filtered_sig)
                        apg = np.gradient(vpg)
                        
                        sig_stack = np.column_stack((filtered_sig, vpg, apg))
                        sig_input_tf = tf.convert_to_tensor(np.expand_dims(sig_stack, axis=0), dtype=tf.float32)
                        
                        preds = model([sig_input_tf, tab_input_tf], training=False)
                        
                        sbp_history.append(float(preds[0][0]))
                        dbp_history.append(float(preds[0][1]))
                        
                        live_buffer.clear()
                        
        except Exception as e:
            print(f"Connection error during stream: {e}")
        finally:
            conn.close()
            print("\n[ESP32 ASLEEP] Session Ended. Calculating Averages...")
            
            # --- PHASE 3: UPLOAD TO SUPABASE ---
            if len(sbp_history) > 0:
                avg_sbp = np.mean(sbp_history)
                avg_dbp = np.mean(dbp_history)
                avg_pp = avg_sbp - avg_dbp
                print(f"RESULTS -> SBP: {avg_sbp:.1f} | DBP: {avg_dbp:.1f} | PP: {avg_pp:.1f}")
                
                now = datetime.now()
                payload = {
                    "hospital_id": HOSPITAL_ID,
                    "ward_id": WARD_ID,
                    "patient_id": PATIENT_ID,
                    "bp_supine_sys": int(round(avg_sbp)),
                    "bp_supine_dia": int(round(avg_dbp)),
                    "pulse_pressure": int(round(avg_pp)),
                    "chart_date": now.strftime("%Y-%m-%d"),
                    "chart_time": now.strftime("%H:%M:%S")
                }
                
                headers = {
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.post(f"{SUPABASE_URL}/rest/v1/chart_vitals", json=payload, headers=headers, timeout=5.0)
                    if response.status_code in [200, 201]:
                        print("Transmission Successful.")
                    else:
                        print(f"API Warning: Received status code {response.status_code}")
                except Exception as e:
                    print(f"Supabase Error: {e}")
            else:
                print("Not enough valid data collected during this session.")
                
            print(f"\nWaiting for next ESP32 wakeup in roughly 120 seconds...")
            print("-" * 50)

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nServer Shutdown manually.")