import serial
import serial.tools.list_ports
import time
from datetime import datetime
import requests

# --- CONFIGURATION ---
SUPABASE_URL = "https://bmvypwtaxhtxrozypdbo.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJtdnlwd3RheGh0eHJvenlwZGJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyMzc5ODYsImV4cCI6MjA4MzgxMzk4Nn0.NAX67YKRr5Mev0hzEGj_cXXLznBEUzu2ClHtr_OUYO0"

HOSPITAL_ID = 6
WARD_ID = 1
PATIENT_ID = 2

SERIAL_BAUDRATE = 115200
CRT_RESULT_PREFIX = "CRT_RESULT:"


def find_esp32_port():
    """Auto-detect the ESP32-C3 USB serial port."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        # Common ESP32-C3 USB identifiers
        if any(keyword in desc for keyword in ["cp210", "ch340", "usb serial", "esp32", "usb jtag"]):
            return port.device
        if any(keyword in hwid for keyword in ["1a86:", "10c4:", "303a:"]):
            return port.device
    # Fallback: try common Raspberry Pi serial device paths
    import os
    for fallback in ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1", "/dev/ttyACM1"]:
        if os.path.exists(fallback):
            return fallback
    return None


def upload_to_supabase(crft_result_str):
    """Upload CRT result to Supabase chart_vitals table (same as original crt.py)."""
    now = datetime.now()
    payload = {
        "hospital_id": HOSPITAL_ID,
        "ward_id": WARD_ID,
        "patient_id": PATIENT_ID,
        "crft": crft_result_str,
        "chart_date": now.strftime("%Y-%m-%d"),
        "chart_time": now.strftime("%H:%M:%S")
    }
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/chart_vitals",
            json=payload,
            headers=headers,
            timeout=5.0
        )
        if response.status_code in [200, 201]:
            print("Transmission Successful.")
        else:
            print(f"API Warning: Received status code {response.status_code}")
    except Exception as e:
        print(f"Supabase Error: {e}")


def main():
    """Main loop: listen on USB serial for CRT results from ESP32-C3."""
    print("=" * 50)
    print("   CRT Serial Receiver (Raspberry Pi)")
    print("=" * 50)

    while True:
        # Auto-detect the ESP32 serial port
        port = find_esp32_port()
        if port is None:
            print("[CRT] No ESP32 device found. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print(f"[CRT] Connecting to ESP32 on {port} @ {SERIAL_BAUDRATE} baud...")

        try:
            ser = serial.Serial(port, SERIAL_BAUDRATE, timeout=1.0)
            print(f"[CRT] Connected. Listening for CRT results...")

            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    # Log all serial output for debugging
                    print(f"[ESP32] {line}")

                    # Check for CRT result
                    if line.startswith(CRT_RESULT_PREFIX):
                        crft_result = line[len(CRT_RESULT_PREFIX):]
                        print(f"\n[CRT] Received result: {crft_result}")
                        print("[CRT] Uploading to Supabase...")
                        upload_to_supabase(crft_result)
                        print("[CRT] Waiting for next test...\n")

                time.sleep(0.01)  # Small sleep to prevent CPU spin

        except (serial.SerialException, OSError) as e:
            print(f"[CRT] Device disconnected: {e}. Reconnecting in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n[CRT] Shutting down.")
            break
        finally:
            try:
                ser.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()