# new_lancher v2
import os
import subprocess
import sys
import threading
import time
import atexit
from datetime import datetime

# Define absolute paths based on the launcher's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRVT_DIR = os.path.join(BASE_DIR, 'crvt')
PCV_DIR = os.path.join(BASE_DIR, 'pcv')
BP_DIR = os.path.join(BASE_DIR, 'bp')
HR_DIR = os.path.join(BASE_DIR, 'hr')

# Track background processes to terminate them cleanly on exit
active_processes = []

def cleanup():
    """Clean up background processes on exit to prevent resource leaks."""
    print("\n[Launcher] Shutting down background services...")
    for proc in active_processes:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

atexit.register(cleanup)

def background_service_worker():
    """Runs BP, HR, and CRT services continuously in the background and logs their output."""
    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    bp_log_path = os.path.join(logs_dir, 'bp_service.log')
    hr_log_path = os.path.join(logs_dir, 'hr_service.log')
    crt_log_path = os.path.join(logs_dir, 'crt_service.log')
    
    with open(bp_log_path, 'a') as bp_log, \
         open(hr_log_path, 'a') as hr_log, \
         open(crt_log_path, 'a') as crt_log:

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        bp_log.write(f"\n=========================================\n")
        bp_log.write(f"STARTING BP SERVICE AT {timestamp}\n")
        bp_log.write(f"=========================================\n")
        bp_log.flush()
        
        hr_log.write(f"\n=========================================\n")
        hr_log.write(f"STARTING HR SERVICE AT {timestamp}\n")
        hr_log.write(f"=========================================\n")
        hr_log.flush()

        crt_log.write(f"\n=========================================\n")
        crt_log.write(f"STARTING CRT SERVICE AT {timestamp}\n")
        crt_log.write(f"=========================================\n")
        crt_log.flush()

        # --- Start BP Service ---
        try:
            bp_proc = subprocess.Popen(
                [sys.executable, 'main.py'],
                cwd=BP_DIR,
                stdout=bp_log,
                stderr=bp_log
            )
            active_processes.append(bp_proc)
        except Exception as e:
            bp_log.write(f"Error starting BP service: {e}\n")
            bp_log.flush()
            bp_proc = None

        # --- Start HR Service ---
        try:
            hr_proc = subprocess.Popen(
                [sys.executable, 'main.py'],
                cwd=HR_DIR,
                stdout=hr_log,
                stderr=hr_log
            )
            active_processes.append(hr_proc)
        except Exception as e:
            hr_log.write(f"Error starting HR service: {e}\n")
            hr_log.flush()
            hr_proc = None

        # --- Start CRT Serial Listener Service ---
        try:
            crt_proc = subprocess.Popen(
                [sys.executable, 'crt.py'],
                cwd=CRVT_DIR,
                stdout=crt_log,
                stderr=crt_log
            )
            active_processes.append(crt_proc)
        except Exception as e:
            crt_log.write(f"Error starting CRT service: {e}\n")
            crt_log.flush()
            crt_proc = None

        # Keep the thread alive and monitor the processes. Re-spawn them if they crash.
        while True:
            time.sleep(5.0)
            
            # Check BP service
            if bp_proc and bp_proc.poll() is not None:
                bp_log.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] BP service exited unexpectedly. Re-spawning...\n")
                bp_log.flush()
                try:
                    if bp_proc in active_processes:
                        active_processes.remove(bp_proc)
                    bp_proc = subprocess.Popen(
                        [sys.executable, 'main.py'],
                        cwd=BP_DIR,
                        stdout=bp_log,
                        stderr=bp_log
                    )
                    active_processes.append(bp_proc)
                except Exception as e:
                    bp_log.write(f"Error re-spawning BP service: {e}\n")
                    bp_log.flush()

            # Check HR service
            if hr_proc and hr_proc.poll() is not None:
                hr_log.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] HR service exited unexpectedly. Re-spawning...\n")
                hr_log.flush()
                try:
                    if hr_proc in active_processes:
                        active_processes.remove(hr_proc)
                    hr_proc = subprocess.Popen(
                        [sys.executable, 'main.py'],
                        cwd=HR_DIR,
                        stdout=hr_log,
                        stderr=hr_log
                    )
                    active_processes.append(hr_proc)
                except Exception as e:
                    hr_log.write(f"Error re-spawning HR service: {e}\n")
                    hr_log.flush()

            # Check CRT service
            if crt_proc and crt_proc.poll() is not None:
                crt_log.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] CRT service exited unexpectedly. Re-spawning...\n")
                crt_log.flush()
                try:
                    if crt_proc in active_processes:
                        active_processes.remove(crt_proc)
                    crt_proc = subprocess.Popen(
                        [sys.executable, 'crt.py'],
                        cwd=CRVT_DIR,
                        stdout=crt_log,
                        stderr=crt_log
                    )
                    active_processes.append(crt_proc)
                except Exception as e:
                    crt_log.write(f"Error re-spawning CRT service: {e}\n")
                    crt_log.flush()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("=" * 50)
        print("         MEDICAL DIAGNOSTIC LAUNCHER")
        print("=" * 50)
        print("\n  Automatic Background Services: ACTIVE")
        print("  - BP Service: Listening on TCP Port 8080 (ESP32)")
        print("  - HR Service: Listening on System Microphone")
        print("  - CRT Service: Listening on USB Serial (ESP32-C3)")
        print("\n  [P] Run Packed Cell Volume (PCV) Test")
        print("  [L] View Background Service Logs")
        print("  [Q] Quit")
        print("\n" + "-" * 50)
        
        choice = input("Enter your choice: ").strip().upper()
        
        if choice == 'P':
            print("\nStarting PCV Test...")
            # We run the command and wait for it to finish
            subprocess.run([sys.executable, 'pcv.py'], cwd=PCV_DIR)
            input("\nPCV Test closed. Press ENTER to return to menu.")
            
        elif choice == 'L':
            clear_screen()
            print("=" * 50)
            print("         BACKGROUND SERVICE LOG MONITOR")
            print("=" * 50)
            
            bp_log_path = os.path.join(BASE_DIR, 'logs', 'bp_service.log')
            hr_log_path = os.path.join(BASE_DIR, 'logs', 'hr_service.log')
            crt_log_path = os.path.join(BASE_DIR, 'logs', 'crt_service.log')
            
            print("\n--- BP Service Log (Last 10 lines) ---")
            if os.path.exists(bp_log_path):
                with open(bp_log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(line, end='')
            else:
                print("No log file found yet.")
                
            print("\n--- HR Service Log (Last 10 lines) ---")
            if os.path.exists(hr_log_path):
                with open(hr_log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(line, end='')
            else:
                print("No log file found yet.")

            print("\n--- CRT Service Log (Last 10 lines) ---")
            if os.path.exists(crt_log_path):
                with open(crt_log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(line, end='')
            else:
                print("No log file found yet.")
                
            input("\nPress ENTER to return to menu.")
            
        elif choice == 'Q':
            print("\nExiting Launcher. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            input("\nPress ENTER to continue.")

if __name__ == "__main__":
    # Start background service thread as daemon so it exits when main thread exits
    bg_thread = threading.Thread(target=background_service_worker, daemon=True)
    bg_thread.start()
    main()