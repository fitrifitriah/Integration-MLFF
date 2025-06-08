#!/usr/bin/env python3
"""
Run system with separated log output - hanya menampilkan subscriber.log di main console
"""

import subprocess
import threading
import sys
import os
import time
import signal

def run_command(command, log_file=None, show_output=False):
    """Run a command and optionally redirect output to a file"""
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                command,
                stdout=f if not show_output else subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True
            )
            
            if show_output:
                # Jika perlu menampilkan output di konsol dan menyimpan ke file
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    f.write(line)
                    f.flush()
                process.wait()
            else:
                return process
    else:
        # Jalankan dan tampilkan output ke konsol
        process = subprocess.Popen(
            command,
            stdout=None if show_output else subprocess.DEVNULL,
            stderr=None if show_output else subprocess.DEVNULL,
            text=True,
            shell=True
        )
        return process

def main():
    # Buat direktori logs jika belum ada
    os.makedirs("logs", exist_ok=True)
    
    print("🚀 Starting MLFF Integration System")
    print("💡 Sensor and detector logs will be saved to separate files")
    
    try:
        # Jalankan sensor.py dengan output disimpan ke file terpisah
        sensor_cmd = "docker exec -it mlff-integration python3 /home/ros_ws/sensor.py"
        sensor_process = run_command(sensor_cmd, "logs/sensor.log")
        print("📡 Started RFID sensor (logs redirected to logs/sensor.log)")
        
        # Jalankan detector dengan output disimpan ke file terpisah
        detector_cmd = "docker exec -it mlff-integration python3 /home/ros_ws/detect_bytetrack_fixed.py"
        detector_process = run_command(detector_cmd, "logs/detector.log")
        print("🎥 Started camera detector (logs redirected to logs/detector.log)")
        
        # Jalankan subscriber dengan output ditampilkan di konsol
        subscriber_cmd = "docker exec -it mlff-integration python3 /home/ros_ws/subscriber.py"
        subscriber_process = run_command(subscriber_cmd, None, show_output=True)
        print("🔄 Started subscriber node (output shown here)")
        
        # Menunggu hingga user menekan Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping all processes...")
        
        # Hentikan semua proses
        for process in [sensor_process, detector_process, subscriber_process]:
            if process and process.poll() is None:  # Proses masih berjalan
                process.terminate()
                process.wait(timeout=5)
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("✅ All processes stopped")

if __name__ == "__main__":
    main()
