#!/usr/bin/env python3
"""
ROS Subscriber for RFID and Camera Detection Integration
Saves data to Firebase Realtime Database when both inputs are detected
"""

import rospy
from std_msgs.msg import String
import json
import os
from datetime import datetime
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Firebase Configuration
FIREBASE_KEY_PATH = "mlff-firebase-key.json"
LOG_FILE = "subscriber.log"
FIREBASE_DB_URL = "https://mlff-e2d86-default-rtdb.firebaseio.com/"

# Global variables to track latest data
latest_rfid = None
latest_camera = None
latest_rfid_time = 0
latest_camera_time = 0

# Cooldown period to prevent duplicate entries (in seconds)
DATA_COOLDOWN = 5.0

# Firebase client
firebase_app = None

def init_firebase():
    """Initialize Firebase Realtime Database connection"""
    global firebase_app
    
    try:
        # Check if the key file exists
        if not os.path.exists(FIREBASE_KEY_PATH):
            log_message(f"❌ Firebase key file not found: {FIREBASE_KEY_PATH}")
            return False
        
        # Initialize Firebase Admin SDK with Realtime Database
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_app = firebase_admin.initialize_app(cred, {
            'databaseURL': FIREBASE_DB_URL
        })
        
        log_message("✅ Firebase Realtime Database initialized successfully")
        return True
    except Exception as e:
        log_message(f"❌ Firebase initialization error: {e}")
        return False

def log_message(message):
    """Log a message to console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Print to stdout (forcing flush to ensure output isn't buffered)
    print(log_entry, flush=True)
    
    try:
        # Ensure log file directory exists
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Use append mode and ensure file is closed after writing
        with open(LOG_FILE, "a") as log_file:
            log_file.write(log_entry + "\n")
            log_file.flush()  # Force flush to disk
            os.fsync(log_file.fileno())  # Ensure data is written to disk
    except Exception as e:
        print(f"Could not write to log file: {e}", flush=True)

def save_last_values():
    """Save the last detected values to a log file"""
    data = {
        "last_rfid": latest_rfid,
        "last_camera": latest_camera,
        "last_rfid_time": datetime.fromtimestamp(latest_rfid_time).strftime("%Y-%m-%d %H:%M:%S") if latest_rfid_time else None,
        "last_camera_time": datetime.fromtimestamp(latest_camera_time).strftime("%Y-%m-%d %H:%M:%S") if latest_camera_time else None
    }
    
    try:
        with open("last_values.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_message(f"❌ Error saving last values: {e}")

def rfid_callback(data):
    """Handle RFID data from the RFID reader"""
    global latest_rfid, latest_rfid_time
    
    rfid_value = data.data.strip()
    current_time = time.time()
    
    # Skip if the same value was received recently
    if latest_rfid == rfid_value and (current_time - latest_rfid_time) < DATA_COOLDOWN:
        return
    
    # Format timestamp for log
    time_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]
    
    # Always print with old and new values
    if latest_rfid:
        log_message(f"📡 RFID Update: {latest_rfid} → {rfid_value} at {time_str}")
    else:
        log_message(f"📡 RFID Received: {rfid_value} at {time_str}")
    
    latest_rfid = rfid_value
    latest_rfid_time = current_time
    save_last_values()
    
    # Always print latest status after update
    log_message(f"📊 Current values - RFID: {latest_rfid}, Camera: {latest_camera}")
    
    # Try to process data if both values are available
    process_detection()

def camera_callback(data):
    """Handle vehicle classification data from the camera"""
    global latest_camera, latest_camera_time
    
    camera_value = data.data.strip()
    current_time = time.time()
    
    # Skip if the same value was received recently
    if latest_camera == camera_value and (current_time - latest_camera_time) < DATA_COOLDOWN:
        return
    
    # Parse camera data (format: "G2|1" for Golongan 2, Lane 1)
    try:
        parts = camera_value.split('|')
        vehicle_class = parts[0]  # G1, G2, etc.
        lane = int(parts[1]) if len(parts) > 1 else 1
        
        # Format timestamp for log
        time_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]
        
        # Always print with old and new values
        if latest_camera:
            log_message(f"🎥 Camera Update: {latest_camera} → {camera_value} at {time_str}")
        else:
            log_message(f"🎥 Camera Received: {camera_value} at {time_str}")
        
        latest_camera = camera_value
        latest_camera_time = current_time
        save_last_values()
        
        # Always print latest status after update
        log_message(f"📊 Current values - RFID: {latest_rfid}, Camera: {latest_camera}")
        
        # Try to process data if both values are available
        process_detection()
    except Exception as e:
        log_message(f"❌ Error parsing camera data: {e}")

def process_detection():
    """Process detection if both RFID and camera data are available"""
    if latest_rfid and latest_camera:
        # Check if both values were received within a reasonable time window (10 seconds)
        time_diff = abs(latest_rfid_time - latest_camera_time)
        if time_diff <= 10.0:
            # Both values are recent, save to Firebase
            save_to_firebase()
        else:
            log_message(f"⚠️ Time difference too large: {time_diff:.1f} seconds")

def save_to_firebase():
    """Save detection data to Firebase Realtime Database"""
    global latest_rfid, latest_camera
    
    if not firebase_app:
        log_message("❌ Firebase not initialized")
        return
    
    try:
        # Parse camera data
        parts = latest_camera.split('|')
        vehicle_class = parts[0]  # G1, G2, etc.
        lane = int(parts[1]) if len(parts) > 1 else 1
        
        # Prepare data for Realtime Database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_data = {
            "timestamp": timestamp,
            "rfid_value": latest_rfid,
            "vehicle_class": vehicle_class,
            "lane": lane,
            "is_processed": False,
            "created_at": {".sv": "timestamp"}  # Server timestamp
        }
        
        # Add data to 'vehicle_validation' node with timestamp as key
        ref = db.reference('/vehicle_validation')
        new_entry = ref.push(detection_data)
        entry_key = new_entry.key
        
        log_message(f"✅ Data saved to Firebase: RFID={latest_rfid}, Class={vehicle_class}, Lane={lane}")
        log_message(f"📋 Firebase entry key: {entry_key}")
        log_message(f"🌐 View in Firebase console: https://console.firebase.google.com/project/mlff-e2d86/database/data/vehicle_validation/{entry_key}")
        
        # Reset the values after saving
        latest_rfid = None
        latest_camera = None
        save_last_values()
        
    except Exception as e:
        log_message(f"❌ Firebase error: {e}")

def main():
    """Main function to initialize ROS node and subscribers"""
    try:
        # Set log file path and clear it at startup
        global LOG_FILE
        LOG_FILE = os.path.join(os.getcwd(), "subscriber.log")
        
        # Clear log file at startup to ensure clean log
        with open(LOG_FILE, 'w') as f:
            f.write("")
        
        # Configure ROS logging to minimize unnecessary output
        import logging
        logging.getLogger('rosout').setLevel(logging.ERROR)
        
        # Print debugging info
        print(f"Starting subscriber node at {datetime.now()}", flush=True)
        print(f"Log file will be saved to: {LOG_FILE}", flush=True)
        
        # Initialize ROS node
        rospy.init_node('mlff_integration_node', anonymous=True, log_level=rospy.ERROR)
        log_message("🚀 MLFF Integration Node started")
        
        # Initialize Firebase
        if not init_firebase():
            log_message("⚠️ Continuing without Firebase connection")
        
        # Subscribe to RFID and camera topics
        rospy.Subscriber('/rfid_sensor_data', String, rfid_callback)
        rospy.Subscriber('/vehicle_classification', String, camera_callback)
        log_message("📡 Subscribed to /rfid_sensor_data and /vehicle_classification topics")
        
        # Print output directly to terminal for testing
        print("📡 Now listening for RFID and camera data...", flush=True)
        
        # Check subscriber status - reduced interval to minimize log spam
        def check_subscribers_status():
            while not rospy.is_shutdown():
                topics = rospy.get_published_topics()
                rfid_found = False
                camera_found = False
                
                for topic_name, topic_type in topics:
                    if topic_name == '/rfid_sensor_data':
                        rfid_found = True
                    if topic_name == '/vehicle_classification':
                        camera_found = True
                
                status_msg = "Topic status: "
                status_msg += "RFID ✅ " if rfid_found else "RFID ❌ "
                status_msg += "Camera ✅" if camera_found else "Camera ❌"
                
                log_message(status_msg)
                
                # Display latest data
                if latest_rfid or latest_camera:
                    log_message(f"Latest values - RFID: {latest_rfid}, Camera: {latest_camera}")
                
                # Check every 60 seconds to reduce spam
                rospy.sleep(60)
        
        # Run checking in a separate thread
        import threading
        status_thread = threading.Thread(target=check_subscribers_status)
        status_thread.daemon = True
        status_thread.start()
        
        # Spin to keep the node alive
        rospy.spin()
    except rospy.ROSInterruptException:
        log_message("⚠️ ROS node interrupted")
    except Exception as e:
        log_message(f"❌ Error in main function: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
    finally:
        log_message("🛑 MLFF Integration Node stopped")

if __name__ == '__main__':
    main()