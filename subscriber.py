#!/usr/bin/env python3
# filepath: subscriber.py

import rospy
from std_msgs.msg import String
import json
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

class MLFFSubscriber:
    def __init__(self):
        # Inisialisasi node ROS
        rospy.init_node('mlff_subscriber', anonymous=True)
        
        # Inisialisasi Firebase
        self.initialize_firebase()
        
        # Variabel untuk menyimpan data
        self.received_camera_data = None
        self.received_sensor_data = None
        self.last_camera_time = 0
        self.last_sensor_time = 0
        
        # Subscribe ke topic kamera dan sensor
        rospy.Subscriber('/vehicle_classification', String, self.camera_callback)
        rospy.Subscriber('/rfid_sensor_data', String, self.sensor_callback)
        
        print("🚀 MLFF Subscriber initialized")
        print("Subscriber topics: /vehicle_classification, /rfid_sensor_data")
        if self.db:
            print("📊 Firebase integration active - using project: mlff-e2d86")
        
        # Run loop
        self.run()
    
    def initialize_firebase(self):
        """Inisialisasi koneksi Firebase"""
        try:
            # Gunakan kredensial dari file service account
            cred = credentials.Certificate("mlff-firebase-key.json")
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("🔥 Firebase connected successfully")
        except Exception as e:
            print(f"❌ Firebase connection error: {e}")
            self.db = None
    
    def camera_callback(self, data):
        """Callback untuk data klasifikasi kamera"""
        self.received_camera_data = data.data
        self.last_camera_time = time.time()
        print(f"📥 Received from camera: {data.data}")
        self.check_data()
    
    def sensor_callback(self, data):
        """Callback untuk data sensor RFID"""
        self.received_sensor_data = data.data
        self.last_sensor_time = time.time()
        print(f"📥 Received from sensor: {data.data}")
        self.check_data()
    
    def check_data(self):
        """Periksa apakah kedua data tersedia dan proses"""
        if self.received_camera_data and self.received_sensor_data:
            print("\n✅ Complete data received!")
            print(f"Camera data: {self.received_camera_data}")
            print(f"Sensor data: {self.received_sensor_data}")
            
            # Proses data untuk Firebase
            self.send_data_to_firebase()
            
            # Reset sensor data, tapi pertahankan camera data
            self.received_sensor_data = None
    
    def send_data_to_firebase(self):
        """Kirim data ke Firebase"""
        if not self.db:
            print("⚠️ Firebase not connected, skipping data upload")
            return
        
        try:
            # Parse data
            try:
                # Format data kamera: "G{golongan}|{jalur}"
                camera_parts = self.received_camera_data.split('|')
                golongan = camera_parts[0].replace('G', '')
                jalur = camera_parts[1] if len(camera_parts) > 1 else "1"
                
                # Format data sensor (tergantung format yang dikirim ESP32)
                sensor_data = self.received_sensor_data
            except Exception as e:
                print(f"⚠️ Error parsing data: {e}")
                golongan = "1"
                jalur = "1"
                sensor_data = self.received_sensor_data
            
            # Buat dokumen untuk dikirim ke Firestore
            transaction_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'golongan': golongan,
                'jalur': jalur,
                'rfid_data': sensor_data,
                'processed': True
            }
            
            # Tambahkan ke koleksi transactions
            self.db.collection('transactions').add(transaction_data)
            print(f"✅ Data sent to Firebase: {transaction_data}")
        
        except Exception as e:
            print(f"❌ Firebase upload error: {e}")
    
    def run(self):
        """Loop utama"""
        rate = rospy.Rate(1)  # 1 Hz
        print("⏳ Waiting for messages...")
        
        while not rospy.is_shutdown():
            # Print status periodically
            if self.received_camera_data:
                age = time.time() - self.last_camera_time
                print(f"Last camera data: {self.received_camera_data} ({age:.1f}s ago)")
            
            if self.received_sensor_data:
                age = time.time() - self.last_sensor_time
                print(f"Last sensor data: {self.received_sensor_data} ({age:.1f}s ago)")
            
            rate.sleep()

if __name__ == "__main__":
    try:
        subscriber = MLFFSubscriber()
    except rospy.ROSInterruptException:
        pass