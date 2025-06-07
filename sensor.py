#!/usr/bin/env python3
# filepath: sensor.py

import rospy
from std_msgs.msg import String
import serial
import time
import random
import csv
import os

class RFIDSensorPublisher:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, csv_file='/home/ros_ws/expanded_vehicle_dataset.csv'):
        # Inisialisasi node ROS
        rospy.init_node('rfid_sensor_publisher', anonymous=True)
        
        # Buat publisher
        self.sensor_pub = rospy.Publisher('/rfid_sensor_data', String, queue_size=10)
        
        # Coba buka koneksi serial dengan ESP32
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"📡 ESP32 UART Serial connected on {port}")
            print(f"Publisher sensor: /rfid_sensor_data")
        except Exception as e:
            print(f"⚠️ Serial connection error: {e}")
            print(f"⚠️ Will continue in simulation mode")
        
        # Load RFID tags from CSV for simulation
        self.rfid_tags = []
        if os.path.exists(csv_file):
            print(f"🔄 Loading RFID tags from {csv_file}")
            try:
                with open(csv_file, 'r') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)  # Skip header
                    for row in csv_reader:
                        if row and len(row) > 0:  # Make sure row isn't empty
                            self.rfid_tags.append(row[0])  # Get only the RFID tag ID
                print(f"✅ Loaded {len(self.rfid_tags)} RFID tags from CSV")
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                # Fallback to simple simulation
                self.rfid_tags = []
        
        # Parameter simulasi (untuk digunakan jika koneksi ESP32 gagal)
        self.simulation_counter = 1000
        
        # Run loop
        self.run()
    
    def run(self):
        """Loop utama untuk membaca data dan publish"""
        rate = rospy.Rate(10)  # 10 Hz
        print("🚀 RFID Sensor Publisher started")
        
        while not rospy.is_shutdown():
            rfid_data = None
            
            # Coba baca data dari ESP32
            if self.ser:
                try:
                    if self.ser.in_waiting > 0:
                        rfid_data = self.ser.readline().decode().strip()
                except Exception as e:
                    print(f"⚠️ Serial read error: {e}")
            
            # Jika tidak ada data dari ESP32, gunakan simulasi
            if not rfid_data:
                # Simulasi data hanya setiap 5 detik untuk tidak membanjiri log
                if int(time.time()) % 5 == 0 and time.time() % 1 < 0.1:
                    # Use CSV data if available, otherwise use simple counter
                    if self.rfid_tags:
                        rfid_data = random.choice(self.rfid_tags)
                        print("⚠️ Using simulated RFID data from CSV")
                    else:
                        self.simulation_counter += 1
                        rfid_data = f"RFID_SIM_{self.simulation_counter}"
                        print("⚠️ Using simple simulated RFID data")
            
            # Publish data jika tersedia
            if rfid_data:
                self.sensor_pub.publish(rfid_data)
                print(f"📤 Published: {rfid_data}")
            
            rate.sleep()

if __name__ == "__main__":
    try:
        publisher = RFIDSensorPublisher()
    except rospy.ROSInterruptException:
        pass