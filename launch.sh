#!/bin/bash

# MLFF System Launcher
echo "===================================================="
echo "🚀 MLFF System Launcher"
echo "===================================================="

# Determine if running on Jetson by checking for CUDA
if [ -d "/usr/local/cuda" ]; then
    DEVICE="cuda:0"
    echo "🖥️ Running on Jetson with CUDA"
else
    DEVICE="cpu"
    echo "🖥️ Running on CPU only"
fi

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Start roscore if not already running
if ! pgrep -x roscore > /dev/null; then
    echo "Starting ROS core..."
    roscore &
    # Wait for roscore to start
    sleep 3
else
    echo "ROS core is already running"
fi

# Buat logs directory
mkdir -p logs

# Copy model weight ke direktori YOLOv9
mkdir -p /home/yolov9
cp /home/ros_ws/best3_sgd3.pt /home/yolov9/

# Set PYTHONPATH untuk YOLOv9
export PYTHONPATH=$PYTHONPATH:/home/yolov9

# Link detect_bytetrack.py ke directory YOLOv9
cp /home/ros_ws/detect_bytetrack_fixed.py /home/yolov9/detect_bytetrack_fixed.py

# Determine source (camera or video)
if [ -e "/dev/video0" ] && [ "$DEVICE" = "cuda:0" ]; then
    SOURCE="0"
    echo "📹 Using camera as source"
else
    SOURCE="/home/ros_ws/waskita_baru_1.mp4"
    echo "🎞️ Using video file"
fi

# Start YOLOv9 detector dengan parameter yang sesuai
echo "1. Starting YOLOv9 detector..."
if [ "$DEVICE" = "cuda:0" ]; then
    # GPU settings
    cd /home/yolov9 && python3 detect_bytetrack_fixed.py --source $SOURCE --device $DEVICE --weights 'best3_sgd3.pt' --half > /home/ros_ws/logs/detector.log 2>&1 &
else
    # CPU settings (lighter)
    cd /home/yolov9 && python3 detect_bytetrack_fixed.py --source $SOURCE --device $DEVICE --weights 'best3_sgd3.pt' --img 416 --vid-stride 4 > /home/ros_ws/logs/detector.log 2>&1 &
fi
DETECTOR_PID=$!
echo "   YOLOv9 detector started (PID: $DETECTOR_PID)"

# Kembali ke direktori utama
cd /home/ros_ws

# Start sensor reader
echo "2. Starting sensor reader..."
python3 sensor.py > logs/sensor.log 2>&1 &
SENSOR_PID=$!
echo "   Sensor reader started (PID: $SENSOR_PID)"

# Start subscriber with Firebase integration
echo "3. Starting MLFF subscriber with Firebase..."
python3 subscriber.py > logs/subscriber.log 2>&1 &
SUBSCRIBER_PID=$!
echo "   MLFF subscriber started (PID: $SUBSCRIBER_PID)"

echo "===================================================="
echo "✅ All MLFF components started successfully!"
echo "Logs are being saved to logs/ directory"
echo "Press Ctrl+C to stop all components"
echo "===================================================="

# Wait a moment for logs to be created
sleep 2

# Show logs in real-time (with error handling)
find logs -type f -name "*.log" | xargs tail -f || echo "No log files found yet"

# Wait for all processes
wait