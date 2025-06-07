FROM osrf/ros:noetic-desktop

# First, update without the key to ensure apt works
RUN apt-get update || true

# Install key management tools and Python pip first
RUN apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    git \
    && apt-get clean

# Clean up existing keys properly
RUN rm -f /etc/apt/sources.list.d/ros*.list && \
    apt-key del F42ED6FBAB17C654 || true

# Add the updated key using the current method
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

# Add ROS repository
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# Update and install ROS dependencies
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    ros-noetic-ros-base \
    ros-noetic-std-msgs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Now we can use pip3 to install Firebase and other packages
RUN pip3 install --no-cache-dir firebase-admin
RUN pip3 install --no-cache-dir pyserial
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir opencv-python

# Install ROS Python packages
RUN pip3 install --no-cache-dir rospkg catkin_pkg

# Install YOLOv9 dependencies
RUN pip3 install --no-cache-dir pandas matplotlib seaborn scipy tqdm pyyaml IPython

# Install a smaller CPU-only PyTorch for build purposes
RUN pip3 install --no-cache-dir torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/torch_stable.html || \
    echo "PyTorch installation failed, but continuing build..."

# Clone YOLOv9 repository
WORKDIR /home
RUN git clone https://github.com/WongKinYiu/yolov9.git

# Create and set working directory
WORKDIR /home/ros_ws

# Copy Firebase credentials
COPY mlff-firebase-key.json /home/ros_ws/

# Copy files
COPY detect_bytetrack_fixed.py /home/ros_ws/
COPY subscriber.py /home/ros_ws/
COPY sensor.py /home/ros_ws/
COPY best3_sgd3.pt /home/ros_ws/
COPY launch.sh /home/ros_ws/
COPY entrypoint.sh /entrypoint.sh
COPY waskita_baru_1.mp4 /home/ros_ws/waskita_baru_1.mp4
COPY expanded_vehicle_dataset.csv /home/ros_ws/expanded_vehicle_dataset.csv

# Make scripts executable
RUN chmod +x /entrypoint.sh
RUN chmod +x /home/ros_ws/launch.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]