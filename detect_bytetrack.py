"""
YOLOv9 + ByteTrack Multi-ROI Vehicle Detection and Counting
Optimized for Jetson AGX Orin with auto camera angle detection

Features:
- Multiple ROI creation with mouse clicking
- ByteTrack vehicle tracking with line crossing detection
- Auto camera angle detection (0-60°)
- YOLOv9 model loading and inference
- Real-time FPS calculation
- CSV logging with vehicle class percentages
- Interactive zone editing
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
import torch
from collections import OrderedDict, deque
import json
# Tambahkan import ROS di bagian awal file
import rospy
from std_msgs.msg import String

# YOLOv9 imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, 
                          non_max_suppression, scale_boxes, increment_path)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Global variables for ROI editing
EDITING_MODE = False
CURRENT_ROI = []
EDITING_LANE = 0
ROIS = []
COUNTING_LINES = []
ROI_FILLED = False  # Toggle for filled/outline ROI visualization
ZONE_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green  
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (128, 128, 128), # Gray
    (255, 192, 203)  # Pink
]

class CameraAngleDetector:
    """Auto-detect camera angle from video feed analysis"""
    
    def __init__(self):
        self.samples_needed = 10
        self.angle_samples = []
        
    def estimate_angle_from_objects(self, detections, frame_height):
        """Estimate camera angle by analyzing object size variation with distance"""
        if len(detections) < 3:
            return 30  # Default moderate angle
        
        # Extract object data: (y_position_ratio, area)
        object_data = []
        for det in detections:
            x1, y1, x2, y2 = det[:4].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center_y_ratio = ((y1 + y2) / 2) / frame_height
            object_data.append((center_y_ratio, area))
        
        if len(object_data) < 2:
            return 30
        
        # Sort by y-position (distance)
        object_data.sort(key=lambda x: x[0])
        
        # Compare objects at different distances
        top_third = [obj for obj in object_data if obj[0] < 0.33]
        bottom_third = [obj for obj in object_data if obj[0] > 0.67]
        
        if not top_third or not bottom_third:
            return 30
        
        # Calculate size ratio (perspective effect)
        avg_top_size = np.mean([obj[1] for obj in top_third])
        avg_bottom_size = np.mean([obj[1] for obj in bottom_third])
        
        if avg_top_size <= 0:
            return 30
        
        size_ratio = avg_bottom_size / avg_top_size
        
        # Convert size ratio to camera angle (empirical mapping)
        if size_ratio < 1.2:
            return 10   # Nearly overhead
        elif size_ratio < 2.0:
            return 20   # Slight angle
        elif size_ratio < 3.5:
            return 30   # Moderate angle (most common)
        elif size_ratio < 6.0:
            return 45   # Steep angle
        else:
            return 55   # Very steep angle
    
    def detect_angle(self, frame, detections):
        """Main angle detection method"""
        if len(self.angle_samples) >= self.samples_needed:
            return int(np.median(self.angle_samples))
        
        if len(detections) > 2:
            angle = self.estimate_angle_from_objects(detections, frame.shape[0])
            self.angle_samples.append(angle)
            print(f"🔍 Angle sample {len(self.angle_samples)}/{self.samples_needed}: {angle}°")
        
        if len(self.angle_samples) == self.samples_needed:
            final_angle = int(np.median(self.angle_samples))
            print(f"✅ Camera angle calibrated: {final_angle}°")
            return final_angle
        
        return None  # Still calibrating


class ByteTrackVehicleCounter:
    """Advanced ByteTrack vehicle counter with line crossing detection"""
    
    def __init__(self, roi_id, camera_angle=30, log_file=None):
        self.roi_id = roi_id
        self.camera_angle = camera_angle
        self.log_file = log_file or f"bytetrack_roi_{roi_id}_counts.csv"
        
        # Tracking parameters adjusted for camera angle
        self._setup_tracking_params()
        
        # Track management
        self.tracked_tracks = OrderedDict()
        self.lost_tracks = OrderedDict()
        self.frame_id = 0
        self.track_id_count = 0
        
        # Vehicle counting
        self.total_count = 0
        self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.counted_tracks = set()
        self.class_names = {0: "G1", 1: "G2", 2: "G3", 3: "G4", 4: "G5"}
        
        # Performance tracking
        self.track_history_length = 8
        
        # Initialize CSV log
        self._initialize_log()
        
        print(f"🚗 ByteTrack Counter initialized for ROI {roi_id}")
        print(f"📐 Camera angle: {camera_angle}° | Track thresh: {self.track_thresh}")
    
    def _setup_tracking_params(self):
        """Setup tracking parameters based on camera angle"""
        if self.camera_angle <= 15:
            self.track_thresh = 0.6
            self.track_buffer = 30
            self.match_thresh = 0.8
            self.min_box_area = 25
        elif self.camera_angle <= 25:
            self.track_thresh = 0.5
            self.track_buffer = 25
            self.match_thresh = 0.7
            self.min_box_area = 20
        elif self.camera_angle <= 35:
            self.track_thresh = 0.4
            self.track_buffer = 20
            self.match_thresh = 0.6
            self.min_box_area = 15
        elif self.camera_angle <= 45:
            self.track_thresh = 0.35
            self.track_buffer = 18
            self.match_thresh = 0.55
            self.min_box_area = 12
        else:  # > 45°
            self.track_thresh = 0.3
            self.track_buffer = 15
            self.match_thresh = 0.5
            self.min_box_area = 10
    
    def _initialize_log(self):
        """Initialize CSV log file"""
        try:
            with open(self.log_file, 'w') as f:
                f.write("Timestamp,ROI_ID,Total,G1,G2,G3,G4,G5,G1%,G2%,G3%,G4%,G5%\n")
        except Exception as e:
            print(f"⚠️  Could not initialize log file: {e}")
    
    def calculate_iou(self, box1, box2):
        """Fast IoU calculation optimized for tracking"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_perspective_iou(self, track_box, det_box):
        """Enhanced IoU for angled cameras with center distance consideration"""
        basic_iou = self.calculate_iou(track_box, det_box)
        
        # Calculate center points
        track_center = [(track_box[0] + track_box[2])/2, (track_box[1] + track_box[3])/2]
        det_center = [(det_box[0] + det_box[2])/2, (det_box[1] + det_box[3])/2]
        
        # Center distance factor
        distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                          (track_center[1] - det_center[1])**2)
        
        # Adjust max distance based on camera angle
        max_distance = 60 + (self.camera_angle * 1.5)
        distance_factor = max(0, 1 - distance / max_distance)
        
        # Combine IoU with center proximity (weight based on angle)
        iou_weight = 0.8 if self.camera_angle <= 30 else 0.6
        distance_weight = 1 - iou_weight
        
        enhanced_iou = basic_iou * iou_weight + distance_factor * distance_weight
        return enhanced_iou
    
    def update_tracks(self, detections, roi_mask):
        """Main ByteTrack update with perspective-aware matching"""
        self.frame_id += 1
        
        if len(detections) == 0 or not roi_mask.any():
            self._age_tracks()
            return []
        
        # Filter detections in ROI
        roi_detections = detections[roi_mask]
        
        if len(roi_detections) == 0:
            self._age_tracks()
            return []
        
        # Get current track data
        track_boxes = []
        track_ids = []
        for track_id, track in self.tracked_tracks.items():
            track_boxes.append(track['bbox'])
            track_ids.append(track_id)
        
        # Association using IoU matching
        if track_boxes and len(roi_detections) > 0:
            det_boxes = roi_detections[:, :4].cpu().numpy()
            
            # Calculate cost matrix
            cost_matrix = np.zeros((len(track_boxes), len(det_boxes)))
            for i, track_box in enumerate(track_boxes):
                for j, det_box in enumerate(det_boxes):
                    if self.camera_angle > 25:
                        iou = self.calculate_perspective_iou(track_box, det_box)
                    else:
                        iou = self.calculate_iou(track_box, det_box)
                    cost_matrix[i, j] = 1 - iou  # Convert to cost
            
            # Hungarian assignment (simplified greedy approach for edge computing)
            matches = []
            matched_tracks = set()
            matched_dets = set()
            
            # Greedy assignment
            for i in range(len(track_boxes)):
                for j in range(len(det_boxes)):
                    if (i not in matched_tracks and j not in matched_dets and 
                        cost_matrix[i, j] < (1 - self.match_thresh)):
                        matches.append((i, j))
                        matched_tracks.add(i)
                        matched_dets.add(j)
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                track_id = track_ids[track_idx]
                det = roi_detections[det_idx]
                self._update_track(track_id, det)
            
            # Create new tracks for unmatched detections
            for j in range(len(roi_detections)):
                if j not in matched_dets:
                    self._create_track(roi_detections[j])
            
            # Move unmatched tracks to lost
            for i in range(len(track_ids)):
                if i not in matched_tracks:
                    track_id = track_ids[i]
                    self.lost_tracks[track_id] = self.tracked_tracks[track_id]
                    del self.tracked_tracks[track_id]
        else:
            # No existing tracks, create new ones
            for det in roi_detections:
                self._create_track(det)
        
        # Clean up old tracks
        self._age_tracks()
        
        return list(self.tracked_tracks.values())
    
    def _create_track(self, detection):
        """Create new track from detection"""
        self.track_id_count += 1
        bbox = detection[:4].cpu().numpy()
        conf = float(detection[4].cpu().item())
        cls = int(detection[5].cpu().item())
        
        # Filter by confidence and size
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if conf >= self.track_thresh and box_area >= self.min_box_area:
            track = {
                'track_id': self.track_id_count,
                'bbox': bbox,
                'score': conf,
                'class': cls,
                'age': 1,
                'time_since_update': 0,
                'history': deque([bbox], maxlen=self.track_history_length),
                'confirmed': False
            }
            self.tracked_tracks[self.track_id_count] = track
    
    def _update_track(self, track_id, detection):
        """Update existing track"""
        track = self.tracked_tracks[track_id]
        bbox = detection[:4].cpu().numpy()
        conf = float(detection[4].cpu().item())
        cls = int(detection[5].cpu().item())
        
        track['bbox'] = bbox
        track['score'] = conf
        track['class'] = cls
        track['age'] += 1
        track['time_since_update'] = 0
        track['history'].append(bbox)
        
        # Confirm track after minimum observations
        if track['age'] >= 3:
            track['confirmed'] = True
    
    def _age_tracks(self):
        """Age tracks and remove old ones"""
        # Age tracked tracks
        for track_id in list(self.tracked_tracks.keys()):
            track = self.tracked_tracks[track_id]
            track['time_since_update'] += 1
            
            # Move to lost if not updated
            if track['time_since_update'] > 1:
                self.lost_tracks[track_id] = track
                del self.tracked_tracks[track_id]
        
        # Remove old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            track = self.lost_tracks[track_id]
            track['time_since_update'] += 1
            
            if track['time_since_update'] > self.track_buffer:
                del self.lost_tracks[track_id]
    
    def count_line_crossings(self, counting_line):
        """Count vehicles crossing the counting line"""
        if not counting_line or len(counting_line) != 4:
            return 0
        
        x1, y1, x2, y2 = counting_line
        new_counts = 0
        
        for track_id, track in self.tracked_tracks.items():
            # Only count confirmed tracks that haven't been counted
            if (not track['confirmed'] or 
                track_id in self.counted_tracks or 
                len(track['history']) < 2):
                continue
            
            # Get last two positions
            history = list(track['history'])
            prev_bbox = history[-2]
            curr_bbox = history[-1]
            
            # Calculate center points
            prev_center = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
            curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
            
            # Check line intersection
            if self._line_intersection(prev_center, curr_center, (x1, y1), (x2, y2)):
                vehicle_class = track['class']
                if vehicle_class in self.class_counts:
                    self.class_counts[vehicle_class] += 1
                    self.total_count += 1
                    self.counted_tracks.add(track_id)
                    new_counts += 1
                    
                    # Log the count
                    self._log_count()
                    
                    class_name = self.class_names.get(vehicle_class, f"Class_{vehicle_class}")
                    print(f"🚗 ROI {self.roi_id}: {class_name} counted (Track {track_id})")
        
        return new_counts
    
    def _line_intersection(self, p1, p2, p3, p4):
        """Check if line segment p1p2 intersects with line segment p3p4"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def get_percentages(self):
        """Calculate class percentages"""
        if self.total_count == 0:
            return {i: 0.0 for i in range(5)}
        return {i: (self.class_counts[i] / self.total_count) * 100 for i in range(5)}
    
    def _log_count(self):
        """Log count to CSV file"""
        try:
            percentages = self.get_percentages()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp},{self.roi_id},{self.total_count}"
            
            # Add counts
            for class_id in range(5):
                log_line += f",{self.class_counts[class_id]}"
            
            # Add percentages
            for class_id in range(5):
                log_line += f",{percentages[class_id]:.1f}"
            
            with open(self.log_file, 'a') as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"⚠️  Logging error: {e}")


def save_rois_to_file(filename="bytetrack_rois.json"):
    """Save ROIs and counting lines to JSON file"""
    roi_data = {
        "rois": [roi.tolist() if isinstance(roi, np.ndarray) else roi for roi in ROIS],
        "counting_lines": COUNTING_LINES,
        "created_timestamp": datetime.now().isoformat(),
        "total_rois": len(ROIS)
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(roi_data, f, indent=2)
        print(f"💾 ROIs saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving ROIs: {e}")
        return False


def load_rois_from_file(filename="bytetrack_rois.json"):
    """Load ROIs from JSON file"""
    global ROIS, COUNTING_LINES
    
    try:
        with open(filename, 'r') as f:
            roi_data = json.load(f)
        
        ROIS = [np.array(roi) for roi in roi_data["rois"]]
        COUNTING_LINES = roi_data["counting_lines"]
        
        print(f"📂 Loaded {len(ROIS)} ROIs from {filename}")
        return True
    except FileNotFoundError:
        print(f"📁 ROI file {filename} not found, starting with empty ROIs")
        return False
    except Exception as e:
        print(f"❌ Error loading ROIs: {e}")
        return False


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for ROI creation"""
    global EDITING_MODE, CURRENT_ROI, ROIS, COUNTING_LINES, EDITING_LANE
    
    if not EDITING_MODE:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point to current ROI
        CURRENT_ROI.append([x, y])
        print(f"📍 Point {len(CURRENT_ROI)} added: ({x}, {y})")
        
        # Complete ROI when 4 points are selected
        if len(CURRENT_ROI) == 4:
            # Create ROI polygon
            roi_points = np.array(CURRENT_ROI, dtype=np.int32)
            ROIS.append(roi_points)
            
            # Create counting line (diagonal across ROI)
            counting_line = [
                CURRENT_ROI[0][0], CURRENT_ROI[0][1],  # Top-left
                CURRENT_ROI[2][0], CURRENT_ROI[2][1]   # Bottom-right
            ]
            COUNTING_LINES.append(counting_line)
            
            # Save and reset
            save_rois_to_file()
            print(f"✅ ROI {len(ROIS)} created with counting line")
            print(f"   Points: {CURRENT_ROI}")
            print(f"   Counting line: {counting_line}")
            
            CURRENT_ROI = []
            
    elif event == cv2.EVENT_RBUTTONDOWN and CURRENT_ROI:
        # Remove last point
        removed = CURRENT_ROI.pop()
        print(f"❌ Removed point: {removed}")


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using OpenCV"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


@smart_inference_mode()
def run_bytetrack_detection(
    weights=ROOT / 'yolov5s.pt',
    source=ROOT / 'data/images',
    data=ROOT / 'data/coco.yaml',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    view_img=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / 'runs/detect',
    name='bytetrack_detection',
    exist_ok=False,
    line_thickness=2,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    camera_angle=None,  # None for auto-detection    
    roi_file='bytetrack_rois.json'
):
    global EDITING_MODE, CURRENT_ROI, ROIS, COUNTING_LINES, ROI_FILLED
    
    print("🚀 YOLOv9 + ByteTrack Multi-ROI Vehicle Detection Starting...")
    print(f"📊 Model: {weights}")
    print(f"📹 Source: {source}")
    
    # Initialize
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    
    if is_url and is_file:
        source = check_file(source)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    
    print(f"🤖 Model loaded: {model.__class__.__name__}")
    print(f"📱 Device: {device}")
    print(f"🎯 Classes: {list(names.values())}")
    
    # Load existing ROIs
    load_rois_from_file(roi_file)
    
    # Setup data loader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Initialize camera angle detector
    angle_detector = CameraAngleDetector() if camera_angle is None else None
    current_camera_angle = camera_angle or 30  # Default
    
    # Inisialisasi ROS publisher untuk klasifikasi kendaraan
    try:
        rospy.init_node('yolo_detection_node', anonymous=True)
        vehicle_pub = rospy.Publisher('/vehicle_classification', String, queue_size=10)
        print("🚀 ROS publisher initialized")
        print(f"Publisher camera: /vehicle_classification")
    except Exception as e:
        print(f"⚠️ ROS initialization error: {e}")
        vehicle_pub = None
    
    # Initialize ByteTrack counters for each ROI
    bytetrack_counters = []
    
    # Performance monitoring
    fps_history = deque(maxlen=30)
    inference_times = deque(maxlen=30)
    
    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    print("🔥 Model warmed up")
    
    # Setup display
    if view_img:
        window_name = 'YOLOv9 + ByteTrack Multi-ROI Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        print("\n🎮 Interactive Controls:")
        print("  E - Toggle editing mode")
        print("  R - Reset current ROI")
        print("  S - Save ROIs")
        print("  L - Load ROIs")
        print("  C - Clear all ROIs")
        print("  Q - Quit")
        print("  Mouse: Left click to add ROI points, Right click to remove last point")
    
    # Main detection loop
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    try:
        for path, im, im0s, vid_cap, s in dataset:
            frame_start_time = time.time()
            
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]
            
            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
            
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
            # Process predictions
            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                p = Path(p)
                save_path = str(save_dir / p.name)
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                
                # Auto-detect camera angle if needed
                if angle_detector and len(det) > 2:
                    detected_angle = angle_detector.detect_angle(im0, det)
                    if detected_angle is not None:
                        current_camera_angle = detected_angle
                        angle_detector = None  # Stop detection after calibration
                        print(f"🎯 Using camera angle: {current_camera_angle}°")
                
                # Initialize counters for ROIs if needed
                while len(bytetrack_counters) < len(ROIS):
                    roi_id = len(bytetrack_counters) + 1
                    counter = ByteTrackVehicleCounter(
                        roi_id=roi_id,
                        camera_angle=current_camera_angle,
                        log_file=f"bytetrack_roi_{roi_id}_counts.csv"
                    )
                    bytetrack_counters.append(counter)
                
                if len(det):
                    # Scale boxes
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    # Print detection results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    
                    # Draw detection boxes
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:
                            c = int(cls)
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    # Process each ROI
                    for roi_idx, (roi, counting_line, counter) in enumerate(zip(ROIS, COUNTING_LINES, bytetrack_counters)):
                        # Check which detections are in this ROI
                        roi_mask = []
                        for det_row in det:
                            bbox = det_row[:4].cpu().numpy()
                            center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                            roi_mask.append(point_in_polygon(center, roi))
                        
                        roi_mask = np.array(roi_mask)
                        
                        if roi_mask.any():
                            # Update ByteTrack for this ROI
                            tracks = counter.update_tracks(det, roi_mask)
                            
                            # Count line crossings
                            new_counts = counter.count_line_crossings(counting_line)
                            # Publish ke ROS topic jika line crossing terdeteksi
                            if new_counts > 0 and vehicle_pub is not None:
                                for class_id, count in counter.class_counts.items():
                                    if count > 0:
                                        golongan = class_id + 1  # Class 0 = Golongan 1
                                        jalur = counter.roi_id
                                        
                                        # Format pesan
                                        msg_data = f"G{golongan}|{jalur}"
                                        vehicle_pub.publish(msg_data)
                                        print(f"📤 Published to /vehicle_classification: {msg_data}")
                            # Draw tracks
                            for track in tracks:
                                if track['confirmed']:
                                    bbox = track['bbox']
                                    track_id = track['track_id']
                                    cls_name = counter.class_names.get(track['class'], f"C{track['class']}")
                                    
                                    # Draw track box
                                    cv2.rectangle(im0, 
                                                (int(bbox[0]), int(bbox[1])), 
                                                (int(bbox[2]), int(bbox[3])), 
                                                ZONE_COLORS[roi_idx % len(ZONE_COLORS)], 2)
                                    
                                    # Draw track ID
                                    cv2.putText(im0, f"ROI{roi_idx+1}_T{track_id}", 
                                              (int(bbox[0]), int(bbox[1])-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                              ZONE_COLORS[roi_idx % len(ZONE_COLORS)], 2)
                
                # Calculate FPS
                frame_time = time.time() - frame_start_time
                fps = 1.0 / frame_time
                fps_history.append(fps)
                
                # Calculate inference time
                inference_time = dt[1].dt * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                avg_fps = np.mean(fps_history)
                avg_inference = np.mean(inference_times)
                  # Draw ROIs and counting lines
                for roi_idx, (roi, counting_line) in enumerate(zip(ROIS, COUNTING_LINES)):
                    color = ZONE_COLORS[roi_idx % len(ZONE_COLORS)]
                    
                    # Draw ROI polygon - filled or outline based on toggle
                    if ROI_FILLED:
                        # Create a copy of the image to apply transparency
                        overlay = im0.copy()
                        cv2.fillPoly(overlay, [roi], color)
                        # Apply transparency (alpha blending)
                        alpha = 0.3  # Transparency level
                        cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0, im0)
                        # Draw outline on top for better visibility
                        cv2.polylines(im0, [roi], True, color, 2)
                    else:
                        cv2.polylines(im0, [roi], True, color, 2)
                    
                    # Draw counting line
                    cv2.line(im0, (counting_line[0], counting_line[1]), 
                            (counting_line[2], counting_line[3]), color, 3)
                    
                    # Draw ROI label
                    roi_center = roi.mean(axis=0).astype(int)
                    cv2.putText(im0, f"ROI {roi_idx+1}", tuple(roi_center-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw current editing ROI
                if EDITING_MODE and CURRENT_ROI:
                    points = np.array(CURRENT_ROI)
                    if len(points) > 1:
                        cv2.polylines(im0, [points], False, (0, 255, 255), 2)
                    for point in CURRENT_ROI:
                        cv2.circle(im0, tuple(point), 5, (0, 255, 255), -1)
                    
                    # Show progress
                    cv2.putText(im0, f"Creating ROI: {len(CURRENT_ROI)}/4 points", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw status panel
                panel_height = 200 + len(bytetrack_counters) * 80
                cv2.rectangle(im0, (10, 50), (400, 50 + panel_height), (0, 0, 0), -1)
                cv2.rectangle(im0, (10, 50), (400, 50 + panel_height), (255, 255, 255), 2)
                
                y_pos = 75
                
                # Performance info
                cv2.putText(im0, f"FPS: {avg_fps:.1f}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 25
                
                cv2.putText(im0, f"Inference: {avg_inference:.1f}ms", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25
                
                cv2.putText(im0, f"Camera Angle: {current_camera_angle}°", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25
                
                cv2.putText(im0, f"ROIs: {len(ROIS)}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25
                
                editing_status = "ON" if EDITING_MODE else "OFF"
                cv2.putText(im0, f"Edit Mode: {editing_status}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if EDITING_MODE else (255, 255, 255), 2)
                y_pos += 25
                
                roi_mode = "FILLED" if ROI_FILLED else "OUTLINE"
                cv2.putText(im0, f"ROI Mode: {roi_mode}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
                y_pos += 35
                
                # ROI counts
                for roi_idx, counter in enumerate(bytetrack_counters):
                    color = ZONE_COLORS[roi_idx % len(ZONE_COLORS)]
                    
                    cv2.putText(im0, f"ROI {counter.roi_id}: {counter.total_count} vehicles", 
                               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 20
                    
                    # Class breakdown
                    for class_id, count in counter.class_counts.items():
                        if count > 0:
                            class_name = counter.class_names[class_id]
                            percentage = counter.get_percentages()[class_id]
                            cv2.putText(im0, f"  {class_name}: {count} ({percentage:.1f}%)", 
                                       (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_pos += 18
                    y_pos += 10
                
                # Display
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(window_name, im0)
                      # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('e'):
                        EDITING_MODE = not EDITING_MODE
                        CURRENT_ROI = []
                        status = "ON" if EDITING_MODE else "OFF"
                        print(f"✏️ Editing mode: {status}")
                    elif key == ord('f'):
                        ROI_FILLED = not ROI_FILLED
                        mode = "FILLED" if ROI_FILLED else "OUTLINE"
                        print(f"🎨 ROI visualization: {mode}")
                    elif key == ord('r') and EDITING_MODE:
                        CURRENT_ROI = []
                        print("🔄 Current ROI reset")
                    elif key == ord('s'):
                        save_rois_to_file(roi_file)
                    elif key == ord('l'):
                        if load_rois_from_file(roi_file):
                            # Reinitialize counters
                            bytetrack_counters = []
                    elif key == ord('c'):
                        ROIS = []
                        COUNTING_LINES = []
                        bytetrack_counters = []
                        save_rois_to_file(roi_file)
                        print("🗑️ All ROIs cleared")
                
                # Save results
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path[i] != save_path:
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()
                            
                            if vid_cap:
                                fps_video = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps_video, w, h = 30, im0.shape[1], im0.shape[0]
                            
                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (w, h))
                        
                        vid_writer[i].write(im0)
                
                # Print progress (less frequent for performance)
                if seen % 30 == 0:
                    print(f"🎬 Frame {seen} | FPS: {avg_fps:.1f} | "
                          f"Inference: {avg_inference:.1f}ms | ROIs: {len(ROIS)}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Detection stopped by user")
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Final summary
    print(f"\n📊 Detection Summary:")
    print(f"   Frames processed: {seen}")
    print(f"   Average FPS: {np.mean(fps_history):.1f}")
    print(f"   Average inference time: {np.mean(inference_times):.1f}ms")
    print(f"   ROIs created: {len(ROIS)}")
    
    for roi_idx, counter in enumerate(bytetrack_counters):
        print(f"\n🚗 ROI {counter.roi_id} Summary:")
        print(f"   Total vehicles: {counter.total_count}")
        for class_id, count in counter.class_counts.items():
            if count > 0:
                class_name = counter.class_names[class_id]
                percentage = counter.get_percentages()[class_id]
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        print(f"   Log file: {counter.log_file}")
    
    # Save final ROIs
    save_rois_to_file(roi_file)
    
    print(f"\n✅ Detection completed. Results saved to {save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser(description='YOLOv9 + ByteTrack Multi-ROI Vehicle Detection')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='bytetrack_detection', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--camera-angle', type=int, help='camera angle in degrees (auto-detect if not specified)')
    parser.add_argument('--roi-file', type=str, default='bytetrack_rois.json', help='ROI save/load file')
    
    return parser.parse_args()


def main(opt):
    """Main function"""
    print("🚀 YOLOv9 + ByteTrack Multi-ROI Vehicle Detection and Counting")
    print("="*70)
    
    # Convert opt to dict and call detection function
    run_bytetrack_detection(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
