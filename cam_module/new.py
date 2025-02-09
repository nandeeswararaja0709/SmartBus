# Previous imports remain the same
import cv2
import numpy as np
import logging
import json
from datetime import datetime
import os
import signal
import sys
from threading import Thread
import time
import csv
from collections import deque

class BusCounterSystem:
    def __init__(self, config_path="config.json"):
        # Initialize logging first for error tracking
        self.setup_basic_logging()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Complete logging setup
        self.setup_full_logging()
        
        # Initialize camera and model
        self.setup_camera()
        self.setup_model()
        
        # Initialize counters and tracking
        self.counts = {
            "up": 0,
            "down": 0,
            "timestamp": [],
            "hourly_stats": {}
        }
        self.previous_centroids = {}
        self.frame_buffer = deque(maxlen=30)
        
        # System status
        self.is_running = True
        self.system_status = {
            "camera_ok": True,
            "model_ok": True,
            "storage_ok": True
        }
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_basic_logging(self):
        """Initial basic logging setup for startup errors"""
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')

    def load_config(self, config_path):
        default_config = {
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 15,
                "exposure": -1,
                "source": 0
            },
            "model": {
                "prototxt": "MobileNetSSD_deploy.prototxt",
                "caffemodel": "MobileNetSSD_deploy.caffemodel",
                "confidence_threshold": 0.8,
                "try_gpu": False  # Default to CPU for compatibility
            },
            "processing": {
                "skip_frames": 2,
                "brightness_adjustment": 1.2,
                "contrast_adjustment": 1.1
            },
            "storage": {
                "log_path": "logs/bus_counter.log",
                "data_path": "data/",
                "backup_interval": 3600
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logging.info(f"Loaded configuration from {config_path}")
                return {**default_config, **config}
        except FileNotFoundError:
            logging.info(f"Config file not found at {config_path}. Using defaults.")
            return default_config
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in config file. Using defaults.")
            return default_config

    def setup_full_logging(self):
        """Complete logging setup with file output"""
        try:
            os.makedirs(os.path.dirname(self.config['storage']['log_path']), exist_ok=True)
            file_handler = logging.FileHandler(self.config['storage']['log_path'])
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logging.error(f"Failed to setup file logging: {str(e)}")

    def setup_camera(self):
        """Initialize the camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(self.config['camera']['source'])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
                
            logging.info("Camera initialized successfully")
            
        except Exception as e:
            self.system_status['camera_ok'] = False
            logging.error(f"Camera initialization failed: {str(e)}")
            raise

    def setup_model(self):
        """Initialize the neural network model with proper error handling"""
        try:
            # Check if model files exist
            if not os.path.exists(self.config['model']['prototxt']):
                raise FileNotFoundError(f"Prototxt file not found: {self.config['model']['prototxt']}")
            if not os.path.exists(self.config['model']['caffemodel']):
                raise FileNotFoundError(f"Caffemodel file not found: {self.config['model']['caffemodel']}")

            # Load the model
            self.net = cv2.dnn.readNetFromCaffe(
                self.config['model']['prototxt'],
                self.config['model']['caffemodel']
            )
            
            # Try GPU only if configured
            if self.config['model']['try_gpu']:
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logging.info("Successfully enabled GPU acceleration")
                except:
                    logging.warning("GPU acceleration not available, using CPU")
            else:
                logging.info("Using CPU for inference (GPU not enabled in config)")
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            self.system_status['model_ok'] = False
            logging.error(f"Model initialization failed: {str(e)}")
            raise

    def preprocess_frame(self, frame):
        # Apply brightness and contrast adjustment
        frame = cv2.convertScaleAbs(
            frame,
            alpha=self.config['processing']['brightness_adjustment'],
            beta=self.config['processing']['contrast_adjustment']
        )
        
        # Apply adaptive histogram equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return frame

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        current_centroids = {}
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config['model']['confidence_threshold']:
                idx = int(detections[0, 0, i, 1])
                if idx != 15:  # Skip if not person (15 is person class in COCO)
                    continue
                    
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate centroid
                centroidX = int((startX + endX) / 2)
                centroidY = int((startY + endY) / 2)
                
                # Apply temporal consistency check
                centroid_id = self.match_centroid((centroidX, centroidY))
                if centroid_id is not None:
                    current_centroids[centroid_id] = (centroidX, centroidY)
                    
                    # Draw visualization
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.circle(frame, (centroidX, centroidY), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {centroid_id}", (startX, startY - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return current_centroids

    def match_centroid(self, current_centroid, max_distance=30):
        # Match current centroid with existing tracks
        for id_, prev_centroid in self.previous_centroids.items():
            distance = np.sqrt(
                (current_centroid[0] - prev_centroid[0])**2 +
                (current_centroid[1] - prev_centroid[1])**2
            )
            if distance < max_distance:
                return id_
        
        # If no match found, create new ID
        new_id = max(self.previous_centroids.keys(), default=-1) + 1
        return new_id

    def update_counts(self, current_centroids, center_line):
        timestamp = datetime.now()
        
        for id_, (cx, cy) in current_centroids.items():
            if id_ in self.previous_centroids:
                prev_cx, prev_cy = self.previous_centroids[id_]
                
                if prev_cy < center_line and cy > center_line:
                    self.counts["down"] += 1
                    self.counts["timestamp"].append({
                        "direction": "down",
                        "time": timestamp.isoformat()
                    })
                    self.update_hourly_stats("down", timestamp)
                    
                elif prev_cy > center_line and cy < center_line:
                    self.counts["up"] += 1
                    self.counts["timestamp"].append({
                        "direction": "up",
                        "time": timestamp.isoformat()
                    })
                    self.update_hourly_stats("up", timestamp)
        
        self.previous_centroids = current_centroids.copy()

    def update_hourly_stats(self, direction, timestamp):
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        if hour_key not in self.counts["hourly_stats"]:
            self.counts["hourly_stats"][hour_key] = {"up": 0, "down": 0}
        self.counts["hourly_stats"][hour_key][direction] += 1

    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = os.path.join(self.config['storage']['data_path'], timestamp)
        os.makedirs(data_path, exist_ok=True)
        
        # Save counts
        with open(os.path.join(data_path, 'counts.json'), 'w') as f:
            json.dump(self.counts, f, indent=4)
        
        # Save hourly stats to CSV
        with open(os.path.join(data_path, 'hourly_stats.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hour', 'Up', 'Down', 'Total'])
            for hour, stats in self.counts["hourly_stats"].items():
                writer.writerow([
                    hour,
                    stats['up'],
                    stats['down'],
                    stats['up'] + stats['down']
                ])

    def signal_handler(self, signum, frame):
        logging.info("Shutdown signal received")
        self.is_running = False

    def run(self):
        frame_count = 0
        last_backup = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                self.system_status['camera_ok'] = False
                continue

            frame_count += 1
            if frame_count % self.config['processing']['skip_frames'] != 0:
                continue

            # Preprocess frame
            frame = self.preprocess_frame(frame)
            
            # Store in buffer
            self.frame_buffer.append(frame.copy())
            
            # Get frame dimensions and draw center line
            height, width = frame.shape[:2]
            center_line = height // 2
            cv2.line(frame, (0, center_line), (width, center_line),
                    (0, 0, 255), 2)
            
            # Detect and track objects
            current_centroids = self.detect_objects(frame)
            self.update_counts(current_centroids, center_line)
            
            # Display counts and status
            cv2.putText(frame, f"UP: {self.counts['up']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"DOWN: {self.counts['down']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display system status
            status_color = (0, 255, 0) if all(self.system_status.values()) else (0, 0, 255)
            cv2.putText(frame, "System OK" if all(self.system_status.values()) else "System Warning",
                       (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Show frame
            cv2.imshow('Bus Counter', frame)
            
            # Periodic backup
            if time.time() - last_backup > self.config['storage']['backup_interval']:
                self.save_data()
                last_backup = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        # Cleanup
        self.save_data()
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("System shutdown complete")

def main():
    try:
        # Create config directory if it doesn't exist
        if not os.path.exists('config.json'):
            default_config = {
                "camera": {
                    "width": 640,
                    "height": 480,
                    "fps": 15,
                    "exposure": -1,
                    "source": 0
                },
                "model": {
                    "prototxt": "MobileNetSSD_deploy.prototxt",
                    "caffemodel": "MobileNetSSD_deploy.caffemodel",
                    "confidence_threshold": 0.8,
                    "try_gpu": False
                },
                "processing": {
                    "skip_frames": 2,
                    "brightness_adjustment": 1.2,
                    "contrast_adjustment": 1.1
                },
                "storage": {
                    "log_path": "logs/bus_counter.log",
                    "data_path": "data/",
                    "backup_interval": 3600
                }
            }
            os.makedirs('config', exist_ok=True)
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            logging.info("Created default config.json file")
        
        counter = BusCounterSystem()
        counter.run()
        
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please ensure you have the following files in your directory:")
        print("1. MobileNetSSD_deploy.prototxt")
        print("2. MobileNetSSD_deploy.caffemodel")
        print("You can download them from: https://github.com/chuanqi305/MobileNet-SSD")
        
    except Exception as e:
        logging.error(f"System error: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()