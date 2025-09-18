import cv2
import numpy as np
import time
from datetime import datetime
import json
import os
import threading
try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("PyAudio not available. Audio monitoring disabled.")
from scipy.spatial import distance as dist

class EnhancedProctoringEngine:
    def __init__(self):
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load YOLO for object detection (if available)
        self.load_yolo_model()
        
        # Tracking variables
        self.last_face_time = time.time()
        self.focus_lost_start = None
        self.no_face_start = None
        self.drowsiness_start = None
        
        # Event thresholds
        self.FOCUS_LOSS_THRESHOLD = 5  # seconds
        self.NO_FACE_THRESHOLD = 10    # seconds
        self.DROWSINESS_THRESHOLD = 3  # seconds
        self.EYE_AR_THRESH = 0.25      # Eye aspect ratio threshold
        
        # Audio monitoring
        self.audio_monitoring = False
        self.background_noise_level = 0
        self.audio_thread = None
        
    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            weights_path = "models/yolov3.weights"
            config_path = "models/yolov3.cfg"
            names_path = "models/coco.names"
            
            if all(os.path.exists(path) for path in [weights_path, config_path, names_path]):
                self.net = cv2.dnn.readNet(weights_path, config_path)
                with open(names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                self.yolo_loaded = True
                print("YOLO model loaded successfully")
            else:
                print("YOLO model files not found. Using basic object detection.")
                self.yolo_loaded = False
                self.classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                              'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
                              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            self.yolo_loaded = False
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio for drowsiness detection"""
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_drowsiness(self, frame, faces):
        """Detect eye closure and drowsiness"""
        if not faces:
            return {'drowsy': False, 'eye_aspect_ratio': 0}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for face in faces:
            x, y, w, h = face['bbox']
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            
            if len(eyes) >= 2:
                # Calculate average eye aspect ratio
                total_ear = 0
                eye_count = 0
                
                for (ex, ey, ew, eh) in eyes:
                    # Simple approximation of eye landmarks for EAR calculation
                    eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Simplified EAR calculation based on eye region dimensions
                    if eh > 0 and ew > 0:
                        ear = eh / ew  # Height to width ratio as proxy for EAR
                        total_ear += ear
                        eye_count += 1
                
                if eye_count > 0:
                    avg_ear = total_ear / eye_count
                    
                    # Check if eyes are closed (low aspect ratio)
                    if avg_ear < self.EYE_AR_THRESH:
                        return {'drowsy': True, 'eye_aspect_ratio': avg_ear}
                    else:
                        return {'drowsy': False, 'eye_aspect_ratio': avg_ear}
        
        return {'drowsy': False, 'eye_aspect_ratio': 0}
    
    def start_audio_monitoring(self):
        """Start audio monitoring in a separate thread"""
        if not AUDIO_AVAILABLE:
            print("Audio monitoring not available - PyAudio not installed")
            return
            
        if not self.audio_monitoring:
            self.audio_monitoring = True
            self.audio_thread = threading.Thread(target=self._monitor_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
    
    def stop_audio_monitoring(self):
        """Stop audio monitoring"""
        self.audio_monitoring = False
        if self.audio_thread:
            self.audio_thread.join()
    
    def _monitor_audio(self):
        """Monitor audio for background voices"""
        if not AUDIO_AVAILABLE:
            return
            
        try:
            # Audio parameters
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            
            p = pyaudio.PyAudio()
            
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            print("Audio monitoring started")
            
            while self.audio_monitoring:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate RMS (Root Mean Square) for volume level
                    rms = np.sqrt(np.mean(audio_data**2))
                    
                    # Update background noise level (running average)
                    if self.background_noise_level == 0:
                        self.background_noise_level = rms
                    else:
                        self.background_noise_level = 0.9 * self.background_noise_level + 0.1 * rms
                    
                    # Detect significant audio activity
                    if rms > self.background_noise_level * 2:
                        # Potential background voice detected
                        pass  # This will be handled in process_frame
                        
                except Exception as e:
                    print(f"Audio monitoring error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"Audio initialization error: {e}")
    
    def detect_faces(self, frame):
        """Detect faces using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8  # Default confidence for Haar cascades
            })
        
        return face_data
    
    def detect_gaze_direction(self, frame, faces):
        """Simple gaze detection using eye position"""
        if not faces:
            return {'looking_at_camera': False, 'horizontal_deviation': 1.0, 'vertical_deviation': 1.0}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for face in faces:
            x, y, w, h = face['bbox']
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            
            if len(eyes) >= 2:
                # Calculate eye positions relative to face center
                face_center_x = w // 2
                face_center_y = h // 2
                
                eye_positions = []
                for (ex, ey, ew, eh) in eyes:
                    eye_center_x = ex + ew // 2
                    eye_center_y = ey + eh // 2
                    eye_positions.append((eye_center_x, eye_center_y))
                
                if len(eye_positions) >= 2:
                    # Calculate average eye position
                    avg_eye_x = sum(pos[0] for pos in eye_positions) / len(eye_positions)
                    avg_eye_y = sum(pos[1] for pos in eye_positions) / len(eye_positions)
                    
                    # Calculate deviation from center
                    horizontal_deviation = abs(avg_eye_x - face_center_x) / (w / 2)
                    vertical_deviation = abs(avg_eye_y - face_center_y) / (h / 2)
                    
                    # Simple threshold for "looking at camera"
                    looking_at_camera = horizontal_deviation < 0.3 and vertical_deviation < 0.3
                    
                    return {
                        'looking_at_camera': looking_at_camera,
                        'horizontal_deviation': horizontal_deviation,
                        'vertical_deviation': vertical_deviation
                    }
        
        return {'looking_at_camera': False, 'horizontal_deviation': 1.0, 'vertical_deviation': 1.0}
    
    def detect_objects_simple(self, frame):
        """Simple object detection using color and shape analysis"""
        detected_objects = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Phone detection (looking for rectangular dark objects)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:  # Phone-like size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it looks like a phone (rectangular)
                if 0.4 < aspect_ratio < 2.5:
                    roi = gray[y:y+h, x:x+w]
                    if np.mean(roi) < 100:  # Dark object
                        detected_objects.append({
                            'class': 'cell phone',
                            'confidence': 0.6,
                            'bbox': [x, y, w, h]
                        })
        
        # Book detection (looking for white/light rectangular objects)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 20000:  # Book-like size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Books are typically rectangular
                if 0.7 < aspect_ratio < 1.5:
                    detected_objects.append({
                        'class': 'book',
                        'confidence': 0.5,
                        'bbox': [x, y, w, h]
                    })
        
        return detected_objects
    
    def detect_objects(self, frame):
        """Detect objects using YOLO or simple detection"""
        if self.yolo_loaded:
            return self.detect_objects_yolo(frame)
        else:
            return self.detect_objects_simple(frame)
    
    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO"""
        try:
            height, width, channels = frame.shape
            
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward()
            
            class_ids = []
            confidences = []
            boxes = []
            
            suspicious_objects = ['cell phone', 'book', 'laptop', 'tablet', 'remote']
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.3:
                        class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
                        
                        if class_name in suspicious_objects:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            
            detected_objects = []
            if len(indexes) > 0:
                for i in indexes.flatten():
                    class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidences[i],
                        'bbox': boxes[i]
                    })
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in YOLO object detection: {e}")
            return self.detect_objects_simple(frame)
    
    def process_frame(self, frame):
        """Process a single frame and return proctoring events"""
        current_time = time.time()
        events = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        num_faces = len(faces)
        
        # Check for multiple faces
        if num_faces > 1:
            events.append({
                'type': 'multiple_faces',
                'timestamp': datetime.now(),
                'details': {'face_count': num_faces},
                'confidence': max([face['confidence'] for face in faces]) if faces else 0.5
            })
        
        # Check for no face
        if num_faces == 0:
            if self.no_face_start is None:
                self.no_face_start = current_time
            elif current_time - self.no_face_start >= self.NO_FACE_THRESHOLD:
                events.append({
                    'type': 'no_face',
                    'timestamp': datetime.now(),
                    'duration': current_time - self.no_face_start,
                    'details': {'threshold_exceeded': True}
                })
                self.no_face_start = current_time
        else:
            self.no_face_start = None
            self.last_face_time = current_time
        
        # Check gaze direction (focus detection)
        if num_faces >= 1:
            gaze_info = self.detect_gaze_direction(frame, faces)
            if not gaze_info['looking_at_camera']:
                if self.focus_lost_start is None:
                    self.focus_lost_start = current_time
                elif current_time - self.focus_lost_start >= self.FOCUS_LOSS_THRESHOLD:
                    events.append({
                        'type': 'focus_loss',
                        'timestamp': datetime.now(),
                        'duration': current_time - self.focus_lost_start,
                        'details': gaze_info
                    })
                    self.focus_lost_start = current_time
            else:
                self.focus_lost_start = None
        
        # Check for drowsiness
        if num_faces >= 1:
            drowsiness_info = self.detect_drowsiness(frame, faces)
            if drowsiness_info['drowsy']:
                if self.drowsiness_start is None:
                    self.drowsiness_start = current_time
                elif current_time - self.drowsiness_start >= self.DROWSINESS_THRESHOLD:
                    events.append({
                        'type': 'drowsiness_detected',
                        'timestamp': datetime.now(),
                        'duration': current_time - self.drowsiness_start,
                        'details': drowsiness_info
                    })
                    self.drowsiness_start = current_time
            else:
                self.drowsiness_start = None
        
        # Detect suspicious objects
        objects = self.detect_objects(frame)
        for obj in objects:
            event_type = 'phone_detected' if 'phone' in obj['class'] else 'notes_detected'
            events.append({
                'type': event_type,
                'timestamp': datetime.now(),
                'confidence': obj['confidence'],
                'details': {'object_class': obj['class'], 'bbox': obj['bbox']}
            })
        
        # Check audio levels for background voices
        if self.audio_monitoring and hasattr(self, 'background_noise_level'):
            # This would be implemented with real-time audio analysis
            # For now, we'll simulate it
            pass
        
        return {
            'events': events,
            'faces': faces,
            'objects': objects,
            'frame_processed': True
        }
    
    def draw_annotations(self, frame, analysis_result):
        """Draw bounding boxes and annotations on frame"""
        annotated_frame = frame.copy()
        
        # Draw face bounding boxes
        for face in analysis_result['faces']:
            x, y, w, h = face['bbox']
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Face: {face['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw object bounding boxes
        for obj in analysis_result['objects']:
            x, y, w, h = obj['bbox']
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"{obj['class']}: {obj['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add status text
        status_text = f"Faces: {len(analysis_result['faces'])}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame