import cv2
import numpy as np
import os
import time
import datetime
import argparse
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import database
import socketio
from config_loader import get_config

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description="Video Analytics Detection")
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
args, _ = parser.parse_known_args()

# --- Load Configuration ---
config = get_config(args.config)

# --- SocketIO Client ---
sio = socketio.Client()
try:
    sio.connect('http://localhost:5000')
    print("Connected to WebSocket Server")
except Exception as e:
    print(f"WebSocket connection failed: {e} (Will try to match logic without real-time updates)")

# --- Configuration from config.yaml ---
CAMERA_SOURCE = config.camera_source
DB_PATH = config.db_path
FACES_DIR = config.faces_dir

# Detection settings
CONFIDENCE_THRESHOLD = config.get('detection.person_confidence', 0.5)
PHONE_CONF_THRESHOLD = config.get('detection.phone_confidence', 0.3)
PHONE_CLASS_ID = 67  # COCO class for cell phone
person_class_id = 0

# Timing settings
LOG_COOLDOWN = config.log_cooldown
PHONE_ALERT_COOLDOWN = config.phone_alert_cooldown
UNAUTHORIZED_FACE_GRACE = config.unauthorized_face_grace
UNAUTHORIZED_NO_FACE_GRACE = config.unauthorized_no_face_grace
TRACK_TIMEOUT = config.track_timeout

# Camera settings
RECONNECT_ATTEMPTS = config.reconnect_attempts
RECONNECT_DELAY = config.reconnect_delay
FRAME_SKIP = config.frame_skip


# --- State ---
class TrackerState:
    def __init__(self):
        # {track_id: {'name': 'Unknown', 'first_seen': time, 'last_seen': time, 'last_log': 0, 'last_phone_log': 0, 'face_attempts': 0}}
        self.person_map = {}
        self.known_embeddings = []
        self.known_names = []
        self.face_check_counter = 0  # Frame counter for face recognition optimization

    def load_faces(self, app):
        print("Loading registered faces...")
        self.known_embeddings = []
        self.known_names = []
        
        users = database.get_users()
        for user in users:
            user_id = user['id']
            name = user['name']
            user_dir = os.path.join(FACES_DIR, str(user_id))
            
            if os.path.exists(user_dir):
                embeddings = []
                for img_name in os.listdir(user_dir):
                    img_path = os.path.join(user_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None: continue
                    
                    faces = app.get(img)
                    if faces:
                        # Largest face
                        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                        embeddings.append(faces[0].normed_embedding)
                        
                if embeddings:
                    # Store all embeddings for better accuracy (multi-template matching)
                    for emb in embeddings:
                        self.known_embeddings.append(emb)
                        self.known_names.append(name)
        print(f"Loaded {len(self.known_names)} face embeddings.")

    def identify_face(self, face_embedding):
        if not self.known_embeddings:
            return "Unknown"

        sims = np.dot(self.known_embeddings, face_embedding)
        best_idx = np.argmax(sims)
        score = sims[best_idx]

        # Use configurable threshold
        if score > config.face_similarity_threshold:
            return self.known_names[best_idx]
        return "Unknown"

# --- Camera Connection with Reconnection Logic ---
def connect_camera(source, max_attempts=None):
    """
    Connect to camera with automatic reconnection support

    Args:
        source: Camera source (int for webcam, str for RTSP)
        max_attempts: Maximum reconnection attempts (None = infinite)

    Returns:
        VideoCapture object or None if failed
    """
    if max_attempts is None:
        max_attempts = RECONNECT_ATTEMPTS

    is_rtsp = isinstance(source, str) and source.lower().startswith('rtsp')
    is_http = isinstance(source, str) and (source.lower().startswith('http://') or source.lower().startswith('https://'))

    for attempt in range(max_attempts):
        try:
            print(f"Connecting to camera... (Attempt {attempt + 1}/{max_attempts})")

            if is_rtsp:
                # RTSP stream - don't use CAP_DSHOW
                cap = cv2.VideoCapture(source)
                print(f"Connecting to RTSP stream: {source}")
            elif is_http:
                # HTTP stream (IP Webcam, MJPEG, etc.) - don't use CAP_DSHOW
                cap = cv2.VideoCapture(source)
                print(f"Connecting to HTTP stream: {source}")
            else:
                # Local webcam - use DirectShow on Windows for better control
                if os.name == 'nt':  # Windows
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(source)
                print(f"Connecting to webcam: {source}")

            # Test if camera opened successfully
            if not cap.isOpened():
                raise Exception("Failed to open camera")

            # Try to read a test frame
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise Exception("Failed to read test frame")

            # For webcams, try to set resolution (won't work for RTSP/HTTP streams)
            if not is_rtsp and not is_http:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Get actual resolution
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ Camera connected! Resolution: {actual_w}x{actual_h}")

            return cap

        except Exception as e:
            print(f"✗ Connection failed: {e}")
            if attempt < max_attempts - 1:
                print(f"Retrying in {RECONNECT_DELAY} seconds...")
                time.sleep(RECONNECT_DELAY)
            else:
                print(f"Failed to connect after {max_attempts} attempts")
                return None

    return None


def reconnect_camera(cap, source):
    """
    Attempt to reconnect camera after connection loss

    Args:
        cap: Existing VideoCapture object (will be released)
        source: Camera source to reconnect to

    Returns:
        New VideoCapture object or None if failed
    """
    print("Camera connection lost. Attempting to reconnect...")

    if cap is not None:
        cap.release()

    return connect_camera(source, max_attempts=RECONNECT_ATTEMPTS)


# --- Main Detection Logic ---
def main():
    yolo_model = config.get('detection.yolo_model', 'yolov8n.pt')
    yolo_imgsz = config.get('detection.yolo_imgsz', 640)
    face_det_size = config.get('detection.face_det_size', 640)
    insightface_model = config.get('detection.insightface_model', 'buffalo_l')

    # Determine InsightFace providers
    # Try OpenVINO first (best for Intel CPUs), fall back to CPU
    providers = config.get('detection.onnx_providers', None)
    if providers is None:
        try:
            import openvino
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
            print("OpenVINO detected - using accelerated inference")
        except ImportError:
            providers = ['CPUExecutionProvider']

    print(f"Initializing YOLO ({yolo_model}, imgsz={yolo_imgsz})...")
    model = YOLO(yolo_model)

    print(f"Initializing InsightFace ({insightface_model}, det_size={face_det_size})...")
    # Only load detection + recognition models (skip genderage, landmarks)
    allowed_modules = config.get('detection.insightface_modules', ['detection', 'recognition'])
    app = FaceAnalysis(name=insightface_model, providers=providers, allowed_modules=allowed_modules)
    app.prepare(ctx_id=0, det_size=(face_det_size, face_det_size))
    
    state = TrackerState()
    state.load_faces(app)

    # Connect to camera with reconnection support
    cap = connect_camera(CAMERA_SOURCE)
    if cap is None:
        print("ERROR: Unable to connect to camera. Exiting...")
        return

    frame_counter = 0
    failed_frames = 0
    max_failed_frames = 30  # Reconnect after 30 consecutive failed frames

    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            failed_frames += 1
            print(f"Failed to read frame ({failed_frames}/{max_failed_frames})")

            if failed_frames >= max_failed_frames:
                print("Too many failed frames. Attempting reconnection...")
                cap = reconnect_camera(cap, CAMERA_SOURCE)

                if cap is None:
                    print("ERROR: Unable to reconnect to camera. Exiting...")
                    break

                failed_frames = 0

            time.sleep(0.1)
            continue

        # Reset failed frame counter on successful read
        failed_frames = 0
        frame_counter += 1

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:  # Update FPS every second
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Frame skipping for performance
        if FRAME_SKIP > 1 and frame_counter % FRAME_SKIP != 0:
            continue

        timestamp = time.time()
        
        # USE BoT-SORT for better tracking stability
        # Added agnostic_nms=True to reduce overlapping boxes (duplicate trackers)
        # Lowered conf=0.3 to catch phones better (default is often 0.25 or 0.5 depending on Ultralytics version, setting explicitly helps)
        results = model.track(frame, persist=True, classes=[0, 67], conf=0.3, tracker="botsort.yaml", verbose=False, agnostic_nms=True, iou=0.5, imgsz=yolo_imgsz)
        
        current_person_boxes = {} # track_id: box
        phone_boxes = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            
            # Manual filtering for heavy overlaps if NMS isn't enough
            person_dets = []
            
            for box in boxes:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                if cls == person_class_id:
                    if box.id is not None:
                        track_id = int(box.id[0])
                        person_dets.append({'id': track_id, 'box': xyxy, 'area': (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])})
                elif cls == PHONE_CLASS_ID:
                    phone_boxes.append(xyxy)
            
            # Simple deduplication: if two boxes overlap > 50%, keep largest
            # (Note: YOLO tracker *should* handle this, but ghost IDs happen)
            kept_ids = set()
            # Sort by area (largest first)
            person_dets.sort(key=lambda x: x['area'], reverse=True)
            
            for p in person_dets:
                is_duplicate = False
                p_box = p['box']
                p_area = p['area']
                
                for k_id in kept_ids:
                    # check overlap with already kept
                    k_box = current_person_boxes[k_id]
                    
                    # IoU calc
                    xA = max(p_box[0], k_box[0])
                    yA = max(p_box[1], k_box[1])
                    xB = min(p_box[2], k_box[2])
                    yB = min(p_box[3], k_box[3])
                    
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    if interArea > 0:
                        union = p_area + ((k_box[2]-k_box[0])*(k_box[3]-k_box[1])) - interArea
                        if interArea / union > 0.6: # High overlap
                            is_duplicate = True
                            break
                            
                if not is_duplicate:
                    current_person_boxes[p['id']] = p['box']
                    kept_ids.add(p['id'])

        # Clean up old tracks and log EXITS
        to_remove = []
        for track_id, p_state in state.person_map.items():
            # If not seen for X seconds, consider them gone
            if timestamp - p_state['last_seen'] > TRACK_TIMEOUT:
                # If they were known/identified, log the exit
                if p_state['name'] != 'Unknown':
                    print(f"EXIT: {p_state['name']} (ID: {track_id})")
                    database.log_event(None, p_state['name'], 'exit')
                    if sio.connected:
                        sio.emit('log_event', {'user_name': p_state['name'], 'event_type': 'exit', 'timestamp': database.get_local_time()})
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del state.person_map[track_id]

        for track_id, bbox in current_person_boxes.items():
            if track_id not in state.person_map:
                state.person_map[track_id] = {
                    'name': 'Unknown', 
                    'first_seen': timestamp,
                    'last_seen': timestamp, 
                    'last_log': 0, 
                    'last_phone_log': 0,
                    'face_attempts': 0,
                    'has_face_seen': False # New flag
                }
            
            p_state = state.person_map[track_id]
            p_state['last_seen'] = timestamp
            
            # --- FACE RECOGNITION ---
            # Try to identify if unknown
            if p_state['name'] == 'Unknown':
                # Performance optimization: Only check face every N frames
                face_check_interval = config.get('detection.face_check_interval', 5)
                state.face_check_counter += 1

                # Only process face recognition every N frames
                if state.face_check_counter % face_check_interval == 0:
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w, _ = frame.shape
                    pad = 15
                    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                    cx2, cy2 = min(w, x2+pad), min(h, y2+pad)

                    person_crop = frame[cy1:cy2, cx1:cx2]

                    # Only check face if crop is decent size
                    if person_crop.size > 0 and (cx2-cx1) > 50:
                        faces = app.get(person_crop)
                        if faces:
                            p_state['has_face_seen'] = True # We saw a face
                            largest_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
                            name = state.identify_face(largest_face.normed_embedding)

                            if name != "Unknown":
                                p_state['name'] = name
                                # LOG ENTRY
                                if timestamp - p_state['last_log'] > LOG_COOLDOWN:
                                    print(f"Identified {name} (ID: {track_id})")
                                    database.log_event(None, name, 'entry')
                                    # Emit to SocketIO
                                    if sio.connected:
                                        sio.emit('log_event', {'user_name': name, 'event_type': 'entry', 'timestamp': database.get_local_time()})
                                    p_state['last_log'] = timestamp
                            else:
                                p_state['face_attempts'] += 1

            # --- LOG UNAUTHORIZED ---
            # Logic Update:
            # 1. If we HAVE seen a face -> Wait UNAUTHORIZED_FACE_GRACE (3s)
            # 2. If we have NOT seen a face -> Wait UNAUTHORIZED_NO_FACE_GRACE (15s) (e.g. arm only)
            time_since_first_seen = timestamp - p_state['first_seen']
            
            is_unauthorized = False
            if p_state['name'] == 'Unknown':
                if p_state['has_face_seen']:
                    if time_since_first_seen > UNAUTHORIZED_FACE_GRACE:
                        is_unauthorized = True
                else:
                    if time_since_first_seen > UNAUTHORIZED_NO_FACE_GRACE:
                        is_unauthorized = True
            
            if is_unauthorized:
                 if timestamp - p_state['last_log'] > LOG_COOLDOWN:
                     print(f"UNAUTHORIZED Person Detected (ID: {track_id})")
                     database.log_event(None, f"Unknown (ID: {track_id})", 'unauthorized')
                     if sio.connected:
                         sio.emit('log_event', {'user_name': f"Unknown (ID: {track_id})", 'event_type': 'unauthorized', 'timestamp': database.get_local_time()})
                     p_state['last_log'] = timestamp

            # --- PHONE DETECTION ---
            has_phone = False
            for ph_box in phone_boxes:
                px1, py1, px2, py2 = map(int, bbox)
                phx1, phy1, phx2, phy2 = map(int, ph_box)
                ph_cx, ph_cy = (phx1 + phx2) / 2, (phy1 + phy2) / 2
                
                # If phone center is roughly inside person box
                if px1 < ph_cx < px2 and py1 < ph_cy < py2:
                    has_phone = True
                    break
            
            if has_phone:
                if timestamp - p_state['last_phone_log'] > PHONE_ALERT_COOLDOWN:
                    msg = f"Phone Detected on {p_state['name']} (ID: {track_id})"
                    print(msg)
                    # REMOVED SNAPSHOT based on feedback
                    database.log_event(None, p_state['name'], 'phone_detected')
                    if sio.connected:
                        sio.emit('log_event', {'user_name': p_state['name'], 'event_type': 'phone_detected', 'timestamp': database.get_local_time()})
                    p_state['last_phone_log'] = timestamp
            
            # --- DRAW UI ---
            color = (0, 255, 0) if p_state['name'] != 'Unknown' else (0, 0, 255)
            # If unknown but within grace period, show Yellow
            if p_state['name'] == 'Unknown':
               if p_state['has_face_seen'] and time_since_first_seen < UNAUTHORIZED_FACE_GRACE:
                   color = (0, 255, 255) # Yellow (Checking face...)
               elif not p_state['has_face_seen'] and time_since_first_seen < UNAUTHORIZED_NO_FACE_GRACE:
                   color = (255, 165, 0) # Orange (Waiting for face...)
                
            label = f"{p_state['name']} ({track_id})"
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if has_phone:
                cv2.putText(frame, "PHONE DETECTED", (int(bbox[0]), int(bbox[1])-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # --- ORPHANED PHONE LISTENING ---
        # Any phone that WASN'T in a person box?
        # Actually, let's just draw ALL phones to be safe and ensure visibility
        for ph_box in phone_boxes:
            phx1, phy1, phx2, phy2 = map(int, ph_box)
            cv2.rectangle(frame, (phx1, phy1), (phx2, phy2), (0, 255, 255), 2)
            cv2.putText(frame, "PHONE", (phx1, phy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Global Alert (if not already handled inside a person track, we could handle it here,
            # but usually it finds a person. If your arm isn't detected as a person, this ensures we at least SEE the phone).
            # Optional: Log "Phone Detected (No Person)" if needed, but drawing it helps user verify sensitivity.
            pass

        # Display FPS counter
        cv2.putText(frame, f"FPS: {current_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # RESIZE FOR DISPLAY if needed (optional)
        # cv2.imshow handles 1080p fine on most screens, but we can make it fit
        cv2.imshow("Video Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    database.init_db()
    main()
