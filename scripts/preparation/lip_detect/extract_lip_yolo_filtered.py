
import cv2
import numpy as np
import pickle
import argparse
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from collections import deque

# Hugging Face & Ultralytics
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Landmark detectors
import face_alignment
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Constants & Configuration ---
LIP_INDICES_MEDIAPIPE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]
# FaceAlignment (68 points) lip indices: Outer (48-59), Inner (60-67)
LIP_INDICES_FACEALIGNMENT = list(range(48, 68))

def calculate_mar(lip_pts):
    """
    Calculates Mouth Aspect Ratio (MAR) from a set of lip points.
    Simple heuristic: (Height) / (Width)
    """
    if len(lip_pts) == 0:
        return 0.0
    
    lip_pts = np.array(lip_pts)
    min_x, min_y = np.min(lip_pts, axis=0)
    max_x, max_y = np.max(lip_pts, axis=0)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w < 1e-5:
        return 0.0
        
    return h / w

def get_lip_bbox(lip_pts):
    """
    Calculates the 1:1 bounding box centered on the lips.
    """
    if len(lip_pts) == 0:
        return None

    lip_pts = np.array(lip_pts)
    min_x, min_y = np.min(lip_pts, axis=0)
    max_x, max_y = np.max(lip_pts, axis=0)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    w = max_x - min_x
    h = max_y - min_y
    
    # Use 1.5x margin of the largest dimension
    size = max(w, h) * 1.5
    half_size = size / 2.0
    
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size
    
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def crop_and_resize(frame, bbox, target_size=(96, 96)):
    """
    Crops the frame based on bbox and resizes it to target_size.
    Handles boundaries by padding with zeros.
    """
    if bbox is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    fh, fw, _ = frame.shape
    x1, y1, x2, y2 = bbox

    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))

    bw = ix2 - ix1
    bh = iy2 - iy1

    if bw <= 0 or bh <= 0:
         return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(fw, ix2)
    src_y2 = min(fh, iy2)

    dst_x1 = src_x1 - ix1
    dst_y1 = src_y1 - iy1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop = np.zeros((bh, bw, 3), dtype=frame.dtype)

    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    try:
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    except Exception:
        resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return resized

class FaceAlignmentHandler:
    def __init__(self, device='cuda'):
        print(f"Initializing FaceAlignment on {device}...")
        try:
             # S3FD is accurate but crashes on too small inputs
             self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        except Exception as e:
             print(f"Error initializing FaceAlignment: {e}")
             sys.exit(1)

    def get_landmarks(self, frame_rgb, bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h, w, _ = frame_rgb.shape
        
        # Add some margin
        margin_x = (x2 - x1) * 0.1
        margin_y = (y2 - y1) * 0.1
        cx1 = max(0, int(x1 - margin_x))
        cy1 = max(0, int(y1 - margin_y))
        cx2 = min(w, int(x2 + margin_x))
        cy2 = min(h, int(y2 + margin_y))
        
        face_crop = frame_rgb[cy1:cy2, cx1:cx2]
        
        # FIX: Avoid tiny crops that crash S3FD
        if face_crop.shape[0] < 32 or face_crop.shape[1] < 32:
            return None

        # OPTIMIZATION: Skip S3FD detection inside face_alignment.
        # Since we already cropped the face, we tell it the face is the entire crop.
        # detected_faces format: [(x1, y1, x2, y2)]
        h_crop, w_crop, _ = face_crop.shape
        detected_faces = [(0, 0, w_crop, h_crop)]
        
        preds = self.fa.get_landmarks(face_crop, detected_faces=detected_faces)
        
        if preds:
            landmarks = preds[0]
            landmarks[:, 0] += cx1
            landmarks[:, 1] += cy1
            return landmarks
        return None

    def get_lip_points(self, landmarks):
        if landmarks is None:
            return []
        return landmarks[LIP_INDICES_FACEALIGNMENT]

class MediaPipeHandler:
    def __init__(self):
        print("Initializing MediaPipe FaceLandmarker...")
        model_filename = "face_landmarker.task"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_asset_path = os.path.join(script_dir, model_filename)
        
        if not os.path.exists(model_asset_path):
             print(f"Downloading MediaPipe model...")
             url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
             import urllib.request
             urllib.request.urlretrieve(url, model_asset_path)

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1, 
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=vision.RunningMode.IMAGE)
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame_rgb, bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h, w, _ = frame_rgb.shape
        
        margin_x = (x2 - x1) * 0.1
        margin_y = (y2 - y1) * 0.1
        cx1 = max(0, int(x1 - margin_x))
        cy1 = max(0, int(y1 - margin_y))
        cx2 = min(w, int(x2 + margin_x))
        cy2 = min(h, int(y2 + margin_y))
        
        face_crop = frame_rgb[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            return None
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
        detection_result = self.landmarker.detect(mp_image)
        
        if detection_result.face_landmarks:
            landmarks_norm = detection_result.face_landmarks[0]
            crop_h, crop_w, _ = face_crop.shape
            
            landmarks = []
            for norm_pt in landmarks_norm:
                px = norm_pt.x * crop_w + cx1
                py = norm_pt.y * crop_h + cy1
                landmarks.append([px, py])
            
            return np.array(landmarks)
        return None

    def get_lip_points(self, landmarks):
        if landmarks is None:
            return []
        lip_pts = []
        for idx in LIP_INDICES_MEDIAPIPE:
             lip_pts.append(landmarks[idx])
        return np.array(lip_pts)


def main():
    parser = argparse.ArgumentParser(description="Extract lip region using YOLOv8 face detection + Landmarks (Filtered).")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--landmark_method", type=str, default="face_alignment", choices=["face_alignment", "mediapipe"], help="Landmark detection method")
    parser.add_argument("--device", type=str, default='cuda', help="Device for FaceAlignment/YOLO (cuda or cpu)")
    parser.add_argument("--min_speaking_threshold", type=float, default=0.01, help="Minimum MAR variance to consider a face as speaking. Below this, output is black.")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    device_name = args.device

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load YOLO Model ---
    print("Loading YOLOv8-Face-Detection model...")
    try:
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        yolo_model = YOLO(model_path)
        yolo_model.to(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)

    # --- 2. Initialize Landmark Detector ---
    if args.landmark_method == "face_alignment":
        landmark_detector = FaceAlignmentHandler(device=device_name)
    else:
        landmark_detector = MediaPipeHandler()

    # --- 3. Process Video ---
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Processing '{input_path}' with {args.landmark_method}")
    print(f"Resolution: {int(width)}x{int(height)}, FPS: {fps}, Frames: {total_frames}")

    # 1. args.input에서 경로를 제외한 '파일명.확장자'만 추출
    base_name = os.path.basename(args.input) 

    # 2. 확장자를 제거하고 이름만 추출
    file_stem = os.path.splitext(base_name)[0]

    # 3. 새로운 파일명 생성 및 출력 폴더와 결합
    out_vid_path = os.path.join(output_dir, f"{file_stem}.lip.mp4")
    # out_vid_path = os.path.join(output_dir, f"{os.path.splitext(args.input)[0]}.lip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_vid_path, fourcc, fps, (96, 96))

    coords_list = []
    
    # Speaker Identification State
    mar_histories = {} # {track_id: deque(maxlen=10)}

    frame_idx = 0
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- YOLO Face Tracking ---
            results = yolo_model.track(frame_rgb, persist=True, verbose=False)
            
            current_frame_mar = {} 
            current_frame_bboxes = {} 
            current_frame_face_bboxes = {} 
            
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Get persistent id
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else int(i + 1000)
                    
                    if track_id not in mar_histories:
                        mar_histories[track_id] = deque(maxlen=10)
                    
                    # Extract Landmarks
                    landmarks = landmark_detector.get_landmarks(frame_rgb, xyxy)
                    if landmarks is not None:
                        lip_pts = landmark_detector.get_lip_points(landmarks)
                        mar = calculate_mar(lip_pts)
                        mar_histories[track_id].append(mar)
                        
                        current_frame_mar[track_id] = mar
                        current_frame_bboxes[track_id] = get_lip_bbox(lip_pts)
                        current_frame_face_bboxes[track_id] = xyxy

            # --- Speaker Selection (MAR Variance) ---
            winner_id = None
            max_var = -1.0
            
            for tid, history in mar_histories.items():
                if tid in current_frame_bboxes: 
                    if len(history) >= 2:
                        # Use standard deviation as the 'speaking' score
                        score = np.std(history)
                    else:
                        score = 0.0
                    
                    if score > max_var:
                        max_var = score
                        winner_id = tid
            
            # Filter: Check if the 'winner' is actually speaking
            if max_var < args.min_speaking_threshold:
                # NO ONE IS SPEAKING (or just reacting/listening)
                winner_id = None

            best_bbox_coords = current_frame_bboxes.get(winner_id) if winner_id is not None else None
            best_face_bbox = current_frame_face_bboxes.get(winner_id) if winner_id is not None else None
            
            if best_bbox_coords is not None:
                coords_list.append(best_face_bbox)
                out_frame = crop_and_resize(frame, best_bbox_coords, (96, 96))
            else:
                coords_list.append(np.zeros(4, dtype=np.float32))
                out_frame = np.zeros((96, 96, 3), dtype=np.uint8)
            
            out_vid.write(out_frame)
            
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out_vid.release()

    # Save Coordinates
    out_pkl_path = os.path.join(output_dir, f"{file_stem}.bbox.pkl")
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(coords_list, f)

    print(f"\nProcessing complete.")
    print(f"Video saved to: {out_vid_path}")
    print(f"Coords saved to: {out_pkl_path}")

if __name__ == "__main__":
    main()
