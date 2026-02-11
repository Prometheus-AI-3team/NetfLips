
import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse
import os
import sys
import urllib.request

# Use the new Tasks API as 'solutions' is not available in this environment
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define lip landmarks indices
# These indices are consistent with the mesh topology used by FaceLandmarker (478 landmarks)
LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]

def get_lip_bbox_and_mar(landmarks, frame_w, frame_h):
    """
    Calculates the bounding box (square, centered) and MAR (Mouth Aspect Ratio).
    Returns:
        bbox: np.array([x1, y1, x2, y2], dtype=float32)
        mar: float (height / width aspect ratio of the lip cloud)
    """
    # Extract lip point coordinates
    lip_pts = []
    for idx in LIP_INDICES:
        # Safety check for index bound
        if idx < len(landmarks):
            pt = landmarks[idx]
            # pt.x and pt.y are normalized [0, 1]
            lip_pts.append([pt.x * frame_w, pt.y * frame_h])
    
    if not lip_pts:
        return None, 0.0

    lip_pts = np.array(lip_pts, dtype=np.float32)

    # Determine bounds of the lip points
    min_x, min_y = np.min(lip_pts, axis=0)
    max_x, max_y = np.max(lip_pts, axis=0)

    w = max_x - min_x
    h = max_y - min_y

    # Calculate MAR
    # Avoid division by zero
    mar = h / w if w > 1e-5 else 0.0

    # Calculate Center
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Determine BBox size
    # 1.5x of the max dimension is a safe margin.
    size = max(w, h) * 1.5
    
    # Ensure square 1:1
    half_size = size / 2.0
    
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size

    return np.array([x1, y1, x2, y2], dtype=np.float32), mar

def get_face_bbox(landmarks, frame_w, frame_h):
    """
    Calculates the bounding box for the entire face based on all landmarks.
    """
    pts = []
    for pt in landmarks:
        pts.append([pt.x * frame_w, pt.y * frame_h])
    
    pts = np.array(pts, dtype=np.float32)
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    # Return as [x1, y1, x2, y2]
    return np.array([min_x, min_y, max_x, max_y], dtype=np.float32)


def crop_and_resize(frame, bbox, target_size=(96, 96)):
    """
    Crops the frame based on bbox and resizes it to target_size.
    Handles boundaries by padding with zeros.
    """
    if bbox is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    fh, fw, _ = frame.shape
    x1, y1, x2, y2 = bbox

    # Convert to integer coordinates for array indexing
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))

    bw = ix2 - ix1
    bh = iy2 - iy1

    if bw <= 0 or bh <= 0:
         return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Calculate intersection with frame
    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(fw, ix2)
    src_y2 = min(fh, iy2)

    # Calculate placement on the destination canvas
    dst_x1 = src_x1 - ix1
    dst_y1 = src_y1 - iy1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Initialize canvas (black padding)
    crop = np.zeros((bh, bw, 3), dtype=frame.dtype)

    # Copy valids pixels
    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    # Resize to target
    try:
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    except Exception:
        resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return resized

def main():
    parser = argparse.ArgumentParser(description="Extract lip region and generate bbox coordinates using MediaPipe Tasks.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Ensure Model Asset Exists ---
    # The new Tasks API requires a binary model bundle.
    model_filename = "face_landmarker.task"
    # Save it in the same folder as this script for reuse
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_asset_path = os.path.join(script_dir, model_filename)

    if not os.path.exists(model_asset_path):
        print(f"Model file '{model_filename}' not found.")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"Downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, model_asset_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)
    
    # --- 2. Initialize MediaPipe FaceLandmarker ---
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5,
        #min_face_detection_confidence=0.5,
        #min_face_presence_confidence=0.5,
        min_face_detection_confidence=0.2, # Lowered for better long-range detection
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        # Use VIDEO mode for temporal consistency
        running_mode=vision.RunningMode.VIDEO)
    
    # --- 3. Process Video ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Processing '{input_path}'")
    print(f"Resolution: {int(width)}x{int(height)}, FPS: {fps}, Frames: {total_frames}")

    # 1. args.input에서 경로를 제외한 '파일명.확장자'만 추출
    base_name = os.path.basename(args.input) 

    # 2. 확장자를 제거하고 이름만 추출
    file_stem = os.path.splitext(base_name)[0]

    # 3. 새로운 파일명 생성 및 출력 폴더와 결합
    out_vid_path = os.path.join(output_dir, f"{file_stem}.lip.mp4")
    # out_vid_path = os.path.join(output_dir, f"{os.path.splitext(args.input)[0]}.lip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
    out_vid = cv2.VideoWriter(out_vid_path, fourcc, fps, (96, 96))

    coords_list = []

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe Tasks requires an RGB MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Timestamp in ms required for VIDEO mode
            # frame_idx / fps * 1000
            if fps > 0:
                timestamp_ms = int((frame_idx / fps) * 1000)
            else:
                timestamp_ms = frame_idx * 33 # assume 30fps fallback

            try:
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                best_bbox = np.zeros(4, dtype=np.float32)
                best_mar = -1.0
                detected = False

                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        # face_landmarks is a list of NormalizedLandmark objects
                        bbox, mar = get_lip_bbox_and_mar(face_landmarks, width, height)
                        
                        if bbox is not None:
                            if mar > best_mar:
                                best_mar = mar
                                best_bbox = bbox
                                detected = True
                                # Calculate face bbox for the best face
                                best_face_bbox = get_face_bbox(face_landmarks, width, height)
                
                # Store coordinates (Store FACE bbox if detected, else zeros)
                if detected:
                    coords_list.append(best_face_bbox)
                else:
                    coords_list.append(np.zeros(4, dtype=np.float32))

                # Write Video Frame
                if detected:
                    out_frame = crop_and_resize(frame, best_bbox, (96, 96))
                else:
                    out_frame = np.zeros((96, 96, 3), dtype=np.uint8)
                
                out_vid.write(out_frame)

                frame_idx += 1
                if frame_idx % 50 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames", end='\r')

            except Exception as e:
                # Basic error handling to keep going
                print(f"\nError processing frame {frame_idx}: {e}")
                coords_list.append(np.zeros(4, dtype=np.float32))
                out_vid.write(np.zeros((96, 96, 3), dtype=np.uint8))
                frame_idx += 1
                continue

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
