
import face_alignment
import cv2
import numpy as np
import pickle
import argparse
import os
import sys
import torch
from tqdm import tqdm

def get_lip_bbox(landmarks, frame_w, frame_h):
    """
    Calculates the bounding box (square, centered) for the lip region.
    landmarks: shape (68, 2)
    Indices for lips in 68-point model:
    Outer: 48-59
    Inner: 60-67
    """
    # Combine outer and inner lip points
    lip_indices = list(range(48, 68))
    lip_pts = landmarks[lip_indices]

    # Bounds
    min_x, min_y = np.min(lip_pts, axis=0)
    max_x, max_y = np.max(lip_pts, axis=0)

    w = max_x - min_x
    h = max_y - min_y

    # Calculate Center
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Determine BBox size
    # 1.5x of using the max dimension is a safe margin usually.
    # For S3FD which is more precise, we can stick to 96x96 relative scale.
    size = max(w, h) * 1.5

    # Ensure square 1:1
    half_size = size / 2.0

    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size

    return np.array([x1, y1, x2, y2], dtype=np.float32)

def get_face_bbox(landmarks, frame_w, frame_h):
    """
    Calculates the bounding box for the entire face based on all landmarks.
    landmarks: shape (68, 2)
    """
    min_x, min_y = np.min(landmarks, axis=0)
    max_x, max_y = np.max(landmarks, axis=0)

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

    # Convert to integer coordinates
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))

    bw = ix2 - ix1
    bh = iy2 - iy1

    if bw <= 0 or bh <= 0:
         return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Calculate intersection
    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(fw, ix2)
    src_y2 = min(fh, iy2)

    # Calculate destination
    dst_x1 = src_x1 - ix1
    dst_y1 = src_y1 - iy1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Initialize canvas
    crop = np.zeros((bh, bw, 3), dtype=frame.dtype)

    # Copy pixels
    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    # Resize
    try:
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    except Exception:
        resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return resized

def main():
    parser = argparse.ArgumentParser(description="Extract lip region using S3FD (face-alignment).")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    device = args.device

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        device = 'cpu'

    print(f"Initializing FaceAlignment on {device}...")
    try:
        # S3FD is the default face detector
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    except Exception as e:
        print(f"Error initializing FaceAlignment: {e}")
        sys.exit(1)

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_vid_path, fourcc, fps, (96, 96))

    coords_list = []
    
    # Read all frames first to process in batch if needed? 
    # face-alignment can process batches, which is faster.
    # But for simplicity and memory safety on large videos, let's do frame by frame or small batches.
    # Let's do frame by frame for now to keep logic simple and consistent with previous script.
    
    frame_idx = 0
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # S3FD expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # get_landmarks returns a list of ndarrays, one for each face
                # or None if no face detected
                preds = fa.get_landmarks(frame_rgb)
                
                best_bbox = np.zeros(4, dtype=np.float32)
                detected = False

                if preds:
                    # If multiple faces, we need a strategy.
                    # Simple strategy: Choose the largest face (S3FD is good at finding faces)
                    # Or closest to center.
                    # Let's pick the one with the largest lip area or just the first one if unsure.
                    # Usually for single speaker videos, first one is fine.
                    # Let's calculate area for all and pick max.
                    
                    max_area = 0
                    for landmarks in preds:
                        bbox = get_lip_bbox(landmarks, width, height)
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        area = w * h
                        
                        if area > max_area:
                            max_area = area
                            best_bbox = bbox
                            detected = True
                            # Calculate face bbox
                            best_face_bbox = get_face_bbox(landmarks, width, height)
                
                if detected:
                    coords_list.append(best_face_bbox)
                else:
                    coords_list.append(np.zeros(4, dtype=np.float32))

                if detected:
                    out_frame = crop_and_resize(frame, best_bbox, (96, 96))
                else:
                    out_frame = np.zeros((96, 96, 3), dtype=np.uint8)
                
                out_vid.write(out_frame)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                coords_list.append(np.zeros(4, dtype=np.float32))
                out_vid.write(np.zeros((96, 96, 3), dtype=np.uint8))
            
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
