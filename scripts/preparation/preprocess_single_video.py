#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single Video Pre-processing Script for AutoAVSR

This script processes a single MP4 video file to extract:
1. Face bounding box coordinates per frame (.bbox.pkl)
2. Lip region video file (.lip.mp4)

Usage:
    python preprocess_single_video.py \
        --input-video path/to/video.mp4 \
        --output-dir path/to/output \
        --detector retinaface \
        --gpu_type cuda
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import torchvision

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preparation.data.data_module import AVSRDataLoader
from preparation.utils import save2vid


def extract_bboxes_mediapipe(video_frames, detector):
    """Extract bounding boxes for each frame using MediaPipe."""
    bboxes = []
    landmarks = []
    
    mp_face_detection = detector.mp_face_detection
    
    for frame in video_frames:
        # Try full range detector first
        results = detector.full_range_detector.process(frame)
        if not results.detections:
            # Fall back to short range detector
            results = detector.short_range_detector.process(frame)
            if not results.detections:
                bboxes.append(None)
                landmarks.append(None)
                continue
        
        # Find largest face
        max_id, max_size = 0, 0
        for idx, detected_face in enumerate(results.detections):
            bboxC = detected_face.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)
            bbox_size = width + height
            if bbox_size > max_size:
                max_id, max_size = idx, bbox_size
        
        # Get the largest detected face
        detected_face = results.detections[max_id]
        bboxC = detected_face.location_data.relative_bounding_box
        ih, iw, ic = frame.shape
        x_min = int(bboxC.xmin * iw)
        y_min = int(bboxC.ymin * ih)
        width = int(bboxC.width * iw)
        height = int(bboxC.height * ih)
        
        # Format: (x_min, y_min, x_max, y_max)
        bbox = (x_min, y_min, x_min + width, y_min + height)
        bboxes.append(bbox)
        
        # Extract landmarks (4 keypoints)
        lmx = [
            [
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(0).value].x * iw),
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(0).value].y * ih)
            ],
            [
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(1).value].x * iw),
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(1).value].y * ih)
            ],
            [
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(2).value].x * iw),
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(2).value].y * ih)
            ],
            [
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(3).value].x * iw),
                int(detected_face.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint(3).value].y * ih)
            ],
        ]
        landmarks.append(np.array(lmx))
    
    return bboxes, landmarks


def extract_bboxes_retinaface(video_frames, detector):
    """Extract bounding boxes for each frame using RetinaFace."""
    bboxes = []
    landmarks = []
    
    for frame in video_frames:
        detected_faces = detector.face_detector(frame, rgb=False)
        if len(detected_faces) == 0:
            bboxes.append(None)
            landmarks.append(None)
            continue
        
        # Find largest face
        max_id, max_size = 0, 0
        for idx, bbox in enumerate(detected_faces):
            bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
            if bbox_size > max_size:
                max_id, max_size = idx, bbox_size
        
        # Format: (x_min, y_min, x_max, y_max)
        bbox = tuple(detected_faces[max_id])
        bboxes.append(bbox)
        
        # Extract landmarks
        face_points, _ = detector.landmark_detector(frame, [detected_faces[max_id]], rgb=True)
        landmarks.append(face_points[0])
    
    return bboxes, landmarks


def main():
    parser = argparse.ArgumentParser(description="Single Video Pre-processing for AutoAVSR")
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Path to input MP4 video file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "mediapipe"],
        help="Type of face detector. (Default: retinaface)",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="GPU type. (Default: cuda)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=25,
        help="Video FPS for output. (Default: 25)",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(args.input_video))[0]
    output_bbox_path = os.path.join(args.output_dir, f"{base_name}.bbox.pkl")
    output_lip_path = os.path.join(args.output_dir, f"{base_name}.lip.mp4")
    
    print(f"Processing video: {args.input_video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Detector: {args.detector}")
    print(f"GPU type: {args.gpu_type}")
    
    # Initialize data loaders
    # Handle GPU type - retinaface expects "cuda" or "mps" format
    # For retinaface, AVSRDataLoader will append ":0" automatically
    # MediaPipe doesn't use GPU type
    if args.detector == "retinaface" and args.gpu_type == "cpu":
        print("Warning: RetinaFace may not work well with CPU. Consider using MediaPipe or GPU.")
        gpu_type_for_detector = "cuda"  # Fallback
    else:
        gpu_type_for_detector = args.gpu_type
    
    vid_dataloader = AVSRDataLoader(
        modality="video",
        detector=args.detector,
        convert_gray=False,
        gpu_type=gpu_type_for_detector
    )
    
    # Load video frames
    print("Loading video frames...")
    video_frames = vid_dataloader.load_video(args.input_video)
    print(f"Loaded {len(video_frames)} frames")
    
    # Extract bounding boxes
    print("Extracting face bounding boxes...")
    if args.detector == "mediapipe":
        bboxes, landmarks = extract_bboxes_mediapipe(video_frames, vid_dataloader.landmarks_detector)
    else:  # retinaface
        bboxes, landmarks = extract_bboxes_retinaface(video_frames, vid_dataloader.landmarks_detector)
    
    # Save bounding boxes
    print(f"Saving bounding boxes to {output_bbox_path}...")
    with open(output_bbox_path, "wb") as f:
        pickle.dump(bboxes, f)
    print(f"Saved {len([b for b in bboxes if b is not None])} bounding boxes")
    
    # Process video to extract lip regions
    print("Processing video to extract lip regions...")
    try:
        # Use the landmarks we extracted
        lip_video = vid_dataloader.load_data(args.input_video, landmarks=landmarks)
        if lip_video is None:
            raise ValueError("Failed to extract lip regions from video")
        
        # Convert to numpy if tensor
        if isinstance(lip_video, torchvision.io.VideoMetaData):
            raise ValueError("Unexpected video metadata returned")
        
        # Ensure it's numpy array
        if hasattr(lip_video, 'numpy'):
            lip_video = lip_video.numpy()
        elif isinstance(lip_video, np.ndarray):
            pass
        elif hasattr(lip_video, 'detach'):
            # PyTorch tensor
            lip_video = lip_video.detach().cpu().numpy()
        else:
            lip_video = np.array(lip_video)
        
        # Ensure video is in correct format (T, H, W, C) for torchvision
        if len(lip_video.shape) == 4:
            # Convert from (T, H, W, C) to torch tensor format
            # torchvision expects uint8 values
            if lip_video.dtype != np.uint8:
                # Clip values to [0, 255] and convert to uint8
                lip_video = np.clip(lip_video, 0, 255).astype(np.uint8)
        
        # Save lip video
        print(f"Saving lip video to {output_lip_path}...")
        # save2vid(output_lip_path, lip_video, args.video_fps)
        print(f"\n*** Video FPS: {args.video_fps}\n")
        save2vid(output_lip_path, lip_video, 25)
        print(f"Saved lip video with shape: {lip_video.shape}")
        
        print("\n=== Processing Complete ===")
        print(f"Bounding boxes: {output_bbox_path}")
        print(f"Lip video: {output_lip_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

