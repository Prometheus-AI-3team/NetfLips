import cv2
import pickle
import numpy as np
import os

def visualize_bbox(video_path, pkl_path, output_path):
    # Load bboxes
    with open(pkl_path, 'rb') as f:
        bboxes = pickle.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {fps} FPS, {frame_count} frames")
    print(f"BBoxes: {len(bboxes)} items")

    # Define codec and create VideoWriter object
    # Using 'mp4v' or 'avc1' for mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(bboxes):
            bbox = bboxes[frame_idx]
            # Handle different formats if necessary. Assuming [x1, y1, x2, y2]
            if bbox is not None:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Draw rectangle (color: green (0, 255, 0), thickness: 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add frame index text
                    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                 cv2.putText(frame, f"Frame: {frame_idx} (No BBox)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved visualized video to {output_path}")

if __name__ == "__main__":
    video_path = "/home/2022113135/gyucheol/NetfLips/data/final_segments/hulk_h264_part2.mp4"
    pkl_path = "/home/2022113135/gyucheol/NetfLips/data/final_bbox/hulk_h264_part2.bbox.pkl"
    output_path = "/home/2022113135/gyucheol/NetfLips/data/final_bbox/hulk_h264_part2_bbox_visualized.mp4"
    
    visualize_bbox(video_path, pkl_path, output_path)
