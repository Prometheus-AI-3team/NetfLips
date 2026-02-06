# Lip Detection and Extraction Scripts

Scripts for detecting faces, extracting lip regions, and managing bounding box metadata for video processing.

## Environment Setup

Create and activate a conda environment for the lip extraction scripts:

```bash
conda env create -f environment.yml
conda activate lip_extraction
```


## Core Extraction Scripts

These scripts process an input video and produce two outputs:
- A `.lip.mp4` video: Cropped and resized (96x96) lip region.
- A `.bbox.pkl` file: A pickle file containing the bounding box coordinates of the detected face for each frame.

### 1. `extract_lip_yolo.py`
Uses YOLOv8 for robust face tracking and combines it with landmark detection to isolate lips.
- **Landmark Methods**: Supports both `face-alignment` (S3FD) and `mediapipe`.
- **Speaker Tracking**: Uses Mouth Aspect Ratio (MAR) variance over time to identify and track the active speaker in multi-person videos.
- **Usage**:
  ```bash
  python extract_lip_yolo.py --input path/to/video.mp4 --output_dir ./output --landmark_method face_alignment --device cuda
  ```

### 2. `extract_lip_yolo_filtered.py`
An extension of the YOLO script that adds a "speaking threshold."
- **Feature**: If the speaker's MAR variance (speaking activity) falls below `--min_speaking_threshold`, the frame is saved as black and coordinates as zeros. This is useful for pruning silent or inactive segments.
- **Usage**:
  ```bash
  python extract_lip_yolo_filtered.py --input path/to/vid.mp4 --output_dir ./out --min_speaking_threshold 0.01
  ```

### 3. `extract_lip_mediapipe.py`
Relies entirely on the MediaPipe Tasks API for face landmarker detection.
- **Feature**: Fast and lightweight. It automatically downloads the necessary `.task` model file. It selects the face with the highest MAR (most active mouth) in each frame.
- **Usage**:
  ```bash
  python extract_lip_mediapipe.py --input path/to/video.mp4 --output_dir ./output
  ```

### 4. `extract_lip_s3fd.py`
Uses the `face-alignment` library (S3FD detector) to detect landmarks.
- **Feature**: Highly accurate landmarking, though slower than MediaPipe. It selects the largest detected face.
- **Usage**:
  ```bash
  python extract_lip_s3fd.py --input path/to/video.mp4 --output_dir ./output --device cuda
  ```

---

## Visualization & Inspection

Tools to verify the accuracy of the extraction process.

### `visualize_bbox.py`
Overlays the bounding boxes from a `.pkl` file onto the original video to check if the detection is correct.
- **Usage**: Edit the `video_path` and `pkl_path` variables in the script and run:
  ```bash
  python visualize_bbox.py
  ```

### `inspect_bbox.py`
A quick diagnostic script to print the structure and sample data of a `.bbox.pkl` file.
- **Usage**: Update the `pkl_path` in the script and run:
  ```bash
  python inspect_bbox.py
  ```

---

## Utility & Metadata Management

Scripts for post-processing and cleaning up metadata.

### `edit_bbox_pickle.py`
Manually "mute" specific segments of a video by setting their bounding box data to `None`.
- **Use Case**: Removing incorrectly detected frames or segments where the speaker is not actually speaking despite being detected.
- **Usage**: Configure the `target_pickle_file` and `ranges_to_set_none` (tuple of start/end frames) in the script and run.

### `change_numpylist_to_py_list.py`
Recursively converts numpy arrays inside all `.pkl` files in a directory into standard Python lists.
- **Use Case**: Eliminating `numpy` dependencies for downstream tasks or ensuring cross-version compatibility for pickle files.
- **Usage**: Set `TARGET_DIR` in the script and run:
  ```bash
  python change_numpylist_to_py_list.py
  ```