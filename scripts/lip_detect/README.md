## `extract_lip_yolo.py`
### 1. Test with Face Alignment (Default)
This uses face_alignment for high-quality landmark detection on the YOLO-detected face crops.

``` bash
python scripts/extract_lip_yolo.py \
  --input data/h264/ariana_h264.mp4 \
  --output_dir data/output_yolo_fa \
  --landmark_method face_alignment \
  --device cuda  # or cpu
```

### 2. Test with MediaPipe
This uses mediapipe for fast landmark detection on the YOLO-detected face crops.

``` bash
python scripts/extract_lip_yolo.py \
  --input data/h264/ariana_h264.mp4 \
  --output_dir data/output_yolo_mp \
  --landmark_method mediapipe \
  --device cuda  # or cpu
```


## `extract_lip_yolo_filtered.py`
``` bash
python scripts/extract_lip_yolo_filtered.py \
  --input data/h264/taylor_h264.mp4 \
  --output_dir data/output_yolo_filtered \
  --landmark_method face_alignment \
  --device cuda \
  --min_speaking_threshold 0.01
  ```