## Inference
```bash
python -m unit2av.inferencePath \
  --in-unit-path "path/to/your/units.txt" \
  --in-vid-path "path/to/original_video.mp4" \
  --in-bbox-path "path/to/modified.bbox.pkl" \
  --out-vid-path "path/to/output_video.mp4" \
  --tgt-lang "en" \
  --unit2av-path "path/to/unit2av_model.pt"
```

## Explanation of Arguments
- `--in-unit-path`: The text file with the number sequence (speech units).
- `--in-vid-path`: Your original input video (used for Speaker Encoder).
- `--in-bbox-path`: Your modified pickle file with the None frames.
- `--unit2av-path`: Path to the .pt checkpoint file you are using.
- `--tgt-lang`: The target language (e.g., en, ko, etc.).