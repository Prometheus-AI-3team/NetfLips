import torch
import os
import glob
import numpy as np
from unit2av.model_speaker_encoder import SpeakerEncoder

# 경로 설정 (본인 환경에 맞게 수정)
encoder_path = "./unit2av/encoder.pt"
pt_dir = "../datasets/zeroth_units"
wav_dir = "./selected_wavs"

# SpeakerEncoder 모델 로드
print("--- Loading Speaker Encoder Model ---")
try:
    use_cuda = torch.cuda.is_available()
    speaker_encoder = SpeakerEncoder(encoder_path)
    if use_cuda:
        speaker_encoder = speaker_encoder.cuda()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# spkr 정보 추출 및 pt 업데이트
pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))
print(f"Found {len(pt_files)} files in {pt_dir}. Starting update...")

for pt_path in pt_files:
    file_name = os.path.basename(pt_path)
    # 확장자만 바꿔 대응하는 wav 경로 생성
    wav_path = os.path.join(wav_dir, file_name.replace(".pt", ".wav"))

    if not os.path.exists(wav_path):
        print(f"⚠️ Skip: Wav file not found for {file_name}")
        continue

    try:
        # spkr 정보 추출 - av2av/unit2av/inference.py와 동일한 로직
        spkr_embed = speaker_encoder.get_embed(wav_path)
        
        checkpoint = torch.load(pt_path)
        
        spkr_tensor = torch.from_numpy(spkr_embed).float()
        
        checkpoint['spkr'] = spkr_tensor
        
        torch.save(checkpoint, pt_path)
        print(f"✅ Success: Updated spkr for {file_name} | Shape: {list(spkr_tensor.shape)}")

    except Exception as e:
        print(f"❌ Error updating {file_name}: {e}")

print("\n--- Test Update Completed! ---")