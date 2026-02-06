import os
import json
import torch
import glob
from TTS.api import TTS
from pathlib import Path
from tqdm import tqdm

def generate_clean_tts():
    # 경로 설정
    DATA_DIR = "/home/2022113135/gyucheol/NetfLips/data/final_segments"
    OUTPUT_DIR = "/home/2022113135/voice_cloning/demo_voice_ko"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    speaker_ref = "/home/2022113135/gyucheol/NetfLips/data/final_segments/timothee_h264_part1_ref.wav"

    # 대상 JSON 파일 목록
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    
    for json_name in tqdm(json_files, desc="TTS 생성 중"):
        stem = Path(json_name).stem
        json_path = os.path.join(DATA_DIR, json_name)
        output_path = os.path.join(OUTPUT_DIR, f"{stem}_ko.wav")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ko_text = data.get("ko", "")

        if not ko_text: continue

        try:
            tts.tts_to_file(
                text=ko_text,
                file_path=output_path,
                speaker_wav=speaker_ref, 
                language="ko"
            )
        except Exception as e:
            print(f"\n❌ 생성 실패 ({stem}): {e}")

if __name__ == "__main__":
    generate_clean_tts()