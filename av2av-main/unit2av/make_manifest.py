import os
import glob
import torch
import random

'''
train_unit2a.py 기대하는 학습 데이터 형식 맞추는 스크립트
train_unit2a.py는 매니페스트 파일(텍스트) 내에 오디오 경로와 코드(Unit) 시퀀스가 텍스트 형태로 구성되어야함
-> 유닛 코드(.pt 파일) + 원본 오디오 경로 -> 텍스트 파일로 변환 
'''

import argparse

# Argument Parser 설정
parser = argparse.ArgumentParser(description='Create manifest file for unit2av training')
parser.add_argument('--audio_root', type=str, required=True, help='Root directory of audio files')
parser.add_argument('--unit_root', type=str, required=True, help='Root directory of unit (.pt) files')
parser.add_argument('--output_file', type=str, default='train_hubert.txt', help='Output manifest file path')

args = parser.parse_args()

# 경로 설정
audio_root = args.audio_root
unit_root = args.unit_root
output_file = args.output_file

# 1. 유닛 파일(.pt) 검색
# 유닛 파일이 "선별된" 데이터이므로, 유닛 파일을 기준으로 오디오를 매칭합니다.
unit_files = sorted(glob.glob(os.path.join(unit_root, '*.pt')))
print(f"Total unit files found: {len(unit_files)}")

# 2. 100개만 선별
target_unit_files = unit_files

lines = []
for unit_path in target_unit_files:
    # unit_path: .../113_003_0012.pt
    fname = os.path.basename(unit_path)[:-16]+ ".pt"
    # fname: 113_003_0012.pt
    
    # 오디오 파일명 추론: 113_003_0012.wav
    wav_fname = fname.replace('.pt', '.wav')
    
    # 폴더 구조 추론: 113_003_0012 -> speaker: 113
    # 오디오 경로는 audio_root + speaker + wav_fname
    parts = fname.split('_')
    if len(parts) >= 1:
        speaker_id = parts[0]
        audio_path = os.path.join(audio_root, speaker_id, wav_fname)
        
        if os.path.exists(audio_path):
            try:
                # 3. .pt 파일 로드 및 코드 추출
                data = torch.load(unit_path)
                # 사용자 데이터 키: 'code' -> 모델이 기대하는 키: 'codes'
                code_tensor = data['code'] 
                
                # 텐서를 공백으로 구분된 문자열로 변환
                code_list = code_tensor.squeeze().tolist()
                if isinstance(code_list, int): code_list = [code_list]
                code_str = ' '.join(map(str, code_list))

                # 4. 딕셔너리 포맷으로 저장
                # unit_path를 저장하여 dataset.py에서 직접 .pt를 로드하도록 함
                entry = {
                    'audio': audio_path,
                    'unit_path': unit_path
                }
                lines.append(str(entry))
            except Exception as e:
                print(f"Error reading {unit_path}: {e}")
        else:
            print(f"Audio file not found for unit: {audio_path}")
    else:
        print(f"Cannot parse speaker id from {fname}")

# 5. 파일 저장
with open(output_file, 'w') as f:
    f.write('\n'.join(lines))

print(f"Saved {len(lines)} samples to {output_file}")