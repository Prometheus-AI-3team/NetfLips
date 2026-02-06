import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # 3번 GPU 사용 
import torch
import torch.nn.functional as F
import argparse
import sys
from pathlib import Path
import librosa
import numpy as np

# unit2av 폴더를 Python path에 추가
sys.path.insert(0, 'av2av')
from unit2av.model_speaker_encoder import SpeakerEncoder

def extract_f0(audio_path, hop_length=200, f0_min=80, f0_max=400):
    """
    오디오 파일에서 F0 (fundamental frequency) 추출
    
    Args:
        audio_path: 오디오 파일 경로
        hop_length: frame shift (samples)
        f0_min: 최소 F0 값 (Hz)
        f0_max: 최대 F0 값 (Hz)
    
    Returns:
        f0: FloatTensor of shape (1, T)
    """
    # 오디오 로드
    wav, sr = librosa.load(audio_path, sr=16000)
    
    # F0 추출 (pYIN algorithm 사용)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        wav,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        hop_length=hop_length
    )
    
    # NaN 값을 0으로 채우기 (unvoiced frames)
    f0 = np.nan_to_num(f0, nan=0.0)
    
    # FloatTensor로 변환
    f0_tensor = torch.FloatTensor(f0).unsqueeze(0)  # (1, T)
    
    return f0_tensor

def get_speaker_embedding(audio_or_video_path, speaker_encoder_model_path, use_cuda=False):
    """
    화자 임베딩 추출
    
    Args:
        audio_or_video_path: 오디오 또는 비디오 파일 경로
        speaker_encoder_model_path: SpeakerEncoder 모델 경로
        use_cuda: GPU 사용 여부
    
    Returns:
        spkr: FloatTensor of shape (1, 1, 256)
    """
    # SpeakerEncoder 로드
    speaker_encoder = SpeakerEncoder(speaker_encoder_model_path)
    if use_cuda:
        speaker_encoder = speaker_encoder.cuda()
    
    # 화자 임베딩 추출 (256-dim)
    embed = speaker_encoder.get_embed(audio_or_video_path)
    
    # FloatTensor로 변환 (1, 1, 256)
    spkr_tensor = torch.from_numpy(embed).view(1, 1, -1)
    
    return spkr_tensor

def load_unit_code(unit_file_path):
    """
    Unit 파일 로드 (.pt 형식)
    
    Args:
        unit_file_path: .pt 파일 경로
    
    Returns:
        code: LongTensor of shape (1, T)
    """
    # PyTorch tensor 파일 로드
    units = torch.load(unit_file_path)
    
    # 1D tensor를 2D로 변환 (1, T)
    if units.dim() == 1:
        units = units.unsqueeze(0)
    
    return units

def preprocess_single_sample(
    unit_file_path,
    audio_path,
    speaker_encoder_path,
    extract_f0_flag=False,
    dur_prediction=True,
    use_cuda=False
):
    """
    단일 샘플 전처리
    
    Args:
        unit_file_path: unit 파일 경로 (.pt)
        audio_path: 화자 임베딩 및 F0 추출을 위한 오디오 파일 경로
        speaker_encoder_path: SpeakerEncoder 모델 경로
        extract_f0_flag: F0를 추출할지 여부
        dur_prediction: Duration prediction 활성화 여부
        use_cuda: GPU 사용 여부
    
    Returns:
        sample: 전처리된 데이터 딕셔너리
    """
    # 1. Unit code 로드
    code = load_unit_code(unit_file_path)
    target_length = code.shape[-1] # unit 길이 (420)
    
    # 2. 화자 임베딩 추출
    spkr = get_speaker_embedding(audio_path, speaker_encoder_path, use_cuda)
    
    # 3. F0 추출 및 길이 조정
    f0 = None
    if extract_f0_flag:
      # 원본 f0 추출 (ex. [1, 673])
        f0_raw = extract_f0(audio_path)
        
        # resampling을 통해 길이를 unit과 맞춤
        f0_resampled = F.interpolate(
          f0_raw.unsqueeze(0),
          size=(target_length,),
          mode='linear',
          align_corners=False
        )
        f0 = f0_resampled.squeeze(0) # [1, 420]으로 복구
    
    # 4. 샘플 딕셔너리 구성
    sample = {
        "code": code,           # LongTensor (1, T)
        "spkr": spkr,           # FloatTensor (1, 1, 256)
        "dur_prediction": dur_prediction  # bool
    }
    
    # F0가 있으면 추가
    if f0 is not None:
        sample["f0"] = f0       # FloatTensor (1, T)
    
    # GPU로 이동 (필요시)
    if use_cuda:
        sample = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                  for k, v in sample.items()}
    
    return sample

def batch_preprocess(
    unit_folder,
    audio_folder,
    output_folder,
    speaker_encoder_path,
    extract_f0_flag=False,
    dur_prediction=True,
    use_cuda=False
):
    """
    폴더 내 모든 파일 배치 전처리
    
    Args:
        unit_folder: unit 파일들이 있는 폴더
        audio_folder: 오디오 파일들이 있는 폴더
        output_folder: 전처리된 데이터를 저장할 폴더
        speaker_encoder_path: SpeakerEncoder 모델 경로
        extract_f0_flag: F0를 추출할지 여부
        dur_prediction: Duration prediction 활성화 여부
        use_cuda: GPU 사용 여부
    """
    unit_path = Path(unit_folder)
    audio_path = Path(audio_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Unit 파일 목록
    unit_files = sorted(unit_path.glob("*.pt"))
    
    print(f"Found {len(unit_files)} unit files")
    
    matched = 0
    not_found = 0
    
    print("Scanning audio files (this may take a while)...")
    audio_files_map = {f.stem: f for f in audio_path.rglob("*.wav")}
    
    for unit_file in unit_files:
        # 파일명에서 base name 추출 (예: 113_003_0012_units.pt -> 113_003_0012)
        base_name = unit_file.stem.replace("_units", "")
        
        # 대응하는 오디오 파일 찾기
        audio_file = audio_files_map.get(base_name)
        
        if audio_file is None or not audio_file.exists():
            print(f"⚠️  Audio file not found for {base_name} in subfolders, skipping...")
            not_found += 1
            continue
        
        print(f"Processing: {base_name}")
        
        try:
            # 전처리
            sample = preprocess_single_sample(
                str(unit_file),
                str(audio_file),
                speaker_encoder_path,
                extract_f0_flag,
                dur_prediction,
                use_cuda
            )
            
            # 저장
            output_file = output_path / f"{base_name}_preprocessed.pt"
            # 이미 파일이 존재하면 스킵
            if output_file.exists():
                print(f"⏩ {base_name} already exists, skipping...")
                matched += 1
                continue
            torch.save(sample, output_file)
            
            print(f"  ✓ Saved to: {output_file}")
            print(f"    Code shape: {sample['code'].shape}")
            print(f"    Speaker shape: {sample['spkr'].shape}")
            if 'f0' in sample:
                print(f"    F0 shape: {sample['f0'].shape}")
            
            matched += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 50)
    print(f"Preprocessing Summary:")
    print(f"  Total unit files: {len(unit_files)}")
    print(f"  Successfully processed: {matched}")
    print(f"  Audio not found: {not_found}")
    print(f"  Output folder: {output_path.absolute()}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for Unit2AV training")
    parser.add_argument(
        "--unit-folder", 
        type=str, 
        default="../datasets/en_mavhubert",
        help="Folder containing unit files (.pt)"
    )
    parser.add_argument(
        "--audio-folder", 
        type=str, 
        default="../voice_cloning/runs/output_audio/en_ffmpeg",
        help="Folder containing audio files (.wav)"
    )
    parser.add_argument(
        "--output-folder", 
        type=str, 
        default="../datasets/en_unit2a_data",
        help="Output folder for preprocessed data"
    )
    parser.add_argument(
        "--speaker-encoder-path", 
        type=str, 
        default="./unit2av/encoder.pt",
        help="Path to SpeakerEncoder model"
    )
    parser.add_argument(
        "--extract-f0", 
        action="store_true",
        help="Extract F0 from audio"
    )
    parser.add_argument(
        "--no-dur-prediction", 
        action="store_true",
        help="Disable duration prediction"
    )
    parser.add_argument(
        "--cpu", 
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    batch_preprocess(
        args.unit_folder,
        args.audio_folder,
        args.output_folder,
        args.speaker_encoder_path,
        args.extract_f0,
        not args.no_dur_prediction,
        use_cuda
    )

if __name__ == "__main__":
    main()