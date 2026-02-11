
"""
AIHub 립리딩(Lip Reading) 데이터셋을 LRS3 데이터셋 형식으로 전처리하는 스크립트.
주요 작업:
1. .tar 압축 파일 해제 (필요한 경우)
2. JSON(라벨링) 및 MP4(영상) 매칭
3. 영상에서 입술/얼굴 영역 크롭 및 리사이즈
4. 영상 프레임 레이트(FPS) 변환
5. 오디오 슬라이싱 및 샘플링 레이트 변환
6. 전처리된 데이터 저장 (MP4, WAV, TXT)
"""

import os
import json
import glob
import tarfile
import argparse
import subprocess
import shutil
import numpy as np
import cv2
import math
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import tempfile

def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess AIHub Lip Reading Dataset for AV2AV")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing .tar files or extracted folders")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--temp-dir", type=str, default="./temp_extract", help="Temporary directory for extracting tar files")
    parser.add_argument("--fps", type=int, default=25, help="Target Video FPS")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target Audio Sample Rate")
    parser.add_argument("--crop-size", type=int, default=96, help="Target Face Crop Size (Square)")
    parser.add_argument("--padding", type=float, default=0.5, help="Padding in seconds to add to start/end of clip")
    parser.add_argument("--no-tar-extract", action="store_true", help="Skip tar extraction if data is already extracted")
    return parser

def extract_tar(tar_path, extract_path):
    """
    tar 압축 파일을 지정된 경로에 해제.
    
    Args:
        tar_path: tar 파일 경로
        extract_path: 압축을 해제할 경로
    """
    try:
        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)
        
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        print(f"Extracted to {extract_path}")
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def resample_audio(audio_path, target_sr=16000):
    """
    ffmpeg을 사용해서 오디오를 목표 샘플링 레이트(target_sr)와 모노 채널로 변환.
    """
    try:
        # Robust한 처리를 위해 ffmpeg 사용
        out_path = audio_path.replace(".wav", f"_{target_sr}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ac", "1", # Mono
            "-ar", str(target_sr),
            "-vn", # No video
            "-loglevel", "error",
            out_path
        ]
        subprocess.run(cmd, check=True)
        return out_path
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None

def process_video_frames(video_path, bboxes, start_time, end_time, src_fps=30, tgt_fps=25, crop_size=96):
    """
    비디오를 읽고, 프레임별 BBox 정보를 바탕으로 얼굴 영역을 크롭한 후 FPS를 변환.
    
    Args:
        video_path: 원본 MP4 파일 경로
        bboxes: 프레임별 바운딩 박스 목록 [[y1, x1, y2, x2], ...]
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        src_fps: 원본 영상의 FPS
        tgt_fps: 목표 FPS (기본 25)
        crop_size: 출력 이미지 크기 (기본 96x96)
        
    Returns:
        전처리된 프레임들의 Numpy array (T, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    frames = []
    
    start_frame = int(start_time * src_fps)
    end_frame = int(end_time * src_fps)
    
    # 시작 프레임 위치로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    curr_frame_idx = start_frame
    
    while curr_frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 해당 프레임의 BBox 정보 확인
        if curr_frame_idx < len(bboxes):
            bbox = bboxes[curr_frame_idx]
            try:
                # AIHub JSON BBox 형식: [y1, x1, y2, x2] (top, left, bottom, right)
                y1, x1, y2, x2 = bbox
                
                # 영상 범위를 벗어나지 않도록 클리핑
                h, w, _ = frame.shape
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                face_img = frame[y1:y2, x1:x2]
                
                # 목표 크기로 리사이즈
                face_img = cv2.resize(face_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                
                # AV-HuBERT 호환을 위해 그레이스케일로 변환
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                frames.append(face_img)

            except Exception as e:
                print(f"Error cropping frame {curr_frame_idx}: {e}")
        else:
            # BBox 정보가 없는 프레임은 건너뜀
            pass
            
        curr_frame_idx += 1
        
    cap.release()
    
    if not frames:
        return None
        
    frames = np.array(frames) # (T_src, H, W)
    
    # Video FPS 변환 (시간축 보간법 사용)
    # 예: 30 FPS -> 25 FPS
    if src_fps != tgt_fps:
        sec = len(frames) / src_fps
        tgt_frames_len = int(sec * tgt_fps)
        
        # 선형 보간을 위한 인덱스 생성
        indices = np.linspace(0, len(frames)-1, tgt_frames_len)
        
        new_frames = []
        for i in indices:
            low = int(math.floor(i))
            high = int(math.ceil(i))
            weight = i - low
            
            if low == high:
                new_frames.append(frames[low])
            else:
                # 두 프레임 사이를 비중(weight)에 따라 혼합
                blended = (frames[low] * (1-weight) + frames[high] * weight).astype(np.uint8)
                new_frames.append(blended)
                
        return np.array(new_frames)
    
    return frames

def save_video(frames, out_path, fps=25):
    """프레임 배열을 MP4 동영상 파일로 저장."""
    if len(frames) == 0: return
    
    h, w = frames.shape[1], frames.shape[2]
    
    # OpenCV VideoWriter를 사용하여 MP4v 코덱으로 저장 (그레이스케일)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), False) # False: isColor=False
    
    for frame in frames:
        out.write(frame)
        
    out.release()

def process_session(json_path, video_path, args, speaker_id):
    """
    하나의 세션(비디오 1개 + JSON 1개)을 처리하여 문장 단위로 데이터를 분리
    """
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # AIHub 데이터 형식에 따라 리스트인 경우 처리
    if isinstance(data, list):
        if len(data) == 0: return
        data = data[0]

    # 메타데이터 파싱
    try:
        # BBox 정보: 'Bounding_box_info' 하위 필드 확인
        bbox_data = data.get('Bounding_box_info', {}).get('Face_bounding_box')
        if isinstance(bbox_data, dict):
            bboxes = bbox_data.get('xtl_ytl_xbr_ybr', [])
        else:
            bboxes = bbox_data
            
        sentences = data.get('Sentence_info', [])
        
    except Exception as e:
        print(f"Error parsing JSON {json_path}: {e}")
        return

    # 오디오 추출 및 임시 저장 (전체 영상을 한 번에 처리 후 메모리에서 슬라이싱)
    temp_wav = video_path.replace(".mp4", "_temp.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "48000", "-loglevel", "error", temp_wav], check=True)
    except:
        return # 오디오 추출 실패 시 세션 스킵

    # 슬라이싱을 위해 오디오 데이터 로드
    sr, audio_data = wavfile.read(temp_wav)
    
    for sent in sentences:
        try:
            sent_id = sent['ID']
            # 패딩(padding) 추가하여 시작/종료 시간 설정
            start_t = max(0, sent['start_time'] - args.padding)
            end_t = sent['end_time'] + args.padding
            text = sent['sentence_text']
            
            # 출력 경로 설정 (save_dir/speaker_id/...)
            spk_dir = os.path.join(args.save_dir, speaker_id)
            os.makedirs(spk_dir, exist_ok=True)
            
            out_vid_name = f"{speaker_id}_{sent_id:04d}.mp4"
            out_wav_name = f"{speaker_id}_{sent_id:04d}.wav"
            out_txt_name = f"{speaker_id}_{sent_id:04d}.txt"
            
            output_vid_path = os.path.join(spk_dir, out_vid_name)
            output_wav_path = os.path.join(spk_dir, out_wav_name)
            output_txt_path = os.path.join(spk_dir, out_txt_name)
            
            if os.path.exists(output_vid_path): continue 

            # 1. 비디오 처리 (크롭 -> 리사이즈 -> FPS 변환)
            frames = process_video_frames(video_path, bboxes, start_t, end_t, src_fps=30, tgt_fps=args.fps, crop_size=args.crop_size)
            
            if frames is None: continue
            
            # 2. 오디오 처리 (슬라이싱 -> 샘플링 레이트 변환)
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            sliced_audio = audio_data[start_sample:end_sample]
            
            # 목표 샘플링 레이트로 리샘플링 (예: 48k -> 16k)
            if sr != args.sample_rate:
                num_samples = int(len(sliced_audio) * args.sample_rate / sr)
                sliced_audio = signal.resample(sliced_audio, num_samples).astype(np.int16)
            
            # 3. 결과 저장
            save_video(frames, output_vid_path, fps=args.fps)
            wavfile.write(output_wav_path, args.sample_rate, sliced_audio)
            
            with open(output_txt_path, 'w', encoding='utf-8') as tf:
                tf.write(text)
                
        except Exception as e:
            print(f"Error processing sentence {sent_id} in {os.path.basename(video_path)}: {e}")
            
    # 임시 오디오 파일 삭제
    if os.path.exists(temp_wav): os.remove(temp_wav)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # 1. 압축 파일(.tar) 해제 처리
    search_root = args.data_root
    tar_files = glob.glob(os.path.join(args.data_root, "**/*.tar"), recursive=True)
    
    if tar_files and not args.no_tar_extract:
        print(f"Found {len(tar_files)} tar files. Extracting...")
        for tar_f in tqdm(tar_files):
            # 파일명으로 서브폴더 생성하여 중복 방지
            tar_name = os.path.splitext(os.path.basename(tar_f))[0]
            extract_to = os.path.join(args.temp_dir, tar_name)
            extract_tar(tar_f, extract_to)
            
        search_root = args.temp_dir

    # 2. JSON(라벨) 및 MP4(원본) 페어 찾기
    json_files = glob.glob(os.path.join(search_root, "**/*.json"), recursive=True)
    
    print(f"Found {len(json_files)} labeling files. Processing...")
    
    for json_path in tqdm(json_files):
        # 동일한 경로 명에서 .json만 .mp4로 변경하여 비디오 파일 탐색
        base_no_ext = os.path.splitext(json_path)[0]
        video_path = base_no_ext + ".mp4"
        
        if not os.path.exists(video_path):
            # AIHub 특성상 '라벨링데이터'와 '원천데이터' 폴더가 나뉜 경우 경로 보정
            # TL(Label) -> TS(Source) 매핑 처리
            video_path = video_path.replace("라벨링데이터", "원천데이터")
            video_path = video_path.replace("TL", "TS")
            
        if not os.path.exists(video_path):
            # 비디오를 찾을 수 없는 경우 건너뜀
            continue
            
        # 화자 ID(Speaker ID) 추출: 파일 경로 또는 JSON 메타데이터에서 확인
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                meta = json.load(f)
                if isinstance(meta, list):
                    meta = meta[0]
                speaker_id = meta.get('speaker_info', {}).get('speaker_ID', 'Unknown')
            except:
                speaker_id = "Unknown"
        
        # 실제 세션 처리 시작
        process_session(json_path, video_path, args, speaker_id)

    print("Preprocessing Complete.")

if __name__ == "__main__":
    main()
