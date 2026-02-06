import os
import sys

# 꼬인 넘파이 무시하고 강제 로드 시도
try:
    import numpy as np
except ImportError:
    os.system("pip install 'numpy<2'")
    import numpy as np
    
# PyTorch가 내부적으로 넘파이를 못 찾는 문제 해결을 위한 패치
import torch
if not hasattr(torch, "numpy"):
    torch.numpy = np
    
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # 3번 GPU 사용 

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath('fairseq'))
sys.path.insert(0, os.path.abspath('avhubert'))
sys.path.insert(0, os.path.abspath('av2unit/avhubert'))
sys.path.insert(0, os.path.abspath('av2unit'))
sys.path.insert(0, base_dir)


from fairseq import checkpoint_utils, utils
from av2unit.task import AVHubertUnitPretrainingTask

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def load_model(model_path, use_cuda=False):
    """Mavhubert 모델 로드"""
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])

    for model in models:
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()            
        model.prepare_for_inference_(cfg)

    # Audio만 사용하도록 설정
    task.cfg.modalities = ["audio"]
    task.load_dataset()

    return models[0], task

def extract_units_from_audio(audio_path, model, task, use_cuda=False):
    """
    단일 오디오 파일에서 unit 추출
    
    Args:
        audio_path: 오디오 파일 경로 (.wav)
        model: Mavhubert 모델
        task: AVHubertUnitPretrainingTask
        use_cuda: GPU 사용 여부
    
    Returns:
        units: LongTensor of unit IDs
    """
    # Audio feature 로드 (video는 None)
    video_feats, audio_feats = task.dataset.load_feature((None, audio_path))
    
    # Audio를 torch tensor로 변환
    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    
    # Normalization (task 설정에 따라)
    if task.dataset.normalize:
        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    
    # Collate (배치 형태로 변환)
    collated_audios, _, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
    
    # Sample 구성 (video는 None)
    sample = {"source": {
        "audio": collated_audios,
        "video": None,
    }}
    
    # GPU로 이동
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    
    # Unit 추출 (inference)
    with torch.no_grad():
        pred = task.inference(model, sample)
    
    return pred

def batch_extract_mavhubert_units(
    audio_folder,
    output_folder,
    model_path,
    use_cuda=False
):
    """
    폴더 내 모든 오디오 파일에서 Mavhubert unit 배치 추출
    
    Args:
        audio_folder: 오디오 파일들이 있는 폴더
        output_folder: unit 파일을 저장할 폴더
        model_path: Mavhubert 모델 경로
        use_cuda: GPU 사용 여부
    """
    audio_path = Path(audio_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    logger.info("Loading Mavhubert model...")
    model, task = load_model(model_path, use_cuda=use_cuda)
    logger.info("Model loaded successfully!")
    
    # 오디오 파일 목록
    audio_files = sorted(audio_path.glob("**/*.wav"))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {audio_folder}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    successful = 0
    failed = 0
    
    for idx, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{idx}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            # Unit 추출
            units = extract_units_from_audio(
                str(audio_file),
                model,
                task,
                use_cuda
            )
            
            # Unit을 LongTensor로 저장 (.pt 형식)
            output_file = output_path / f"{audio_file.stem}_units.pt"
            torch.save(units.cpu(), output_file)
            
            logger.info(f"  ✓ Saved to: {output_file}")
            logger.info(f"    Units shape: {units.shape}")
            logger.info(f"    Unit range: [{units.min().item()}, {units.max().item()}]")
            logger.info(f"    Sample units: {units[:20].tolist()}")
            
            successful += 1
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 요약
    logger.info("=" * 60)
    logger.info(f"Extraction Summary:")
    logger.info(f"  Total audio files: {len(audio_files)}")
    logger.info(f"  Successfully processed: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Output folder: {output_path.absolute()}")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Extract Mavhubert units from audio files"
    )
    parser.add_argument(
        "--audio-folder",
        type=str,
        default="../voice_cloning/demo_voice_ko_16k",
        help="Folder containing audio files (.wav)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="../voice_cloning/demo_unit/ko",
        help="Output folder for extracted units"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoints/mavhubert_large_noise.pt",
        help="Path to Mavhubert model checkpoint"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    if use_cuda:
        logger.info("Using GPU for extraction")
    else:
        logger.info("Using CPU for extraction")
    
    batch_extract_mavhubert_units(
        args.audio_folder,
        args.output_folder,
        args.model_path,
        use_cuda
    )

if __name__ == "__main__":
    main()