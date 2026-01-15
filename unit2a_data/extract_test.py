import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # GPU 설정
import sys
import torch
import librosa
import numpy as np
from omegaconf import OmegaConf

# 경로 설정
standard_paths = [p for p in sys.path if "site-packages" in p or "lib/python" in p]
for p in standard_paths:
    if p in sys.path:
        sys.path.remove(p)
        sys.path.insert(0, p)

av2av_root = "/home/2022113135/av2av"
fairseq_root = "/home/2022113135/av2av/fairseq"
task_dir = "/home/2022113135/av2av/av2unit"
hubert_pretrain_path = "/home/2022113135/av2av/av2unit/avhubert"

# 시스템 경로에 추가
for p in [av2av_root, fairseq_root, task_dir, hubert_pretrain_path]:
    if p not in sys.path:
        sys.path.insert(0, p)
        
from fairseq import tasks

# 태스크 별칭 등록
try:
    from hubert_pretraining import AVHubertPretrainingTask, AVHubertPretrainingConfig
    @tasks.register_task("av_hubert_unit_pretraining", dataclass=AVHubertPretrainingConfig)
    class AVHubertUnitPretrainingTaskAlias(AVHubertPretrainingTask):
        pass

    print("✅ Successfully registered 'av_hubert_unit_pretraining' using class inheritance!")
except Exception as e:
    print(f"⚠️ Registration Bypass: {e}")

from fairseq import checkpoint_utils

CHECKPOINT_PATH = "/home/2022113135/av2av/checkpoints/mavhubert_large_noise.pt"
LIST_FILE = "/home/2022113135/av2av/selected_files.txt"
OUT_DIR = "/home/2022113135/datasets/zeroth_units"
os.makedirs(OUT_DIR, exist_ok=True)

def load_mavhubert():
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    
    # 체크포인트 로드
    state = checkpoint_utils.load_checkpoint_to_cpu(CHECKPOINT_PATH)
    cfg = state["cfg"]
    
    # OmegaConf 엄격 모드 해제
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg.model, False)
    
    # 태스크 생성
    task_dict = OmegaConf.to_container(cfg.task, resolve=True)

    task_dict["labels"] = [] 
    task_dict["label_dir"] = "/tmp" # 아무 의미 없는 경로로 설정
    
    for k in ["pretrained_checkpoint", "noise_wav", "noise_prob", "noise_snr", "noise_num"]:
        task_dict.pop(k, None)
    task_obj = tasks.setup_task(OmegaConf.create(task_dict))
    
    # 클래스 레벨에서 dictionaries 프로퍼티를 빈 리스트로 고정
    type(task_obj).dictionaries = property(lambda self: [])
    
    # state 내부에 직접 주입
    if hasattr(task_obj, 'state'):
        task_obj.state.dictionaries = []
    
    # 모델 설정 수정
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_dict.update({"_name": "av_hubert", "final_dim": 256, "audio_embed_dim": 104})
    
    # Fairseq의 최신 Transformer/Wav2Vec2 코드가 요구하는 모든 기본 설정값
    missing_defaults = {
        # 차원 불일치 해결
        "audio_embed_dim": 104, 
        "encoder_embed_dim": 1024,    
        "conv_pos": 128,
        "conv_pos_groups": 16,
        
        # 유사도 관련
        "sim_type": "cosine",
        "logit_temp": 0.1,
        "target_glu": False,
        "final_dim": 256,
        "untie_final_proj": True,
        
        # Masking 관련 
        "mask_selection": "static",   
        "mask_other": 0.0,
        "mask_length": 10,
        "mask_prob": 0.8,
        "no_mask_overlap": False,
        "mask_min_space": 1,
        "mask_channel_selection": "static",
        "mask_channel_other": 0.0,
        "mask_channel_length": 10,
        "mask_channel_prob": 0.0,
        "no_mask_channel_overlap": False,
        "mask_channel_min_space": 1,
        
        # Transformer & Activation
        "activation_fn": "gelu",
        "layer_type": "transformer", 
        "layerdrop": 0.0,
        "checkpoint_activations": False,
        "offload_activations": False,
        
        # Convolution & ResNet
        "required_seq_len_multiple": 1,
        "conv_pos": 128,
        "conv_pos_groups": 16,   
        "resnet_relu_type": "prelu",
        "resnet_weights": None,
        
        # 기타 필수 파라미터
        "sub_encoder_layers": model_dict.get("encoder_layers", 24),
        "layer_norm_first": True,
        "feature_grad_mult": 1.0,
        "encoder_layerdrop": 0.0,
        "dropout_input": 0.0,
        "dropout_features": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "dropout": 0.0,
        "no_seed_provided": False, 
        "cond_on_norm": False,
        "reproducible": False,
        "encoder_layers_B": 0,
    }
    
    for k, v in missing_defaults.items():
        if k not in model_dict:
            model_dict[k] = v
    
    from hubert import AVHubertModel
    model = AVHubertModel.build_model(OmegaConf.create(model_dict), task_obj)
    
    # 특징 추출 후 [Batch, Time, Dim] -> [Batch, Time, 1024]로 바꿔주는 역할
    if hasattr(model.feature_extractor_audio, 'proj'):
        print("🔧 Patching audio feature extractor projection layer...")
    
    # 체크포인트 반영 : 256차원 임베딩 레이어 강제 생성
    model.final_proj = torch.nn.Linear(1024, 256)
    
    print("--- Loading Weights & Injecting Codebook ---")
    ckpt_state = state["model"]
    model_state = model.state_dict()
    new_state_dict = {}
    
    model.unit_codebook = None
    model.label_predictor_weight = None

    for k, v in ckpt_state.items():
        # 일반 레이어 매핑
        if k in model_state and v.shape == model_state[k].shape:
            new_state_dict[k] = v
        
        # 256차원 출력 레이어 찾기
        if "final_proj" in k or "label_predictor" in k:
            if v.shape == torch.Size([256, 1024]):
                new_state_dict["final_proj.weight"] = v
                print(f"🎯 Mapped {k} -> final_proj.weight")
            elif v.shape == torch.Size([256]):
                new_state_dict["final_proj.bias"] = v
                print(f"🎯 Mapped {k} -> final_proj.bias")

        # 유닛 코드북 찾기 ([2008, 256] 모양)
        if v.shape == torch.Size([2008, 256]):
            model.unit_codebook = v.cuda() # GPU로 미리 전송
            print(f"🔥 Found Codebook: {k} (Size: 2008 units)")
        
        # final_proj가 없을 때를 대비해 가중치 보관
        if "label_predictor.weight" in k or "final_proj.weight" in k:
            if v.shape[0] == 256:
                model.label_predictor_weight = v.cuda()

    model.load_state_dict(new_state_dict, strict=False)
    
    if model.unit_codebook is None:
        raise ValueError("❌ Critical: Could not find [2008, 256] codebook in checkpoint!")

    return model.cuda().eval()

print("\n--- Step 1: Loading Model ---")
model = load_mavhubert()

print("\n--- Step 2: Testing ---")
with open(LIST_FILE, "r") as f:
    all_files = [line.strip() for line in f.readlines()]

print("\n--- Step 3: Extracting Units (Audio-Only Mode) ---")
for i, f in enumerate(all_files):
    try:
        # --- 전처리 강화 ---
        y, _ = librosa.load(f, sr=16000)
        y = librosa.util.normalize(y) # 전체 볼륨 최적화
        y, _ = librosa.effects.trim(y, top_db=20) 
        y = librosa.effects.preemphasis(y, coef=0.98) # 고주파 더 강하게 강조
        
        # --- 멜 스펙트로그램 해상도 조정 ---
        mel = librosa.feature.melspectrogram(
            y=y, sr=16000, n_mels=104, 
            n_fft=2048, hop_length=640, win_length=1280 # FFT 창을 키워 해상도 확보
        )
        log_mel = librosa.power_to_db(mel)
        
        # Instance Normalization (강화된 버전)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        
        mel_input = torch.from_numpy(log_mel).float().cuda().unsqueeze(0)
        x = mel_input.transpose(1, 2)

        with torch.no_grad():
            res_x = model.feature_extractor_audio.proj(x)            
            x_in = res_x.transpose(0, 1)
            
            # 레이어 앙상블 (4, 6, 8, 12층 - 고차원 정보 살짝 추가)
            layer_outputs = []
            target_layers = [4, 6, 8, 12]
            for idx, layer in enumerate(model.encoder.layers):
                x_in, _ = layer(x_in, self_attn_padding_mask=None)
                if idx + 1 in target_layers:
                    layer_outputs.append(x_in.transpose(0, 1))
                if idx + 1 == 12: break
            
            inter_x = torch.mean(torch.stack(layer_outputs), dim=0)
            
            # Embedding & Linear Projection
            if hasattr(model, 'final_proj'):
                emb = model.final_proj(inter_x) 
            else:
                emb = torch.nn.functional.linear(inter_x, model.label_predictor_weight)
                
            # 임베딩 화이트닝: 분포를 강제로 사방으로 펼침
            emb = (emb - emb.mean(dim=1, keepdim=True)) / (emb.std(dim=1, keepdim=True) + 1e-6)
            
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            codebook = torch.nn.functional.normalize(model.unit_codebook, p=2, dim=-1)
            
            # 유사도(logits) 계산: [T, 256] @ [256, 2008] -> [T, 2008]
            logits = torch.matmul(emb.squeeze(0), codebook.T)
            
            logits /= 0.1
            
            units = torch.argmax(logits, dim=-1).flatten().cpu().numpy()
            
            # Median Filtering
            from scipy.signal import medfilt
            
            smoothed_units = medfilt(units, kernel_size=3) 
            
            units_tensor = torch.from_numpy(smoothed_units).to(torch.long)
            
            # 중간 지점 계산
            mid = len(units_tensor) // 2
            # 중간 지점부터 10개 출력 (범위 초과 방지 위해 min 사용)
            mid_pattern = units_tensor[mid : mid + 10].tolist()
            
            unique_units = torch.unique(units_tensor)
          
            print(f"DEBUG: Layers Avg | Unique: {len(unique_units)} | Mid-Pattern: {mid_pattern}")
            
        # 저장
        save_data = {
            'code': units_tensor,
            'spkr': torch.zeros(256).float(),
            'f0': torch.zeros(len(units_tensor)).float(),
            'dur_prediction': False
        }
        
        save_path = os.path.join(OUT_DIR, os.path.basename(f).replace(".wav", ".pt"))
        torch.save(save_data, save_path)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(all_files)}] Progressing...")
        
    except Exception as e:
        import traceback
        print(f"[{i+1}/10] ❌ Error: {e}")
        traceback.print_exc()