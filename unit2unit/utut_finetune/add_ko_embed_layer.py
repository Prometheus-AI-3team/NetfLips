import torch

# 1. 파일 경로 설정
input_ckpt_path = "/home/2022113135/jjs/av2av/ckpts/utut_sts_ft.pt"
output_ckpt_path = "/home/2022113135/jjs/av2av/ckpts/utut_sts_add_ko_idx1004.pt"

# 2. 체크포인트 로드
ckpt = torch.load(input_ckpt_path, map_location='cpu')
state_dict = ckpt['model']

# 3. 복사 대상 언어의 인덱스 파악 (중요)
# Fairseq Dictionary 기준으로 [en] 등의 태그가 몇 번 인덱스인지 확인해야 합니다.
# 1024 모델의 경우 대개 마지막 부근에 언어 태그가 위치합니다.

""" 사전 구성 추정

기본 유닛: 1000개 (0~999)
특수 토큰: 4개 (BOS, PAD, EOS, UNK)
기존 언어 태그: 20개 (공식 지원 5종 + 기타 사전학습 언어 15종 등)
신규 추가 태그: 1개 ([ko])
합계: 1025개
"""
src_lang_idx = 1005  # [en]의 인덱스가 1004라고 기대

# 4. 확장할 레이어 리스트
target_layers = [
    'encoder.embed_tokens.weight', 
    'decoder.embed_tokens.weight', 
    'decoder.output_projection.weight'
]

print(f"Initializing [ko] embedding from index {src_lang_idx}...")

for layer in target_layers:
    if layer in state_dict:
        old_weight = state_dict[layer]  # Shape: [1024, 1024]
        
        # 특정 언어의 가중치 행 추출 (1, 1024) 후 복제(.clone)
        # clone()을 써야 기존 가중치와 메모리 참조가 분리됩니다.
        new_lang_row = old_weight[src_lang_idx:src_lang_idx+1].clone()
        # new_lang_row = torch.randn(1, 1024) * 0.02 #랜덤 초기화
        
        # [1024, 1024] + [1, 1024] = [1025, 1024] 결합
        state_dict[layer] = torch.cat([old_weight, new_lang_row], dim=0)
        print(f" - {layer}: {old_weight.shape} -> {state_dict[layer].shape}")

# 5. 결과 저장
torch.save(ckpt, output_ckpt_path)
print(f"\nSuccessfully saved expanded checkpoint to: {output_ckpt_path}")