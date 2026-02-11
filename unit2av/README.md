# 1. UNIT2A
## 2-1. Train

### 1. 학습 데이터 경로 포함하는 manifest 생성

`make_manifest.py` 실행해서 원본 오디오, 해당하는 유닛코드 파일 묶은 `.txt`  파일 생성(매니페스트)

```bash
python unit2av/make_manifest.py \
  --audio_root /home/2022113135/datasets/zeroth/train_data_01/003 \
  --unit_root /home/2022113135/jihye/preprocessed_mavhubert_unit2a \
  --output_file train_hubert_new.txt
```

### 2. config.json 수정

먼저 `config.json` 에 학습/검증 데이터 매니페스트 `.txt` 파일 경로 넣어줘야함

```json
{
    "input_training_file": "train_hubert_2000.txt",
    "input_validation_file": "train_hubert_2000.txt",
    /// ... ///
}
```

### 3. 학습 실행

```bash
cd gyucheol/NetfLips/av2av-main

CUDA_VISIBLE_DEVICES=<GPU번호설정> python train_unit2a.py \
    --config unit2av/config_hubert.json \
    --checkpoint_path unit2av/checkpoint/seamless-unit-2000 \
    --validation_interval 20 \
    --training_steps 200000 \
    --checkpoint_interval 1000
```

## 2-2. Inference

```bash
python inference_unit2a.py 
  --checkpoint "path/to/your/checkpoint" 
  --config "path/to/your/config.json" 
  --input_file "path/to/your/input.pt" 
  --output_folder "path/to/output/folder"
```

# 3. UNIT2AV 
## Inference
```bash
python inference_unit2av.py 
  --in-unit-path "path/to/your/units.txt" 
  --in-vid-path "path/to/original_video.mp4" 
  --in-bbox-path "path/to/modified.bbox.pkl" 
  --out-vid-path "path/to/output_video.mp4" 
  --tgt-lang "en" 
  --unit2av-path "path/to/unit2av_model.pt"
```

## Explanation of Arguments
- `--in-unit-path`: The text file with the number sequence (speech units).
- `--in-vid-path`: Your original input video (used for Speaker Encoder).
- `--in-bbox-path`: Your modified pickle file with the None frames.
- `--unit2av-path`: Path to the .pt checkpoint file you are using.
- `--tgt-lang`: The target language (e.g., en, ko, etc.).

# 4. 원본 코드 수정한 부분

### `unit2av/model.py`

1. **불필요한 루프 주석 처리 및 삭제**
    
    ```python
    class CodeHiFiGANModel_spk(CodeHiFiGANModel):
        def forward(self, **kwargs):
        # ... 중략 ...
    	  for k, feat in kwargs.items():
            if k in ["spkr", "code", "f0", "dur_prediction"]:
                continue
    					
            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)
    
        return super(CodeHiFiGANModel, self).forward(x), torch.repeat_interleave(kwargs["code"], dur_out.view(-1))
    ```
    
    - **원본**: kwargs를 돌며 spkr, code, f0 등을 제외한 나머지 특징량을 모두 업샘플링해서 x에 이어붙이는(concatenate) 로직
        - 목적 : "나머지 처음 보는 데이터(feat)가 들어오면, 무조건 **오디오(x) 길이에 맞춰 늘려서(upsample) 모델 입력에 이어붙여(concat) 버리자!**"
    - **수정본**: 이 부분 주석 처리
2. **학습 시 Duration Loss 계산 로직 추가**
    
    ```python
    if self.dur_predictor and self.training:
    # ... 중략 ...
    return super(CodeHiFiGANModel,self).forward(x), dur_losses
    ```
    
    `self.dur_predictor`가 있고 모델이 학습 상태(`self.training`)일 때 실행되는 분기 추가
    
    - `process_duration` 함수를 사용해 실제 Duration 값을 추출
    - `self.dur_predictor`를 통해 예측된 값과 실제 값 사이의 **MSE Loss(dur_losses)**를 계산
    - 결과값으로 `super().forward(x)`와 함께 계산된 **`dur_losses`를 반환**
3. **반환값(Return Value)의 세분화**
    
    상황에 따라 모델이 반환하는 두 번째 인자값이 달라지도록 변경되었습니다.
    
    - **학습 시**: `dur_losses` 반환
    - **추론 시 (`dur_prediction=True`)**: `dur_out`에 맞춰 확장된(repeat_interleave)  code 반환 (FaceRenderer가 사용)
    - **기본/평가 시**: 확장되지 않은 원본  `kwargs["code"]`반환 (이전에는 무조건 확장을 시도했으나 이제 조건부로 바뀜)

### Dur_prediction에 관하여…

**1. 첫 번째 분기: `if self.dur_predictor and self.training:`**

```python
if self.dur_predictor and self.training:
# ... 코드 ...
return super(CodeHiFiGANModel,self).forward(x), dur_losses

```

- **언제**: 모델을 **학습시킬 때**
- **이유**:
    - 오디오 생성(HiFi-GAN)과 길이 예측(Duration Predictor)을 **동시에** 학습하고 있음
    - 오디오 생성은 **forward(x)**로 수행하고, 얼마나 길게 말해야 할지 맞추는 연습은 `dur_losses`로
    - 그래서 학습 결과로 "오디오 신호"와 "길이 예측 오차(Loss)"를 모두 반환해야 학습이 됨

**2. 두 번째 분기: `if self.dur_predictor and kwargs.get("dur_prediction", False):`**

```python
if self.dur_predictor and kwargs.get("dur_prediction",False):
# ... 코드 ...
return..., torch.repeat_interleave(...)

```

- **언제**: 학습이 끝난 후 Inference 단계
- **조건**: 코드를 호출할 때 `dur_prediction=True`라고 명시했을 때 (UTUT에서 번역된 후 Unit2A 수행할 때)
- **이유**:
    - 입력받은 유닛 코드에는 시간 정보가 없음 ← *UTUT로 번역돼서 왔기 때문*
    - 그래서 "이 유닛은 3프레임, 저 유닛은 5프레임..." 하고 모델이 직접 길이를 예측해서 유닛을 복사함
    - 그래야 이 늘려진 코드를 받아서 얼굴 생성기(FaceRenderer) 등이 영상의 길이를 맞출 수 있습니다.
