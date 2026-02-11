## Data Preparation for UTUT Fine-tuning


<aside>

### 🗣 Workflow Summary

1. a2a Parallel audio (en/*.wav, ko/*.wav)
↓ [AV2Unit]
2. Unit text files (units/en/*.txt, units/ko/*.txt)
↓ [Concatenate]
3. Raw text files (train.en, train.ko, valid.en, valid.ko)
↓ [fairseq-preprocess]
4. Binarized data (*.bin, *.idx)
↓ [finetune_en_ko.py]
5. Fine-tuned model
</aside>
<hr>



### Step 1. 병렬 오디오 데이터 준비

영어-한국어 1:1 대응되는 오디오 파일 쌍이 필요.

```bash
audio/
├── en/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── ko/
    ├── sample_001.wav ←en/sample_001.wav의 한국어 번역 음성
    ├── sample_002.wav
    └── ...
```

- 파일명은 반드시 **언어 간 동일하게 유지**되어야 함.
- 각 `(en, ko)` 오디오 쌍이 하나의 번역 샘플을 구성.
- 우리 팀은 한국어 wav에 대응되는 영어 wav를 보이스 클로닝 tts로 구축 !

### Step 2. AV2Unit을 이용한 오디오 → Discrete Unit 추출

각 오디오 파일을 mavHuBERT 기반 AV2Unit 모델로 처리하여

정수 시퀀스(unit sequence)로 변환

- 영어 오디오 처리
    
    ```bash
    PYTHONPATH=fairseq python av2unit/inference.py \
        --in-vid-path audio/en/sample_001.wav \
        --out-unit-path units/en/sample_001.txt \
        --ckpt-path modelckpt/mavhubert_large_noise.pt \
        --modalities audio
    ```
    
- 한국어 오디오 처리
    
    ```bash
    PYTHONPATH=fairseq python av2unit/inference.py \
        --in-vid-path audio/ko/sample_001.wav \
        --out-unit-path units/ko/sample_001.txt \
        --ckpt-path modelckpt/mavhubert_large_noise.pt \
        --modalities audio
    ```
    
- 생성 결과 예시
    
    ```bash
    #units/en/sample_001.txt
    45 78 123 456 789 234 567 890 12 34 56 78
    
    #units/ko/sample_001.txt
    23 89 156 234 567 890 123 456 78 90 12
    ```
    
    - 각 숫자는 **quantized speech unit token**
    - 텍스트 번역이 아니라 **unit-to-unit translation** 문제로 변환됨.
    - 우리는 가연 mavhubert_units 추출 코드로 자동화
        
### Step 3. Fairseq용 Raw Text 데이터 구성

Fairseq 전처리를 위해, 개별 unit 파일들을 **하나의 텍스트 파일로 병합** (한 줄 = 하나의 샘플)

1. 디렉토리 생성

    ```bash
    mkdir -p unit2unit/utut_finetune/raw_data
    ```

2. train.en 생성 (영어 unit 시퀀스) 

    ```bash
    for f in units/en/train_*.txt; do
        cat "$f"
        echo ""
    done > raw_data/train.en
    ```

    1. train.ko 생성 (한국어 unit 시퀀스)

    ```bash
    for f in units/ko/train_*.txt; do
        cat "$f"
        echo ""
    done > raw_data/train.ko
    ```

    ```bash
    # 실제로 서버에서 실행시킨 터미널 명령어
    for f in /home/2022113135/datasets/final_unit2a_split/train/*.pt; do
        cat "$f"
        echo ""
    done > raw_data/train.ko
    ```

- `valid.en / valid.ko`, `test.en / test.ko`도 동일 방식으로 생성

- 결과 파일 형식 예시

        ```bash
        # train.en
        45 78 123 456 789 234 567
        12 34 56 78 90 123 456 789
        ...

        # train.ko
        23 89 156 234 567 890
        78 90 12 34 56 78 90
        ...
        ```

⚠️ **중요**

- `train.en`의 N번째 줄 ↔ `train.ko`의 N번째 줄은 **반드시 병렬 쌍**
- 순서가 어긋나면 학습이 의미를 잃음

---

### Step 4. fairseq-preprocess (Binarization)

Raw text 데이터를 **fairseq 내부에서 사용하는 binary format**으로 변환. 

    ```bash
    fairseq-preprocess \
    --source-lang en \
    --target-lang ko \
    --trainpref raw_data/train \
    --validpref raw_data/valid \
    --testpref raw_data/test \
    --destdir ./data/dataset_mbart_ft_bin_data/en/ko \
    --srcdict unit2unit/utut_pretrain/dataset/dict.txt \
    --tgtdict unit2unit/utut_pretrain/dataset/dict.txt \
    --workers 4
    ```


- (cf) dict는 사전학습(pretraining) 단계에서 이미 만들어진 것(→utut_pretrain/dataset/dict) 을 재사용
- UTUT (unit-to-unit) **pretraining 데이터셋에서 생성된 vocabulary**
- mavHuBERT unit space와 **정합된 dict** (사전학습 모델의 embedding 크기와 일치)
    - source / target이 동일한 unit space이므로 `srcdict == tgtdict` 가 개념적으로 일치
    
- 생성되는 파일들

    ```
    data/dataset_mbart_ft_bin_data/en/ko/
    ├── dict.en.txt
    ├── dict.ko.txt
    ├── train.en-ko.en.bin
    ├── train.en-ko.en.idx
    ├── train.en-ko.ko.bin
    ├── train.en-ko.ko.idx
    ├── valid.en-ko.en.bin
    ├── valid.en-ko.en.idx
    ├── valid.en-ko.ko.bin
    ├── valid.en-ko.ko.idx
    ├── test.en-ko.en.bin
    ├── test.en-ko.en.idx
    ├── test.en-ko.ko.bin
    └── test.en-ko.ko.idx
    ```

- (cf) TSV Manifest는 다른 태스크용 데이터 포맷
    
    
    | Config의 task 설정에 따라… | 요구되는 데이터 포맷 |
    | --- | --- |
    | `translation_from_pretrained_bart` | **Binarized (.bin / .idx)** |
    | `utut_pretraining` | TSV manifest + raw unit files |
    
    av2av 저자분께 전달받은 스크립트 기반에서는 `translation_from_pretrained_bart` task를 사용하므로 **TSV 필요 없다**.
    

---

## Finally run UTUT Fine-tuning

**To execute fine-tuning:**

```bash
cd /Users/jisu/Desktop/dev/cli/av2av/unit2unit/utut_finetune

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=path/to/fairseq OMP_NUM_THREADS=1 python finetune_en_ko.py data/dataset_mbart_ft_bin_data/en/aihub_ko \
    --arch mbart_large \
    --task translation_from_pretrained_bart \
    --criterion focal_label_smoothed_cross_entropy \
    --user-dir ./

```