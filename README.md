# 🎬 NetfLips

**Unit-based Audiovisual Translation for Korean**  
*Text-free Direct Speech Translation with Synchronized Lip Movement*

<div align="center">
  
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  
</div>

---

## 📋 Overview

NetfLips는 영어 영상을 입력받아 **음성과 입 모양이 동기화된 한국어 번역 영상**을 생성하는 프로젝트입니다.

### ✨ Key Features

- **🎯 Unit-based Translation**: 텍스트 중간 표현 없이 음성과 시각 정보를 공통 유닛(Unit) 표현으로 직접 모델링
- **🔊 Speech & Visual Sync**: 음성과 비디오를 공통 특징 공간의 Unit 단위로 정렬하여 강건한 번역 구현
- **🇰🇷 Korean Fine-tuning**: 기존에 지원되지 않던 한국어 capability를 위한 Fine-tuning
- **💬 Natural Synthesis**: 자연스러운 음성 합성 및 립싱크 생성

### 🎯 Keywords

`#Unit-based Audiovisual Translation` `#Text-free Direct Speech Translation` `#Lip Sync` `#Speech Translation`

---

## 🎥 Demo
🌐 **[Demo Link](https://www.miricanvas.com/v2/design2/v/db519087-2eae-4b07-9b24-9d925d81469f)**
---

## 🏗️ Architecture

NetfLips는 3단계 파이프라인으로 구성됩니다:

### 1️⃣ Unit Extraction
- FLAC 복원 (wav)
- 특징 추출 (Mel Spectrogram)
- K-means 분류
- 정수 sequence로 변환

### 2️⃣ Unit Translation
- **Base Model**: AV2AV (Choi, J., et al., 2024)
- **Translation**: 영어 unit → 한국어 unit
- **Framework**: Fairseq toolkit 기반 unit sequence 학습
- **Backbone**: 대규모 사전 학습 모델 mBART 활용

### 3️⃣ AV Generation
- Unit → Audio 변환
- 한국어 unit & 화자 임베딩 활용
- Speech Resynthesis

---

## 📊 Dataset

본 프로젝트는 다음 데이터셋을 활용하여 학습되었습니다:

| Dataset | Description | Size |
|---------|-------------|------|
| **Zeroth Korean ASR** | 한국어 음성 인식 데이터 | 12,245 문장 |
| **AIHub Ko-X 통번역 음성** | 한국어-영어(미국) 병렬 음성 데이터 | 169,488 문장 |


## 🚀 Getting Started

### Prerequisites

```bash
# 1. 레포지토리 클론
git clone https://github.com/Prometheus-AI-3team/NetfLips.git

cd NetfLips

# 2. 서브모듈(fairseq) update
git submodule init
git submodule update

# 2. Conda 기본 환경 생성
conda env create -f environment.yml
conda activate unit2a

# 3. Pip 다운그레이드 (메타데이터 에러 방지)
pip install "pip<24.1"

# 4. PyTorch 설치 (CUDA 11.7 기준)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Installation

```bash
# 5. 나머지 라이브러리 설치
pip install -r requirements.txt

# 6. Fairseq 설치
cd av2av-main/fairseq
pip install -e .
```

---

## 💻 Usage
### Checkpoints
| Model | Name | link |
| --- | --- | --- |
| AV2Unit | `mav_hubert_large_noise.py` | [download]() |
| Unit2Unit | `utut_sts_ft.pt` | [download]() |
| Unit2AV | `unit_av_renderer_withKO.pt` | [download](https://drive.google.com/file/d/1vNaJGWqqC8VAzEXTEYb33fq5PsfE74F6/view?usp=drive_link) |

### Quick Start

```python
# 사용 예제 코드 (추후 업데이트)
```

### Advanced Usage

```bash
# 커맨드라인 사용법 (추후 업데이트)
```

---

## 📁 Project Structure

```
NetfLips/
├── av2unit/                  # Audio-Visual to Unit Extraction
│   ├── avhubert/             # Feature extraction using AV-HuBERT
│   └── inference.py          # Unit extraction inference script
├── unit2unit/                # Unit to Unit Translation
│   ├── utut_pretrain/        # Pre-training modules
│   ├── utut_finetune/        # Fine-tuning modules
│   └── inference.py          # Translation inference script
├── unit2av/                  # Unit to Audio-Visual Generation
│   ├── model.py              # Unit2AV model definition
│   ├── train_unit2a.py       # Training script for Unit2Audio
│   └── inference_unit2av.py  # Inference scripts
├── fairseq/                  # Fairseq Toolkit (Submodule)
├── scripts/                  # Utility Scripts for Data Preparation
├── inference_av2av.py        # Main End-to-End Inference Script
├── environment.yml           # Conda Environment Configuration
└── requirements.txt          # Python Dependencies
```

---

## 🔬 Methodology

### Data Preprocessing
- FLAC 파일 복원 및 wav 변환
- Mel Spectrogram 기반 특징 추출
- K-means 클러스터링을 통한 Unit 분류

### Model Training
- mBART 기반 sequence-to-sequence 학습
- Fairseq toolkit 활용
- Unit-to-Unit translation 최적화

### Audio-Visual Generation
- 한국어 unit에서 음성 재합성
- 화자 임베딩을 활용한 자연스러운 음성 생성
- 립싱크가 동기화된 비디오 생성

---

## 🛠️ Technical Details

### Base Model
- **AV2AV**: Audio-Visual to Audio-Visual translation model
- **Reference**: Choi, J., et al., 2024

### Fine-tuning Strategy
- 한국어 미지원 문제 해결을 위한 Fine-tuning
- 병렬 한-영 음성 데이터 활용
- Unit-level translation 학습

---

## 👥 Team Members From Prometheus(AI club)

| Name | batch |
|------|------|
| **장지수** | 6th |
| **유지혜** | 6th |
| **신규철** | 8th |
| **이가연** | 8th |

---

## 📝 Citation

```bibtex
@misc{netflips2024,
  title={NetfLips: Unit-based Audiovisual Translation for Korean},
  author={장지수, 유지혜, 신규철, 이가연},
  year={2024}
}
```

### References
- Choi, J., et al. (2024). AV2AV: Audio-Visual to Audio-Visual Translation

---

## License

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## Acknowledgments

This repository is built upon [AV2AV](https://github.com/choijeongsoo/av2av?tab=readme-ov-file) and [Fairseq](https://github.com/pytorch/fairseq). We appreciate the open-source of the projects.
