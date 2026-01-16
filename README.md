# NetfLips
[2025-2] textless direct audio-video speech translation
---

This repository is built upon [AV2AV](https://github.com/choijeongsoo/av2av?tab=readme-ov-file) and [Fairseq](https://github.com/pytorch/fairseq). We appreciate the open-source of the projects.

---
## Unit2A 학습 위한 환경 설정 - 규철
### 1. 가연 서버(리눅스) 가상환경

```bash
conda activate gyucheol
```

### 2. 다른 리눅스/윈도우에서

```bash
# 1. 규철 브랜치 클론 서브모듈 다운로드
git clone -b gyucheol --single-branch https://github.com/Prometheus-AI-3team/NetfLips.git

cd NetfLips

# 2. 서브모듈(fairseq) update
git submodule init
git submodule update

# 2. Conda 기본 환경 생성
conda env create -f environment.yml
conda activate gyucheol

# 3. [핵심] Pip 다운그레이드 (메타데이터 에러 방지)
pip install "pip<24.1"

# 4. PyTorch 설치 (CUDA 11.7 기준)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. 나머지 라이브러리 설치
pip install -r requirements.txt

# 6. Fairseq 및 Speech-Resynthesis 설치 (반드시 이 순서대로!)
# fairseq 설치
cd av2av-main/fairseq
pip install -e .
```

---

### 실행 테스트 방법

이거 되면 성공

```bash
# av2av-main 폴더로 이동 후
python train_unit2a.py --help
```
