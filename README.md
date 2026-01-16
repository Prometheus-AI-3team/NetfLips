# NetfLips
[2025-2] textless direct audio-video speech translation
---

This repository is built upon [AV2AV](https://github.com/choijeongsoo/av2av?tab=readme-ov-file) and [Fairseq](https://github.com/pytorch/fairseq). We appreciate the open-source of the projects.

---
## Unit2A 학습 위한 환경 설정 - 규철
### 1. 가연 서버(리눅스) 가상환경

```cpp
conda activate gyucheol
```

### 2. 다른 리눅스 서버에서

```bash
# 1. 프로젝트 및 서브모듈 다운로드
git clone --recursive [레포주소]
cd netflips_team

# 2. Conda 기본 환경 생성
conda env create -f environment.yml
conda activate gyucheol

# 3. [핵심] Pip 다운그레이드 (메타데이터 에러 방지)
pip install "pip<24.1"

# 4. PyTorch 설치 (CUDA 11.7 기준)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. 나머지 라이브러리 설치
pip install -r requirements_fixed.txt

# 6. Fairseq 및 Speech-Resynthesis 설치 (반드시 이 순서대로!)
# fairseq 설치
cd av2av-main/fairseq
pip install -e .
```

---

### 🪟 윈도우(Windows) 사용자용 가이드

```bash
# 1. 프로젝트 다운로드 (Git Bash 등 사용)
# git clone --recursive [레포주소]
# cd netflips_team

# 2. Conda 기본 환경 생성
conda env create -f environment.yml
conda activate gyucheol

# 3. [핵심] Pip 다운그레이드
python -m pip install "pip<24.1"

# 4. PyTorch 설치 (윈도우용 CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. 나머지 라이브러리 설치
pip install -r requirements_fixed.txt

# 6. Fairseq 설치
# fairseq 폴더로 이동 (탐색기 경로 확인 필수)
cd .\av2av-main\fairseq
pip install -e .
cd ..\..
```

---

### 3단계: 실행 테스트 방법

이거 되면 성공

```bash
# av2av-main 폴더로 이동 후
python train_unit2a.py --help
```