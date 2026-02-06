# 1. 환경 설정

### 1. 가연 서버(리눅스) 가상환경

```bash
conda activate unit2a
```

### 2. 다른 리눅스/윈도우에서

```bash
# 1. 규철 브랜치 클론 서브모듈 다운로드
git clone -b unit2a --single-branch https://github.com/Prometheus-AI-3team/NetfLips.git

cd NetfLips

# 2. 서브모듈(fairseq) update
git submodule init
git submodule update

# 2. Conda 기본 환경 생성
conda env create -f environment.yml
conda activate unit2a

# 3. [핵심] Pip 다운그레이드 (메타데이터 에러 방지)
pip install "pip<24.1"

# 4. PyTorch 설치 (CUDA 11.7 기준)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. 나머지 라이브러리 설치
pip install -r requirements.txt

# 6. Fairseq 설치
cd av2av-main/fairseq
pip install -e .
```