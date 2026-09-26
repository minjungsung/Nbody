# Setup Guide

## 사전 요구사항

| 도구 | 최소 버전 | 설치 방법 |
|------|----------|----------|
| **Python** | 3.8+ | [python.org](https://www.python.org/) |
| **pip** | 최신 | Python에 포함 |

## 1단계: 클론 및 가상환경

```bash
git clone https://github.com/minjungsung/Nbody.git
cd Nbody

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

## 2단계: 의존성 설치

```bash
pip install rebound numpy h5py torch matplotlib
```

### 의존성 상세

| 패키지 | 용도 | 설치 |
|--------|------|------|
| `rebound` | N-body 시뮬레이션 | `pip install rebound` |
| `numpy` | 수치 연산 | `pip install numpy` |
| `h5py` | HDF5 데이터 파일 | `pip install h5py` |
| `torch` | LSTM 딥러닝 | `pip install torch` |
| `matplotlib` | 결과 시각화 | `pip install matplotlib` |

### PyTorch 설치 (GPU 지원)

GPU를 사용하려면 CUDA 버전에 맞는 PyTorch를 설치합니다:

```bash
# CPU 전용
pip install torch

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 3단계: 전체 파이프라인 실행

### 3.1 시뮬레이션 데이터 생성

```bash
python nbody.py
```

**출력:** `nbody_data.h5` (500 × 12 시뮬레이션 데이터)

```
시뮬레이션 설정:
├── 태양 (m=1.0, 중심)
├── 행성 1 (m=0.001, a=1.0 AU)
├── 행성 2 (m=0.001, a=1.5 AU)
├── dt = 0.01
└── 500 스텝
```

### 3.2 데이터 전처리

```bash
python predata.py
```

**처리:** HDF5 → Sliding Window → PyTorch DataLoader

```
입력: nbody_data.h5 (500, 12)
출력: X (495, 5, 12), y (495, 12)
      DataLoader (batch=16)
```

### 3.3 모델 정의

```bash
python model_definition.py
```

**출력:** LSTM 모델 인스턴스 생성

```
LSTM(input=12, hidden=64, layers=2)
└── Linear(64, 12)
```

### 3.4 모델 학습

```bash
python inference.py
```

**출력:** 학습된 모델 (50 epochs)

```
Epoch 1/50, Loss: 0.XXXXXX
Epoch 2/50, Loss: 0.XXXXXX
...
Epoch 50/50, Loss: 0.XXXXXX
```

### 3.5 결과 시각화

```bash
python result.py
```

**출력:** Matplotlib 그래프 (실제 vs 예측)

## 전체 실행 (한 번에)

파일들 간의 의존성으로 인해, 실제로는 하나의 스크립트에서 순차적으로 실행하는 것이 편합니다:

```python
# run_all.py
exec(open("nbody.py").read())
exec(open("predata.py").read())
exec(open("model_definition.py").read())
exec(open("inference.py").read())
exec(open("result.py").read())
```

또는 Jupyter Notebook에서 각 파일을 순서대로 실행합니다.

## 파라미터 조정

### 시뮬레이션 파라미터 (`nbody.py`)

```python
sim.dt = 0.01       # 시간 스텝 (작을수록 정밀)
n_steps = 500       # 시뮬레이션 스텝 수
# 천체 추가:
sim.add(m=1e-3, a=2.0)  # 행성 3 추가
```

### 모델 파라미터

```python
# predata.py
window_size = 5     # 과거 스텝 수 (5 → 10 등)
batch_size = 16     # 배치 크기

# model_definition.py
hidden_size = 64    # LSTM 히든 크기 (64 → 128)
num_layers = 2      # LSTM 레이어 수 (2 → 3)

# inference.py
epochs = 50         # 학습 에포크 수
lr = 1e-3           # 학습률
```

## 트러블슈팅

### rebound 설치 오류

```bash
# macOS에서 C 컴파일러 필요
xcode-select --install
pip install rebound
```

### CUDA 메모리 부족

```python
# CPU로 전환
device = torch.device('cpu')
model = model.to(device)
```

### HDF5 파일 확인

```python
import h5py
with h5py.File("nbody_data.h5", "r") as f:
    print(f["positions"].shape)  # (500, 12) 확인
    print(f["positions"][:3])    # 처음 3스텝 확인
```

### Matplotlib 디스플레이 오류

```bash
# macOS에서 백엔드 설정
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Agg' (파일 저장만)
import matplotlib.pyplot as plt
```

### 메모리 효율적 학습

데이터가 큰 경우:
```python
# DataLoader에서 num_workers 설정
dataloader = DataLoader(dataset, batch_size=16,
                       shuffle=True, num_workers=4)
```
