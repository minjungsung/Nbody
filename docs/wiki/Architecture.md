# Architecture

## 전체 파이프라인

```mermaid
graph LR
    A[nbody.py<br/>시뮬레이션] --> B[nbody_data.h5<br/>HDF5 파일]
    B --> C[predata.py<br/>데이터 전처리]
    C --> D[DataLoader<br/>PyTorch]
    D --> E[model_definition.py<br/>LSTM 모델]
    E --> F[inference.py<br/>모델 학습]
    F --> G[result.py<br/>시각화]
```

## Step 1: N-body 시뮬레이션 (`nbody.py`)

### rebound 시뮬레이션 설정

```python
sim = rebound.Simulation()
sim.add(m=1.0)        # 태양 (질량 = 1.0 태양질량)
sim.add(m=1e-3, a=1.0) # 행성 1 (질량 = 목성의 ~1/1000, 궤도 = 1 AU)
sim.add(m=1e-3, a=1.5) # 행성 2 (질량 = 같음, 궤도 = 1.5 AU)

sim.dt = 0.01          # 시간 스텝
n_steps = 500          # 총 시뮬레이션 스텝
```

### 데이터 구조

각 시간 스텝에서 3개 천체의 (x, y, vx, vy)를 기록합니다.

```
시간 스텝 t = 0, 1, 2, ..., 499

각 스텝의 데이터:
[x₀, y₀, vx₀, vy₀,    ← 태양
 x₁, y₁, vx₁, vy₁,    ← 행성 1
 x₂, y₂, vx₂, vy₂]    ← 행성 2

최종 데이터 shape: (500, 12)
```

### HDF5 저장

```python
with h5py.File("nbody_data.h5", "w") as f:
    f.create_dataset("positions", data=data)
# data.shape = (500, 12)
```

## Step 2: 데이터 전처리 (`predata.py`)

### Sliding Window 방식

과거 `window_size` 스텝의 데이터로 다음 스텝을 예측하는 구조입니다.

```
window_size = 5

X (입력):                    y (정답):
[t₀, t₁, t₂, t₃, t₄]  →   [t₅]
[t₁, t₂, t₃, t₄, t₅]  →   [t₆]
[t₂, t₃, t₄, t₅, t₆]  →   [t₇]
    ...                       ...
```

### NBodyDataset 클래스

```python
class NBodyDataset(Dataset):
    def __init__(self, file, window_size=5):
        # HDF5에서 데이터 로드
        with h5py.File(file, "r") as f:
            data = f["positions"][:]

        # Sliding Window 적용
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])    # (5, 12)
            y.append(data[i+window_size])       # (12,)

        self.X = torch.tensor(X, dtype=torch.float32)  # (495, 5, 12)
        self.y = torch.tensor(y, dtype=torch.float32)   # (495, 12)
```

### 데이터 Shape

```
전체 시뮬레이션: (500, 12)
                    │
        Sliding Window (size=5)
                    │
                    ▼
X: (495, 5, 12)    y: (495, 12)
 │   │   │          │    │
 │   │   └── 12 features (x, y, vx, vy × 3 bodies)
 │   └──── 5 timesteps (window)
 └──────── 495 samples

DataLoader: batch_size=16, shuffle=True
```

## Step 3: LSTM 모델 (`model_definition.py`)

### 모델 아키텍처

```
Input: (batch, 5, 12)
        │
        ▼
┌───────────────────┐
│  LSTM Layer 1      │  hidden_size=64
│  LSTM Layer 2      │  num_layers=2
│                    │  batch_first=True
└────────┬──────────┘
         │ output: (batch, 5, 64)
         │
         ▼
  마지막 timestep 선택
  out[:, -1, :]  →  (batch, 64)
         │
         ▼
┌───────────────────┐
│  Linear Layer      │  64 → 12
└────────┬──────────┘
         │
         ▼
Output: (batch, 12)
  = 다음 시간 스텝의 [x, y, vx, vy] × 3 천체
```

### 모델 코드

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                           num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq_len, hidden)
        out = out[:, -1, :]          # 마지막 timestep
        out = self.fc(out)           # (batch, input_size)
        return out
```

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `input_size` | 12 | 3 천체 × 4 features (x, y, vx, vy) |
| `hidden_size` | 64 | LSTM 히든 유닛 수 |
| `num_layers` | 2 | LSTM 레이어 수 |
| `batch_first` | True | 배치 차원이 첫 번째 |

## Step 4: 모델 학습 (`inference.py`)

### 학습 설정

```python
criterion = nn.MSELoss()           # 평균 제곱 오차
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
```

### 학습 루프

```
Epoch 1:
  ├── DataLoader에서 배치 로드 (16 samples)
  ├── Forward: model(X_batch) → pred
  ├── Loss: MSELoss(pred, y_batch)
  ├── Backward: loss.backward()
  └── Update: optimizer.step()

  ... (모든 배치 반복)

Epoch 2 ~ 50: 반복
```

### 손실 함수

```
MSE Loss = (1/N) × Σ(예측값 - 실제값)²

예측: [x̂₀, ŷ₀, v̂x₀, v̂y₀, x̂₁, ŷ₁, ...]  (12개)
실제: [x₀, y₀, vx₀, vy₀, x₁, y₁, ...]  (12개)
```

## Step 5: 결과 시각화 (`result.py`)

마지막 테스트 샘플의 예측값과 실제값을 비교합니다.

```python
X_test, y_test = dataset[-1]
with torch.no_grad():
    y_pred = model(X_test.unsqueeze(0)).numpy()[0]

plt.plot(y_test.numpy()[:6], label="True positions")
plt.plot(y_pred[:6], "--", label="Predicted")
plt.legend()
plt.show()
```

### 시각화 결과

```
       True (실선) vs Predicted (점선)
  값
   │    ___/\___
   │  /         \___    True
   │ /     .---.    \
   │/   .-'     '-.  \  Predicted
   ├──────────────────── features
   x₀  y₀  vx₀ vy₀ x₁ y₁
```

## 전체 데이터 흐름 요약

```
rebound 시뮬레이션
├── 3 천체 설정 (태양 + 2 행성)
├── dt=0.01, 500 스텝 적분
└── 출력: (500, 12) → nbody_data.h5

NBodyDataset
├── Sliding Window (size=5)
├── X: (495, 5, 12), y: (495, 12)
└── DataLoader (batch=16, shuffle)

LSTM 모델
├── 2-layer LSTM (hidden=64)
├── Linear (64 → 12)
├── 50 epochs, Adam (lr=1e-3)
└── MSE Loss

결과 비교
├── 실제 위치 vs 예측 위치
└── Matplotlib 시각화
```
