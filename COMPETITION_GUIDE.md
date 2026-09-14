# 블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회 정리

> DACON Competition ID: `236753`  
> 작성 기준일: **2026-09-13**  
> 공식 페이지: https://dacon.io/competitions/official/236753/overview/description

---

## 0. 한눈에 보는 대회

| 항목 | 내용 |
|---|---|
| 대회명 | 블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회 |
| 분야 | Computer Vision / Video Understanding / Temporal Event Detection / Classification |
| 주제 | 블랙박스 영상만으로 재녹화 여부, 사고 주요시점·상황, 차량 거동 특성을 분석 |
| 총 상금 | **4,200만 원** |
| 참가 자격 | 대한민국 국민 누구나 |
| 팀 인원 | 개인 또는 팀, **최대 5명** |
| 운영 | DACON |
| 주최 | 행정안전부, 한국지능정보사회진흥원 |
| 주관 | 국립과학수사연구원 |
| 제출 방식 | `submit.zip` 코드 제출 |
| 일일 제출 | **최대 3회** |
| 사용 언어 | Python |
| 1차 선발 | Private Leaderboard 기준 **상위 15팀** |
| 최종 수상 | 2차 평가를 거쳐 **상위 7팀** |

---

# 1. 대회의 핵심 문제

이 대회는 단순한 사고영상 분류가 아니다.

하나의 제출 코드가 **세 개의 서로 다른 비디오 분석 문제**를 동시에 해결해야 한다.

```text
Blackbox Video
     │
     ├── Stage 1 ── 재녹화 여부 판별
     │
     ├── Stage 2 ── 사고 주요시점 + 사고상황 분석
     │
     └── Stage 3 ── 차량 가감속 + 조향 상태 추정
```

최종 종합점수의 Stage 가중치는 다음과 같다.

```text
Total Score
 = 0.20 × Stage1
 + 0.40 × Stage2
 + 0.40 × Stage3
```

따라서 **Stage 2와 Stage 3가 전체 점수의 80%**를 차지한다.

---

# 2. Stage 1 — 재녹화 여부 판별

## 2.1 Task

영상 하나를 입력받아 다음 두 클래스 중 하나를 예측한다.

```text
ORIGINAL
RERECORDED
```

즉,

> 이 영상이 실제 블랙박스 원본인가,  
> 아니면 다른 화면/기기에서 재생된 영상을 다시 촬영한 것인가?

를 판단하는 binary video classification 문제이다.

## 2.2 입력

```text
data/stage1/videos/
├── TEST_S1_001.mp4
├── TEST_S1_002.avi
└── ...
```

영상 하나가 하나의 평가 샘플이다.

실제 비공개 데이터의

- 사고 유형
- 촬영 환경
- 영상 길이
- 해상도
- FPS

는 공개 예시와 다를 수 있다.

## 2.3 참고 가능한 재녹화 특징

공식 안내에서 예시로 제시한 특징:

- 화면 테두리 / 재생 화면 일부 노출
- 디스플레이 주사 패턴
- 반사광 및 주변 환경 노출
- 밝기·색상·명암 변화
- 추가 압축 및 화질 저하
- 재촬영 기기의 움직임
- 원근 변화
- 해상도 / aspect ratio 변화

중요한 점:

> 특정 artifact 하나에 의존하기보다 spatial + temporal 특징을 함께 활용해야 한다.

## 2.4 출력

```csv
ID,answer
TEST_S1_001,ORIGINAL
TEST_S1_002,RERECORDED
```

## 2.5 평가

**Macro-F1**

ORIGINAL과 RERECORDED의 F1을 동일한 비중으로 평균한다.

### 모델링 후보

- VideoMAE
- TimeSformer
- MViT
- SlowFast
- CNN/ViT frame encoder + temporal pooling
- EfficientNet / ConvNeXt + temporal statistics
- frequency-domain artifact detector
- optical-flow / camera-motion statistics

Stage 1은 전체 점수의 20%이므로, 매우 무거운 모델 하나보다  
**robust한 pretrained video/image encoder + temporal aggregation**이 비용 대비 효율적일 가능성이 높다.

---

# 3. Stage 2 — 사고 주요시점·상황 분석

Stage 2는 이 대회의 핵심 task 중 하나이며 전체 종합점수의 **40%**를 차지한다.

## 3.1 입력

원본 사고 영상을 프레임 단위 이미지로 추출해 제공한다.

```text
data/stage2/images/
├── TEST_S2_001/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── TEST_S2_002/
```

같은 폴더의 프레임들은 하나의 연속된 사고 영상이다.

## 3.2 예측해야 하는 것

### ① collision_frame

피의차량과 피해차량이 **실제로 접촉한 최초 시점**.

주의:

- 위험이 발생한 순간 X
- 가까워진 순간 X
- 실제 물리적 접촉 시점 O

---

### ② entry_frame

피해차량이 피의차량의 차선에 **최초로 진입한 시점**.

공식 기준:

> 피해차량의 바퀴가 차선에 처음 닿는 시점.

차량이 영상에 처음 등장한 시점이 아니다.

---

### ③ evasion_space

충돌 시점에 피의차량이 진행하거나 충돌을 피할 수 있는 공간이 존재했는지 예측.

허용 값:

```text
0
1
```

**정수형 0 또는 1만 허용**된다.

판단에 활용 가능한 요소:

- 주변 차선
- 도로 경계
- 주변 차량
- 중앙분리대
- 연석
- 방호벽

---

### ④ entry_side

피해차량의 진입 방향.

```text
LEFT
RIGHT
```

도로의 절대 방위가 아니라 **블랙박스 영상 화면 기준**이다.

```text
LEFT  = 화면 왼쪽에서 진입
RIGHT = 화면 오른쪽에서 진입
```

공식 답변에 따르면 Stage 2에서는 LEFT / RIGHT 두 범주만 존재한다.

## 3.3 출력

```csv
ID,collision_frame,entry_frame,evasion_space,entry_side
TEST_S2_001,120,85,1,RIGHT
TEST_S2_002,244,198,0,LEFT
```

### 매우 중요한 제출 규칙

`collision_frame`과 `entry_frame`은 새로 생성한 frame index가 아니라

> **파일명에 적힌 원본 프레임 번호**

를 제출해야 한다.

예:

```text
frame_001249.jpg
```

에서 충돌이 발생했다면

```text
collision_frame = 1249
```

이다.

---

# 4. Stage 2 평가 기준

충돌 시점과 진입 시점은 제출 frame을 영상의 실제 시간으로 변환한 후 정답 시간과 비교한다.

```text
|prediction_time - ground_truth_time| <= 0.3 sec
```

이면 정답 처리된다.

Stage 2 내부 가중치는 공식 토크/평가 안내 기준으로 다음과 같이 알려져 있다.

```text
collision time : 0.35
entry time     : 0.35
evasion_space  : 0.15
entry_side     : 0.15
```

즉 Stage 2에서는 **collision + entry timestamp가 70%**를 차지하므로 시점 검출 성능이 특히 중요하다.

### 오답 처리되는 값

- 예측값 누락
- NaN / 비수치값
- 음수 frame
- 실제 영상 범위를 벗어난 frame
- 정의되지 않은 classification value

공식 답변에 따르면 비공개 Stage 2 데이터에는

- 무충돌 영상이 없고
- 블랙박스 장착 차량과 피해차량이 직접 충돌하는 사고만 포함되며
- 제3자 차량 간 사고는 포함되지 않는다.

---

# 5. Stage 2 모델링 아이디어

Stage 2는 사실상 다음 문제들의 조합이다.

```text
Object Detection
      +
Object Tracking
      +
Lane / Road Understanding
      +
Temporal Event Localization
      +
Scene Classification
```

## 추천 pipeline

```text
Frames
  │
  ├─ Object Detector
  │     └─ vehicle detection
  │
  ├─ Multi Object Tracking
  │     └─ victim vehicle trajectory
  │
  ├─ Lane / Road Segmentation
  │
  ├─ Temporal Encoder
  │     └─ collision / entry likelihood
  │
  └─ Scene Head
        ├─ evasion_space
        └─ entry_side
```

### 가능한 모델

Detection:
- YOLO
- RT-DETR
- DINO / Grounding DINO 계열
- Faster R-CNN

Tracking:
- ByteTrack
- BoT-SORT
- OC-SORT

Lane / segmentation:
- SegFormer
- Mask2Former
- YOLO segmentation
- lane-specific pretrained models

Temporal modeling:
- VideoMAE
- transformer
- TCN
- BiLSTM
- temporal convolution
- frame-level classifier + smoothing

### 중요한 아이디어

충돌 차량을 collision 시점에서 먼저 찾은 뒤,

```text
collision frame
      ↓
victim vehicle identification
      ↓
track backward
      ↓
lane entry frame detection
```

방식으로 피해차량을 역추적하는 구조가 자연스럽다.

공식 가이드에서도 이 방향을 권장한다.

---

# 6. Stage 3 — 차량 거동 특성 분석

Stage 3 역시 전체 종합점수의 **40%**를 차지한다.

## 6.1 입력

실차 주행 영상을 **10 Hz**로 구성한 영상 파일.

```text
data/stage3/videos/
├── TEST_S3_001.mp4
├── TEST_S3_002.mp4
└── ...
```

Stage 3에서는 이미지 폴더가 아니라 video file이 입력이다.

CAN 데이터는 평가 정답 생성에 사용되지만 참가자에게는 제공되지 않는다.

즉 영상만 보고 차량 dynamics를 추정해야 한다.

---

# 7. Stage 3 예측값

각 0.1초 sample마다 두 가지 상태를 예측한다.

## accel_label

```text
ACCELERATING
DECELERATING
CONSTANT
STOPPED
```

## steer_label

```text
LEFT
STRAIGHT
RIGHT
```

## sample_index

10 Hz이므로

```text
sample_index 0 → 0.0 sec
sample_index 1 → 0.1 sec
sample_index 2 → 0.2 sec
sample_index 3 → 0.3 sec
...
```

비공개 평가영상에서는

> decoded frame 수 == sample_index 수

이다.

## 출력

```csv
ID,sample_index,accel_label,steer_label
TEST_S3_001,0,CONSTANT,STRAIGHT
TEST_S3_001,1,ACCELERATING,STRAIGHT
TEST_S3_001,2,DECELERATING,LEFT
```

---

# 8. Stage 3 평가

평가 기본 지표는 **Macro-F1**.

Stage 3 내부 가중치는 공식 토크에 공개된 기준으로

```text
acceleration : 0.7
steering     : 0.3
```

으로 알려져 있다.

### STOPPED 관련 중요 규칙

정답 accel label이 `STOPPED`인 frame은 steer score 계산에서 제외된다.

하지만 제출 시에는 STOPPED frame에도

```text
steer_label
```

을 반드시 출력해야 한다.

Macro-F1은 평가 데이터에 실제 등장한 클래스만이 아니라  
**정의된 전체 클래스 집합 기준**으로 산정된다.

따라서 class imbalance 대응이 중요하다.

---

# 9. Stage 3 모델링 아이디어

영상만으로 ego vehicle의

- acceleration
- braking
- steering

을 추정해야 한다.

핵심 signal 후보:

### Ego-motion

- optical flow
- vanishing point movement
- road texture flow
- lane line motion
- camera translation / rotation

### Scene geometry

- 차선 curvature
- horizon 변화
- 차량 간 상대 거리
- depth 변화

### Temporal dynamics

한 프레임만 보는 것보다 연속 프레임 변화량이 중요하다.

추천 구조:

```text
Frames
   │
Image / Video Encoder
   │
Temporal Encoder
   │
   ├── Acceleration Head
   └── Steering Head
```

또는

```text
Optical Flow / Ego Motion
        +
Video Transformer Feature
        ↓
Temporal Fusion
        ↓
Accel / Steering
```

### 후보 모델

- VideoMAE
- MViT
- SlowFast
- TimeSformer
- CNN/ViT + GRU/LSTM
- CNN/ViT + TCN
- CNN/ViT + Temporal Transformer

### 데이터 구축 시 고려

Stage 3는 공식 설명상 공개 데이터가 **소규모·희소 라벨 예제 수준**이다.

따라서 외부 공개 driving dataset을 활용해

- ego speed
- acceleration
- steering angle

을 category로 재가공하는 전략이 중요할 수 있다.

예시로 조사할 가치가 있는 공개 데이터셋:

- BDD100K
- KITTI
- nuScenes
- Waymo Open Dataset
- comma.ai / comma2k19
- Honda Research Institute Driving Dataset
- D²-City
- A2D2
- Udacity Self Driving Car Dataset

단, 실제 사용 전 반드시 각 데이터셋의 **라이선스와 비영리 사용 가능 여부**를 확인해야 한다.

---

# 10. 학습 데이터 — 대회의 가장 중요한 특징

## 공식 학습 데이터셋이 없다

DACON은 별도의 완성된 Train dataset을 제공하지 않는다.

배포 데이터는 주로

> 입력 형식 / 출력 형식 / 코드 동작 확인을 위한 예제

이다.

공식 안내에서도 공개 예제와 실제 평가 데이터는

- 데이터 구성
- 분포

가 다를 수 있다고 명시한다.

따라서 이 대회는 모델 architecture만큼이나

# **학습 데이터 구축 능력**

이 중요하다.

---

# 11. 외부 데이터 / 사전학습 모델 규정

다음은 원칙적으로 사용할 수 있다.

- 공개 외부 데이터
- pretrained model
- API
- 직접 생성한 데이터
- synthetic data

조건:

> 사용에 법적 제한이 없어야 하고, 누구나 접근 가능한 공개 자원이며 최소 비영리 목적 사용이 허용돼야 한다.

또한 2차 평가 진출 시

- 데이터 출처
- pretrained model 출처
- API
- 외부 데이터
- 생성 데이터

등 **사용한 모든 요소의 출처를 명확히 기술**해야 한다.

따라서 실험 시작 시부터 다음을 기록하는 것이 좋다.

```text
dataset_registry.csv

name
url
license
download_date
purpose
stage
preprocessing
```

그리고

```text
model_registry.md
```

에 pretrained weight와 license를 함께 기록한다.

---

# 12. 매우 중요한 금지 사항

## 12.1 비공개 평가 데이터로 추가 학습 금지

평가 서버에서 test data를 보고 다음을 하면 안 된다.

- fine-tuning
- online learning
- self-training
- pseudo-labeling
- model update
- test-time training

전처리·후처리·순수 추론은 가능하지만 **모델을 갱신하면 안 된다.**

---

## 12.2 평가 파일 간 정보 공유 금지

각 test file은 독립적으로 예측해야 한다.

### 허용

한 영상 내부에서

```text
video
 ├ segment 1
 ├ segment 2
 ├ segment 3
 └ aggregate
```

처럼 여러 구간을 분석한 뒤 합치는 것.

### 금지

다른 평가 영상의

- prediction
- 평균
- class distribution
- 통계
- feature
- 결과

를 현재 영상 보정에 사용하는 것.

즉 아래와 같은 transductive calibration은 피해야 한다.

```text
전체 test prediction 평균 계산
          ↓
분포 맞춤
          ↓
영상별 prediction 재보정
```

---

## 12.3 평가 데이터 유출 시도 금지

코드 제출 기능을 이용해

- test 파일 복사
- hidden data 출력
- 외부 서버 전송
- encoding을 통한 정보 유출

등을 시도하면 즉시 실격될 수 있다.

---

# 13. 코드 제출 구조

반드시 다음 구조를 사용한다.

```text
submit.zip
├── model/
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
├── inference.py
└── requirements.txt
```

`inference.py`에는 반드시 아래 세 함수가 존재해야 한다.

```python
def predict_stage1(data_dir, model_dir):
    pass

def predict_stage2(data_dir, model_dir):
    pass

def predict_stage3(data_dir, model_dir):
    pass
```

반환값은 `pandas.DataFrame`.

### 반환 컬럼

Stage 1

```text
ID
answer
```

Stage 2

```text
ID
collision_frame
entry_frame
evasion_space
entry_side
```

Stage 3

```text
ID
sample_index
accel_label
steer_label
```

---

# 14. 평가 서버 환경

## Hardware

```text
GPU       NVIDIA L40S
GPU VRAM  44.7 GiB

CPU       7 vCPU
RAM       60 GB
Shared Mem 30 GB
```

## 시간 제한

```text
package install : 10 min
inference       : 60 min
```

60분에는 다음이 모두 포함된다.

- 데이터 로딩
- 영상 decoding
- preprocessing
- model inference
- postprocessing
- submission 저장

즉 실제 모델 설계 단계부터 **latency budget**을 고려해야 한다.

## 파일 제한

```text
submit.zip <= 10 GB

압축 해제 후 <= 32 GB
```

따라서 여러 거대 모델 ensemble은 저장공간과 inference time 때문에 현실적으로 제한될 수 있다.

---

# 15. 인터넷 제한

평가 서버에서는 inference 중 인터넷 연결이 불가능하다.

따라서 다음 코드는 위험하다.

```python
model = timm.create_model(
    "...",
    pretrained=True
)
```

또는 Hugging Face에서 runtime에 weight를 받는 구조.

권장 방식:

```python
model = create_model(weights=None)
model.load_state_dict(
    torch.load("model/stageX/model.pt")
)
```

즉 **모든 weight를 submit.zip 내부에 포함**해야 한다.

---

# 16. 평가 서버 기본 주요 라이브러리

주요 버전:

```text
torch==2.8.0+cu128
torchvision==0.23.0+cu128

pandas==2.2.2
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.5.2

opencv-python-headless==4.10.0.84
pillow==10.4.0
av>=15,<17

albumentations==1.4.20
timm==1.0.15
einops==0.8.1

transformers==4.57.6
accelerate==1.9.0
huggingface-hub==0.34.4
safetensors==0.6.2

ultralytics==8.3.170
lap==0.5.12
filterpy==1.4.5

ffmpeg
```

가능하면 서버 기본 패키지를 활용하고  
`requirements.txt`에는 꼭 필요한 추가 dependency만 넣는 것이 안전하다.

---

# 17. 제출 오류와 설치 오류

DACON에서는 두 오류를 구분한다.

## 설치 오류

예:

- zip 구조 오류
- requirements 설치 실패

→ **일일 제출 횟수 차감 X**

## 제출 오류

`script.py` 실행 이후 발생하는 오류.

예:

- inference crash
- CUDA OOM
- 잘못된 return dataframe
- runtime error

→ **일일 제출 횟수 차감 O**

일일 제출이 3회뿐이므로 local dry-run이 매우 중요하다.

---

# 18. 일정

| 일정 | 날짜 |
|---|---|
| 참가 신청 시작 | 2026-08-18 10:00 |
| 대회 시작 | 2026-08-26 10:00 |
| **팀 병합 마감** | **2026-09-23 23:59** |
| **리더보드 제출 마감** | **2026-09-29 10:00** |
| **대회 종료** | **2026-09-30 10:00** |
| 2차 평가 자료 제출 | 2026-09-30 12:00 ~ 2026-10-05 10:00 |
| 2차 평가 및 검증 | 2026-10-05 12:00 ~ 2026-10-15 10:00 |
| 최종 결과 발표 | 2026-10-16 10:00 |
| 오프라인 시상식 | 2026-11-27 예정 |

> 일정 페이지 본문의 대회 종료 연도에 `2025`로 적힌 부분이 있으나, 같은 페이지의 나머지 일정 및 대회 전체 문맥상 **2026-09-30**이 맞는 것으로 보인다.

---

# 19. 2차 평가

1차 평가에서 Private leaderboard 기준 **상위 15팀**이 대상.

제출해야 하는 자료:

- Private Score를 재현할 수 있는 학습 코드
- 모델 개발 보고서 `.hwp`
- 학습데이터 구성 보고서 `.hwp`
- 학습에 사용한 데이터 파일 일체
- 팀 구성원 정보

보고서 분량 제한은 없지만 2차 평가 항목을 모두 포함해야 한다.

따라서 실험 후에 정리하려고 하지 말고 처음부터

```text
experiments/
data_registry/
models/
configs/
reports/
```

를 관리하는 것이 좋다.

---

# 20. 추천 프로젝트 구조

```text
project/
│
├── configs/
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
│
├── data/
│   ├── raw/
│   ├── external/
│   ├── processed/
│   └── labels/
│
├── src/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── common/
│
├── weights/
│
├── experiments/
│   └── experiment_log.csv
│
├── data_registry.csv
├── model_registry.md
│
├── inference.py
├── requirements.txt
└── README.md
```

---

# 21. 추천 validation 전략

이 대회는 hidden test distribution이 공개 sample과 다를 수 있기 때문에 random split 하나만 쓰면 위험하다.

## Stage 1

가능하면 다음 domain으로 split:

```text
camera
device
resolution
compression
environment
source video
```

같은 원본에서 생성된 rerecorded pair가 train / valid에 동시에 들어가지 않도록 주의.

---

## Stage 2

사고 영상 단위 split.

절대로 같은 영상에서 나온 frame이 train / validation에 섞이면 안 된다.

추천 metric:

```text
collision accuracy within ±0.3 sec
entry accuracy within ±0.3 sec
evasion Macro-F1
entry-side Macro-F1
```

그리고 실제 Stage 2 score와 동일한 weighted score를 로컬에서 계산.

---

## Stage 3

주행 sequence / 원본 drive 단위 split.

frame random split 금지.

같은 주행 trajectory가 train/valid에 섞이면 temporal leakage가 매우 심해질 수 있다.

평가:

```text
Accel Macro-F1
Steer Macro-F1
```

STOPPED 구간의 steer는 공식 metric과 동일하게 제외.

---

# 22. 이 대회에서 특히 조심해야 할 Leakage

### Stage 1

같은 source video의 ORIGINAL과 파생 RERECORDED 영상이 서로 다른 split에 들어가는 문제.

### Stage 2

같은 accident video의 adjacent frame이 train과 valid에 동시에 들어가는 문제.

### Stage 3

같은 continuous drive sequence의 frame이 train과 valid에 섞이는 문제.

따라서 validation split의 기본 단위는 항상

> **frame이 아니라 원본 영상 / 주행 sequence**

로 잡는 것이 안전하다.

---

# 23. 현재 기준 우선순위 제안

종합 가중치를 기준으로 하면 단순하게

```text
Stage 2 + Stage 3 >>> Stage 1
```

이다.

추천 개발 우선순위:

## Priority 1 — 데이터 구축

특히 Stage 3 외부 driving dataset + 라벨 변환 전략.

## Priority 2 — Stage 2 temporal event detector

- collision localization
- entry localization

Stage 2 내부에서도 두 시점 항목이 핵심.

## Priority 3 — Stage 3 ego-motion model

가감속이 Stage 3 내부 비중이 더 크므로 accel 성능을 먼저 확보.

## Priority 4 — Stage 1 robust baseline

강한 pretrained backbone으로 빠르게 안정적 baseline 확보.

## Priority 5 — inference optimization

60분 제한과 10GB 제한을 일찍 확인.

---

# 24. 추천 초기 Baseline

처음부터 거대한 시스템을 만들기보다 **3개 Stage가 모두 정상 작동하는 제출**을 먼저 확보한다.

## Stage 1

```text
ConvNeXt / EfficientNet / ViT
frames N개 sampling
→ frame embedding
→ temporal mean/max pooling
→ binary classifier
```

## Stage 2

초기 버전:

```text
Frame Encoder
→ temporal feature sequence
→ collision probability per frame
→ entry probability per frame
→ scene classification heads
```

추후 YOLO + tracking을 추가.

## Stage 3

```text
Video / frame encoder
→ temporal transformer / GRU
→ per-frame accel head
→ per-frame steer head
```

### 첫 목표

```text
submit.zip
      ↓
3 stages 실행 성공
      ↓
60분 이내
      ↓
submission format 검증
```

그 후 모델 성능을 올리는 것이 안전하다.

---

# 25. 실험 로그에 반드시 남길 것

```text
experiment_id
date
stage
dataset_version
split_version
model
pretrained_weight
input_resolution
clip_length
sampling
augmentation
optimizer
lr
epochs
seed
validation_score
inference_time
model_size
notes
```

외부 데이터는 추가로

```text
source
URL
license
usage_scope
```

까지 기록.

2차 평가 진출 시 이 기록이 그대로 보고서의 기반이 된다.

---

# 26. 제출 전 체크리스트

## Code

- [ ] `predict_stage1` 구현
- [ ] `predict_stage2` 구현
- [ ] `predict_stage3` 구현
- [ ] 세 함수 모두 DataFrame 반환
- [ ] 예상치 못한 빈 폴더 / 영상 길이에 대응

## Stage 1

- [ ] answer ∈ {ORIGINAL, RERECORDED}

## Stage 2

- [ ] collision_frame 정상 범위
- [ ] entry_frame 정상 범위
- [ ] 파일명의 원본 frame 번호 사용
- [ ] evasion_space ∈ {0, 1}
- [ ] entry_side ∈ {LEFT, RIGHT}

## Stage 3

- [ ] 모든 sample_index 출력
- [ ] accel label 범주 검증
- [ ] steer label 범주 검증
- [ ] STOPPED frame에도 steer_label 출력

## Runtime

- [ ] 인터넷 없이 실행
- [ ] pretrained weight 자동 다운로드 없음
- [ ] model weight가 zip 내부에 존재
- [ ] zip <= 10GB
- [ ] unzip <= 32GB
- [ ] 전체 inference <= 60분
- [ ] package install <= 10분

## Rules

- [ ] test-time learning 없음
- [ ] pseudo-labeling 없음
- [ ] 평가 파일 간 통계 공유 없음
- [ ] 외부 데이터 license 기록
- [ ] pretrained model 출처 기록

---

# 27. 핵심 결론

이 대회의 난이도는 단순히 좋은 Video Transformer를 사용하는 데 있지 않다.

실질적으로는 다음 네 가지가 승부처다.

### 1. 학습 데이터 구축

공식 train dataset이 없기 때문에  
외부 데이터 수집·라벨 설계·synthetic data 생성 능력이 매우 중요하다.

### 2. Domain Generalization

공개 예제와 private test의 환경이 다를 수 있다.

```text
camera
resolution
fps
weather
road
compression
accident type
```

변화에 강해야 한다.

### 3. Temporal Modeling

Stage 2와 Stage 3 모두 단일 이미지가 아니라  
시간에 따른 변화가 핵심이다.

### 4. Engineering

최종 시스템은

```text
3개의 Stage
+ video decoding
+ preprocessing
+ inference
+ postprocessing
```

을 모두 합쳐 **60분 안에 L40S 한 장에서 실행**되어야 한다.

따라서 성능뿐 아니라

```text
accuracy × robustness × runtime × reproducibility
```

의 균형이 중요하다.

---

# 28. 다음 실험 순서 제안

```text
[1] 공식 baseline 실행 / 제출 구조 완전 이해
            ↓
[2] 로컬 평가 코드 구현
            ↓
[3] Stage별 external dataset 후보 정리
            ↓
[4] 가장 단순한 3-Stage baseline 제작
            ↓
[5] inference end-to-end 테스트
            ↓
[6] Stage 2 timestamp 모델 강화
            ↓
[7] Stage 3 ego-motion / temporal 모델 강화
            ↓
[8] Stage 1 domain-robust classification 강화
            ↓
[9] ensemble / TTA 검토
            ↓
[10] L40S 60분 runtime 최적화
```

---

# 29. 공식 참고 링크

- 대회 개요  
  https://dacon.io/competitions/official/236753/overview/description

- 데이터  
  https://dacon.io/competitions/official/236753/data

- 평가  
  https://dacon.io/competitions/official/236753/overview/evaluation

- 규칙  
  https://dacon.io/competitions/official/236753/overview/rules

- 일정  
  https://dacon.io/competitions/official/236753/overview/schedule

- Stage별 분석 대상 및 예측 기준 안내  
  https://dacon.io/competitions/official/236753/talkboard/417186

---

## 문서 사용 메모

이 문서는 **2026-09-13 기준 DACON 공식 대회 페이지 및 공식 운영진 답변**을 기반으로 정리했다.

대회 중 규정·평가지표·일정이 변경될 수 있으므로 실제 제출 전 반드시 공식 공지와 평가/규칙 탭을 다시 확인한다.
