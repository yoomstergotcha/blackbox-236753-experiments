# 실험 설계 문서 — 블랙박스 영상 기반 지능형 고의사고 분석 모델

> 기준일: 2026-09-13 (대회 시작 2026-08-26, 팀 병합 마감 2026-09-23, LB 제출 마감 2026-09-29)
> 근거: [COMPETITION_GUIDE.md](COMPETITION_GUIDE.md) + `data/` 실사용 예제 + 베이스라인 노트북 2종 실측

이 문서는 배포된 공개 예제 데이터를 직접 열어본 결과를 바탕으로, 무엇이 "지금 학습 가능한 신호"이고 무엇이 "외부 데이터 없이는 학습 불가능한 신호"인지 구분하고, 그에 따라 앞으로 진행할 실험을 우선순위와 함께 설계한다.

---

## 1. 데이터 탐색 결과

### 1.1 Stage 1 — 재녹화 판별

```text
data/stage1/original/000001~5.mp4     (5개, 286KB~527KB)
data/stage1/rerecorded/000001~5.mp4   (5개, 616KB~1.03MB)
```

**핵심 발견: `stage1/original/*.mp4`은 `stage2/videos/*.mp4`와 MD5가 완전히 동일한 파일이다.**

```text
955094b2...  stage1/original/000001.mp4 == stage2/videos/000001.mp4
0ef4ff47...  stage1/original/000002.mp4 == stage2/videos/000002.mp4
... (5개 전부 동일)
```

즉 Stage 1의 ORIGINAL 5건은 별도로 촬영된 데이터가 아니라 Stage 2 사고 영상 5건을 그대로 재사용한 것이다. RERECORDED 5건은 베이스라인 노트북 주석에 명시된 대로 "재녹화 과정에서 발생할 수 있는 특성을 모사한 파생 예제"이며 실제 다른 기기로 재촬영한 데이터가 아니다 (원본보다 대략 1.5~2배 큰 용량 — 재인코딩/화질 저하 흔적으로 추정).

**의미:**
- 이 10개 샘플로는 "재녹화 vs 원본"을 일반화 학습할 수 없다. 진짜 재녹화(다른 기기로 화면을 재촬영)의 spatial artifact(모아레, 베젤, 반사광, 주사선)가 전혀 반영되어 있지 않다.
- Stage 1 데이터는 자체 제작(synthetic rerecording pipeline)이 사실상 필수다.
- Validation 시 Stage 1과 Stage 2가 같은 원본 소스를 공유할 수 있다는 전제를 유지해야 한다 (실제 대회 데이터에서도 유사한 소스 중복이 있을 가능성 → §22의 leakage 주의사항과 직결).

### 1.2 Stage 2 — 사고 시점/상황

```csv
ID,path,t_collision,t_entry,evasion_space,entry_side
S2_001,videos/000001.mp4,32,-1,-1,-1
...
```

**핵심 발견: `t_collision`(=collision_frame)만 실제 값이고, `t_entry`/`evasion_space`/`entry_side`는 전부 `-1` placeholder다.**

이는 가이드 §3에서 "공개 예제는 충돌시점 라벨만 활용"이라 명시한 것과 정확히 일치한다. 베이스라인 학습 노트북도 실제로 `t_collision`만 GRU 헤드 학습에 쓰고 `scene` 헤드(evasion/entry_side)는 구조 검증용으로만 존재한다 (학습 안 됨, 랜덤 초기화 수준의 출력).

**의미:**
- collision_frame 검출 로직은 5건이나마 실제 신호가 있어 프로토타입 검증이 가능하다.
- entry_frame, evasion_space, entry_side 세 항목은 **공개 데이터로는 전혀 학습 신호가 없다.** 이 셋이 Stage 2 가중치의 65%(0.35+0.15+0.15)를 차지하므로, 외부 사고영상 데이터셋 확보 또는 직접 라벨링이 Stage 2 점수의 사실상 전제조건이다.

### 1.3 Stage 3 — 차량 거동

```csv
ID,sample_index,frame_index,time_seconds,accel_label,steer_label
OPEN_001,0,0,0.0,CONSTANT,STRAIGHT
OPEN_001,60,120,6.0,CONSTANT,RIGHT
...  (영상당 10개 행, 6초 간격)
```

**핵심 발견 두 가지:**

1. `frame_index = 2 × sample_index` — 즉 공개 예제 영상은 **~20fps로 디코딩**되고, `sample_index`는 10Hz(0.1초) 기준이라 프레임 2개당 sample 1개꼴이다. 반면 가이드 §7은 "비공개 평가영상에서는 decoded frame 수 == sample_index 수"라고 명시한다 → **비공개 평가 영상은 정확히 10fps로 제공되어 프레임과 sample이 1:1**이라는 뜻이며, 공개 예제(20fps)와 디코딩 특성이 다르다. 전처리 파이프라인이 fps를 가정하지 말고 항상 `sample_index ↔ decoded frame index` 1:1 매핑으로 짜여 있어야 한다 (현재 베이스라인 `predict_stage3`는 이미 이 가정을 따름 — `sample_index = enumerate(frames)` 순서와 동일).
2. 라벨이 **6초 간격으로만** 존재한다 (영상당 10개 행). 즉 대부분의 프레임에는 라벨이 없는 "형식 확인용" 샘플이다. 실제 비공개 평가는 모든 sample_index에 대해 조밀한 정답이 있다고 가정해야 한다(§7, §8).
3. 파일 크기가 stage1/2 샘플(<1.1MB, 수 초 분량)과 달리 **~37MB/개**로 훨씬 길다 — 실제 주행 시퀀스 전체 분량에 가깝다.

**의미:**
- 5건 x 10 sparse label로는 4-class accel / 3-class steer 분류기를 의미 있게 학습할 수 없다. class 분포도 이미 불균형(STRAIGHT/CONSTANT 압도적, STOPPED 희귀 — §5 참고).
- 외부 주행 데이터셋(BDD100K, comma2k19 등)에서 CAN/IMU 기반 raw 가감속도·조향각을 threshold로 범주화하는 라벨 변환 전략이 Stage 3 성능의 핵심 병목이다 (가이드 §9의 지적과 일치, 데이터로 직접 재확인됨).

### 1.4 Stage 3 라벨 분포 (5개 샘플, 50 rows)

```text
accel_label   CONSTANT 26 · ACCELERATING 13 · DECELERATING 8 · STOPPED 3
steer_label   STRAIGHT 37 · LEFT 5 · RIGHT 8
```

극단적인 class imbalance가 이미 5건짜리 샘플에서도 드러난다. STOPPED·LEFT가 특히 희귀 → 외부 데이터로 재구성할 때 이 두 클래스를 의도적으로 충분히 확보해야 하며, 학습 시 class weight 또는 oversampling이 필요할 가능성이 높다.

---

## 2. 재확인된 핵심 제약 (실험 설계에 직접 영향)

| 항목 | 값 | 실험 설계 영향 |
|---|---|---|
| Stage 2 collision/entry 판정 | \|pred_time − gt_time\| ≤ 0.3초 | 프레임 단위가 아니라 **영상 자체 fps로 환산한 초 단위**로 로컬 metric을 짜야 함 |
| Stage 2 내부 가중치 | collision 0.35 / entry 0.35 / evasion 0.15 / entry_side 0.15 | timestamp 두 개가 70% → EXP 우선순위 최상위 |
| Stage 3 내부 가중치 | accel 0.7 / steer 0.3 | accel 정확도가 steer보다 2배 이상 중요 |
| STOPPED 처리 | steer 점수 계산에서 제외되지만 출력은 필수 | 로컬 metric도 STOPPED 행을 accel만 채점, steer 예측값은 형식상 반드시 채워야 함 |
| collision/entry frame 제출값 | **파일명에 박힌 원본 프레임 번호** (새 인덱스 아님) | 데이터 로더가 프레임 정렬/재넘버링을 하더라도 원본 번호를 별도로 들고 있어야 함 |
| 런타임 | 60분 / L40S 1장 / zip ≤10GB / 압축해제 ≤32GB | 실험 단계부터 stage별 inference 시간을 5개 샘플 기준 측정 → 예상 평가 볼륨으로 외삽 |
| 금지 | test-time learning, pseudo-labeling, 파일 간 통계 공유 | 로컬 실험 파이프라인도 이 규칙을 어기지 않는 구조로 처음부터 설계 (나중에 걷어내기 어려움) |

---

## 3. Validation 전략 (Leakage 방지 우선)

### Stage 1
- Split 단위: **원본 소스 영상 ID**. 같은 소스에서 파생된 ORIGINAL/RERECORDED 쌍은 반드시 같은 split에.
- 자체 제작한 rerecording 파이프라인을 쓸 경우, "재녹화 방식"(기기/디스플레이/압축 프로파일)별로도 최소 1개는 validation 전용으로 완전히 분리 — 실제 평가 환경/기기가 학습 시 못 본 조합일 가능성이 높다(가이드 §21).

### Stage 2
- Split 단위: **사고 영상 ID** (frame 단위 절대 금지).
- 로컬 metric은 실제 대회 공식과 동일하게 구현:
  ```text
  score = 0.35 * collision_within_0.3s
        + 0.35 * entry_within_0.3s
        + 0.15 * evasion_macro_f1
        + 0.15 * entry_side_macro_f1
  ```
- collision/entry는 "정답 프레임 번호를 그 영상의 실제 fps로 초 환산 후 ±0.3초 이내인가"로 판정 (프레임 오차 허용치가 아니라 시간 오차 허용치라는 점 주의).

### Stage 3
- Split 단위: **원본 주행 시퀀스/드라이브 ID** (연속된 sample_index를 자르지 말 것).
- 로컬 metric:
  ```text
  score = 0.7 * accel_macro_f1(전체 sample)
        + 0.3 * steer_macro_f1(accel != STOPPED인 sample만)
  ```
- Macro-F1은 "데이터에 등장한 클래스"가 아니라 "정의된 전체 클래스 집합"(accel 4개, steer 3개) 기준으로 계산 — 클래스가 한 번도 안 나온 split이라도 4/3개 클래스 전부로 나눠야 함.

---

## 4. 실험 로드맵 (Priority 순, 가이드 §23·§28 기반)

각 실험은 `experiments/experiment_log.csv`에 한 행씩 기록한다 (컬럼은 §5 참고).

### Priority 1 — 데이터 구축

| ID | 내용 | 산출물 |
|---|---|---|
| `EXP-S3-DATA-001` | 외부 주행 데이터셋 후보(BDD100K, comma2k19, KITTI, nuScenes, Waymo, HDD, D²-City, A2D2, Udacity) 라이선스·비영리 사용 가능 여부·CAN/IMU 신호 유무 조사 | `dataset_registry.csv` 초안 |
| `EXP-S3-LABEL-001` | 선정한 외부 데이터의 raw accel/yaw-rate/steering-angle을 4-class accel_label / 3-class steer_label로 변환하는 threshold 규칙 설계 + 샘플 프레임 육안 검증 | 라벨 변환 스크립트 + threshold 근거 노트 |
| `EXP-S1-DATA-001` | 재녹화 시뮬레이션 파이프라인 설계 (디스플레이 베젤 합성, 모아레/주사선 노이즈, 재압축, 원근 왜곡, 반사광) — 실제 재촬영 데이터 확보 가능성도 병행 조사 | synthetic rerecording augmentation 스크립트 |
| `EXP-S2-DATA-001` | entry_frame/evasion_space/entry_side 실 라벨이 있는 외부 사고영상(CCD 계열 등) 소싱 또는 공개 5건 직접 라벨링(sanity check용 소량) | 라벨링된 소규모 검증셋 |

### Priority 2 — Stage 2 시점 검출 고도화 (내부 가중치 70%)

| ID | 내용 |
|---|---|
| `EXP-S2-TIME-001` | 현재 "단일 GRU + argmax" 방식을 가이드 §5 권장 구조(vehicle detection → tracking → collision frame → backward tracking → lane entry frame)로 교체, ablation 비교 |
| `EXP-S2-TIME-002` | collision frame 기준 backward tracking으로 entry_frame 추정 — 별도 head 없이 궤적 기반 규칙으로 우선 baseline 확보 후 학습형으로 전환 |
| `EXP-S2-SCENE-001` | evasion_space/entry_side는 실 라벨 확보 전까지 lane/도로 경계 segmentation 기반 룰베이스 heuristic으로 대체 (학습 헤드보다 먼저 동작 검증) |

### Priority 3 — Stage 3 ego-motion 모델 (accel 비중 0.7)

| ID | 내용 |
|---|---|
| `EXP-S3-BASE-001` | 현재 MViT-only clip 분류기 대비 optical-flow/ego-motion 특징 추가 버전 ablation (RGB-only vs flow-only vs fused) |
| `EXP-S3-CLASS-001` | STOPPED/LEFT 등 희귀 클래스 대응 (class weight, oversampling) — §1.4에서 확인한 불균형 기준으로 튜닝 |
| `EXP-S3-STEER-001` | steer는 accel보다 비중 낮지만(0.3) STOPPED 프레임 제외 규칙이 채점에 영향 → 로컬 metric에 정확히 반영 후 별도 튜닝 |

### Priority 4 — Stage 1 robust baseline

| ID | 내용 |
|---|---|
| `EXP-S1-BASE-001` | synthetic rerecording 데이터로 재학습한 baseline과 현재 베이스라인(5쌍만 학습) 비교 |
| `EXP-S1-ROBUST-001` | 압축/해상도/기기 도메인을 인위적으로 다양화한 validation set에서 일반화 확인 |

### Priority 5 — Inference 최적화

| ID | 내용 |
|---|---|
| `EXP-INFRA-001` | **로컬 평가 하네스** 구현 — §3의 세 metric 공식을 코드로 고정 (다른 모든 실험이 이 위에서 비교됨) — **완료**, [src/eval/](src/eval/) 참고 |
| `EXP-INFRA-002` | 5개 공개 샘플 기준 stage별 inference 시간 측정 → 예상 평가 볼륨으로 외삽, 60분 예산 대비 여유 확인 |
| `EXP-INFRA-003` | submit.zip 용량/압축해제 용량 추적 (ensemble 도입 시 10GB/32GB 한도 재확인) |

---

## 5. 지금 시작하는 레지스트리

가이드 §11·§25는 2차 평가 진출 시 데이터·모델 출처 전체를 기술해야 한다고 명시한다. 실험을 시작하는 시점부터 아래 세 파일에 기록한다 (본 커밋에서 헤더만 생성):

- [`dataset_registry.csv`](dataset_registry.csv) — 외부 데이터셋별 `name, url, license, download_date, purpose, stage, preprocessing`
- [`model_registry.md`](model_registry.md) — pretrained weight별 출처·라이선스
- [`experiments/experiment_log.csv`](experiments/experiment_log.csv) — 실험별 `experiment_id, date, stage, dataset_version, split_version, model, pretrained_weight, input_resolution, clip_length, sampling, augmentation, optimizer, lr, epochs, seed, validation_score, inference_time, model_size, notes`

---

## 6. 로컬 평가 하네스 사용법 (`EXP-INFRA-001`)

```text
src/eval/
├── metrics.py       # 순수 채점 로직 — pandas만 필요, cv2 의존 없음
├── video_stats.py   # Stage 2 fps/최대 프레임 계산 (cv2, 지연 import)
└── harness.py       # CLI: gt/pred CSV 두 개로 즉시 점수 확인
```

```bash
# 채점 공식 자체가 맞는지 수기 계산값과 대조 (실행 환경에 pandas만 있으면 됨)
python -m src.eval.metrics

# 실제 검증 split에 대해 채점 (pred/gt는 predict_stageX와 동일한 컬럼 스키마)
python -m src.eval.harness --stage 1 --pred stage1_pred.csv --gt stage1_gt.csv
python -m src.eval.harness --stage 2 --pred stage2_pred.csv --gt stage2_gt.csv --video-dir data/stage2/videos
python -m src.eval.harness --stage 3 --pred stage3_pred.csv --gt stage3_gt.csv
```

`metrics.py`의 세 `score_stageN` 함수는 각각 breakdown(하위 metric별 점수, 무효값 개수, fps 누락 ID 등)까지 dict로 반환하므로, `total_score(s1, s2, s3)`로 종합점수를 바로 계산할 수 있다. 이 위에서 §4의 모든 실험을 비교한다.

2026-09-13: 이 편집 머신에 Python 3.12(`winget`, `.venv/`)를 설치해 실제로 실행·확인했다 — `python -m src.eval.metrics` PASS, 손계산값과 정확히 일치 (§9-1 참고). `.venv/Scripts/python.exe -m pip install -r requirements-dev.txt`로 numpy/pandas/opencv-python-headless/PyYAML을 넣으면 이 저장소의 `src/eval`, `src/data/comma2k19` 모두 이 머신에서 바로 실행 가능하다.

## 7. 다음 액션 체크리스트

- [x] `EXP-INFRA-001` 로컬 평가 하네스 구현 — [src/eval/](src/eval/), self-test 실제 실행 PASS
- [x] `EXP-S3-DATA-001` 외부 데이터셋 1차 조사(라이선스 + CAN/steering 신호) — §8, [dataset_registry.csv](dataset_registry.csv)
- [x] `EXP-S3-LABEL-001` comma2k19 라벨 변환 파이프라인 — §9, [src/data/comma2k19/](src/data/comma2k19/). Python/awk 이중 검증 일치
- [x] `EXP-S3-LABEL-002` 64-segment/21-route 확대 검증 — §10. class distribution 개선, route split 구현·테스트, STEER_SIGN 근거 대폭 강화. **§11 Human Review 대기 중 — 답 오기 전까지 학습 착수 금지**
- 참고: 이 머신에 Python 3.12(`.venv/`)와 NVIDIA RTX 3060(12GB)이 확인됨 — `EXP-S3-BASE-001` 실행 환경 자체는 준비돼 있음
- [ ] Stage 1/2 소스 중복(§1.1) 을 반영해 validation split 로직에 그룹 ID 강제
- [ ] Stage 3 sample_index/frame_index 1:1 가정(§1.3)을 전처리 코드에 명시적으로 검증하는 assert 추가
- [ ] 베이스라인 `predict_stage1`/`predict_stage3`의 CUDA 필수 제약(§ 확인됨) 때문에 로컬 GPU 없는 환경에서는 dry-run 불가 — 실행 환경 확보 필요

---

## 8. Stage 3 외부 데이터셋 조사 결과 (`EXP-S3-DATA-001`)

가이드 §9가 예시로 든 9개 후보를 실제 라이선스 원문과 "CAN/조향각 신호가 실제로 있는가"를 기준으로 조사했다. 전체 근거는 [dataset_registry.csv](dataset_registry.csv)에 기록. 판단 기준은 가이드 §11의 조건("법적 제한 없음 + 누구나 접근 가능 + 최소 비영리 사용 허용")이다.

### 요약

| 데이터셋 | 라이선스 | 조향/CAN 신호 | 비고 |
|---|---|---|---|
| **comma2k19** | **MIT (상업적 사용도 허용)** — 최초 조사 때 이름이 비슷한 다른 comma.ai 데이터셋과 혼동해 CC BY-NC-SA로 잘못 기록했었음, §9에서 재확인 후 정정 | steering_angle(deg), speed(m/s), wheel speed — **직접 제공** | 33시간/94.6GB, 규모 충분. Stage3 라벨 변환에 가장 바로 쓸 수 있고 라이선스도 가장 유연 |
| **A2D2** | **CC BY-ND 4.0 (상업적 사용도 허용)** | steering wheel angle, throttle, brake, 가감속 — **직접 제공** | 조사 후보 중 라이선스가 가장 유연 |
| **Udacity (CH2)** | **MIT (상업적 사용 허용)** | steering angle, throttle, brake, speed — **직접 제공** (driving_log.csv) | 규모는 작음, 저장소 archived(2021) → 다운로드 경로 재확인 필요 |
| **nuScenes (CAN bus expansion)** | CC BY-NC-SA 4.0 (비영리) | steering angle feedback(rad), wheel speed, throttle, brake — 제공되나 **일부 scene에만 존재** | 사용 전 커버리지 확인 필수 |
| **KITTI** | CC BY-NC-SA 3.0 (비영리) | 조향각 없음, OXTS로 속도·가속도·yaw rate만 | steer_label은 yaw rate로 근사, 규모 작음 |
| **HDD (Honda)** | 비영리, **대학 이메일 필요** | throttle, brake, steering angle, yaw rate, speed — 직접 제공 | 접근 장벽(대학 소속) 확인 필요 |
| **BDD100K** | 연구용 무료 / 상업적 사용은 BDD·BAIR 회원 한정 | CAN 없음, GPS+IMU 융합 course/speed만 | steer_label 근사 정밀도 낮음. Stage1/2 소스로서의 가치가 더 큼 |
| **Waymo Open Dataset** | 비영리(연구/출판/벤치마킹) | Motion/WOD-E2E가 ego pose·velocity·acceleration **직접 제공**, 신호 품질 최상 | ⚠️ 아래 참고 |
| **D²-City** | **확인 실패** (공식 페이지가 내부 IP로 리다이렉트) | 없음 (대시캠 영상 + object annotation뿐) | Stage3에는 부적합, Stage1/2용으로만 재검토 |

### ⚠️ Waymo Open Dataset — 법적 리스크 플래그

Waymo의 라이선스 원문은 비영리 목적을 "research, teaching, scientific publication, personal experimentation"으로 정의하면서, **"litigation, licensing, or enforcement를 향한 목적은 일부라도 비영리 목적에서 제외한다"**고 명시한다.

이 대회는 **국립과학수사연구원(포렌식 기관)이 주관하는 고의사고(보험사기 등) 분석 모델** 과업이다. 신호 품질만 보면 Waymo가 가장 매력적인 후보지만, 최종 활용 목적이 사실상 법적 판단/수사 지원에 가깝기 때문에 라이선스의 enforcement/litigation 제외 조항에 저촉될 가능성이 있다. **법무 확인 또는 주최측 문의 없이는 사용하지 않는 것을 권장**한다.

### 권장 우선순위

1. **comma2k19** — CAN 신호 완비 + 규모 충분(94.6GB, 33시간) + 라이선스 MIT(상업적 사용도 허용, §9에서 재확인). 실제로 `EXP-S3-LABEL-001`에 투입해 파이프라인 검증까지 마쳤다 — §9 참고.
2. **A2D2** — 라이선스가 가장 유연하고 신호도 완비. comma2k19와 함께 조합해 다양성 확보.
3. **Udacity CH2** — 라이선스 매우 단순, 신호 직접 제공. 다만 archived 저장소라 다운로드 경로부터 재확인.
4. **nuScenes CAN bus expansion** — 커버리지 확인 후 보조 데이터로 편입.
5. **KITTI / HDD** — KITTI는 조향각 부재로 근사치만, HDD는 접근 장벽 확인 후 판단.
6. **BDD100K** — Stage3 signal은 근사 수준이라 우선순위 낮음. Stage1/2(도로 다양성) 쪽 활용을 별도로 검토.
7. **Waymo** — 법적 확인 전까지 보류.
8. **D²-City** — 이번 조사에서 접속 실패, Stage3 용도로는 애초에 신호가 없어 제외. Stage1/2 후보로 재조사할 가치만 있음.

---

## 9. `EXP-S3-LABEL-001` 결과 — comma2k19 라벨 변환 파이프라인

> 이 섹션은 이 문서의 이전 버전을 대체한다. 이전 버전은 comma2k19 전체 데이터셋의 라이선스를 CC BY-NC-SA 3.0으로 잘못 기록했었다 — 아래 "라이선스" 항목에서 정정 경위를 그대로 남겨둔다.

### 1. Metric self-test — **PASS**

이 편집 머신에 Python이 전혀 없어서 이전까지는 `_self_test()`를 손으로 계산한 값으로만 검증했다. 이번에 `winget`으로 Python 3.12를 설치하고(`C:\projects\BLACKBOX_LB\.venv`, numpy/pandas/opencv-python-headless/PyYAML) 실제로 실행했다.

```text
$ python -m src.eval.metrics
모든 self-test 통과
  stage1=0.5000 stage2=0.7250 stage3=0.5167
  total=0.5967
```

손계산값과 정확히 일치 (exit code 0). 수정 사항 없음.

### 2. Python subset pipeline — awk 검증값과 **완전히 일치**

```text
$ python -m src.data.comma2k19.run_subset --segment data/external/comma2k19_example/segment --out output/comma2k19_subset
sample 수=600, duration=59.907s, timestamp range=[46408.590, 46468.497]

accel_label   ACCELERATING=143(23.83%) DECELERATING=123(20.50%) CONSTANT=334(55.67%) STOPPED=0(0.00%)
steer_label   LEFT=0(0.00%) STRAIGHT=593(98.83%) RIGHT=7(1.17%)
```

이전에 awk로 독립 구현해 실제 다운로드한 숫자로 실행했던 결과와 소수점까지 동일하다 — timestamp interpolation, (N,1) squeeze, 10Hz resampling, threshold 구현 네 가지 모두 python/awk 두 구현이 일치하므로 로직 자체의 신뢰도가 높다고 판단.

### 3. STEER_SIGN — **motion-signal로 검증 완료, video 검증은 미완**

**방법 A — 영상 육안 확인 (8.1): 불확정.** steering_angle이 가장 큰 구간(sample 96~104, -3.0~-4.6도)의 overlay 프레임을 직접 열어 봤다 ([output/comma2k19_subset/overlay_steer_check/](output/comma2k19_subset/overlay_steer_check/)). 이 구간은 편도 다차선 직선 도로(주차차량 줄지어 있는 도심 대로)를 ~20m/s로 그대로 직진하는 장면이고, 프레임 85→91→95→98(피크)→105→112까지 vanishing point/차선 위치가 거의 변하지 않는다. **즉 이 60초 세그먼트에는 눈으로 확인할 만한 실제 회전이 없다** — steering_angle 변동폭(-4.6~+2.5도)이 차선 유지 수준의 미세 보정이라 화면상 좌/우를 시각적으로 판별할 수 없었다. 이건 원래 이 문서(개정 전 §11)가 예상했던 리스크와 정확히 일치한다: "해당 60초 영상 자체가 거의 직진 주행일 수 있다."

**방법 B — motion signal 교차검증 (8.2): 완료, 강한 근거.** 같은 세그먼트의 `processed_log/IMU/gyro/{t,value}`(z축 = 요레이트)를 받아 steering_angle과의 관계를 실측했다.

```text
n=4973 (겹치는 구간)
pearson corr(steering_angle, gyro_z) = -0.699
|steer|>1deg & |gyro_z|>1deg/s인 39개 구간에서 부호 일치율 = 0% (=100% 반대 부호)
```

좌표계 확인: comma2k19 IMU는 **forward-right-down(FRD)** 축이다(README 확인). FRD는 우수좌표계(F×R=D)이므로 각속도 공식 `d(F)/dt = ω × F`에 `ω = ω_z·D`를 대입하면 `d(F)/dt = ω_z·(D×F) = ω_z·R`이 나온다 — **양의 gyro_z는 차량 정면(F)이 R(우측) 방향으로 회전한다는 뜻, 즉 우회전이다.**

steering_angle은 gyro_z와 거의 항상 반대 부호이므로: **steering_angle이 양수일 때는 gyro_z가 음수 → 차량이 좌측으로 회전 → LEFT.** 이는 현재 코드의 `STEER_SIGN=+1`(양수 steering_angle → LEFT)과 정확히 같은 결론이며, 자동차 업계에서 흔한 SAE 스타일 관례(steering wheel 반시계=좌회전=양수)와도 부합한다(참고로 nuScenes CAN bus expansion 문서도 "positive = left turn"이라고 명시 — 다른 차량이지만 같은 업계 관례).

**결론:** `STEER_SIGN=+1`은 물리적으로 유도되고 실측 데이터로 뒷받침되는 근거를 갖췄다 — 하지만 이 세그먼트가 실제 회전을 포함하지 않아 "영상 육안 확인"이라는 원래 요구조건은 아직 충족하지 못했다. §8.2 원칙("영상 육안 판단과 motion signal이 같은 결론을 지지할 때 최종 결정")을 문자 그대로 지키면, **아직 "COMPLETE"로 닫을 수 없다** — `EXP-S3-LABEL-002`에서 실제 회전이 포함된 세그먼트로 재확인하는 것을 조건으로 "provisionally verified"로 표시한다.

`labeling.py`와 `configs/stage3/comma2k19_label.yaml`에 위 근거를 그대로 기록해뒀다.

### 4. Label conversion — config로 분리 완료

`src/data/comma2k19/labeling.py`를 `configs/stage3/comma2k19_label.yaml`에서 파라미터를 읽도록 리팩터링했다 (`LabelConfig` dataclass). 기존 하드코딩 값과 정확히 동일한 기본값을 유지해 회귀 없음을 재실행으로 확인(§2 숫자 그대로 재현됨).

```yaml
stopped_speed_threshold_mps: 0.5
accel_positive_threshold_mps2: 0.3
accel_negative_threshold_mps2: -0.3
steering_deadzone_deg: 3.0
steer_sign: 1              # §9-3 근거로 확정 (provisional)
speed_smoothing_window: 1  # smoothing 없음 — 아직 튜닝 안 함
steering_smoothing_window: 1
```

- speed → (옵션 smoothing) → 중심차분 → accel_mps2 → 4-class
- steering_angle → (옵션 smoothing) → deadzone threshold → 3-class

smoothing window는 이번 60초 데이터로 튜닝하지 않았다(§11 원칙 그대로 유지 — "이 데이터에서 예쁘게 나오라고 threshold를 억지로 조정하지 않는다"). 기본값 1(=smoothing 없음)로 두어 §2/§3 결과와 동일하게 재현되는 것만 확인했다.

**미확정 threshold**: 4개 값 모두 이 세그먼트 하나(고속도로 직진, STOPPED/LEFT 예시 0개)로는 튜닝할 근거가 없다. `EXP-S3-LABEL-002`에서 다양한 route가 섞인 청크를 받은 뒤 재검토.

### 5. Full dataset 계획 (`Task E`, 다운로드는 아직 안 함)

**라이선스 재조사 결과 — 정정.** 이전 버전에서 전체 comma2k19 데이터셋을 "CC BY-NC-SA 3.0"으로 기록했던 것은 **오류였다.** 그 근거로 인용한 Academic Torrents 페이지(`58c41e8b...`)는 comma2k19가 아니라 이름이 비슷한 **다른** comma.ai 데이터셋("comma.ai driving dataset", 2016년 공개, dog/emily/frodo 드라이버, HDF5 포맷, 7시간 분량, CC BY-NC-SA 3.0)이었다. comma2k19를 정확히 지칭하는 Academic Torrents 페이지(`65a2fbc964078aff62076ff4e103f18b951c5ddb`)와 HuggingFace `commaai/comma2k19` dataset card(YAML `license: mit`)를 다시 확인한 결과, GitHub LICENSE·Academic Torrents·HuggingFace 세 곳 모두 **MIT**로 일치한다. `dataset_registry.csv`와 [data/external/comma2k19_example/README.md](data/external/comma2k19_example/README.md)를 함께 정정했다.

| 항목 | 내용 |
|---|---|
| license | **MIT** (상업적 사용 포함 허용) |
| 예상 용량 | 전체 94.6GB, `Chunk_1.zip`~`Chunk_10.zip` (각 ~8.7~9.9GB), 2019개 1분 세그먼트를 10개 청크로 분할 |
| 다운로드 방식 | Academic Torrents(등록 불필요, torrent client 필요) 또는 HuggingFace `commaai/comma2k19`(HTTP resolve 가능성 있음 — 이번 조사에서는 개별 파일 단위 접근 여부까지는 확인 못함) |
| 우선 사용할 subset | **Chunk 1개(~9GB)만 먼저** — 그 안에 여러 route가 섞여 있어 route-level split 검증과 STOPPED/LEFT 예시 확보에 충분할 가능성이 높음. 전체 94.6GB는 파이프라인이 청크 1개에서 문제없이 돈 뒤에 결정 |
| CAN/steering/speed 유무 | 이미 §2에서 실측 확인 완료 (모든 세그먼트가 동일 폴더 구조를 따를 것으로 예상되나, chunk 1개를 받으면 2~3개 세그먼트를 골라 구조가 실제로 동일한지 재확인할 것) |

### 6. Train/validation split 설계

comma2k19는 폴더명이 `<device-hash>|<datetime>/<segment-번호>` 구조다(예: `b0c9d2329ad1606b|2018-08-02--08-34-47/40`). `<device-hash>|<datetime>` 부분이 하나의 연속 주행(**route**)이고 그 안의 `<segment-번호>`(0,1,2,...)들이 그 route를 1분 단위로 자른 조각이다.

**Split 단위는 반드시 route(`<device-hash>|<datetime>`) 전체**로 잡는다 — 같은 route의 segment 0과 segment 1은 시간적으로 이어져 있어 각각 다른 split에 들어가면 temporal leakage가 발생한다(가이드 §22의 Stage3 leakage 주의사항과 동일한 논리). GroupKFold 등에서 group key = route id.

아직 route가 1개(`Example_1`)뿐이라 실제 분할을 적용해보지는 못했다 — chunk를 받으면 route 목록을 뽑아 group split 코드를 추가한다.

### Training gate (`§15`) 현재 상태

```text
[x] src.eval.metrics self-test PASS
[x] Python sync 결과가 real demo 검증값(awk)과 일치
[x] speed/value (N,1) shape 처리 확인
[x] speed / steering / frame timestamp 독립 보간 확인
[~] STEER_SIGN 영상 기반 검증 완료      <- motion-signal 근거는 강하나 video 검증은 미완 (회전이 없는 세그먼트라 판정 불가)
[x] STEER_SIGN 코드/문서에 근거 기록      <- labeling.py, configs/stage3/comma2k19_label.yaml, 본 섹션
[~] Stage 3 label conversion 결과 육안 sanity check 완료  <- 위와 동일한 이유로 부분 완료
[ ] train/validation route split 설계 완료  <- 설계는 §6에 기록, 실제 다중 route 데이터로 적용은 아직
```

**`EXP-S3-LABEL-001`은 아직 COMPLETE가 아니다.** 모델 학습은 계속 보류한다. 남은 항목은 모두 "실제 회전이 있는 route가 포함된 데이터가 없어서" 생기는 gap이라, `EXP-S3-LABEL-002`(청크 1개 확보)가 사실상 이 gap을 메우는 다음 실험이다.

---

## 10. `EXP-S3-LABEL-002` 결과 — 64-segment/21-route 대규모 검증

### Objective
`EXP-S3-LABEL-001`이 1개 세그먼트(60초, 회전 없음)로만 검증됐던 gap — 다양한 route에서 dense label을 만들어 class distribution을 재확인하고, STEER_SIGN을 더 큰 규모로 재검증하고, route-level split을 실제로 구현·테스트한다.

### Current Bottleneck
route 다양성 부족 (§9 Training gate의 `[~]`/`[ ]` 3항목 모두 이 한 가지 원인).

### 데이터 소스 결정 — Chunk 대신 HF demo parquet mirror

원래 계획(§9-5)은 comma2k19 `raw_data/Chunk_1.zip`(~9GB)을 받는 것이었다. 실행 전에 HuggingFace `commaai/comma2k19` 저장소 구조를 다시 확인해보니, 같은 저장소에 **`data/demo-*.parquet` 3개 파일(총 227.9MB)로 구성된 "demo" split이 별도로 존재**했다 — 64개 실제 세그먼트(21개 route)의 CAN/IMU/pose 신호 전체(`log` struct: speed, steering_angle, gyro, accelerometer, wheel_speed, GNSS 등)를 담고 있고, 다만 영상은 전체 프레임이 아니라 세그먼트당 `preview` 1장뿐이다.

이 mirror는 (a) MIT 라이선스로 이미 확인된 것과 동일한 배포물이고, (b) 크기가 228MB로 이미 이번 세션에서 반복적으로 받아온 개별 파일들과 같은 스케일이며, (c) torrent client 없이 `curl`만으로 받을 수 있어서 — `AUTONOMOUS_RESEARCH_AGENT.md` §15/§26이 사람 승인을 요구하는 "비용이 큰 전체 데이터 다운로드"에 해당한다고 보지 않고 그대로 진행했다. 9GB `Chunk_*.zip`이나 torrent 전체 다운로드는 여전히 손대지 않았다.

### Hypothesis
현재 문제는 **route 다양성 부족** 때문이다. 세그먌트 수를 1개→64개(21 route)로 늘리면 (1) STOPPED/LEFT 같은 희귀 클래스가 실제로 나타날 것이고, (2) STEER_SIGN의 gyro_z 교차검증이 통계적으로 훨씬 강해질 것이며, (3) route-level split을 실제 데이터로 구현·검증할 수 있을 것으로 예상한다.

### Single Change
`EXP-S3-LABEL-001`의 로직(`sync.py`/`labeling.py`/`report.py`)은 그대로 두고, 입력 소스만 "디스크의 세그먼트 폴더 1개" → "parquet의 64개 세그먼트"로 바꿨다. 이를 위해 `sync.py`에 `Comma2k19Segment.from_arrays()`를 추가(로딩 로직은 `_prepare_signal`을 공유해 중복 없음)하고, 새 스크립트 `src/data/comma2k19/analyze_demo64.py`와 route-split 모듈 `src/data/comma2k19/split.py`를 추가했다. **labeling 로직 자체는 한 글자도 바꾸지 않았다** — 1-세그먼트 결과가 그대로 재현되는 것으로 회귀 확인.

### Result

**Class distribution (64 세그먼트, 38,370 sample = 63.95분):**

```text
accel_label   ACCELERATING 17.1%  DECELERATING 15.3%  CONSTANT 60.7%  STOPPED 7.0%
steer_label   LEFT 15.7%  STRAIGHT 64.4%  RIGHT 20.0%   (STOPPED 제외해도 거의 동일)
```

1개 세그먼트에서는 0%였던 **STOPPED(7.0%)와 LEFT(15.7%)가 이제 실제로 관측됨** — §1.4에서 지적한 class imbalance가 여전히 있지만(STRAIGHT/CONSTANT 우세), 4/3개 클래스 전부가 의미 있는 비율로 나타난다.

**gyro_z 교차검증 (64/64 세그먼트 전부):**

```text
corr(steering_angle, gyro_z) 평균 = -0.907  (범위: -0.997 ~ -0.358)
corr < 0인 세그먼트: 64/64 (100%, 예외 없음)
뚜렷한 구간(|steer|>1deg, |gyro_z|>1deg/s) 96,036개 중 반대부호: 96,036개 (100%)
```

21개의 서로 다른 route(날짜·시간대 모두 다름)에서 예외 없이 같은 결론이 나왔다 — §9-3의 물리적 유도(FRD 좌표계, +gyro_z=우회전)와 결합하면 **STEER_SIGN=+1(양수 steering_angle=LEFT)에 대한 근거가 크게 강화**됐다.

**Route-level split (route id로 group split, 실제 21 route에 적용):**

```text
train: 51 segments / 17 routes
val:   13 segments / 4 routes
assert_no_route_leakage() 통과 (겹치는 route 0개)
```

`src/data/comma2k19/split.py`의 `group_train_val_split()`을 실제 데이터로 테스트해 route가 절대 섞이지 않는 것을 확인했다.

**회전 후보 세그먼트 식별:** `b0c9d2329ad1606b|2018-08-17--14-55-39/2` — steering_angle 최대 **258도**, RIGHT 84.3%/LEFT 12.5%/STRAIGHT 3.2%. 이 정도 각도는 고속도로 미세 보정이 아니라 실제 급회전(교차로/유턴 등)일 가능성이 매우 높다. **영상 육안 확인을 위한 최우선 타겟.**

### Baseline Comparison
`EXP-S3-LABEL-001`(1세그먼트) 대비: STOPPED/LEFT 클래스 최초 관측, gyro 교차검증 표본이 39개(1세그먼트) → 96,036개(64세그먼트, 21route)로 확대, route split이 설계 단계에서 실제 구현·테스트 단계로 진전.

### Self-Critique
- **Leakage 가능성**: 없음 — 아직 모델 학습을 하지 않았으므로 train/val leakage 자체가 발생할 수 없는 단계다. 다만 route-split 코드가 실제로 route를 안 섞는지는 `assert_no_route_leakage()`로 확인했다(통과).
- **같은 route가 섞였는가**: 아니오, 확인됨.
- **단일 seed 우연**: 해당 없음 — 실측 신호에 대한 결정론적 계산(상관계수, 임계값 분류)이라 seed에 의존하지 않는다(route-split의 셔플만 seed=20260825 고정, 재현 가능).
- **특정 class만 개선되고 다른 게 무너졌는가**: 해당 없음(모델 비교가 아님).
- **validation 과적합**: 해당 없음.
- **runtime/model size 증가**: 해당 없음(모델 없음). 64 세그먼트 라벨 생성은 이 CPU 머신에서 1분 이내.
- **데이터 라벨 오류가 gain처럼 보이는 것 아닌가**: 가능성 점검 — parquet의 `speed/value`가 여전히 (N,1) 형태(object dtype)인 것을 실측으로 재확인하고 동일한 squeeze 로직(`_prepare_signal`)을 강제했다. `b0c9d2329ad1606b|2018-08-17--14-55-39/2`의 258도라는 극값이 버그(예: 단위 오류)가 아니라 실제 급회전일 가능성이 높다고 판단한 근거는 (a) 물리적으로 타당한 범위(자동차 조향각은 풀락 시 수백 도까지 감 — 예: ±540도가 흔한 스펙), (b) 같은 세그먼트의 steer_label 분포(RIGHT 84%)가 gyro_z 부호와도 일관됨(개별 세그먼트 corr 확인은 안 했지만 전체 64개 중 하나이므로 같은 corr<0 population에 속함).
- **metric 구현 오류 가능성**: `labeling.py`/`sync.py`의 핵심 함수는 EXP-S3-LABEL-001에서 이미 python/awk 이중 검증을 마쳤고, 이번엔 그 함수를 코드 변경 없이 재호출만 했다(입력 adapter만 추가). 회귀 테스트로 1-세그먼트 결과가 그대로 재현되는 것을 확인.
- **규정 리스크**: 없음. hidden test 데이터 접근이나 대회 평가 파일과 무관한 외부 공개 데이터 라벨링이며, MIT 라이선스 데이터를 비영리·연구 목적으로 사용했다. §3(가이드 §12 대응) 절대 규칙 위반 없음.

### Decision
**KEEP** — class distribution 개선, route split 구현·검증 완료, STEER_SIGN 근거 대폭 강화. 단, 아래 Human Review 항목이 해소되기 전까지 **Training은 계속 BLOCKED**다.

### What We Learned
- route 다양성이 진짜 병목이었다는 가설이 맞았다 — 데이터 소스만 바꿔서(로직은 그대로) STOPPED/LEFT 클래스 관측, split 코드 실동 테스트까지 한 번에 해결됨.
- 9GB 청크를 받지 않고도 "pipeline 검증"이라는 목적은 이미 100% 충족됐다 — 남은 것은 순수하게 "영상 육안 확인"이라는, 데이터양이 아니라 **영상 유무**의 문제다. 무작정 더 큰 데이터를 받는 것은 이 시점에서는 기대효용이 낮다(§12 Expected Utility 관점: video 없이는 gain=0).
- 급회전 후보 세그먼트를 구체적으로 특정할 수 있게 됐다 — 다음에 영상이 필요하다면 9GB 전체가 아니라 이 세그먼트 하나(~37MB급)만 targeted로 받는 것이 압도적으로 효율적이다.

### Next Highest-Value Experiment
영상 없이 할 수 있는 일은 사실상 다 했다. 다음으로 기대효용이 가장 높은 행동은 **새 실험이 아니라 아래 Human Review 요청**이다 — `AUTONOMOUS_RESEARCH_AGENT.md` §15 조건 3·4("영상 육안 판단 필요", "STEER_SIGN처럼 잘못되면 전체 라벨이 뒤집힘")에 정확히 해당하기 때문에, 이 시점에서 더 진행하는 것보다 사람의 판단을 받는 것의 기대효용이 더 크다고 판단했다.

---

## 11. Human Review 요청 (`AUTONOMOUS_RESEARCH_AGENT.md` §15)

**해당 조건:** §15의 3번("영상 육안 판단이 필요한데 자동으로 확정할 수 없는 경우")과 4번("STEER_SIGN처럼 잘못 결정하면 전체 라벨이 뒤집히는 경우")에 정확히 해당한다. §26은 이런 경우를 추측으로 확정하는 것을 명시적으로 금지한다.

### 지금까지의 근거 요약
- 물리 유도(FRD 좌표계, `d(forward)/dt = ω × forward`): +gyro_z = 우회전
- 실측: 21개 route, 64개 세그먼트 **전부**에서 steering_angle과 gyro_z가 반대 부호(평균 corr -0.907, 최악 -0.358, 뚜렷한 구간 96,036개 중 반대부호 100%)
- 결론: **positive steering_angle = LEFT**, 즉 현재 코드의 `STEER_SIGN=+1`이 맞다는 강력한 통계적/물리적 근거

이 근거는 매우 강하지만, **실제 영상에서 "차가 화면상 좌회전하는 것을 눈으로 보고 확인"한 적은 아직 없다** — 지금까지 확보한 실제 영상 2개(Example_1, compression_challenge 세그먼트) 모두 회전이 없거나(전자) CAN 데이터가 없다(후자).

### 선택지 A — 지금의 motion-signal 근거만으로 STEER_SIGN=+1을 확정하고 학습 진행
- 장점: 즉시 `EXP-S3-BASE-001`(baseline 학습) 착수 가능. 이 GPU 머신(RTX 3060, 12GB VRAM 확인됨)에서 바로 실행 가능.
- 단점: 만약 이 근거가 (예: comma2k19 특정 차량의 CAN 부호가 SAE 관례와 반대인 경우처럼) 틀렸다면, Stage 3 steer_label 전체가 뒤집혀 처음부터 다시 학습해야 함 — §15/§26이 정확히 이런 리스크 때문에 사람 승인을 요구한다.

### 선택지 B — segment `2018-08-17--14-55-39/2` 영상만 targeted로 받아 최종 확인 후 진행
- 장점: 9GB 전체가 아니라 이 세그먼트 하나(~37MB 추정, 이미 받은 개별 영상들과 같은 스케일)만 필요. HuggingFace CDN이 `Accept-Ranges: bytes`를 지원하는 것은 확인했지만, 이 세그먼트가 `Chunk_1~10.zip` 중 어디에 들어있는지 매핑은 아직 확인 못했다 — 매핑을 못 찾으면 결국 특정 청크(~9GB) 전체를 받아야 할 수 있음.
- 단점: 추가 조사·다운로드 시간이 필요하고, 매핑 확인이 안 되면 사실상 선택지 C(전체 청크 다운로드)로 이어질 수 있음.

### 추천안
**선택지 B를 먼저 시도**하되, chunk 매핑을 못 찾으면 **선택지 A(현재 근거로 학습 진행)** 로 넘어가는 것을 권장한다. 이유: 근거가 이미 매우 강하고(21 route, 0 예외), 훈련 자체는 작은 baseline부터 시작할 것이므로 Stage 3 steer 방향이 틀렸을 경우에도 초기 발견 비용이 크지 않다 — 단, **이 판단(A로 갈지, B를 더 파볼지)은 사람이 결정**해야 한다는 것이 `AUTONOMOUS_RESEARCH_AGENT.md`의 명시적 규칙이다.

### 필요한 사람의 판단
1. 선택지 A(지금 근거로 진행) vs B(targeted 영상 추가 확보) 중 무엇을 원하는지
2. B를 선택할 경우, chunk 매핑 조사에 추가 시간을 써도 되는지, 아니면 (필요하다면) 해당 청크 하나(~9GB) 전체 다운로드까지 승인하는지

### 결정 — 2026-09-14, 선택지 A 승인

사용자가 **선택지 A**를 승인했다: 21route/64segment motion-signal 근거(gyro_z corr 평균 -0.907, 64/64 예외 없음)만으로 `STEER_SIGN=+1`을 확정하고 `EXP-S3-BASE-001`로 진행한다. `labeling.py`와 `configs/stage3/comma2k19_label.yaml`에 "VERIFIED 2026-09-14, 사용자 승인" 근거를 기록해뒀다. 영상 육안 확인은 여전히 못한 상태로 남지만, 이는 사람이 명시적으로 감수하기로 한 리스크다.

### Training gate 최신 상태 — 전부 통과 (STEER_SIGN 이제 영상으로도 직접 확인됨)

```text
[x] src.eval.metrics self-test PASS
[x] Python sync 결과가 real demo 검증값(awk)과 일치
[x] speed/value (N,1) shape 처리 확인
[x] speed / steering / frame timestamp 독립 보간 확인
[x] STEER_SIGN 영상 기반 검증 완료      <- 2026-09-14 Chunk_1 실제 영상(b0c9d2329ad1606b|2018-08-17--14-55-39/2, t=1.2~7.9s)으로 직접 확인: "FWY" 램프에서 실제로 우회전해 완전히 다른 도로로 진입하는 장면을 프레임으로 확인, steer_label=RIGHT와 일치
[x] STEER_SIGN 코드/문서에 근거 기록
[x] Stage 3 label conversion 결과 육안 sanity check 완료  <- 위 영상 확인으로 완전히 종료
[x] train/validation route split 설계 완료  <- 설계+구현+실제 21 route 테스트까지 완료 (src/data/comma2k19/split.py)
```

STEER_SIGN은 이제 (1) 물리적 유도, (2) 64세그먼트 motion-signal 통계(예외 0), (3) **실제 영상 육안 확인** 세 가지가 모두 같은 결론을 지지한다 — `AUTONOMOUS_RESEARCH_AGENT.md` §8.2 기준을 완전히 충족했다. 상세: `src/data/comma2k19/labeling.py`, [output/turn_check/overlay/](output/turn_check/overlay/)의 sample_00012~00075.jpg.

---

## 12. `EXP-S3-BASE-001` 결과 — 첫 Stage 3 baseline (smoke-test)

### Objective
Training gate가 통과됐으니 실제 학습 루프(dataloader → 모델 forward/backward → checkpoint)가 정상 동작하는지 확인한다. 성능이 아니라 **엔지니어링 정합성**이 목표라고 사전에 명시했다.

### Current Bottleneck
영상+CAN 라벨이 모두 있는 세그먼트가 1개(Example_1)뿐 — route 1개, STOPPED/LEFT 샘플 0개.

### Hypothesis
`src/train/{stage3_dataset,stage3_model,train_stage3_baseline}.py`로 구성한 학습 루프가 실제로 돌 것이고, loss가 계산되고 checkpoint가 저장될 것이다. 단, 데이터가 극단적으로 부족하고(1 route, 600 sample) 편향돼(STRAIGHT 98.8%, STOPPED 0%) 있어 **의미 있는 성능은 나오지 않을 것**으로 예상한다 — 이 예상이 맞는지까지 확인하는 것이 이 실험의 진짜 목적이다.

### Single Change
베이스라인 노트북의 `Stage3MViT`(mvit_v2_s, weights=None, dual-head) 구조를 그대로 재사용했다 — 새 실험에서 모델 구조를 바꾸지 않는다는 원칙을 지켰다. 새로 만든 것은 학습 루프/데이터로더뿐.

### Result

```text
$ python -m src.train.train_stage3_baseline --labels output/comma2k19_subset/labels_10hz.csv \
    --video data/external/comma2k19_example/segment/video.hevc --out output/exp_s3_base_001 --epochs 3

epoch 0: train_loss=1.344 val_loss=0.903 val_accel_acc=0.633 val_steer_acc=1.000  (98.2s)
epoch 1: train_loss=1.183 val_loss=1.078 val_accel_acc=0.633 val_steer_acc=1.000  (98.0s)
epoch 2: train_loss=1.186 val_loss=0.903 val_accel_acc=0.633 val_steer_acc=1.000  (98.3s)
peak VRAM: 6.29 GB (RTX 3060 12GB 중)
```

- Primary metric: 위 accel/steer accuracy (단, 아래 self-critique로 무효화됨)
- Runtime: ~98초/epoch (RTX 3060, batch=4, CPU 프레임 캐시 1회 디코딩 포함하면 최초 로딩에 추가 시간)
- VRAM: 6.29GB peak — 12GB 예산에 여유 있음
- Size: mvit_v2_s dual-head, 체크포인트 `output/exp_s3_base_001/smoke_test.pt`

### Baseline Comparison
이전 baseline 없음(첫 모델 실험).

### Self-Critique
- **class collapse 확인**: val_steer_acc=1.0이 나온 게 의심스러워 checkpoint를 다시 불러와 validation set의 실제 confusion을 직접 찍어봤다.
  ```text
  steer  true={STRAIGHT:60}            pred={STRAIGHT:60}   <- val set 자체에 STRAIGHT만 있었고, 모델도 전부 STRAIGHT로만 예측
  accel  true={CONSTANT:38,ACC:16,DEC:6} pred={CONSTANT:60}  <- 모델이 전부 CONSTANT로만 예측 (38/60=0.633, 정확히 관측된 accuracy와 일치)
  ```
  **두 head 모두 완전히 majority-class로 collapse했다.** `val_accel_acc=0.633`과 `val_steer_acc=1.0`은 실제 학습 신호가 아니라 클래스 불균형 때문에 생긴 가짜 지표다.
- **가장 강한 반대 해석**: 이 수치를 "베이스라인 성능"으로 잘못 인용하면 이후 실험의 개선 여부를 완전히 잘못 판단하게 된다 — 반드시 무효 처리한다.
- **이 결과가 가짜 개선일 가능성**: 100% — 위에서 직접 confusion matrix로 확인함. "개선"이라고 부를 지표 자체가 없다(첫 실험이니 비교 대상도 없음).
- **leakage 가능성**: 없음(route가 1개뿐이라 오히려 leakage를 논할 수조차 없음 — train/val이 같은 route라는 것 자체가 이미 알고 있는 제약).
- **runtime/모델 크기 트레이드오프**: 해당 없음(비교 대상 없음).
- **규정 리스크**: 없음.

### Decision
**BLOCKED** (`AUTONOMOUS_RESEARCH_AGENT.md` §9) — 모델/하이퍼파라미터를 더 조정해봐야 소용없다. §17 원칙("3회 연속 개선 없으면 모델보다 데이터가 문제일 가능성을 먼저 검토")을 3회씩 기다릴 필요도 없이 confusion matrix에서 바로 확인됨: **이건 모델 문제가 아니라 데이터 문제**다. 파이프라인 자체는 KEEP(정상 동작 확인됨)이지만, 이 데이터로는 어떤 하이퍼파라미터를 바꿔도 같은 결과가 나올 것이다.

### What We Learned
- 학습 파이프라인(dataset/model/train loop/checkpoint/VRAM 측정)은 전부 정상 동작한다 — `EXP-S3-BASE-001`의 엔지니어링 목표는 달성됐다.
- 예상대로, 1개 route·클래스 불균형 데이터로는 의미 있는 baseline이 나올 수 없다는 것이 확인됐다(가설 확인).
- **더 이상 이 데이터로 모델을 실험하는 것은 기대효용이 0에 가깝다** — `AUTONOMOUS_RESEARCH_AGENT.md` §16 Stop Condition 4("외부 데이터가 추가로 필요함")에 해당한다.

### Next Highest-Value Experiment
STOPPED/LEFT 샘플이 실제로 포함된 다른 route의 **영상**이 필요하다. 현재 갖고 있는 라벨(64 세그먼트/21 route, §10)에는 라벨은 있지만 영상이 없고, 영상을 구하려면 `raw_data/Chunk_*.zip`(각 ~9GB) 중 하나를 받아야 한다 — 이건 오늘의 "선택지 A" 승인과는 다른, **새로운 비용 결정**이라 별도로 사람에게 확인이 필요하다(§13 참고).

---

## 13. Stop — 다음 결정이 필요한 지점

`AUTONOMOUS_RESEARCH_AGENT.md` §16 Stop Condition에 해당해 여기서 자율 루프를 멈춘다: **"외부 데이터가 추가로 필요함."**

### 지금까지 확인된 것
- 학습 파이프라인 자체는 문제없이 동작한다(§12).
- 지금 있는 데이터(1 route)로는 어떤 모델/하이퍼파라미터를 써도 majority-class collapse 이상으로 나아갈 수 없다.
- 64-segment parquet mirror(§10)로 route 다양성과 STEER_SIGN 검증은 이미 충분히 확보했지만, **영상이 없어 실제 학습에는 못 쓴다.**

### 선택지
- **A. Chunk 1개(~9GB, MIT) 다운로드** — `raw_data/Chunk_1.zip` 등. 여러 route의 실제 영상+라벨을 동시에 얻어 진짜 baseline 학습이 가능해진다. 대신 9GB 다운로드+디스크(현재 293GB 중 여유 충분)+압축해제 시간이 든다.
- **B. 여기서 멈추고 다른 작업(Stage 1/2)으로 전환** — Stage 3 영상 확보를 잠시 미루고 가이드 §23 Priority 2(Stage 2 시점 검출)나 Priority 4(Stage 1 baseline) 쪽으로 자원을 돌린다.
- **C. 더 작은 단위로 시도** — 청크 zip의 중앙 디렉터리를 HTTP Range 요청으로 읽어 특정 segment(예: §10에서 찾은 `2018-08-17--14-55-39/2`, 258도 회전) 하나만 부분 추출 — 시도 가치는 있지만 어느 chunk에 있는지 매핑을 못 찾으면 결국 A로 귀결될 수 있다(이전 턴에서 시도해보지 않고 보류한 항목).

### 추천안
A(Chunk 1개 다운로드)를 권장한다 — 지금 데이터로는 어떤 추가 실험도 기대효용이 0이라는 게 이미 confusion matrix로 확인됐고, C는 성공 여부가 불확실한데 반해 A는 확실하게 문제를 해결한다. 다만 9GB는 이번 세션에서 다뤄온 다른 다운로드보다 훨씬 크므로, 진행 여부는 사람이 결정하는 것이 맞다.

---

## 14. Stage 3 실제 학습 결과 요약 + 첫 LB 결과 (2026-09-15 ~ 09-17)

§13 이후 사용자가 Chunk_1(8.7GB) 다운로드를 승인해 진행한 내용의 요약. 실험별 상세는 [experiments/experiment_log.csv](experiments/experiment_log.csv), 제출 전략은 [SUBMISSION_STRATEGY.md](SUBMISSION_STRATEGY.md).

| 실험 | 변경 | val_stage3_score(공식 채점식) | 판정 |
|---|---|---|---|
| EXP-S3-BASE-002 | MViT scratch, lr 1e-4, Chunk_1 47seg/13route | steer 3 epoch 내내 100% STRAIGHT, accel 0.682→0.611 퇴행 | DROP |
| EXP-S3-BASE-003 | lr만 2e-5 | 동일 패턴 — lr 문제가 아님 | DROP |
| EXP-S3-BASE-004/005 | **아키텍처 교체**: ImageNet ResNet18(frozen)+MLP head(13만 파라미터), best-by-macro-F1 저장 | **0.518** (epoch 12; accel_f1 0.571, steer_f1 0.394) | KEEP |

교훈 두 가지: (1) 4000 sample이 실제론 47 segment/13 route라 대형 video transformer를 scratch로 학습하기엔 데이터가 부족했다 — 모델 용량을 데이터 규모에 맞추는 게 답이었다. (2) **accuracy는 majority-class collapse를 못 걸러낸다** — steer 100% STRAIGHT가 accuracy 0.658인데 macro-F1은 0.26. 그 뒤로 학습 루프가 매 epoch `src/eval/metrics.py`의 공식 채점식을 계산하고 best checkpoint를 그 기준으로 저장한다.

### 첫 LB 결과 (2026-09-16 제출, 09-17 확인)

| 제출 | Stage1 | Stage2 | Stage3 | Total(0.2/0.4/0.4) | 서버 시간 |
|---|---|---|---|---|---|
| #91244 `submit.zip` (공식 baseline) | 0.40456 | 0.13129 | 0.10721 | 0.1763 | 14분 19초 |
| #91246 `submit_v2.zip` (Stage3만 교체) | 0.40456 | 0.13129 | **0.37034** | **0.2816** | 11분 46초 |
| 1등 | 0.95145 | 0.54346 | 0.75456 | 0.7095 | — |

- Stage1/2 점수가 소수점 10자리까지 동일 → LB 열 순서가 Stage1/2/3임을 확인. 현재 231등.
- Stage3 0.107→0.370(3.45배)은 로컬 official-label 비교(0.140→0.409)와 방향·규모가 일치한다 → **로컬 검증 파이프라인이 LB를 잘 예측한다.**
- Stage1 0.4046은 "전부 ORIGINAL 예측" 시 macro-F1 `p/(1+p)`와 정확히 맞는다(역산하면 테스트셋 ORIGINAL 비율 ≈ 68%). 즉 Stage1 baseline은 degenerate.
- 1등 대비 가중 갭: **S1 0.109 / S2 0.165 / S3 0.154.** 갭 자체는 S2가 최대지만 S2는 라벨 없는 3개 항목이 65%라 비용이 가장 크다. S1은 degenerate 상태에서 출발하고 외부 데이터 없이 자체 합성이 가능해 **기대효용/비용이 가장 좋다** → 다음 실험은 Stage1.
- 런타임 12~14분/60분 → 무거운 모델을 써도 여유가 크다.

### 부수 발견 — 공식 Stage3 샘플은 comma2k19 계열 영상이다

`data/stage3/videos/OPEN_00*.mp4`는 **1164×874, 1200프레임(60초@20fps)** — comma2k19 EON 카메라 규격과 정확히 같다. comma2k19 학습 모델이 official Stage3 라벨/LB에서 잘 일반화된 이유가 이것이다. 비공개 Stage3 평가셋도 같은 계열일 가능성이 높으므로, **comma2k19 데이터를 더 쓰는 것(Chunk 추가, 전체 세그먼트 사용, backbone unfreeze)이 Stage3에서 가장 직접적인 다음 이득**이다.

### 공식 Stage1 RERECORDED 샘플 실측

`data/stage1/rerecorded/*.mp4` 5건: 해상도(1280×720)·fps(10)·프레임수(50)·프레이밍 모두 원본과 동일. 차이는 **강한 재압축(블록 노이즈)뿐** — 라플라시안 분산이 원본 대비 2~3배(예: 000005: 110.7→247.4), 파일 크기 ~2배. 즉 DACON의 공개 예제는 "재촬영 시뮬레이션 = 저품질 재인코딩"이다. 비공개 평가셋은 실제 재촬영(베젤·모아레·반사광·원근·흔들림)을 포함할 수 있어(가이드 §2.2~2.3), 합성 파이프라인은 둘 다 커버하도록 설계했다(§15).

---

## 15. `EXP-S1-SYNTH-001` — 합성 재녹화로 Stage1 학습

### Objective / Bottleneck
Stage1 baseline이 degenerate(all-ORIGINAL, LB 0.4046). 가중치 0.2라도 0.40→0.8이면 total +0.08 — 현재 가장 싼 큰 이득.

### Hypothesis
재녹화 흔적(재압축 블록, 모아레/스캔라인, 색·감마 변화, 반사광, 베젤, 원근, 흔들림, 해상도 저하)은 장면 내용과 무관한 **텍스처 단서**다. 같은 원본 프레임을 (원본, 합성 재녹화) 쌍으로 만들어 학습하면 모델은 내용이 아니라 흔적만으로 구분하도록 강제되고, 이는 실제 재촬영에도 일반화될 것이다.

### Single Change
Stage1만 교체. 데이터: comma2k19 59세그먼트×10프레임 + OPEN 5개×20프레임(ORIGINAL 소스, 무손실 PNG). RERECORDED는 `src/data/stage1/synth_rerecord.py`로 즉석 합성(영상 단위 파라미터 고정, 프레임마다 흔들림·노이즈·JPEG만 변화). 모델: ImageNet ResNet18 전체 fine-tune, fc→1 logit BCE, 원본 해상도 224 랜덤 크롭(고주파 단서 보존) + 양 클래스 공통 랜덤 스케일 0.6~1.2.
검증(그룹 단위): (a) 공식 5쌍(실제 ORIGINAL + DACON식 RERECORDED) 영상 단위 macro-F1, (b) comma2k19 route 3개 hold-out(원본 실제 + 합성 재녹화). 채점 단위와 같은 영상 단위(프레임 5크롭 평균→프레임 평균→0.5). best = (a)+(b) 평균.
추론(`src/train/predict_stage1_synth.py`): 영상당 균등 8프레임×5크롭=40장, 원본 해상도, sigmoid 평균.

### Result — 001/002 실패, 원인 실측으로 규명

| 실험 | 변경 | synthetic hold-out F1 | 공식 5쌍 F1 | 판정 |
|---|---|---|---|---|
| 001 | 위 설계 그대로 | 1.0 | **0.333** (10건 전부 RERECORDED 예측) | DROP |
| 002 | 양 클래스 공통 '기저 열화'(JPEG 45~85) 추가 | 1.0 | **0.231** (원본 prob 0.73~0.97 > 재녹화 0.39~0.87 — 역전) | DROP |

합성 과제는 자명하게 풀리는데(train loss 0.003) 공식 원본이 공식 재녹화본보다 **더 재녹화답게** 나온다. 그래서 공식 5쌍의 변환 자체를 측정했다(프레임 중앙, 000001/3/5):

| 지표 | 원본 | DACON 재녹화 | 해석 |
|---|---|---|---|
| phaseCorrelate 이동 | — | (0.0, 0.0), 응답 1.05~1.14 | **정렬 이동·스케일 없음** → 베젤/원근/흔들림 없음 |
| blockiness(8px 격자 경계 대비) | 2.5~3.7 | 2.0~3.0 (**−25%**) | 기존 블록이 **부드러워짐** — 원본이 이미 유튜브 재인코딩급으로 블록이 심함 |
| 라플라시안 분산 | 76/122/111 | 166/212/247 (**×2**) | 미세 그레인 노이즈 추가 |
| 고주파 에너지 비율 | 0.44~0.62 | 0.53~0.71 (**+15%**) | 그레인이 인코딩 후에도 살아 있음 |
| 밝기 / 대비(std) | — | −1~−3 / +2~+4 | 화면 재촬영 특유의 감마·대비 변화 |
| 프레임 간 차이 | 5.6~9.4 | +0.5~0.9 | 프레임마다 다른 노이즈(플리커) |

즉 DACON식 재녹화 = **블러(블록 완화) + 그레인 + 대비 소폭 증가 + 가벼운 재인코딩**이고, v1/v2 합성(강한 JPEG 블록·다운스케일·큰 색 변화)은 정확히 반대 방향이었다 — 모델은 "블록이 심하고 날카로움 = 재녹화"를 배웠고, 원래 블록이 심한 공식 원본이 그쪽으로 분류됐다. 기저 열화(002)만으로는 이 방향성이 안 바뀐다.

**v3 합성(EXP-S1-SYNTH-003/004)**: 원본 클래스 기저 = 블록 심한 JPEG q18~50(테스트 원본 수준). 재녹화 핵심(항상) = 블러 0.3~0.8 → 확률 0.5로 센서 격자 재샘플링(미세 스케일+서브픽셀 이동, 블록 경계 뭉개기) → 대비 1.0~1.12·감마 0.95~1.08·밝기 −5~+1 → 고품질 재인코딩(q75~95) → **인코딩 뒤** 그레인 σ2~6. 실제 재촬영 흔적(모아레·반사광·베젤·원근·스캔라인·강한 재압축)은 0.15~0.25 확률의 선택 항목으로 유지해 비공개셋이 실제 재촬영일 가능성도 커버한다. 합성 후 위 지표가 실측과 같은 방향·규모로 움직이는지 확인한 뒤 학습한다.

### EXP-S1-SYNTH-003/004 — 방향은 잡혔지만 실제 흔적을 재현 못함

| 실험 | 합성 | 공식 5쌍 |
|---|---|---|
| 003 (v3.0: 블러 0.4~1.0, 백색 그레인 σ5~14, 재샘플링 항상) | synth F1 0.85~0.95 | F1 0.333(전부 ORIGINAL)이지만 **쌍 내 순위는 재녹화>원본으로 반전**(4번 0.34 vs 0.12) |
| 004 (v3.2: 블러 0.1~0.35, 상관 그레인 σ3~8 사후, 재샘플링 p0.35) | synth F1 1.0 | AUC 0.56~0.80, 재녹화−원본 차이 +0.03 수준, 확률 전부 0.5 미만 |

합성 통계를 실측에 맞춰도(라플라시안·밝기는 일치, 블록 경계는 여전히 과소) 모델은 DACON 실제 흔적을 거의 못 본다 — 합성 그레인과 실제 코덱 거친 그레인이 질적으로 다르거나, 단일 프레임 회색조 통계로 못 잡는 요소(크로마 서브샘플링/코덱 특성)가 있는 것으로 판단. **추측 반복을 멈추고 실제 DACON 처리 샘플(공식 5쌍)을 학습에 넣기로 결정.**

### EXP-S1-SYNTH-005 — 공식 5쌍 포함 + leave-one-pair-out — **KEEP**

- 학습: 합성 소스(위와 동일) + 공식 5쌍 전체 프레임(50/영상; 원본→(원본, 합성 재녹화), 실제 재녹화→(재녹화)), 공식 프레임 ×2 오버샘플링, 공식 프레임엔 기저 열화 30%만.
- 검증: 쌍 단위 leave-one-pair-out 5-fold — 각 fold는 해당 쌍(원본+재녹화 영상 2개)을 한 번도 보지 않은 모델로 예측.

```text
LOPO out-of-fold (10 영상):  macro-F1 = 1.0,  AUC = 1.0
  원본   000001 0.008 | 000002 0.061 | 000003 0.044 | 000004 0.488 | 000005 0.099
  재녹화 000001 0.724 | 000002 0.894 | 000003 0.917 | 000004 0.982 | 000005 0.936
```

4쌍만 보고도 나머지 1쌍을 전부 맞춘다 → DACON 처리 시그니처는 소스 영상 내용과 무관하게 일반화된다. 최종 모델은 5쌍 전부+합성으로 3 epoch 학습(`output/exp_s1_synth_005/best.pt`), 임계값 0.5 유지(OOF 최적 0.606도 F1 동일).

### Self-Critique
- **N=10 영상, 5쌍**: 통계적으로 작다. 다만 각 fold가 진짜 held-out 소스라 leakage는 없다(같은 원본에서 나온 두 영상은 항상 같은 fold).
- **비공개셋의 재녹화 방식**: 공개 예제와 같은 처리라면 LB Stage1 0.9+ 기대. 실제 재촬영(베젤·모아레·반사광)이 섞여 있으면 합성 브랜치(저확률 선택 흔적)에 의존 → 0.5~0.8 사이로 나올 수 있다. **LB 결과 자체가 이 질문의 답이 된다.**
- **4번 원본 0.488**: 임계값 0.5와 여유가 작다. 테스트 원본 중 이런 경계 사례가 있으면 FP 발생. 테스트 원본 비율이 ~68%라 원본 오분류가 macro-F1을 더 깎는다.
- **규정**: 공개 라벨 데이터로 학습/보정만 했고 평가 영상 간 통계 공유 없음. 추론은 영상마다 독립적으로 고정 임계값 비교.

### 제출 후보 3
`build_submission_v3.py` → `submit_v3.zip`(127MB): Stage1 = 005 모델, Stage2 = baseline, Stage3 = 제출 2와 동일. Stage2/3 smoke-test 출력 CSV가 제출 2와 바이트 동일(`cmp`)임을 확인 → LB 델타는 순수 Stage1 변경분.

### LB 결과 — submit_v3: Stage1 0.4008 (baseline 0.4046) → **비공개 재녹화는 공개 예제와 다른 종류**

- Stage2/3는 예상대로 소수점 10자리까지 동일. Stage1만 0.4046→0.4008로 **소폭 하락**.
- macro-F1 역산(원본 비율 p≈0.68): 전부 ORIGINAL이면 0.4046. 0.4008은 "약 1%를 RERECORDED로 찍었는데 거의 전부 오답(TP≈0)"에 해당 → 모델이 비공개 RERECORDED 영상을 **하나도 못 잡았다.**
- 같은 모델이 공개 5쌍은 LOPO 10/10이었으므로, 비공개 재녹화는 공개 예제(재압축 파생본)와 처리가 다르다. 가이드 §2.2가 "비공개 데이터는 공개 예시와 다를 수 있다"고 했고 §2.3이 열거한 특징(화면 테두리·주사 패턴·반사광·원근·기기 움직임·해상도 변화)이 곧 실제 화면 재촬영의 모습이다. **비공개 RERECORDED = 카메라로 화면을 찍은 영상**이 가장 그럴듯한 가설.
- 1등의 Stage1 0.951은 그 신호가 강하게 존재함을 뜻한다.

### EXP-S1-SYNTH-006 — v4 '실제 화면 재촬영' 합성 (프로브)

가설: 비공개 RERECORDED는 실제 재촬영이며, 그 외형(화면이 프레임의 55~95%를 차지 + 베젤/배경 + 모아레 2중 격자 + 주사 밴딩 + 원근 2~8% + 손떨림 + 카메라 색온도/감마/비네팅/반사광 + 초점 블러 + 센서 노이즈)을 강하게 합성해 학습하면 잡을 수 있다. 공개 예제식(미세) 합성과 50:50으로 섞고 공식 5쌍도 계속 포함해 두 종류를 모두 커버한다. 추론 크롭에 모서리 4개를 추가(9크롭)해 베젤/배경 영역이 보이게 했다. **LB Stage1이 0.4→0.6+ 로 오르면 가설 확인, 그대로면 다른 종류(예: 다른 코덱 재인코딩)로 재해석.**

---

### LB 결과 — submit_v4: Stage1 0.4237 (baseline 0.4046, v3 0.4008) → 재촬영 가설 '약하게만' 확인, Stage1 보류

- 역산: all-ORIGINAL macro-F1 0.4046 ⇒ 비공개 Stage1의 ORIGINAL 비율 ≈ 0.68. v3의 0.4008은 RERECORDED 재현율 0에 FP율 ≈ 1.6%와 일치(공개 5쌍을 완벽히 잡는 모델이 비공개에선 오탐만 냄). v4의 0.4237은 FP 0%→재현율 1.8% / FP 1%→재현율 2.1% / FP 2%→재현율 2.4% / FP 5%→재현율 3.3% 범위 — **비공개 RERECORDED의 2~5%만 검출**.
- 해석: v4가 v3보다 나은 것은 방향이 아주 틀리진 않았다는 뜻이지만, 대부분의 비공개 재녹화는 v4의 '화면 속 화면(베젤·모아레·원근)' 합성으로도 잡히지 않는다. 남는 후보: (a) 전체화면 재촬영이라 공간 흔적이 미세하고 **시간축 흔적**(프레임 중복/블렌딩, 주사율 비트, 밝기 깜빡임)이 주 단서, (b) 다른 코덱/해상도/fps로의 재인코딩(공개 예제와 종류는 같되 강도가 달라 threshold가 안 맞음). 둘 다 비공개 출력 분포를 볼 수 없어 1슬롯 프로브가 필요.
- 결정: 해석 규칙(0.6 미만 → 보류)대로 **Stage1 보류**, v4 모델은 LB 최고이므로 제출의 Stage1 자리에 유지. 다음 제출은 `submit_v5b.zip`(Stage1 v4 + Stage3 CLASS-001).

## 16. Stage 3 확장 — Chunk_1+2, 291세그먼트/42route, 특징 캐시 기반 head 학습

### 데이터·파이프라인
- Chunk_2(9.05GB) 추가 다운로드 후 Chunk_1·2에서 route당 최대 8세그먼트 추출 → **291세그먼트/42route, 174,074 10Hz 샘플**(005 대비 5배). 클래스 비율: accel CONSTANT 62/ACC 15/DEC 15/STOP 8%, steer STRAIGHT 63/RIGHT 21/LEFT 16%.
- 이 규모에선 프레임 캐시(세그먼트당 180MB→52GB)가 불가능. backbone이 frozen이므로 **프레임별 ResNet18 특징을 한 번만 추출**(`stage3_features.py`, float16 715MB, 35분)하고 head만 학습(`train_stage3_head.py`). 클립 특징 = 16프레임 특징 평균이라 기존 `Stage3ResNetHead`와 수학적으로 동일하고, export 포맷도 같아 추론 snippet을 바꾸지 않는다. head 학습 40 epoch ≈ 1분 → 실험 회전이 초 단위.
- val = 63세그먼트/8route(37,771 샘플) — 005의 val(12seg/3route)보다 5배 커서 점수 신뢰도가 높다.

### 실험 (단일 변경 원칙)

| 실험 | 단일 변경 | val_stage3_score | accel_f1 | steer_f1 | 판정 |
|---|---|---|---|---|---|
| EXP-S3-BASE-006 | 데이터 5배 (005 레시피 그대로) | 0.550 (epoch1) / 후반 0.536 | 0.576→0.52 | 0.49→0.57 | KEEP(파이프라인) — accel이 CONSTANT 과다 예측 |
| EXP-S3-CLASS-001 | CE에 클래스 빈도 역수 가중 | **0.566** (epoch13), 마지막 0.562 | 0.559 | 0.583 | **KEEP** — 예측 분포가 실제 분포에 근접 |
| EXP-S3-TEMP-001 | + 앞8/뒤8 반윈도 평균 concat(1024-d) | 0.559 | 0.566 | 0.543 | DROP — accel +0.007, steer −0.04 |

### Self-Critique
- 005(0.518, 12seg val)와 006(0.550, 63seg val)은 val이 달라 직접 비교가 아니다. 같은 val 안에서의 비교(006→CLASS-001 +0.016)만 개선으로 셈한다.
- class weight 효과는 macro-F1이 소수 클래스에 민감한 데서 나온 것으로, 정확도 자체가 오른 것은 아니다 — 대회 지표가 macro-F1이므로 정당한 최적화.
- 남은 병목: accel_f1 0.56 — 프레임 평균 특징은 속도 변화를 담기 어렵다. TEMP-001의 실패는 "순서 정보를 주는 방식"(concat)의 문제일 수 있어 차이 벡터만 추가하는 좁은 변형이 남아 있다. 근본적으로는 optical-flow/ego-motion 특징(가이드 §9)이 필요.
- 최종 확인은 공식 Stage3 라벨(50행) + LB로 한다(§17 제출 후보 5).

### 공식 Stage3 라벨 평가 — 정렬 버그 수정 후

로컬 official-label 평가에 버그가 있었다: 공개 OPEN 영상은 20fps라 라벨의 `frame_index = 2 × sample_index`인데, 예측은 디코딩 프레임마다 1행(`sample_index` = 프레임 번호)이라 `sample_index`로 merge하면 **라벨과 2배 어긋난 프레임을 비교**하고 있었다(비공개는 10fps라 이 문제가 없음). `frame_index`로 merge하도록 고친 뒤:

| 모델 | 잘못된 정렬 | **올바른 정렬** | accel_f1 | steer_f1 |
|---|---|---|---|---|
| baseline | 0.140 | 0.140 | 0.076 | 0.289 |
| candidate2 (BASE-005) | 0.409 | **0.584** | 0.687 | 0.343 |
| CLASS-001 | 0.360 | **0.743** | 0.770 | 0.679 |

잘못된 정렬로는 "CLASS-001이 official에서 더 나쁘다"는 거짓 결론이 나올 뻔했다. 수정 후엔 내부 val(0.518→0.566)과 같은 방향이다. candidate2의 로컬 0.584 vs LB 0.370 갭은 N=50 노이즈 + 비공개셋 라벨 정의/도메인 차이로 본다.

**윈도우 시간 불일치 점검**: 비공개는 10fps라 16프레임 윈도우가 학습(20fps, 0.8초)과 달리 1.6초가 된다. 캐시된 OPEN 특징에서 10fps를 시뮬레이션(프레임 하나 걸러)해 윈도우 16/12/8/6을 비교 → CLASS-001 0.756/0.746/0.737/0.760, 차이가 노이즈 수준이고 16이 불리하지 않음 → **추론 코드 변경 없음**(제출 5는 Stage3 모델만 바뀐 단일 변경).

### 제출 후보 5
- `submit_v5a.zip`: Stage1 baseline + Stage2 baseline + **Stage3 CLASS-001** — v4가 실패(S1≈0.40)했을 때 올린다. 직전 유효 제출(v2) 대비 Stage3만 변경.
- `submit_v5b.zip`: Stage1 v4 모델 + Stage2 baseline + **Stage3 CLASS-001** — v4가 성공(S1≥0.6)했을 때 올린다. v4 대비 Stage3만 변경.
- 기대: LB Stage3 0.370 → 0.45~0.55 (로컬 official-label 개선 비율 0.584→0.743 ≈ +27%를 보수적으로 적용).

## 17. `EXP-S3-MOTION-001` — optical-flow ego-motion 특징 (Stage3 accel 병목 해소)

### Objective / Bottleneck
Stage3 accel_f1이 내부 val 0.56에서 정체(CLASS-001). frozen ImageNet ResNet18 프레임 특징의 평균은 '장면이 무엇인가'만 담고 '카메라가 어떻게 움직이는가'는 거의 담지 못한다 — 가감속/조향은 본질적으로 ego-motion.

### Hypothesis / Single Change
Farneback dense optical flow(160×120 gray, stride 1)에서 프레임당 13개 요약 특징(흐름 크기·수평/수직 평균·방사 팽창률(div)·좌/우·상/하 분해·정지 비율·프레임 차)을 뽑아 16프레임 윈도우의 mean/std/delta(39-d)를 head 입력에 concat. 대조군 = 같은 split/seed에 `--fps-aug`만 추가(FPSAUG-001).

### 사전 분석 (학습 전 특징 검증, `src/train/stage3_motion.py`)
- 83세그먼트 48,811샘플: u_mean vs 조향각(+=LEFT) 상관 +0.28 (STEER_SIGN과 부호 일치), div vs 속도 +0.28, 정지 구간 흐름 크기 중앙값 0.066 vs 주행 0.58.
- 그러나 16프레임 delta vs CAN 가속도 상관 0.04–0.06 — ±0.3 m/s²는 0.8s에 속도 1.6% 변화라 짧은 윈도우에선 보이지 않음.
- 깊이 스케일에 불변인 log-ratio(뒤 H프레임 평균 / 앞 H프레임 평균)로 바꾸면 div 기준 H=16 0.15 → 32 0.24 → 64 0.34(3.2s) → 128 이상에서도 상승. **가속 정보는 긴 지평에 있다** → 지평 16/32/64/128의 log-ratio(mag_mean/div/mag_bottom/mag_top)와 static_frac 차이 20-d를 추가(59-d).

### fps 문제
공개 OPEN/comma2k19는 20fps, 비공개 Stage3는 10fps인데 컨테이너 fps(CAP_PROP_FPS=479.78)는 못 믿는다. 흐름 크기는 fps에 반비례하므로 학습을 두 fps에 노출: 캐시에 stride-2 흐름도 저장하고, 짝수 프레임+stride-2로 10fps를 시뮬레이션한 복제본을 학습에 추가(`--fps-aug`), val은 native/sim10 둘 다 보고하고 평균으로 best 선택.

### Result (내부 val 63seg/8route 37,771샘플 | official OPEN 50행)

| 실험 | 입력 | val native | val sim10 | official 20fps | official 10fps-sim |
|---|---|---|---|---|---|
| CLASS-001 (제출5) | 512 | 0.566 | – | 0.743 | 0.756 |
| FPSAUG-001 (대조군) | 512 | 0.571 | 0.584 | 0.766 | 0.735 |
| MOTION-001a | 512+39 | 0.630 | 0.645 | 0.793 | 0.765 |
| **MOTION-001b** | **512+59** | **0.655** (accel 0.667, steer 0.626) | **0.665** | 0.768 | 0.765 |
| MOTION-001c (motion만) | 59 | 0.588 | 0.599 | 0.725 | 0.673 |

### Self-Critique
- 내부 val +0.084는 8route 37k샘플이라 신뢰할 만하고 native/sim10에서 일관. official(N=50)은 1행=0.02라 001a/001b를 구분 못 하며, 20fps-native에선 001a가 더 높게 나왔다(노이즈로 판단, 001b 선택 근거는 내부 val).
- 흐름 특징은 해상도/코덱/노출에 민감할 수 있다. 공식 Stage3 샘플이 comma2k19 규격(1164×874)임을 확인했으므로 비공개도 같을 가능성이 높지만, 다른 카메라라면 표준화(학습셋 mean/std)가 어긋날 수 있음 — LB로 확인.
- motion만(001c)은 steer 0.50으로 급락 → appearance가 조향의 맥락(차선/도로 형상)을 제공. 둘 다 필요.
- 추론 시간 8.0s/영상(1200프레임; flow는 CPU 단일 스레드). 비공개 영상 수를 모르지만 기존 13분 예산에 여유가 크다.

### Decision
MOTION-001b 채택 → **제출 후보 6**(`submit_v6.zip`, Stage1 v4·Stage2 baseline 유지, Stage3만 변경). snippet(`src/train/predict_stage3_motion.py`)이 캐시 경로와 5,992행 100% 일치함을 확인.

### 후속 (같은 날) — 지평 확장 DROP, 시간축 smoothing KEEP
- `MOTION-002a`(지평 +256) select 0.6597, `MOTION-002b`(32~512) 0.6596 vs 001b 0.6599 → 지평 128 이상은 정보 없음, 001b 유지.
- 진단(001b 내부 val 혼동행렬): STOP F1 0.94, CON 0.80, ACC 0.46, DEC 0.48 — ACC/DEC↔CON 경계(±0.3 m/s²) 혼동이 손실의 대부분, ACC↔DEC 부호 혼동은 적음. steer는 LEFT 0.52 / STR 0.78 / RIGHT 0.59로 LEFT↔STR 혼동. 세그먼트별 accel macro-F1 분산이 큼(0.21~0.84).
- `SMOOTH-001`: 라벨 구간이 수 초 단위인데 예측은 프레임별로 flicker → 영상 내에서 logit을 프레임 축 이동평균 후 argmax(파일 간 통계 없음). 내부 val(10Hz 행 폭) w=1 0.655/0.665(native/sim10) → w=21 0.672/0.675 → w=31 0.676/0.677 → w=41 0.678/0.677 → w=61 0.675/0.676: 2~4초 plateau. 추론 폭 31프레임(10fps 3.1s, 20fps 1.55s — 둘 다 plateau 안). official: native 0.768(불변), 10fps-sim 0.765 → **0.794**(accel 0.833).
- 제출 후보 6 = MOTION-001b + smooth31 로 재빌드(snippet–캐시 경로 5,992행 100% 일치 재확인, 9.2s/영상).

### Next
- ACC/DEC↔CON 경계: 프레임별 head 대신 특징 시퀀스 위 1D temporal conv(수 초 문맥을 학습으로 통합) — 캐시 특징으로 수 분 실험. 방향별 흐름 히스토그램, 로그 스케일 입력은 후순위.
- 비공개 10fps에서 1200프레임=120s? 영상 길이가 다르면 지평 프레임 수의 시간 의미가 달라진다 — LB 결과와 함께 재검토.

## 18. 1등 갭 재분석 (2026-09-17 밤) 과 `EXP-S1-META-001` / `EXP-S2-HEUR-001`

### 갭 (1등 S1 0.9699 / S2 0.64 / S3 0.7823 vs 우리 0.4237 / 0.1313 / 0.3703, 234등)
가중 갭 **S2 0.203 > S3 0.165(v5b/v6 대기) > S1 0.109**. 세 Stage 모두 "무엇이 다른가"를 실측으로 좁혔다.

### Stage1 — 컨테이너 서명 (`EXP-S1-META-001`, 제출 7)
- 실측: 공개 5쌍·샘플 10영상 모두 ORIGINAL = `FMP4(mp4v)/Lavf58.12.100` ~0.5Mbps, RERECORDED = `h264(avc1)/Lavf63.1.100` ~1.3Mbps. 20/20.
- 해석: 재녹화는 재촬영이든 재인코딩이든 반드시 새 인코딩을 거치므로 코덱/먹서 서명이 남는다. 픽셀 모델(v3 완벽 → 비공개 0)과 달리 파이프라인 수준 단서라 비공개에도 남아 있을 가능성이 있고, 1등 S1 0.97이 이것으로 설명된다.
- 규칙: `mp4v` → ORIGINAL, `avc1/hev1/...` → RERECORDED, 애매하면 v4 CNN. 파일별 자기 헤더만 읽음(§12.2 준수). **리스크**: (a) 비공개가 일괄 재먹싱돼 서명이 균일하면 전부 한 클래스(S1 0.24 또는 0.40), (b) 메타데이터 의존을 2차 평가에서 어떻게 볼지 — 가이드 §2.3 "추가 압축 및 화질 저하/해상도·fps 변화"와 같은 계열로 설명 가능하지만 명시적으로 문서화해 둔다.

### Stage2 — 학습 없는 물리 휴리스틱 (`EXP-S2-HEUR-001`, 제출 8)
- 데이터 실측: 비공개는 프레임 이미지 폴더(파일명 = 원본 프레임 번호), 공개 라벨은 t_collision 5개뿐(entry/evasion/side = -1). 베이스라인은 5클립으로 학습한 GRU라 사실상 랜덤(LB 0.131).
- collision: 접촉 순간 카메라 피칭 → phase-correlation 수직 이동량의 robust z(영상 내 중앙값/MAD). 최대 jolt에 앵커하고 앞쪽 5프레임 내 z>5 jolt로 옮겨 가며 시작점을 찾음. **공개 5/5 (±3프레임=±0.3s@10fps)**; 예측-라벨 차 −1,−1,+1,+1,+2.
- entry: 몽타주 판독상 진입 후 0.2~0.6s 안에 접촉하는 사례가 많아 `collision − 5프레임`(10fps 가정; 비공개 fps 불명이 리스크).
- evasion_space: 접근 구간 [c−12, c−2) 변화 에너지의 중앙(가로 30~70%) 집중도 — 추돌(앞차 급정거) 0.92~0.93 vs 측면 진입 0.52~0.71 → 임계 0.8, 검출 c ±1프레임에 안정.
- entry_side: 절대 무게중심은 카메라/도로 기하 편향으로 전부 RIGHT, 에고모션 보정 잔여 에너지 무게중심은 전부 0.5(무정보). 상대 좌측비율(접근 구간 vs 영상 전체)은 수동 판독과 2/4 일치 — 코인플립 수준이지만 두 클래스를 모두 내므로 macro-F1 기대값은 상수 예측(≈0.33)보다 높음(≈0.45). **약한 부분으로 명시**; 라벨 있는 외부 사고 데이터셋(entry side)이 있어야 개선 가능.
- 기대: collision 50~70% × 0.35 + entry 15~30% × 0.35 + evasion/side F1 0.4~0.5 × 0.3 ≈ **S2 0.30~0.45** (현재 0.13).

### 다음
- LB로 v7(S1), v8(S2) 확인 → 채택분을 하나로 합친 v9. Stage2 외부 데이터셋(사고 시점/진입 방향 라벨) 라이선스 조사 재개.

## 19. Stage3 역상관의 원인 규명 — OPEN 원본 세그먼트 발견과 주최측 라벨 정의 역산 (2026-09-18)

### 발견
- 공개 OPEN_001~005는 comma2k19 **Chunk_1 route `b0c9d2329ad1606b|2018-07-27--06-03-57`의 세그먼트 10/11/3/4/5**(첫 프레임 픽셀차 0.0). 이 route는 291세그먼트 학습셋의 TRAIN에 들어 있었다 → §16~17의 "official-label" 점수는 학습 데이터 평가였고, LB와의 역상관은 여기서 온다. candidate2(chunk1 앞 59세그먼트)도 같은 route를 포함했을 가능성이 높다.
- 세그먼트의 CAN으로 공식 라벨 40행을 대조: accel 임계 **0.33~0.35 m/s²**(우리 0.3), steer deadzone **4~5°**(우리 3°), stopped < 0.47 m/s. 우리 라벨은 소수 클래스를 더 자주 붙여 모델이 ACC/DEC/LEFT/RIGHT를 과예측(OPEN 전체 프레임 분포: STRAIGHT 0.52~0.66 vs 공식 0.78).
- LB 순서(candidate2 0.370 > motion+cw 0.355 > CLASS-001+cw 0.225)는 "소수 클래스 과예측이 적은 순서"와 일치.

### 조치
1. 라벨 v2: stopped 0.3 / accel ±0.35 / deadzone 4.5° (`configs/stage3/comma2k19_label_v2.yaml`) — 40행 97.5%/100% 일치.
2. 학습에서 OPEN route를 **완전 제외**(`--holdout-route`) → official 50행 + 원본 CAN으로 재구성한 dense 라벨이 진짜 held-out 검증이 된다.
3. class weight 제거(MOTION-003) + 공개 라벨 비율로 prior 보정(PRIOR-001, |bias|≤1) + smoothing → 제출 후보 9.
4. 비공개가 다른 차량(Civic, Chunk_3+)일 가능성 대비 청크 추가 다운로드 검토.

## 20. 두 번째 차량(Civic, Chunk_3) 추가와 차량 도메인 시프트 측정 (2026-09-18)

- Chunk_3(9.41GB, HF `raw_data/Chunk_3.zip`, MIT) = dongle `99c94dc769b5d96e`(Civic) 202세그먼트/21route. 우선 12route/115세그먼트 추출·캐시(라벨 v3 = RAV4 291 + Civic 115), 나머지 9route는 추출 중(라벨 v4).
- **Cross-car**: RAV4 전용 MOTION-004를 Civic 115세그먼트에 평가 → native 0.587 / sim10 0.605 (RAV4 내부 held-out 0.636/0.654). 차량·카메라 시프트 ≈ −0.05. **LB 0.37 vs 로컬 0.78~0.82의 갭을 설명하기엔 작다** → 비공개는 차량 외 요소(10Hz 재구성 파이프라인의 재인코딩/해상도, 또는 다른 라벨 후처리)도 다를 가능성.
- **MOTION-005**(두 차량, OPEN route holdout, no cw, motion, fps-aug): 내부 val 0.645/0.669, official 0.780/0.797, dense OPEN 0.770 (004: 0.821/0.819, 0.776 — RAV4 route에선 동급). → 제출 후보 11.
- Civic 라벨 분포: STOPPED 10.5% (RAV4 7.5%), STRAIGHT 83.8% — 도심 정지가 더 많음.

- **재인코딩 강건성**(`EXP-S3-REENCODE-001`): OPEN을 10fps mp4v로 재구성해 v11 snippet 실행 → 원해상도 0.765(캐시 0.770, 예측 일치 98%), 1280×720 리사이즈 0.749(steer −0.07). 파이프라인 차이도 갭을 설명 못 함.
- 남는 가설: 비공개 Stage3는 comma2k19가 아닌 **별도 촬영**(한국 도로/다른 카메라/자체 CAN). 그렇다면 appearance 특징의 전이가 나쁘고 motion 특징의 절대 스케일도 어긋남 → 영상별 정규화 motion-only 변형이 후보.

### 다음
- v10 vs v11 LB S3 비교 → 비공개 차량 판단. 둘 다 0.4 근처면 "10Hz 재구성" 파이프라인 차이를 재현하는 실험(OPEN 영상을 10fps로 재인코딩해 예측 변화 측정)으로 전환.
- 나머지 Civic 9route 캐시 완료 후 MOTION-006(전체 493seg).
