# 다음 에이전트 작업 프롬프트 — Stage 3 라벨 파이프라인 검증

우리는 DACON 「블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회」를 준비 중이다.

현재까지 대회 규정 조사, 공개 데이터 탐색, 로컬 평가 metric 구현, Stage 3 외부 데이터셋 조사, comma2k19 예제 데이터 기반 timestamp synchronization 및 라벨 변환 코드 작성까지 완료되어 있다.

기존 작업을 처음부터 다시 하지 말고, 저장소의 현재 상태를 먼저 확인한 후 아래 작업을 이어서 수행하라.

---

# 0. 먼저 읽을 문서

작업 시작 전에 반드시 다음 파일을 읽어 현재 구현 상태와 실험 설계를 파악하라.

```text
EXPERIMENT_DESIGN.md
dataset_registry.csv
src/eval/metrics.py

src/data/comma2k19/
    sync.py
    labeling.py
    report.py
    overlay.py
    run_subset.py
```

가능하다면 관련 README, 기존 baseline 코드, experiment log도 함께 확인한다.

이미 완료된 작업을 새로 다시 작성하지 말고 기존 코드를 검증·수정·확장한다.

---

# 1. 현재까지 확인된 사실

## Stage 3 공개 데이터

공개 Stage 3 데이터는 학습용으로 충분하지 않다.

- 영상 5개
- 영상당 sparse label 약 10개
- 총 약 50개 label
- accel 4-class / steer 3-class를 제대로 학습하기에는 매우 부족

따라서 Stage 3는 외부 주행 데이터를 활용해 학습 데이터를 구축하는 방향으로 진행한다.

---

# 2. comma2k19 관련 현재 상태

Stage 3 외부 데이터 후보로 comma2k19를 조사했다.

라이선스는 반드시 두 항목을 구분한다.

```text
commaai/comma2k19 GitHub demo
→ MIT

comma2k19 full dataset / Academic Torrents distribution
→ CC BY-NC-SA 3.0
```

`dataset_registry.csv`에는 이미 두 항목을 별도 row로 기록해두었다.

이 내용을 다시 하나의 MIT 라이선스로 합치지 마라.

---

# 3. 실제 데이터로 이미 확인한 내용

공식 GitHub demo segment를 실제 다운로드하여 분석했다.

확인 사항:

### speed

```text
CAN/speed/value
shape = (4974, 1)
```

즉 flat array가 아니므로 반드시 squeeze/reshape 처리해야 한다.

코드에서는 최종적으로:

```python
values.ndim == 1
```

을 보장하도록 defensive check를 둔다.

### timestamp

다음 timestamp array는 서로 동일하지 않다.

```text
speed/t
steering_angle/t
frame_times
```

따라서 절대 array index끼리 직접 zip하지 않는다.

항상:

```text
timestamp interpolation
→ common timeline
```

방식으로 정렬한다.

현재 코드는 이 구조를 사용하고 있다.

이를 유지하라.

---

# 4. 현재 comma2k19 demo 검증 결과

약 60초 segment를 10Hz로 변환했을 때:

```text
samples = 600
```

현재 threshold 기준 분포:

```text
Accel
ACCELERATING   23.8%
DECELERATING   20.5%
CONSTANT       55.7%
STOPPED         0.0%

Steer
LEFT            0.0%
STRAIGHT       98.8%
RIGHT           1.2%
```

이 값은 pipeline sanity check 결과이지 최종 threshold 근거가 아니다.

특히 이 60초 segment만으로 steering threshold를 결정하면 안 된다.

---

# 5. 현재 가장 중요한 blocker

현재 Stage 3 학습은 의도적으로 중단되어 있다.

이유:

```text
comma2k19 steering_angle의 부호 방향
```

즉,

```text
positive steering_angle = LEFT ?
positive steering_angle = RIGHT ?
```

가 공식 comma2k19 문서에서 명확히 확인되지 않았다.

현재 `labeling.py`의:

```python
STEER_SIGN
```

은 검증되지 않은 가정이다.

### 매우 중요

이 방향을 확인하기 전에 모델 학습을 시작하지 마라.

잘못된 방향을 사용하면 LEFT/RIGHT 학습 라벨 전체가 뒤집힐 수 있다.

---

# 6. Task A — metric self-test 실행

가장 먼저 다음을 실행한다.

```bash
python -m src.eval.metrics
```

`_self_test()` 내부의 hand-calculated assert가 모두 통과하는지 확인한다.

검증 대상에는 최소 다음이 포함되어야 한다.

### Stage 2

```text
collision ±0.3 sec
entry ±0.3 sec
evasion
entry_side
weighted score
```

### Stage 3

```text
0.7 × accel Macro-F1
+
0.3 × steer Macro-F1
```

그리고:

```text
GT accel == STOPPED
```

인 sample은 steer metric 계산에서 제외되어야 한다.

단, prediction에는 steer label이 존재해야 한다.

## 실패하면

모델 실험으로 넘어가지 말고 metric 구현부터 수정한다.

## 성공하면

결과를 experiment log 또는 작업 보고서에 기록한다.

---

# 7. Task B — comma2k19 subset pipeline 실행

다음을 실행한다.

```bash
python -m src.data.comma2k19.run_subset \
    --segment data/external/comma2k19_example/segment \
    --out output/comma2k19_subset
```

아래 산출물이 정상 생성되는지 확인한다.

```text
10Hz synchronized samples
label report
class distribution
overlay images
```

현재 awk 기반으로 검증된 600 sample 결과와 Python 결과가 실질적으로 일치해야 한다.

큰 차이가 발생하면:

```text
timestamp interpolation
array squeeze
time range
10Hz resampling
threshold implementation
```

중 무엇이 다른지 확인한다.

---

# 8. Task C — STEER_SIGN 검증

이 작업이 이번 단계에서 가장 중요하다.

## 8.1 영상 육안 검증

steering angle이 큰 sample을 추출한다.

최소:

```text
가장 큰 positive steering_angle 5~10개
가장 큰 negative steering_angle 5~10개
```

가능하면 STRAIGHT에 가까운 값은 제외한다.

각 sample에 대해 overlay 또는 원본 영상을 확인한다.

overlay에는 최소 다음을 표시한다.

```text
timestamp
speed
acceleration
steering_angle raw value
current predicted steering label
```

영상상 차량이 실제로:

```text
LEFT
RIGHT
```

중 어느 방향으로 조향하는지 확인한다.

---

## 8.2 motion 기반 추가 sanity check

가능하면 영상 육안 확인 외에도 독립적인 motion signal을 사용한다.

예:

```text
yaw rate
gyro_z
vehicle trajectory
lane/vanishing-point motion
```

comma2k19에 이용 가능한 IMU/yaw 관련 신호가 있다면 steering angle과의 부호 상관관계를 측정한다.

예:

```text
corr(steering_angle, yaw_rate)
```

또는

```text
sign agreement
```

을 계산한다.

주의:

coordinate convention을 확인하지 않은 채 상관계수 부호 하나만으로 LEFT/RIGHT를 확정하지 않는다.

영상 육안 판단과 motion signal이 같은 결론을 지지할 때 최종 결정한다.

---

# 9. STEER_SIGN 확정 후 해야 할 것

확정되면 `labeling.py`에 명확하게 기록한다.

예:

```python
# Verified against comma2k19 demo video on YYYY-MM-DD.
# Positive steering_angle corresponds to LEFT.
STEER_SIGN = ...
```

또는 반대 방향.

그리고 근거를:

```text
EXPERIMENT_DESIGN.md
experiment log
```

에도 남긴다.

다시는 이 값이 "가정"으로 보이지 않도록:

```text
verified / evidence
```

를 명시한다.

---

# 10. Task D — Stage 3 label conversion audit

STEER_SIGN 확인 후에만 진행한다.

현재 pipeline:

```text
speed(t)
→ smoothing
→ derivative
→ acceleration
→ 4-class accel label

steering_angle(t)
→ smoothing
→ 3-class steer label
```

을 검증한다.

라벨 파라미터는 코드에 박지 말고 config로 분리한다.

최소:

```yaml
stopped_speed_threshold:
accel_positive_threshold:
accel_negative_threshold:
steering_deadzone:
speed_smoothing_window:
steering_smoothing_window:
```

---

# 11. threshold는 아직 최적화하지 말 것

현재 demo 60초 데이터는 threshold tuning용 데이터가 아니다.

목적:

```text
pipeline correctness
timestamp correctness
sign correctness
label direction sanity check
```

이다.

따라서 이 데이터에서 class distribution이 예쁘게 나오도록 threshold를 억지로 조정하지 않는다.

특히:

```text
LEFT 0%
RIGHT 1.2%
```

라는 이유만으로 steering threshold를 줄이지 마라.

해당 60초 영상 자체가 거의 직진 주행일 수 있다.

---

# 12. Task E — full dataset 사용 전 조사

full comma2k19를 무조건 전부 다운로드하기 전에 다음을 확인한다.

```text
전체 데이터 크기
segment 구조
CAN availability
steering signal availability
speed availability
download method
license implications
required storage
expected processing time
```

그리고 Stage 3 training용으로 필요한 subset만 우선 가져올 수 있는지 조사한다.

가능하면 첫 단계에서는:

```text
몇 개 route / segment
```

만 가져와 pipeline을 검증한다.

전체 다운로드는 pipeline correctness 확인 후 결정한다.

---

# 13. 데이터 split 설계

comma2k19 기반 학습 데이터를 만들 경우 절대 frame random split을 사용하지 않는다.

split 단위:

```text
route
drive
continuous sequence
```

같은 주행 sequence의 인접 frame이 train과 validation에 동시에 들어가면 temporal leakage가 발생한다.

반드시 Group split을 사용한다.

---

# 14. 생성해야 하는 보고서

이번 작업이 끝나면 최소 다음을 보고한다.

```markdown
# EXP-S3-LABEL-001 결과

## 1. Metric self-test
- PASS / FAIL
- 수정 사항

## 2. Python subset pipeline
- sample 수
- duration
- timestamp range
- accel 분포
- steer 분포
- awk 검증값과 차이

## 3. STEER_SIGN
- positive angle이 의미하는 방향
- 영상으로 확인한 sample
- motion/yaw 추가 근거
- 최종 결론
- confidence

## 4. Label conversion
- speed preprocessing
- acceleration 계산 방식
- steering preprocessing
- 현재 threshold
- 아직 미확정인 threshold

## 5. Full dataset 계획
- license
- 예상 용량
- 다운로드 방식
- 우선 사용할 subset

## 6. 다음 실험
- 추천하는 정확한 experiment ID
- 이유
```

---

# 15. Training gate

아래 조건을 모두 만족하기 전에는 모델 훈련을 시작하지 않는다.

```text
[ ] src.eval.metrics self-test PASS
[ ] Python sync 결과가 real demo 검증값과 일치
[ ] speed/value (N,1) shape 처리 확인
[ ] speed / steering / frame timestamp 독립 보간 확인
[ ] STEER_SIGN 영상 기반 검증 완료
[ ] STEER_SIGN 코드/문서에 근거 기록
[ ] Stage 3 label conversion 결과 육안 sanity check 완료
[ ] train/validation route split 설계 완료
```

모두 완료된 뒤에만:

```text
EXP-S3-LABEL-001 → COMPLETE
```

로 표시한다.

---

# 16. 그 이후 다음 단계

위 작업이 모두 끝났을 경우 바로 대규모 모델 학습을 시작하지 말고 먼저:

```text
EXP-S3-LABEL-002
```

를 새로 정의한다.

목표:

```text
larger comma2k19 subset
→ dense 10Hz labels 생성
→ class distribution 분석
→ label transition 분석
→ threshold sensitivity 분석
```

그 결과가 충분히 정상적일 때:

```text
EXP-S3-BASE-001
```

로 넘어가 첫 RGB temporal baseline을 학습한다.

---

# 17. 규정 관련 주의

절대 다음을 하지 않는다.

```text
hidden test prediction으로 threshold 튜닝
hidden test pseudo-labeling
test-time training
평가 영상 간 통계 공유
평가 데이터 전체 class distribution에 맞춘 calibration
```

모든 threshold와 preprocessing 규칙은 train / validation 외부 데이터에서 고정한다.

---

# 18. 작업 원칙

- 기존 코드를 먼저 읽고 최소 수정한다.
- 실제 데이터로 검증 가능한 것은 추측하지 않는다.
- 문서와 실제 데이터가 충돌하면 실제 관찰 내용을 기록하되 공식 대회 규정은 별도로 유지한다.
- 라이선스는 데이터 배포 단위별로 기록한다.
- 구현한 모든 중요한 가정에는 assert 또는 validation check를 추가한다.
- 한 번에 여러 실험을 섞지 않는다.
- 실패한 실험도 기록한다.
- 현재 단계에서는 모델 성능보다 **라벨의 정확성과 데이터 파이프라인의 신뢰성**이 최우선이다.

작업을 완료한 뒤, 변경한 파일 목록과 실험 결과, 발견한 문제, 다음 권장 실험을 구체적으로 보고하라.
