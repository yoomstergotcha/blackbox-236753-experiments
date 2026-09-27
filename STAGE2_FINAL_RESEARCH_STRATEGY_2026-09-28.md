# DACON 236753 — 최종 개선 전략 정리
**기준 시점:** 2026-09-28 01:09 KST  
**대상 저장소:** `yoomstergotcha/blackbox-236753-experiments`  
**목표:** 마감 직전 남은 실험을 “검증 가능한 개선” 중심으로 압축하고, 이미 실패한 방향의 반복을 막는다.

---

## 0. 현재 스냅샷

### 현재 최고 LB
| 제출 | Stage1 | Stage2 | Stage3 | Total |
|---|---:|---:|---:|---:|
| `submit_v33.zip` | **0.97382** | **0.35887** | 0.54024 | **0.5544** |

최신 확인 커밋:
- `a02d2f8` — v33 LB 최고 갱신: S1 0.974 / S2 0.359 / S3 0.540 / Total 0.5544
- `a99e6c2` — Stage2 카메라 도메인 이동 강건 확인, tail-exclude 후보 v35/v36
- `6f5cd4a` — Stage3 혼합 앙상블·Viterbi 기각

### 현재 확정된 구성
- **Stage1:** Qwen2-VL 기반 rerecord 판별 고정. 추가 연구 우선순위 매우 낮음.
- **Stage2:** `stride=3` 고정 + `entry = collision - 21` + VLM side/evasion.
- **Stage3:** 현재 LB 기준은 v13(0.54024). v34가 전차종 + median61 + steer bias 개선 후보로 준비 완료.

### 가장 중요한 최근 발견
1. **Stage2 private 시간축은 10fps 계열보다 3배 스케일에 가깝다.**
   - v26 S2 0.28331
   - v30 (`stride3 + entry-21`) S2 0.33987
   - v33 (+ VLM side/evasion) S2 0.35887
2. **VLM side/evasion은 실제 LB에서도 유효했다.**
   - v30 → v33에서 S2가 약 +0.019 상승.
3. **카메라/화질 domain shift는 현재 Stage2 실패의 주원인이 아닌 것으로 보인다.**
   - gamma, jpeg25, low-res, zoom, noise, blur, hflip 변환에서도 OOF 성능이 대체로 유지.
4. **Stage3에서 Viterbi와 v13 혼합 앙상블은 이미 기각됐다.**
5. **Stage2 tail exclusion은 아직 LB로 확인할 가치가 있다.**
   - v30은 stride3 기준 마지막 3 sampled frame = 원본 약 9 frame을 제외.
   - v35는 이를 1 sampled frame = 원본 약 3 frame으로 축소.

---

# 1. 전략 원칙

남은 시간에는 새 데이터셋을 찾거나 source를 추측하기보다 아래 원칙을 유지한다.

> **현재 LB에서 확인된 강한 구성(v33)을 anchor로 고정하고, 한 번에 하나의 실패 모드만 수정한다.**

특히 Stage2에서는:
- 강한 peak가 맞는 영상은 절대 건드리지 않는다.
- **ambiguous / low-confidence 영상에서만 보조 신호를 사용한다.**
- 모든 신규 방법은 `fallback = v33/v30 baseline`을 가져야 한다.

이 방식은 전체 모델을 갈아엎는 것보다 마감 직전 리스크가 낮다.

---

# 2. 지금 당장 확인해야 하는 기존 후보

## P0-A. `v35` — tail exclusion 축소
**목적:** 충돌이 영상 끝에 가까운 경우 현재 `tail exclude = 9 original frames` 때문에 정답 peak를 버리고 있는지 확인.

- baseline: v30
- 변경: sampled tail exclude `3 → 1`
- Stage2만 판독
- **판정**
  - `S2(v35) > 0.33987`: tail 축소 채택
  - 동률 수준: 기존 유지
  - 하락: 폐기

이 실험은 구조 변경이 거의 없고 해석도 명확하므로 우선 실행 가치가 높다.

## P0-B. `v34` — Stage3 개선 후보
구성:
- RAV4 + Civic 전차종 3-seed
- median window 61
- steer logit bias: LEFT +1.25 / RIGHT +0.5

held-out:
- Civic: 0.378 → 약 0.459
- RAV4: 0.702 → 약 0.741

**판정:** v33과 비교해 Stage3 열만 본다.  
Stage3가 오르면 최종 backbone은 v34 계열로 교체하고, 이후 Stage3 신규 실험은 중단한다.

---

# 3. 아직 충분히 탐구하지 않았고 가능성이 높은 Stage2 전략

## P1. Top-K collision 후보 + confidence-conditional reranker
**추천도: 최상**

현재 localizer는 사실상 최종 score curve의 **argmax 하나**를 선택한다.  
가장 먼저 확인할 것은 “정답 collision이 top-1에는 없지만 top-2/top-3에는 자주 존재하는가?”이다.

### 3.1 1차 진단
OOF에서 다음을 계산:
- top-1 hit
- top-2 oracle hit
- top-3 oracle hit
- top-5 oracle hit

peak 간 최소 거리도 둔다(예: 3~5 sampled frames).

### 3.2 해석
- top-1과 top-3 차이가 작음 → reranking 가치 낮음
- **top-3 oracle이 크게 높음 → 현재 병목은 detection이 아니라 candidate selection**

이 경우 새 backbone보다 reranker가 훨씬 효율적이다.

### 3.3 candidate feature
각 후보 peak `p`에 대해:
- base ensemble score
- 1위/2위 score margin
- peak width / sharpness
- local temporal curvature
- pre/post `mag_mean` 변화
- radial expansion
- vertical/global jerk
- flow residual energy
- candidate가 clip tail에서 얼마나 떨어졌는지
- post-impact stillness
- YOLO proximity / disappearance signal
- phase agreement(아래 P4)

### 3.4 핵심: 보조 신호를 전체 영상에 적용하지 않는다
기존 YOLO 실험에서:
- 전체 적용은 중립/열세
- **weak-peak subset에서는 약한 개선이 관측됨**

따라서:
```text
if localizer_confidence >= threshold:
    use baseline argmax
else:
    rerank top-K candidates with auxiliary features
```

### 3.5 학습
복잡한 모델 불필요:
- logistic regression
- LightGBM/작은 MLP
- pairwise ranking loss

positive = true collision 근처 candidate  
negative = 모델이 실제로 헷갈린 top false peak

### Go / No-Go
**Go:** robust OOF에서 top-1 대비 개선 + shift 조건에서 worst-case 유지  
**No-Go:** top-K oracle gap 자체가 작음

---

## P2. Entry 독립 refiner — `collision - 21`을 안전하게 보정
**추천도: 상**

`entry = collision - 21`은 현재 강한 baseline이지만 모든 사고에 동일한 lead time을 강제한다.

Stage2에서 collision과 entry의 비중이 둘 다 크기 때문에, entry를 조금만 독립적으로 개선해도 가치가 있다.

### 3.6 설계
먼저 collision은 현재 v30/v33 값을 그대로 사용한다.

그 앞 구간에서:
- radial expansion 증가
- center-region residual motion
- flow divergence
- global motion jerk
- looming/TTC proxy
- sustained motion onset

의 **change point**를 찾는다.

예:
```text
search_window = [collision - 45, collision - 6]  # 30fps 기준 후보
default_entry = collision - 21

if change_point_confidence > threshold:
    entry = detected_change_point
else:
    entry = default_entry
```

### 3.7 중요한 점
완전한 새 entry detector가 아니라 **고신뢰일 때만 -21에서 벗어나는 refiner**여야 한다.

### 검증
- 수동 entry label이 있는 CCD subset 활용
- 10fps 원본 + 3x 시간축 변형 양쪽에서 확인
- offset MAE뿐 아니라 official-style tolerance 기준 hit 측정

### Go / No-Go
**Go:** baseline `-21` 대비 entry hit 상승 + collision은 완전히 동일  
**No-Go:** confidence gate 후에도 worst-domain 성능 감소

---

## P3. VLM side/evasion의 flip-consistency TTA
**추천도: 상 / 구현비용 낮음**

VLM side/evasion은 이미 LB에서 **실제 + 효과가 확인된 신호**다.  
따라서 temporal localization에 VLM을 다시 쓰는 것보다, 이미 성공한 task를 더 안정화하는 쪽이 낫다.

### 3.8 Entry side
원본:
```text
d_orig = logP(LEFT) - logP(RIGHT)
```

수평 반전 영상에서는 LEFT/RIGHT 의미를 뒤집어:
```text
d_flip_corrected = -(logP_flip(LEFT) - logP_flip(RIGHT))
```

최종:
```text
d_final = 0.5 * (d_orig + d_flip_corrected)
```

이 방식은 현재 존재하는 LEFT bias를 물리적으로 상쇄할 가능성이 있다.

### 3.9 Evasion
YES/NO는 horizontal flip invariant이므로:
- original
- flipped

두 logit을 평균.

### 3.10 런타임 보호
전체 클립에 2배 VLM inference를 하지 않는다.

```text
if abs(current_logit_margin) < uncertainty_threshold:
    run flip TTA
else:
    keep original result
```

즉 **uncertain cases only**.

### 추가로 가능한 저비용 개선
2개 prompt ensemble:
- 현재 prompt
- 동일 의미의 더 짧고 구체적인 prompt

단, manual label validation에서 둘 다 살아남을 때만 채택.

---

## P4. stride3 temporal phase consensus
**추천도: 중상 / LB probe 성격**

현재 v30은 30fps 영상에서 사실상 한 phase만 본다.

- phase 0: 0, 3, 6, ...
- phase 1: 1, 4, 7, ...
- phase 2: 2, 5, 8, ...

실제 30fps에서는 세 phase 모두 진짜 frame이다.

기존 `EXP-S2-PHASE-001`은 10fps CCD를 **선형보간으로 30fps화**했기 때문에 phase 1/2가 blended fake frame이었고, 실험이 실제 30fps phase 문제를 제대로 검증하지 못했다.

### 새 방식
score curve를 무조건 평균하지 않는다.

각 phase에서:
```text
candidate_0
candidate_1
candidate_2
```
를 얻고 원본 frame 좌표로 복원한다.

- 2개 이상 candidate가 가까움 → cluster weighted median
- 세 후보가 갈림 → baseline phase0 유지
- phase1/2가 phase0보다 훨씬 강하면서 서로 동의할 때만 이동

즉 **conservative consensus**.

### 주의
로컬에 genuine 30fps collision label validation이 없다면 이 방법은 강한 offline 증거를 만들기 어렵다.  
따라서 **경량 Stage2-only LB probe 한 번** 이상 투자하지 않는다.

---

## P5. frame duplicate 정리
**추천도: 중 / 구현비용 매우 낮음**

긴 클립 시뮬에서 장면 경계가 허위 jolt를 만들었고, 실제 데이터에 frame duplication이 있으면 motion feature에도 왜곡이 생길 수 있다.

### 안전한 범위만 적용
연속 frame 간 low-res difference가 거의 0인 경우만 duplicate로 판단:
```text
if mean_abs_diff(frame[t], frame[t-1]) < eps:
    collapse duplicate
```

collision을 정리된 시간축에서 찾은 후 **원본 frame index로 역매핑**.

### 하지 말 것
큰 frame difference를 무조건 frame-drop으로 간주해 제거하지 않는다.  
진짜 collision도 큰 difference를 만들기 때문이다.

---

## P6. Hard-negative 학습 / ranking loss
**추천도: 중 / P1 결과가 좋을 때만**

현재 BCE soft target에서는 pothole, 급브레이크, camera shake 같은 어려운 오답 peak가 일반 negative와 크게 다르게 취급되지 않는다.

P1의 top-K oracle gap이 크다면 그다음에 실시한다.

### loss 예시
```text
L = L_BCE + λ * max(0, margin - score_true + score_hardneg)
```

`hardneg`:
- 현재 모델이 높게 점수 준 false peak
- 정답에서 충분히 떨어진 위치

### 우선순위
새 GRU/CNN을 처음부터 탐색하기보다:
1. 기존 checkpoint로 hard negatives 추출
2. 소규모 fine-tune
3. worst-domain 기준 비교

---

# 4. Stage2에서 지금 반복하지 말아야 할 전략

아래는 이미 충분한 반증이 있으므로 마감 직전 다시 파지 않는다.

| 전략 | 이유 |
|---|---|
| VLM coarse-to-fine collision localization | 10fps/30fps 모두 jolt보다 크게 열세 |
| YOLO collision detector 단독 | collision hit가 jolt보다 낮음 |
| source/FPS dataset matching 재탐색 | LB상 source matching보다 시간축 처리 효과가 훨씬 컸음 |
| 카메라/화질 augmentation 추가 탐색 | gamma/jpeg/lowres/zoom/noise/blur/hflip에서 이미 강건 |
| z-score multi-stride fusion 반복 | 기존 실험에서 ratio selection보다 열세 |
| post-impact stillness를 전체에 강제 | CCD에서 중립/소폭 하락; 별도 probe 이상 투자 불필요 |
| synthetic interpolated 30fps phase 평균 | blended-frame artifact 때문에 실제 30fps를 대표하지 못함 |

---

# 5. Stage3 전략

## 우선 v34 LB 결과를 먼저 본다
지금 Stage3는 새 연구보다 **v34가 private에서 실제로 이득인지 확인하는 것**이 우선이다.

이미 기각:
- v13 + allcars 혼합 ensemble
- Viterbi decoding
- prior matching
- accel bias
- appearance centering
- video/steer normalization 계열 일부

따라서 v34가 개선되면 **Stage3는 lock**하는 것이 좋다.

### v34가 하락할 경우
새 대규모 학습보다:
- v13 + median61
- v13 + validated steer bias

처럼 **한 요소씩 분리한 경량 probe**가 더 합리적이다.

---

# 6. Stage1 전략

**고정.**

S1 = 0.97382로 이미 상위권 기준을 충족/상회한다.  
Stage1에서 새로운 물리 feature, metadata, 합성 재촬영 classifier를 다시 파는 것은 현재 기대값이 낮다.

---

# 7. 추천 실험 순서

## Tier A — 바로 수행
1. **v35 LB** — tail exclude 9 → 3 frame 효과 확인
2. **v34 LB** — Stage3 개선 판정
3. **EXP-S2-RERANK-001** — top-K oracle gap 측정

## Tier B — top-K gap이 충분할 때
4. **EXP-S2-RERANK-002** — confidence-conditional candidate reranker
5. **EXP-S2-VLM-TTA-002** — uncertain-case horizontal flip TTA

## Tier C — 추가 시간이 있을 때
6. **EXP-S2-ENTRY-CHANGE-001** — confidence-gated entry change-point refiner
7. **EXP-S2-PHASE-TTA-002** — conservative 3-phase consensus LB probe
8. **EXP-S2-TIMECLEAN-001** — duplicate collapse

## Tier D — 조건부
9. **EXP-S2-HARDNEG-001** — P1에서 top-K oracle gap이 클 때만 hard-negative fine-tune

---

# 8. 각 실험의 Go / No-Go 기준

| 실험 | Go 기준 | 실패 시 |
|---|---|---|
| v35 tail | v30 S2 0.33987 초과 | tail=3 sampled 유지 |
| v34 S3 | v33 S3 0.54024 초과 | v13 유지 |
| Top-K oracle | top3가 top1보다 명확히 높음 | reranker 중단 |
| conditional reranker | robust OOF + worst-domain 동시 개선 | baseline argmax 유지 |
| VLM flip TTA | manual label CV 개선, runtime 허용 | 기존 threshold 유지 |
| entry refiner | `c-21`보다 entry hit 개선 | `c-21` 고정 |
| phase consensus | LB에서 v30/v33 collision 계열 초과 | phase0 고정 |
| duplicate cleanup | transformed validation 중립 이상 | 제거 |

**중요:** 평균 성능이 아니라 worst-domain이 악화되지 않는지 같이 본다.

---

# 9. 제출 슬롯 운영 원칙

Stage별 isolate build를 우선한다.

### Stage2 probe
- S1 = 가벼운 기존 고정값
- S3 = v13
- **S2만 변경**

이렇게 해야 LB 숫자를 바로 해석할 수 있다.

### 최종 build
최종 구성은 각 stage에서 **LB로 실제 확인된 요소만** 합친다.

예:
```text
Stage1 = v33 VLM
Stage2 = best confirmed among v33 / tail / reranker / entry-refiner
Stage3 = best confirmed among v13 / v34
```

마감일 마지막 슬롯은 새 가설 probe가 아니라 **확정 요소 결합 최종본**에 남긴다.

---

# 10. 최종 추천

현재 가장 큰 기대값은 새 backbone이 아니라 다음 두 군데에 있다.

## 1순위
**collision candidate selection 개선**
- top-K oracle 진단
- strong case는 baseline 유지
- weak/ambiguous case만 rerank

이 전략은 이미 버린 YOLO/stillness/VLM 신호도 “전체 대체”가 아니라 **fallback expert**로 재활용할 수 있다는 점이 핵심이다.

## 2순위
**VLM side/evasion 안정화**
- 이미 private LB에서 +0.019가 확인된 신호
- horizontal-flip consistency + uncertain-case-only TTA는 비용 대비 합리적

## 3순위
**entry `-21`의 confidence-gated refinement**
- 현재 강한 prior를 유지하면서 개별 사고 차이를 반영
- 실패해도 baseline fallback 가능

---

## 한 줄 결론

> **v33을 버리지 말고, “argmax가 애매한 영상만 더 똑똑하게 처리하는 조건부 보정”에 집중한다.**  
> 현재 시점에서 가장 가능성이 높은 신규 전략은 `Top-K collision reranking → VLM flip-consistency → entry change-point refiner` 순서다.

---

## 실행 결과 반영 (2026-09-28 새벽, 실험 로그 EXP-S2-TOPK-001 / RERANK-001 / NONEGO-001 / ALL-001)

- **Top-K reranking**: 배포 11모델 진짜 OOF 0.788, 오라클 top-2 0.881 / top-3 0.941(오답의 72%가 top-3 안). 그러나 curve/phys/app/det 특징의 listwise 재순위기는 최대 0.790 — 후보를 가르는 정보가 현재 특징에 없어 **기각**.
- **라벨/도메인 불일치**: 공개 Stage2 5개 = CCD 000001~5, t_collision = CCD onset 완전 일치. 배포 앙상블(ego 전용)은 CCD non-ego(관찰자형) 699클립에서 0.282 → LB 역산 collision 적중(0.27~0.33)과 같은 크기. **가장 큰 식별 가능한 불일치 = 사고 유형(ego 관여 여부)**.
- **채택**: ego+non-ego 통합 학습 3-seed와 배포 11의 그룹 가중 혼합(w=0.5) → ego 0.793 / non-ego 0.421, 30fps 긴 클립 이동에서도 유지. `submit_v37.zip` = v34 + 이 혼합 (릴리스 submissions-2026-09-28d).
