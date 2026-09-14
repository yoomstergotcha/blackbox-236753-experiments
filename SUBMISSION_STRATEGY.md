# 제출 전략 — 하루 3회 한도, LB 점수 향상 목표

> 작성일: 2026-09-15 | 리더보드 제출 마감: 2026-09-29 (약 14일, 최대 ~42회 제출 가능)
> 근거: [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md), [COMPETITION_GUIDE.md](COMPETITION_GUIDE.md) §17

## 0. 먼저 밝혀야 할 제약

**나는 DACON 웹사이트에 직접 제출 버튼을 누를 수 없다.** 브라우저 조작이나 DACON API 접근 도구가 없다 — `submit.zip`을 만들고 검증하는 것까지가 내 역할이고, 실제 업로드는 사용자가 dacon.io에서 직접 해야 한다. 아래 전략은 "언제 무엇을 준비해서 사용자에게 제출을 요청할지"에 대한 계획이다.

---

## 1. 점수 구조 재확인 — 어디에 여지가 있는가

```text
Total = 0.20×Stage1 + 0.40×Stage2 + 0.40×Stage3
Stage2 = 0.35×collision_acc + 0.35×entry_acc + 0.15×evasion_f1 + 0.15×side_f1
Stage3 = 0.7×accel_macroF1 + 0.3×steer_macroF1
```

| 항목 | 전체 가중치 | 현재 상태 |
|---|---|---|
| Stage1 전체 | 0.20 | 공식 베이스라인 그대로 — 10샘플 중 예측 전부 ORIGINAL로 collapse |
| Stage2 collision (0.35) | 0.14 | 5샘플 실 라벨로 학습됨 — 유일하게 baseline보다 나을 수 있는 부분 |
| Stage2 entry/evasion/side (0.65) | 0.26 | **학습 신호 전혀 없음** — 전부 추정(현재 evasion=1, side=LEFT로 collapse) |
| Stage3 accel (0.7) | 0.28 | comma2k19 파이프라인 완성, 실제 학습 진행 중 |
| Stage3 steer (0.3) | 0.12 | 위와 동일 |

**Stage2의 entry/evasion/side(전체의 26%)가 가장 크고 가장 손대지 않은 영역**이지만, 사고 충돌/진입 시점에 대한 외부 공개 데이터가 comma2k19처럼 바로 가져다 쓸 만한 게 없다(정상 주행 데이터와 실제 "사고" 데이터는 근본적으로 다른 도메인). 반면 **Stage3(전체의 40%)는 이미 검증된 파이프라인이 있어 가장 빠르게 현재 baseline보다 나아질 수 있는 영역**이다.

## 2. 우선순위 (기대효용 = 영향 × 신뢰도 ÷ 비용)

| 순위 | 변경 | 영향 | 신뢰도 | 비용 | 비고 |
|---|---|---|---|---|---|
| 1 | Stage3에 comma2k19 학습 모델 적용 | 0.40 | 중간 | 낮음(이미 90% 완성) | **도메인 시프트 리스크**: comma2k19는 정상 고속도로/도심 주행, 실제 평가 데이터는 사고 직전/중 영상 — 일반화 안 될 가능성 있음. 그래도 현재 baseline(collapse)보다는 나을 것으로 예상 |
| 2 | Stage1 synthetic 재녹화 augmentation | 0.20 | 중간 | 낮음(외부 데이터 불필요, 자체 생성) | 화면 베젤/모아레/재압축 시뮬레이션 — 아직 미착수, 하루 이내 착수 가능 |
| 3 | Stage2 evasion/entry_side 룰베이스 heuristic | 0.13(0.4×0.15×2) | 낮음 | 낮음 | lane/도로 경계 segmentation 필요, 학습 없이도 collapse보다 나은 규칙 가능성 |
| 4 | Stage2 entry_frame 검출기 | 0.14 | 낮음 | 높음 | 실 라벨 없음 — 사고영상 데이터소싱 또는 직접 라벨링부터 필요, 가장 오래 걸림 |

## 3. 제출 슬롯 배분 원칙

1. **한 번에 한 Stage만 바꾼다.** 여러 축을 동시에 바꾸면 LB 변화의 원인을 알 수 없다 — `AUTONOMOUS_RESEARCH_AGENT.md`의 "single change" 원칙을 제출에도 그대로 적용.
2. **로컬 검증 없이는 슬롯을 쓰지 않는다.** `src/eval/metrics.py` + confusion matrix로 majority-class collapse 등 자명한 실패를 먼저 걸러낸다 — 이미 EXP-S3-BASE-001에서 이 습관 덕에 가짜 개선(val_steer_acc=1.0)을 걸러낸 적 있음.
3. **하루 3번을 다 쓸 필요는 없다.** 남은 기간이 약 14일(~42회)이라 여유가 있다 — 검증된 변경이 없으면 슬롯을 아낀다. 단, install/runtime 오류는 매일 3번 중 소진되므로(가이드 §17) 제출 전 로컬 smoke-test를 반드시 통과해야 한다.
4. **첫 제출은 "성능"이 아니라 "정합성 확인"이 목적.** 실제 비공개 평가 데이터·60분 러닝타임·평가서버 환경에서 무너지지 않는지 확인하는 것이 가장 중요 — 로컬 smoke-test(5개 샘플)로는 이걸 100% 보장 못 하기 때문.

## 4. 구체적 일정

### Day 1 (오늘, 2026-09-15) — 제출 1: 공식 baseline 그대로

- 이미 준비됨: `submit.zip`(303MB) — Stage1/2/3 전부 공식 베이스라인, 5개 공개 샘플로 로컬 smoke-test 통과.
- 목적: LB 점수 확인 + 평가서버 install/runtime 호환성 확인(이후 모든 실험의 기준선).
- **사용자 액션 필요**: dacon.io에 이 zip 업로드.

### Day 1~2 — 제출 2: Stage3만 comma2k19 모델로 교체

- 필요 작업(진행 중): `EXP-S3-BASE-002` 학습 완료 → 공식 `predict_stage3(data_dir, model_dir)` 인터페이스에 맞는 wrapper 작성(comma2k19 폴더 구조가 아니라 임의의 `data_dir/videos/*.mp4`를 받아야 함) → 로컬 smoke-test로 baseline과 비교.
- Stage1/2는 baseline 그대로 유지 — LB 델타가 순수하게 Stage3 변경분이 되도록.
- **예상 리스크**: 도메인 시프트로 기대보다 개선이 적거나 오히려 나쁠 수 있음 — 그래도 시도할 가치 있음(현재 baseline이 collapse 상태라 하한이 낮음).

### Day 2~3 — 제출 3: + Stage1 synthetic augmentation

- Stage1 재녹화 시뮬레이션 파이프라인(디스플레이 베젤, 모아레, 재압축, 원근왜곡) 구축 → 재학습 → baseline과 비교.
- Stage2/3는 제출 2 상태 유지.

### Day 3 이후 — Stage2 착수

- entry_frame/evasion_space/entry_side용 데이터 소싱(사고 영상 데이터셋 재조사, 또는 공개 5건 직접 라벨링으로 최소 검증셋 확보) — 가장 오래 걸리는 트랙이라 병렬로 조사 시작.

## 5. 각 제출 전 체크리스트 (가이드 §26 그대로)

- [ ] `src/eval/metrics.py` self-test PASS
- [ ] 바뀐 stage만 baseline과 다름 (나머지 두 stage는 이전 제출과 동일)
- [ ] confusion matrix로 collapse 여부 확인
- [ ] 로컬 smoke-test(5개 공개 샘플)로 3개 함수 모두 에러 없이 DataFrame 반환
- [ ] submit.zip 구조/용량(≤10GB, 압축해제 ≤32GB) 확인
- [ ] requirements.txt에 불필요한 의존성 없음
- [ ] inference.py가 런타임에 인터넷 접근을 시도하지 않음(pretrained weight 자동 다운로드 없음)

## 6. 기록

모든 제출은 [experiments/experiment_log.csv](experiments/experiment_log.csv)에 실험으로 기록하고, 실제 LB 점수를 사용자에게 받으면 이 문서와 로그에 소급 반영한다.
