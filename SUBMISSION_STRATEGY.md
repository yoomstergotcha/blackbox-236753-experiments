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

### 제출 2: Stage3만 comma2k19 모델로 교체 — **완료, 제출 대기 중**

- `EXP-S3-BASE-002/003`(MViT scratch, lr 1e-4·2e-5)은 둘 다 majority-class collapse — 진단 후 아키텍처를 ImageNet-pretrained ResNet18(frozen)+작은 head로 교체(`EXP-S3-BASE-004/005`).
- `EXP-S3-BASE-005`(15 epoch, val_stage3_score 기준 best checkpoint 자동 저장) → 공식 `predict_stage3(data_dir, model_dir)` 인터페이스 wrapper 작성(`src/train/predict_stage3_comma2k19.py`, comma2k19 폴더 구조가 아니라 `data_dir/videos/*.mp4` 임의 입력을 받음).
- **실제 official Stage3 라벨(50행, comma2k19와 무관한 도메인)로 baseline과 직접 비교**: baseline `stage3_score=0.140` → candidate2 `stage3_score=0.409` (약 3배). Stage1/2는 baseline과 100% 동일.
- `submit_v2.zip`(203MB) 생성·로컬 smoke-test 통과 완료 — `build_submission_v2.py`.
- **도메인 시프트 리스크는 우려보다 작았다** — comma2k19(정상 주행)로 학습했지만 official 사고영상 라벨에서도 개선을 보임. N=50이라 확정적이진 않지만 방향은 뚜렷함.
- **사용자 액션 필요**: `submit_v2.zip`을 dacon.io에 업로드 (제출 1과 별도 슬롯 — 오늘 3회 중 아직 안 씀).

### LB 결과 (2026-09-17 확인) — 두 제출 모두 정상 채점

| 제출 | S1 | S2 | S3 | Total | 서버 시간 |
|---|---|---|---|---|---|
| #91244 submit.zip | 0.40456 | 0.13129 | 0.10721 | 0.1763 | 14:19 |
| #91246 submit_v2.zip | 0.40456 | 0.13129 | **0.37034** | **0.2816** (231등) | 11:46 |
| submit_v3.zip (Stage1 005) | 0.40076 | 0.13129 | 0.37034 | 0.2809 | 13:30 |
| submit_v4.zip (Stage1 v4 프로브) | **0.42366** | 0.13129 | 0.37034 | **0.2854** | 13:56 |
| submit_v5b.zip (Stage3 CLASS-001) | 0.42366 | 0.13129 | **0.22468** | 0.2371 | 14:16 |
| submit_v6.zip (Stage3 MOTION+smooth) | 0.42366 | 0.13129 | 0.35539 | 0.2794 | 13:19 |
| submit_v7.zip (Stage1 서명 규칙) | **0.39303** | 0.13129 | 0.35539 | 0.2734 | 13:41 |
| submit_v8.zip (Stage2 휴리스틱) | 0.42366 | **0.24988** | 0.35539 | **0.3268** | 13:01 |
| submit_v9.zip (Stage3 no-cw + prior) | 0.42366 | 0.24988 | 0.50120 | 0.3851 | 12:39 |
| submit_v10.zip (Stage3 라벨 v2 + holdout) | 0.42366 | 0.24988 | **0.51512** | **0.3907** | 12:51 |
| submit_v11.zip (Stage3 + Civic) | 0.42366 | 0.24988 | 0.47633 | 0.3752 | 12:51 |
| submit_v12.zip (Stage2 CCD 후반부 제한) | 0.42366 | **0.17469** | 0.51512 | 0.3606 | |
| submit_v13.zip (Stage3 RAV4 373seg 3-seed) | 0.42366 | 0.17469 | **0.54024** | 0.3707 | |
| 1등 (2026-09-18) | 0.9699 | 0.64 | 0.7823 | 0.7629 | |
| 1등 (2026-09-17) | 0.95145 | 0.54346 | 0.75456 | 0.7095 | |

Stage3 델타(3.45배)가 로컬 예측(2.9배)과 일치 → 로컬 official-label 검증을 믿고 슬롯을 써도 된다. 1등 대비 가중 갭 S1 0.109 / S2 0.165 / S3 0.154. **우선순위 재조정: Stage1(degenerate, 자체 합성으로 해결 가능, 가장 싼 큰 이득) → Stage3 확장(공식 Stage3 샘플이 comma2k19 규격임을 확인, 데이터 추가가 직접 효과) → Stage2.** 상세: [EXPERIMENT_DESIGN.md §14](EXPERIMENT_DESIGN.md).

### 제출 3: Stage1만 교체 — **완료, 제출 대기 중** (2026-09-17)

- 합성만으로는(EXP-S1-SYNTH-001~004) 공식 5쌍을 못 맞춤 → 공식 5쌍의 변환을 실측(블러+그레인+대비, 정렬 이동 없음)하고, **공식 5쌍을 학습에 포함 + leave-one-pair-out 검증**(EXP-S1-SYNTH-005): **OOF 10영상 macro-F1 1.0, AUC 1.0.** 상세 [EXPERIMENT_DESIGN.md §15](EXPERIMENT_DESIGN.md).
- `submit_v3.zip`(127MB): Stage1 = 005 모델(+threshold.json), Stage2 = baseline, Stage3 = 제출 2와 동일 — Stage2/3 smoke 출력이 제출 2와 바이트 동일함을 `cmp`로 확인.
- 기대: S1 0.40 → 0.9+ (비공개 재녹화가 공개 예제와 같은 처리일 때, total +0.10). 실제 재촬영이 섞여 있으면 0.5~0.8 → LB 값 자체가 비공개 처리 방식을 알려주는 진단이 된다.
- **LB 결과 (12:19)**: S1 **0.4008** (baseline 0.4046), S2/S3 동일. 역산하면 RERECORDED를 하나도 못 잡음 → **비공개 재녹화는 공개 예제(재압축 파생본)와 다른 종류.** 공개 5쌍을 완벽히 잡는 모델이 전혀 못 보므로, 실제 화면 재촬영(가이드 §2.3 특징들)일 가능성이 높다.

### 제출 4 (프로브): Stage1 = 실제 화면 재촬영 합성(v4) 모델 — **제출 완료, LB 0.4237**

- `EXP-S1-SYNTH-006`: 화면이 프레임의 55~95%를 차지하는 모니터 합성(베젤/배경) + 모아레 2중 격자 + 주사 밴딩 + 원근 + 손떨림 + 카메라 색/비네팅/반사광 + 초점 블러 + 센서 노이즈. 공개 예제식 합성과 50:50, 공식 5쌍 포함. 추론 크롭에 모서리 4개 추가.
- `submit_v4.zip`(127MB): Stage2/3 출력 제출 2와 바이트 동일.
- **해석 규칙**: S1 ≥ 0.6 → 재촬영 가설 확인, v4를 더 정교화(합성 다양성·해상도/fps 변화) 하면 0.9 근접 가능. S1 ≈ 0.40 유지 → 비공개 재녹화는 또 다른 종류 → Stage1 보류, Stage3 확장(Chunk_2 다운로드 진행 중)으로 전환.
- **LB 결과 (20:38)**: S1 **0.4237** (baseline 0.4046, v3 0.4008), S2/S3 동일. 역산(비공개 ORIGINAL 비율≈0.68): FP 0%→재현율 1.8% / FP 1%→재현율 2.1% / FP 2%→재현율 2.4% / FP 5%→재현율 3.3% — 비공개 RERECORDED의 **2~5%만** 잡음. 판정: 0.6 기준 미달 → 규칙대로 **Stage1 보류**. 재촬영 가설은 완전 기각은 아니고(v3는 0개, v4는 소수 검출) '비공개 재녹화 대부분은 v4 합성과도 다른 외형'. 다만 v4 Stage1이 LB 최고(+0.019, total +0.004)이므로 **Stage1 자리는 v4 모델로 유지**.

### 제출 5: Stage3 확장 (Chunk_1+2, 291세그먼트, class-weighted head) — **완료, 제출 대기**

- Chunk_2 추가 → 291세그먼트/42route(5배). frozen backbone이라 프레임 특징을 캐시(715MB)하고 head만 학습 → 실험 1분. `EXP-S3-CLASS-001`(class-weighted CE) 내부 val 0.566(63seg/8route), **공식 라벨(정렬 버그 수정 후) 0.584→0.743**. 상세 [EXPERIMENT_DESIGN.md §16](EXPERIMENT_DESIGN.md).
- 두 변형 준비: **v4 실패 시 `submit_v5a.zip`**(Stage1 baseline), **v4 성공 시 `submit_v5b.zip`**(Stage1 v4 모델). 어느 쪽이든 직전 제출 대비 Stage3만 변경.
- 기대: LB S3 0.370 → 0.45~0.55.
- **결정 (2026-09-17 20:40): `submit_v5b.zip` 제출** — v4 Stage1(0.4237)이 baseline(0.4046)보다 높으므로 v5b가 v5a를 지배. v4 대비 Stage3만 변경이라 S3 델타 그대로 귀속. 오늘 3번째 슬롯.

### 제출 6: Stage3 = appearance + optical-flow ego-motion (EXP-S3-MOTION-001b) + 31프레임 logit smoothing — **완료, 제출 대기** (2026-09-17 21:50)

- 가설: accel 병목은 frozen ResNet 특징이 카메라 움직임을 못 담아서. Farneback 광류 요약 13-d × (16프레임 mean/std/delta + 지평 16/32/64/128 log-ratio) = 59-d를 head에 concat. 10fps 비공개 대비로 학습에 10fps 시뮬레이션 복제(`--fps-aug`).
- 내부 val 0.571(대조군) → 0.655 (motion) → **0.676** (+영상 내 logit 이동평균 31프레임; 파일 간 통계 없음), official 10fps-sim 0.735 → 0.794 (accel 0.833). 지평 확장(002a/b)은 효과 없음. 상세 [EXPERIMENT_DESIGN.md §17](EXPERIMENT_DESIGN.md).
- `submit_v6.zip`(127MB, MD5 `da039061c17fbf52243b89e48feb2222`; MOTION-001b + SMOOTH-001): Stage1 = v4 모델(제출 4/5b와 동일), Stage2 = baseline, Stage3만 변경 → v5b 대비 S3 델타만 귀속. stage1/2 smoke 출력 v5b와 `cmp` 동일 확인. Release `submissions-2026-09-17`에 16조각으로 업로드.
- 제출 순서: v5b → v6 (둘 다 v4 대비 Stage3만 변경이라 순서가 바뀌어도 각각 귀속 가능). v5a는 폐기.
- 기대: LB S3 0.37 → (v5b 결과 × 1.05~1.15). v5b LB가 나오면 v6 예측치를 갱신.

### 제출 7 (프로브): Stage1 = 컨테이너 서명 규칙 + v4 폴백 (EXP-S1-META-001) — **완료, 제출 대기** (2026-09-17 22:40)

- 실측: 공개 5쌍·샘플 10영상 모두 ORIGINAL = `FMP4(MPEG-4 Part2)/Lavf58.12`, RERECORDED = `h264/Lavf63.1`(비트레이트 2배). 재녹화는 반드시 재인코딩을 거치므로 코덱/먹서 서명이 남는다는 가설 — 1등 S1 0.97이 이것으로 설명될 수 있음.
- 규칙: 코덱 `mp4v` → ORIGINAL, `avc1/hev1/...` → RERECORDED, 애매하면 v4 CNN. 영상별 자기 파일 헤더만 읽음(파일 간 통계 없음). 가이드 §2.3의 "추가 압축 및 화질 저하 / 해상도·fps 변화"와 같은 계열의 단서로 해석하지만, **메타데이터 의존이라는 점은 2차 평가 때 설명이 필요** — 문서화함.
- `submit_v7.zip`(127MB, MD5 `abb81bd4e0673b77bdd71b75db30fe3c`): Stage2/3 = v6와 동일(Stage1만 변경).
- 해석 규칙: S1 ≈ 1.0 → 채택(이후 모든 제출의 Stage1). S1 ≈ 0.24(전부 RERECORDED 판정) 또는 0.40 → 비공개는 서명이 균일, 규칙 폐기.

### 제출 8: Stage2 = 학습 없는 물리 휴리스틱 (EXP-S2-HEUR-001) — **완료, 제출 대기** (2026-09-17 23:20)

- 비공개 Stage2는 프레임 이미지 폴더, 공개 라벨은 t_collision 5개뿐 → 학습 대신 물리 신호: 충돌 = 카메라 수직 jolt(phase-correlation) onset(**공개 5/5**), entry = 충돌 − 0.5s, 회피공간 = 접근 구간 변화 에너지의 중앙집중도(추돌이면 0), 진입방향 = 상대 좌측비율(약함). 상세 [EXPERIMENT_DESIGN.md §18](EXPERIMENT_DESIGN.md).
- `submit_v8.zip`(127MB, MD5 `a4f0c834b6017d0d9de5b22bebfccb40`): Stage1 = v4, Stage3 = v6, Stage2만 변경.
- 기대: S2 0.13 → 0.30~0.45 (total +0.07~0.13). 1등 갭 중 가장 큰 항목.

### 제출 9: Stage3 = MOTION-003(class weight 없음) + prior 보정 + smoothing — **완료, 제출 대기** (2026-09-18 01:10)

- 원인 규명([EXPERIMENT_DESIGN.md §19](EXPERIMENT_DESIGN.md)): 공개 OPEN 5영상이 comma2k19 Chunk_1 route `2018-07-27--06-03-57`(학습셋 포함)이라 그동안의 official-label 로컬 점수는 학습 데이터 평가였고, 주최측 라벨 임계는 accel ±0.35 m/s²·deadzone 4.5°(우리 0.3/3°)라 우리 모델이 소수 클래스를 과예측. class weight가 이를 증폭 → v5b 붕괴.
- v9 Stage3: class weight 제거(MOTION-003) + 공개 라벨 비율로 고정 logit bias(|b|≤1) + 31프레임 smoothing. OPEN 전체 프레임 예측 분포 accel .60/.19/.14/.07, steer .76/.13/.11 (공식 .60/.16/.18/.06, .78/.12/.10).
- `submit_v9.zip`(127MB, MD5 `faa780a2bc54efff90b55bdfaf438a5d`): Stage1 v4, Stage2 v8, Stage3만 변경.
- 기대: S3 0.355(v6) → 0.40+ (candidate2 0.370 초과 목표). 후속 v10 = 재라벨(v2 임계) + OPEN route holdout 학습.

### 제출 10: Stage3 = 라벨 v2 재학습 + OPEN route holdout (EXP-S3-MOTION-004) — **완료, 제출 대기, 1순위** (2026-09-18 01:40)

- 주최측 임계(accel ±0.35, deadzone 4.5°, stopped 0.3)로 291세그먼트 재라벨 → 분포가 공식 비율에 근접(STRAIGHT .74, CONSTANT .67). OPEN route 8세그먼트를 학습/검증에서 완전 제외 → **정직한 official 50행 0.821(native)/0.819(10fps-sim)**, 예측 분포 보정 없이 L1 0.13/0.07.
- `submit_v10.zip`(127MB, MD5 `f45efbd9dbf46537fd5549a8ee518986`): Stage1 v4, Stage2 v8, Stage3만 변경(class weight 없음, motion, smoothing 31, prior 보정 생략).
- 남은 의문: 로컬 0.82 vs LB 최고 0.37. v10의 LB가 여전히 0.4 근처면 비공개는 다른 차량/도메인(Chunk_3 = 다른 차량 다운로드 중) 또는 라벨 정의의 다른 요소(smoothing/시간 정렬)를 의심.

### 제출 11: Stage3 = 두 차량(RAV4+Civic) 학습 (EXP-S3-MOTION-005) — **완료, 제출 대기, 2순위** (2026-09-18 22:10)

- Chunk_3(Civic) 115세그먼트 추가, 라벨 v2 임계, OPEN route holdout, class weight 없음. RAV4 전용 모델의 Civic 점수 0.59~0.61(내부 0.64~0.65) → 차량 시프트 −0.05.
- `submit_v11.zip`(127MB, MD5 `e3ffbe20a40dcefb7210575653689e75`): v10과 Stage3만 다름. held-out official 0.780/0.797, dense 0.770.
- 판단 규칙: LB S3 v11 > v10이면 비공개는 다른 차량 비중이 큼 → Civic 전체(202seg)·추가 청크로 확장. 둘 다 0.4 근처면 재인코딩 파이프라인 재현 실험으로 전환. 상세 [EXPERIMENT_DESIGN.md §20](EXPERIMENT_DESIGN.md).

### 제출 12: Stage2 = CCD 사전정보 반영 휴리스틱 (EXP-S2-HEUR-002) — **완료, 제출 대기** (2026-09-19 18:00)

- 공개 Stage2 5클립 = CCD 000001~005, 충돌 = CCD 첫 사고 프레임(5/5 일치) → 비공개도 CCD일 가능성이 큼. CCD onset은 항상 후반(30~49/50) → 검색을 ≥0.55N으로 제한, 미검출 시 0.72N, 파라미터를 N에 비례 스케일. 상세 [EXPERIMENT_DESIGN.md §21](EXPERIMENT_DESIGN.md).
- `submit_v12.zip`(127MB, MD5 `9c7088e85e691c530cc66309b095853c`): v10과 Stage2만 다름. CCD 라벨로 클립별 학습은 하지 않음(테스트 라벨 사용 회피, 문서 명시).
- 다음: `submit_v13.zip` = v12 + Stage3 RAV4 전체(291→~380seg) 3-seed 앙상블 (빌드 중).

### 제출 13: Stage3 = RAV4 전체(373seg) 3-seed 앙상블 (EXP-S3-MOTION-007) — **완료, 제출 대기** (2026-09-19 18:20)

- v11 결과(Civic 추가 -0.039)에 따라 RAV4(Chunk_1/2)만 전부 추출(291→373seg, OPEN route 제외). seed 3개 학습 후 logit 평균(단일 0.821/0.821/0.735 → 앙상블 0.819, 분산 감소 목적).
- `submit_v13.zip`(207MB, MD5 `4217588cfb95a2397a56698d51b2e728`): v12와 Stage3만 다름.
- 제출 순서 제안: v12(S2 사전정보) → v13(S3 데이터+앙상블). LB 열이 분리돼 있어 순서 무관.

### 그 다음 (예정)
- LB로 v7(S1)·v8(S2) 확인 후 채택분을 합친 v9. Stage2: 사고 시점/진입 방향 라벨이 있는 외부 데이터셋 라이선스 조사 재개(DoTA/CCD/DAD/Nexar). Stage3: 1D temporal conv, backbone 부분 unfreeze.
- Stage1: 보류(v4 0.4237). 재개 조건: 비공개 재녹화 외형에 대한 새 가설(예: 코덱/해상도/fps 재인코딩, 프레임 중복·깜빡임 등 시간축 흔적)이 생겼을 때 1슬롯 프로브.

### 이후 — Stage2 착수

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

### 제출물 전달 (2026-09-17)

- 사용자가 외부에 있어 zip을 원격으로 전달. 이 PC는 ~10MB/10초를 넘는 업로드가 끊기므로(SendUserFile 30MiB 한도, Release 자산 직접 업로드 실패) `split -b 8m`으로 나눠 조각별 재시도 업로드.
- GitHub Release **[submissions-2026-09-17](https://github.com/yoomstergotcha/blackbox-236753-experiments/releases/tag/submissions-2026-09-17)**: `submit_v4.zip`(16조각), `submit_v5b.zip`(16조각), `submit_v5a.zip`(27조각) + `MD5SUMS.txt`. 조각 합계 크기·MD5는 로컬 원본과 일치 확인(v4 `8fd0362d…`, v5b `4d526a27…`, v5a `7686a8fd…`).
- 합치기: Windows `copy /b submit_v4.zip.part* submit_v4.zip` / macOS·Linux `cat submit_v4.zip.part* > submit_v4.zip`, 이후 MD5 대조.
- 제출 순서: v4 먼저 → S1 ≈ 0.40이면 v5a, S1 ≥ 0.6이면 v5b (§ 제출 4·5).
- 부수 수정: 실수로 커밋됐던 `data/external/comma2k19_multi/`(14GB, 11,400파일)를 미푸시 커밋 2개에서 제거하고 `.gitignore`에 추가(원본은 dataset_registry.csv 출처에서 재획득 가능).

### LB 결과 (2026-09-18 00:13) — 제출 5b/6/7/8

- **Stage3 역상관**: candidate2(v2) 0.370 > MOTION+smooth(v6) 0.355 > CLASS-001(v5b) 0.225. 로컬(내부 val / official 50행)은 정반대 순서. v5b는 전부 CONSTANT/STRAIGHT 예측 수준(0.219)에 근접 → 비공개에서 무정보. 공통 의심: class weight(소수 클래스 과예측: official 50행에서도 LEFT+RIGHT 24 vs 정답 11), comma2k19 과적합(도메인/라벨 임계값 차이). **로컬 검증 신뢰도 하락 → 검증 체계 재정비가 최우선.**
- **Stage1 규칙 실패**: 0.393 < 0.4237. 비공개 코덱 서명은 라벨과 무관. v4 유지.
- **Stage2 휴리스틱 성공**: 0.131 → 0.250. 채택.
- 현재 stage별 최고 조합(v4 S1 + v8 S2 + v2 S3) = 0.3328. 다음 제출은 이 조합 + Stage3 재정비분.

### LB 결과 (2026-09-19 16:00) — 제출 9/10/11

- **v10 S3 0.515** (+0.145 vs candidate2). 원인 규명(§19)이 맞았다: class weight 제거 + 주최측 임계 재라벨 + smoothing. v9(라벨 v1 + prior 보정) 0.501 → 라벨 v2가 보정보다 낫고, 보정은 보류.
- **v11(Civic 추가) 0.476 < v10** → 비공개는 RAV4 쪽. Civic 제외, RAV4(Chunk_1/2) 잔여 세그먼트 전부 추가.
- total 0.3907. 가중 갭: S2 0.156 > S1 0.109 > S3 0.107. **Stage2가 최대 갭** — 공개 Stage2 클립이 CCD(Car Crash Dataset)와 동일 형식(10fps·50프레임·5초·러시아 블랙박스)이라 원본 데이터셋 확인 중.

## 7. 대회 규칙·평가 방식 재확인 (2026-09-19, dacon.io 설명/규칙/평가 탭 직접 대조)

| 항목 | 공식 문구(요지) | 우리 제출물 상태 |
|---|---|---|
| 총점 | Stage1 0.2 / Stage2 0.4 / Stage3 0.4 | ✓ `src/eval/metrics.py` 동일 |
| Stage1 | ORIGINAL/RERECORDED Macro-F1 | ✓ |
| Stage2 | "제출된 프레임 번호를 **영상별 프레임–시간 대응정보**로 초 단위 변환 후 정답시각과 비교"; 누락·결측·비수치·음수·범위 초과·허용 외 범주는 오답 | ✓ 파일명 프레임 번호 출력, 범위 내, evasion 0/1·side LEFT/RIGHT. **리스크**: 영상별 fps가 다를 수 있음 → 휴리스틱은 클립 5초 가정으로 N 스케일(HEUR-002) |
| Stage3 | STOPPED 정답 프레임은 조향 평가 제외, 그래도 모든 프레임에 steer_label 출력 | ✓ |
| 1차 평가 | "Private Score(대회 종료 시점의 Public Score의 종합 점수)" | 별도 private split 언급 없음 → 현재 LB 총점이 최종. **최종 제출 선택 기능이 있는지 제출 탭에서 확인 필요** |
| 2차 평가 | 상위 15팀: 모델 개발 보고서 + 학습데이터 구성 보고서 제출·검증 → 상위 7팀 | 준비물: dataset_registry.csv, model_registry.md, EXPERIMENT_DESIGN.md가 원자료. 보고서 초안 작성 예정 |
| 외부 데이터·사전학습 | "사용에 법적 제한이 없는 모든 방법론 가능", 라이선스 참가자 확인, 2차 평가 시 출처 기술 | comma2k19 MIT ✓, CCD MIT ✓, ImageNet ResNet18(torchvision BSD-3; 베이스라인 zip이 동일 가중치 포함 → 주최측 허용으로 판단) ✓ |
| 비공개 평가 데이터 | 추가 학습·튜닝·Pseudo-Labeling 금지 | ✓ 전혀 사용 안 함. CCD/comma2k19가 비공개와 겹칠 가능성은 있으나 공개 데이터셋 자체 사용은 규칙상 허용. CCD는 클립별 라벨 학습을 하지 않고 분포 사전정보만 사용(문서 §21) |
| 파일 간 정보 공유 | "다른 파일 샘플의 정보·예측값·통계 활용 금지", 한 파일을 세그먼트로 나눠 종합은 허용 | ✓ 모든 Stage가 파일별 독립(z-score·smoothing은 영상 내부만). prior bias·threshold는 빌드 시 상수 |
| 실행 환경 | L40S, 60분, 오프라인, 읽기 model/·data/, 쓰기 output/만 | ✓ ~13분, 인터넷 접근 없음, 파일 쓰기 없음 |
| 메타데이터 | 코드 제출 가이드: 제공 구조 밖의 파일·메타데이터 접근 불가 | v7의 컨테이너 헤더 읽기는 제공된 영상 파일 내부라 위반은 아니나 회색지대 + 실패 → **폐기 상태 유지** |
| 제출 한도 | 1일 3회, Python | ✓ |
| 실격 | 코드 제출 기능을 악용한 평가 데이터 유출 시도 | 해당 없음 (출력 외 기록·전송 없음) |

가이드 문서(COMPETITION_GUIDE.md)에 있는 세부 수치(허용오차 0.3s, Stage2 내부 가중치 0.35/0.35/0.15/0.15, Stage3 0.7/0.3)는 평가 탭이 아니라 [필독] 공지·토크 출처이므로 **공지 원문 재확인 권장**(토크 게시판은 로그인/렌더링 문제로 자동 조회 불가).

### LB 결과 (2026-09-20) — 제출 12/13
- **v12 S2 0.175 (−0.075)**: CCD 후반부 제한이 해로움 → 비공개 Stage2 클립은 50프레임 트림이 아니라 충돌이 앞쪽에도 있는 더 긴 구간. HEUR-002 폐기, Stage2는 v8(HEUR-001, 0.250)로 원복.
- **v13 S3 0.540 (+0.025)**: RAV4 전체 + 3-seed 앙상블 채택.
- stage별 최고 조합(v4 S1 + v8 S2 + v13 S3) = **0.4008**. 사용자 목표(0.959/0.641/0.798 ≈ 현재 1등) 대비 갭: S1 0.107, S2 0.156, S3 0.103 (가중).
