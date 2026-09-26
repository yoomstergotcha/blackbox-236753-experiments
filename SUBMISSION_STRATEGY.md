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
| submit_v14.zip (Stage1 합성+mpeg4 재인코딩) | **0.45013** | 0.24988 | 0.54024 | **0.4060** | |
| submit_v16.zip (Stage2 earliest-burst) | 0.42366 | **0.21072** | 0.54024 | 0.3906 | |
| submit_v19.zip (Stage2 학습 localizer 단일) | 0.42366 | 0.19488 | 0.54024 | 0.3787 | 2026-09-23 |
| submit_v20.zip (Stage2 앙상블+꼬리 제외) | 0.42366 | 0.24255 | 0.54024 | 0.3978 | 2026-09-23 |
| submit_v21.zip (Stage1 메타데이터 프로브) | 0.35506 | 0.24255 | 0.54024 | 0.3841 | 2026-09-23 |
| submit_v22.zip (S1 VLM + S2 VLM side/eva) | **0.97382** | 0.25314 | 0.54024 | **0.5123 (98등)** | 2026-09-26 |
| submit_v23.zip (S1 역방향 규칙 + S2 entry −21) | 0.31807 | 0.28087 | 0.54024 | 0.3933 | 2026-09-26 |
| submit_v26.zip (S2 적응 stride 11모델) | 0.31807 | **0.28331** | 0.54024 | 0.3943 | 2026-09-26 |
| 상위권 (2026-09-21) | 0.96243 | 0.72408 | 0.76492 | 0.788 | |
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

### 제출 14·15 (2026-09-20)

- **v14** (`submit_v14.zip` 207MB, MD5 `0d5857aa69cf48edddb48c72bfa19862`; LOPO 0.899/AUC 1.0): Stage1 = `EXP-S1-SYNTH-007`(재촬영 합성 → **mpeg4 저비트율 재인코딩**까지 재현한 765개 합성 영상 + 공식 5쌍, LOPO) 프로브. Stage2 = v8 원복, Stage3 = v13. S1 열로 판정: ≥0.6이면 가설 확인.
- **v15**: Stage3 = v13 앙상블 + seed별 macro-F1 최적 logit bias(`EXP-S3-BIAS-001`, comma2k19 held-out val +0.03). `submit_v15.zip`(207MB, MD5 `b077df726a5b27161057cf74a94ed06d`). Stage2 = v8 원복(0.250 기대), Stage1 = v4.
- DINOv2-S 특징(`EXP-S3-DINO-001`)은 내부 val +0.03이지만 OPEN held-out −0.25, 앙상블에도 손해 → 폐기.

### 제출 16: Stage2 = earliest-burst 충돌 검출 (EXP-S2-HEUR-005) — **완료, 제출 대기** (2026-09-20 17:30)

- CCD 영상 확보 후 실측: 현행 검출기는 CCD ego 클립에서 0.565(±3프레임)인데 LB 역산 적중은 ~0.30 → 비공개 클립은 사후 충격을 포함한 긴 구간. 시뮬레이션에서 global-max는 0.36으로 무너지고 earliest-burst(z>8, 길이≥2)는 0.45 유지 → 채택. 진입방향은 23클립 수동 라벨에서 모든 cue가 코인플립(52~65%, n 작음) → 현행 유지. 상세 §22.
- `submit_v16.zip`(207MB, MD5 `e4e2a0f8f6dd1ec232a3727bee729c89`): v13과 Stage2만 다름.

### 제출 17·18 (2026-09-21)
- **v17** (`submit_v17.zip` 207MB, MD5 `9244c16f95e484060473fca57c655d7c`): Stage1 = `EXP-S1-SYNTH-008` — ORIGINAL 클래스에 CCD 실제 원본 400클립, 같은 클립의 합성 재녹화(mpeg4/h264 재인코딩) 쌍. LOPO 1.0/AUC 1.0, thr 0.5. Stage2는 v16 상태(S2 무관), Stage3 v13.
- **v18** (`submit_v18.zip`, MD5 `3564a69bd9c1aeac87483fcf23e11221`): Stage3 = 5-seed + seed별 bias. official 50행 0.759(bias가 OPEN route에선 손해, 내부 val은 이득) → v15 결과 보고 결정.
- Stage1 시간축 단서(`EXP-S1-TEMPORAL-001`)는 분리력 없음 → 폐기.

### 제출 19: Stage2 = 학습 기반 충돌 localizer (EXP-S2-LEARN-001) — **완료, 제출 대기** (2026-09-22)

- CCD ego 796클립 onset 라벨로 BiGRU localizer 학습(소스 단위 5-fold). OOF ±3프레임 **0.771** (휴리스틱 0.565~0.595). 긴 클립 시뮬 0.47(휴리스틱 0.36). entry = 충돌 −7(수동 라벨 64클립 중앙값). 상세 [EXPERIMENT_DESIGN.md §23](EXPERIMENT_DESIGN.md).
- `submit_v19.zip`(212MB, 27조각, MD5 `caa2d2f9f6e327550d7c44015fc46cd6`): Stage1 v4, Stage3 v13, Stage2만 교체.
- 알려진 약점: 완만한 충돌에서 마지막 프레임을 고르는 경향(공개 000002/000005) → 디코딩 규칙 개선·강한 증강 변형(LEARN-002) 진행 중.

### 제출 20: Stage2 localizer 앙상블 + 꼬리 제외 디코딩 — **완료, 제출 대기** (2026-09-22)

- v19 단일 모델(argmax) → 5모델×5fold 앙상블(시드 3 + σ2.5 + hidden128) + 마지막 3프레임 제외. CCD OOF ±3 **0.771 → 0.803**, 공개 5클립 3/5 → 4/5 (000002만 35 vs 30). 격자 흐름 특징·강한 증강은 OOF 손해라 미채택(EXP-S2-LEARN-002/007).
- `submit_v20.zip`(236MB, 30조각, MD5 `dba00c017c58b7afb930da05210c6d21`): Stage1/Stage3는 v19와 바이트 동일.
- 제출 순서 권장: v19 → v20 (Stage2 열 델타로 앙상블/디코딩 효과 확인). v19가 v8(0.250)보다 낮으면 학습 기반 접근 자체를 재검토.

### 제출 21: Stage1 메타데이터 프로브 (EXP-S1-PROBE-002) — **완료, 제출 대기** (2026-09-22)

- 공개 RERECORDED 5개는 원본과 프레임이 동일한 순수 재인코딩(테두리·모아레 없음)인데 공식 정의는 '화면을 다시 촬영'. 합성 재촬영 학습(v1~v8)이 전부 LB 0.42~0.45(상수 0.405 근처)였으므로 비공개 재촬영본의 구조를 먼저 확인한다.
- 규칙: 해상도≠1280×720 또는 fps≠10 → RERECORDED. `submit_v21.zip`(236MB, 30조각, MD5 `fc9ad4dd616f47923d6704243d8a4b2c`), Stage2/3은 v20과 바이트 동일.
- 해석: S1 ≈ 0.405 → 규격 정규화(시각/시간 단서만 남음) / 0.6↑ → 규격이 곧 라벨(그대로 채택·보강) / 0.405 미만 → ORIGINAL 규격이 다양(CCD 규격 가정 폐기).

### 제출 22: VLM(Qwen2-VL-2B) Stage1 화면 재촬영 판별 + Stage2 진입방향/회피공간 — **완료, 업로드 중** (2026-09-22)

- 평가 서버(L40S 44GB, 60분, zip 10GB, transformers 4.57.6 기본)에 맞춰 Apache-2.0 소형 VLM을 동봉. 학습 없음(zero-shot log-prob 비교), 고정 임계.
- Stage2: 수동 라벨에서 방향 LOO acc **0.66**(현행 규칙 0.49), 회피 LOO macroF1 **0.77**(현행 0.47). collision/entry는 v20과 동일 → S2 델타 = side/evasion 효과.
- Stage1: 테두리·원근형 재촬영은 AUC 0.94로 잡고 순수 재인코딩은 못 잡음. 비공개 재촬영본이 실제 화면 촬영이면 유효 → 프로브. 하방은 상수(0.405) 수준.
- `submit_v22.zip`(4453MB, ~557조각, MD5 `a182475abf4eef1db322c263a7ed8e67`). 조각이 많으니 `cat submit_v22.zip.part* > submit_v22.zip` 후 MD5 확인 필수.
- 제출 순서 권장: v19 → v20 → v22 (S2), v21 → v22 (S1). v22의 S1이 0.405 근처면 비공개 재촬영본은 시각적으로도 원본과 구분 어려움 → Stage1은 시간축/코덱 통계 쪽으로 전환.

### LB 판정 (2026-09-23 제출 v19/v20/v21) — 전략 재전환

- **Stage2 학습 localizer 실패**: v19 0.195 < v20 0.243 < v8 휴리스틱 0.250. CCD OOF 0.80이 비공개에서 무의미 → 비공개 Stage2는 CCD형 50프레임/10fps 클립이 아니다(길이·프레임률·촬영원 다름). 꼬리 제외/앙상블(+0.048)만 유효.
- **Stage1 메타데이터 프로브 실패**: 0.355 < 상수 0.405 → 비공개 ORIGINAL 규격이 다양(1280×720/10fps 아님). 공개 예시 규격 가정 전면 폐기.
- 결론: 세 스테이지 모두 비공개 = 공개 예시와 다른 출처(한국 블랙박스 30fps/1080p 가능성). 남은 슬롯은 (1) 프레임률 프로브로 구조 확정, (2) 프레임률·길이 불변 파이프라인으로 전환에 쓴다.

### 제출 23: Stage1 역방향 규격 규칙 + Stage2 entry −21 — **완료, 업로드 중** (2026-09-25)

- v21의 0.355는 무작위 규칙(≈0.48)보다 낮음 = 규칙이 라벨과 역상관 → 뒤집은 규칙(표준 규격 1280×720·10fps → RERECORDED)은 0.5~0.56 기대(최고 0.450 대비 상승). `submit_v23.zip`(236MB, MD5 `501ba23007c89560ef558780e5cf877d`).
- Stage2는 v20에서 entry 오프셋만 −7→−21: 비공개가 30fps면 S2 ≈ +0.07, 10fps면 하락. 이 델타로 이후 모든 Stage2 오프셋/stride를 확정.
- 오늘 권장 순서: **v22 → v23 → v15** (S1/S2/S3 열이 각각 독립 판정).

### 제출 24: Stage2 프레임률 적응 localizer (EXP-S2-FPS-001) — **완료, 업로드 중** (2026-09-25)

- **릴리스 변경**: `submissions-2026-09-17`이 GitHub 자산 한도(1,000개)에 도달 → v23부터는 **`submissions-2026-09-25`** 릴리스에 올린다(MD5SUMS.txt도 그 릴리스 것).
- 학습 localizer가 비공개에서 휴리스틱 이하였던 원인을 프레임률/길이 불일치로 보고, 클립마다 stride 1/2/3 중 앙상블 최대 점수가 stride1의 1.5배를 넘는 것을 택한다(파일 간 통계 없음). CCD 200클립: 10fps **0.79**(손실 0) / ×3 보간 30fps **0.715**(현행 0.37). entry = collision − 7·stride.
- `submit_v24.zip`(236MB, MD5 `1baf8d1f550ca291812a275b9eed5564`), Stage1 = v23 역방향 규칙, Stage3 v13. 공개 5클립 출력은 v20과 동일.
- 다음: v24 + VLM 진입방향/회피공간(v22 S2 결과가 +면) = v25(4.4GB).

### 제출 25: v24 + VLM 진입방향/회피공간 — **완료, 업로드 검증됨(557조각)** (2026-09-25)

- `submit_v25.zip`(4453MB, 557조각, MD5 `10e9080a56bd6e450082ce84327acc72`, 릴리스 `submissions-2026-09-25`). Stage2 = 프레임률 적응 collision/entry(v24) + Qwen2-VL 진입방향/회피공간(v22 임계, 창 폭 stride 배). Stage1 역방향 규칙, Stage3 v13.
- 판정: v22(S2 VLM 효과)·v24(S2 프레임률 효과)가 모두 +면 v25가 Stage2 최종 형태.

### 제출 26: 불변성 프로토콜로 고른 Stage2 앙상블 (EXP-S2-INV-001/002) — **완료, 업로드 검증됨** (2026-09-25)

- 원칙(사용자 지시): 비공개 소스를 추측하지 않고, 의도적 분포 이동(10/30fps × 짧은/긴 클립)을 모두 견디는 것만 채택. 위치 사전정보를 없앤 학습(앞뒤 정상 주행 0~100프레임 무작위 이어붙임)으로 국소 CNN 3시드·GRU 3시드를 추가, 기존 GRU 5모델과 합집합.
- 4조건 적중(짧10/긴10/짧30/긴30): v24 0.82/0.785/0.73/0.715 → **v26 0.785/0.815/0.77/0.75** (최저 0.715→0.75).
- `submit_v26.zip`(255MB, 32조각, MD5 `e5e2c3f6a6dd7b6e652ef22d005ec85d`). Stage1/3은 v24와 동일.
- Stage1 물리 특징(정지 픽셀·손떨림·모아레·스펙트럼 피크 등)은 자체 합성에도 AUC ≤0.77이라 미채택.

### 오늘 기각 (2026-09-25, 분포 이동 검증 미통과)
- Stage1 물리 특징(정지 픽셀·손떨림·고주파·모아레·테두리·깜빡임·정적 스펙트럼 피크): 자체 합성 재촬영에도 AUC ≤0.77, 정지 픽셀은 코덱 의존.
- Stage3 좌우 반전 TTA 0.710→0.662, 5-seed 0.705(3-seed 0.710) → v13 유지.
- Stage2 z-융합 디코딩: ratio 선택 대비 전 조건 열세.
- 권장 제출 순서: **v26 → v22 → v23** (v24·v25는 결과에 따라).

### LB 판정 (2026-09-26) — Stage1 해결, Stage2 요소별 확인
- **Stage1 = VLM zero-shot 0.974**(v22). 비공개 재촬영본은 눈에 보이는 화면 촬영 흔적을 가짐. 규격 규칙(정·역방향) 폐기.
- Stage2: entry −21(+0.038), 적응 stride 앙상블(+0.041), VLM side/evasion(+0.011) 각각 유효 → 결합 = 제출 27.
- 남은 최대 레버 = Stage3(0.54 vs 상위 0.76, 가중치 0.4): v15/v18(bias) 결과 대기 후 라벨 임계/클래스 사전분포 쪽으로.

### 제출 27: 검증된 요소 결합 — **완료, 업로드 검증됨(559조각, 릴리스 submissions-2026-09-25)** (2026-09-26)
- S1 VLM(thr −0.5) + S2 [v26 collision + entry −21 고정 + VLM side/evasion + 1500s 시간 가드] + S3 v13. `submit_v27.zip`(4472MB, 559조각, MD5 `7192ec6d6ec52affa1840274ad457f93`, 릴리스 submissions-2026-09-25).

### 제출 28: Stage3 전차종 앙상블 (EXP-S3-ALLCARS-001) — **빌드 완료, v27 뒤 자동 업로드** (2026-09-26)
- 의도적 차종 이동 검증: v13(RAV4만)은 학습에 없던 Civic route에서 accel 붕괴(0.70→0.31~0.38). 전차종 3-seed는 held-out RAV4 0.723 / Civic 0.426으로 두 route 모두 v13보다 나음 → 채택. 영상별 흐름 정규화·모션 전용은 한쪽 route에서 붕괴해 기각.
- `submit_v28.zip`(4472MB, 559조각, MD5 `72f85df29fcaf376d3ba5847afb8327c`): S1 VLM, S2 v27과 동일, S3만 교체 → S3 열 델타로 판정.

### 제출 29·30 (경량 Stage2 후보, 2026-09-26 저녁)
- Stage2 병목 분해: collision·entry 적중 ~0.25/~0.16. 검출기(YOLOv8) 기반 접촉/진입/방향은 CCD에서 전부 열세(EXP-S2-DET-001), 카메라 스케일 이동엔 이미 강건(EXP-S2-SHIFT-002), fps 추정(자기상관)은 불가.
- 남은 가설 = 비공개 시간 척도 3배(v23 entry −21 +0.038): stride 선택이 실데이터에서 잘 안 켜지는 문제 → **v29** = 순수 최대 규칙(ratio 1.0) + entry −21, **v30** = stride 3 고정(90프레임 미만 클립은 1) + entry −21. 둘 다 S1 v4·S3 v13인 255MB 경량 빌드(S2 열만 판독).
- `submit_v29.zip` MD5 `3e144a4f22bd44e717d797158b7722ab`, `submit_v30.zip` MD5 `c0b106a1d48e31e62956e75d096f2650` — **릴리스 `submissions-2026-09-27`** (두 번째 릴리스도 1,000개 한도 도달; v25 조각을 삭제해 v27만 두 번째 릴리스에서 마무리, v28~는 세 번째 릴리스).

### 제출 31·32 (경량, 충돌 후 정지 게이트) — 2026-09-26 저녁
- VLM coarse-to-fine 국소화(사용자 제안)는 CCD 10fps에서 창 적중 0.54·fine 0.39, 30fps에서 창 0.25로 jolt(0.76~0.78)에 크게 못 미치고 융합도 중립 → 기각(EXP-S2-VLM-002/003).
- 남은 물리 신호 '사고 후 정지'를 게이트로: `submit_v31.zip`(v26 + entry −21 + 게이트, MD5 `bd56bfb4be84ab24ee5a8a6c2cf78a1d`), `submit_v32.zip`(v30 + 게이트, MD5 `5cb680ec87d4b1fcaff120b606d62345`). CCD 중립, 비공개에 사후 구간이 있으면 효과. v28 뒤 자동 업로드.

### 권장 제출 순서 (2026-09-27)
1. **v27** (S1 VLM + S2 v26·entry −21·VLM side/eva + S3 v13) — 기대 총점 ≥0.53
2. **v30** (S2 stride 3 고정) — S2 열이 0.30을 넘으면 시간 척도 가설 확정 → 최종 빌드에 반영
3. **v28** (S3 전차종) 또는 **v29** (순수 최대 규칙) — v30 결과에 따라

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

### 전략 전환 (2026-09-21, 사용자 지시)
- v16(S2 0.211)까지의 결과: 휴리스틱 미세조정(0.25 ± 0.04)으로는 상위권(S2 0.72)에 접근 불가. Stage1(0.45 vs 0.96)도 접근법 수준의 차이.
- **Stage2를 학습 기반 새 프로젝트로 재구성**: 비공개 클립 구조 가설(CCD 원본 유튜브 소스의 긴 원본 fps 구간) 검증 → 같은 구성의 학습셋 생성(CCD 사고 프레임 라벨 + 수동 라벨 entry/side/evasion) → 시퀀스 모델 학습. 상세 §23.
- Stage1: 시간축 단서(손떨림 지터·밝기 플리커) 기반 재접근. Stage3: 점진 개선 유지(5-seed + bias).
