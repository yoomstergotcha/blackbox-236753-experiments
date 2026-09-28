# Model Registry

2차 평가 진출 시 사용한 모든 pretrained weight의 출처와 라이선스를 기술해야 한다 (COMPETITION_GUIDE.md §11).

| model | source / url | license | usage_scope | stage | notes |
|---|---|---|---|---|---|
| mvit_v2_s (torchvision, weights=None) | torchvision.models.video | BSD-3-Clause (torchvision) | 구조만 사용, 가중치는 자체 학습 | stage1, stage3 | 베이스라인 노트북 기준, ImageNet pretrained 미사용 |
| resnet18 (ImageNet1K_V1) | torchvision.models | BSD-3-Clause (torchvision); ImageNet 가중치는 공식 베이스라인 zip(model/stage2/resnet18-f37072fd.pth)이 동일하게 포함 → 주최측 허용으로 판단(2026-09-19 규칙 재확인) | frame feature extractor (stage2: baseline 그대로; stage3: frozen backbone + 자체 학습 MLP head; stage1: 초기화 후 fine-tune) | stage1, stage2, stage3 | submit.zip에 가중치 파일 포함 필수 (인터넷 다운로드 금지) |
| Farneback optical flow (OpenCV `cv2.calcOpticalFlowFarneback`) | opencv-python-headless | Apache-2.0 (OpenCV) | 알고리즘만 사용, 학습 가중치 없음 | stage3 | EXP-S3-MOTION-001: 160x120 gray 프레임 간 흐름 요약 13-d → head 입력 (제출 6부터) |
| Qwen2-VL-2B-Instruct | Hugging Face `Qwen/Qwen2-VL-2B-Instruct` (snapshot 2026-09-22, data/external/hf_models/) | Apache-2.0 (모델 카드·LICENSE 파일 확인) | zero-shot 질의(학습 없음): Stage2 진입방향(LEFT/RIGHT)·회피공간(YES/NO) log-prob 비교, Stage1 '화면 재촬영' 판별 후보 | stage2 (검증 중), stage1 (후보) | 4.4GB bf16, 평가 서버 transformers 4.57.6 기본 탑재로 requirements 추가 없음. 가중치 전체를 submit.zip에 동봉(model/stage2/qwen2vl/). 로컬 검증: output/vlm_side_eval_*.csv |
| YOLOv8s (COCO) | Ultralytics `yolov8s.pt` (data/external/hf_models/yolo/, ultralytics 8.3.170) | AGPL-3.0 (Ultralytics; 가중치·라이브러리 모두) | 차량 검출(car/motorcycle/bus/truck) → 충돌 직전 8프레임 피해차량 트랙 횡위치로 entry_side 판정(EXP-S2-SIDEEVA-001, 제출 41부터). 학습 없음 | stage2 | 22.6MB, model/stage2/yolov8s.pt 동봉. 평가 서버에 ultralytics 8.3.170 기본 탑재. 오프라인 캐시: output/stage2_ccd_det |
