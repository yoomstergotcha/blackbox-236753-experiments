# Model Registry

2차 평가 진출 시 사용한 모든 pretrained weight의 출처와 라이선스를 기술해야 한다 (COMPETITION_GUIDE.md §11).

| model | source / url | license | usage_scope | stage | notes |
|---|---|---|---|---|---|
| mvit_v2_s (torchvision, weights=None) | torchvision.models.video | BSD-3-Clause (torchvision) | 구조만 사용, 가중치는 자체 학습 | stage1, stage3 | 베이스라인 노트북 기준, ImageNet pretrained 미사용 |
| resnet18 (ImageNet1K_V1) | torchvision.models | BSD-3-Clause (torchvision) / ImageNet 사용조건 확인 필요 | frame feature extractor (stage2: baseline 그대로; stage3: frozen backbone + 자체 학습 MLP head; stage1: 초기화 후 fine-tune) | stage1, stage2, stage3 | submit.zip에 가중치 파일 포함 필수 (인터넷 다운로드 금지) |
| Farneback optical flow (OpenCV `cv2.calcOpticalFlowFarneback`) | opencv-python-headless | Apache-2.0 (OpenCV) | 알고리즘만 사용, 학습 가중치 없음 | stage3 | EXP-S3-MOTION-001: 160x120 gray 프레임 간 흐름 요약 13-d → head 입력 (제출 6부터) |
