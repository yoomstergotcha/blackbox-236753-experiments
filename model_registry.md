# Model Registry

2차 평가 진출 시 사용한 모든 pretrained weight의 출처와 라이선스를 기술해야 한다 (COMPETITION_GUIDE.md §11).

| model | source / url | license | usage_scope | stage | notes |
|---|---|---|---|---|---|
| mvit_v2_s (torchvision, weights=None) | torchvision.models.video | BSD-3-Clause (torchvision) | 구조만 사용, 가중치는 자체 학습 | stage1, stage3 | 베이스라인 노트북 기준, ImageNet pretrained 미사용 |
| resnet18 (ImageNet1K_V1) | torchvision.models | BSD-3-Clause (torchvision) / ImageNet 사용조건 확인 필요 | frame feature extractor | stage2 | submit.zip에 가중치 파일 포함 필수 (인터넷 다운로드 금지) |
