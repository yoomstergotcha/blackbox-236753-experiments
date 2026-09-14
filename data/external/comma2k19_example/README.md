# comma2k19 실제 예제 세그먼트 (검증용)

- 출처: https://github.com/commaai/comma2k19 (`Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/`)
- 다운로드일: 2026-09-13
- 용도: `src/data/comma2k19/` 파이프라인의 timestamp 동기화·라벨 변환 로직을 실제 데이터로 검증하기 위한 소규모 샘플. **전체 33시간 comma2k19 데이터셋이 아니라 공식 저장소에 데모용으로 커밋되어 있는 1개 세그먼트(약 60초)다.**

## 라이선스

`commaai/comma2k19` GitHub 저장소 루트의 `LICENSE` 파일은 **MIT License**다. 전체 33시간/2019 세그먼트 데이터셋(Academic Torrents `65a2fbc964078aff62076ff4e103f18b951c5ddb`, HuggingFace `commaai/comma2k19`)도 동일하게 **MIT**로 확인됐다 — 세 소스(GitHub LICENSE, Academic Torrents 페이지, HF dataset card YAML `license: mit`)가 모두 일치한다.

이 문서는 이전 버전에서 전체 데이터셋을 "CC BY-NC-SA 3.0"으로 잘못 기록한 적이 있다 — 그건 이름이 비슷한 **다른** comma.ai 데이터셋("comma.ai driving dataset", 2016년, dog/emily/frodo 드라이버, HDF5, 7시간 분량, Academic Torrents `58c41e8bcc8eb4e2204a3b263cdf728c0a7331eb`)을 comma2k19로 착각해 인용한 것이었다. 재확인 후 정정했다 — [dataset_registry.csv](../../../dataset_registry.csv)도 함께 정정함.

## 실측 데이터 특성 (이 세그먼트 기준)

- `processed_log/CAN/speed/{t,value}`: 4974 샘플, ~83Hz(불균일), value shape은 `(4974, 1)` — 2차원이라 코드에서 반드시 squeeze 필요
- `processed_log/CAN/steering_angle/{t,value}`: 4974 샘플, value shape `(4974,)`, 범위 -4.6~+2.5도 (이 세그먼트는 고속도로 직선 위주 구간)
- `global_pose/frame_times`: 1200 샘플, ~20.02fps, 전체 구간 ~59.95초
- CAN speed/steering_angle과 frame_times는 모두 "device boot time(초)" 기준 같은 클럭 — 값 범위가 46408~46468초로 서로 겹침을 확인함
- speed/t, steering_angle/t는 서로 다른 배열이며 값도 정확히 일치하지 않음 → 반드시 timestamp 기준 보간으로 동기화해야 하며 인덱스로 zip하면 안 됨 (실측으로 확인)
