

# ---------------------------------------------------------------------------
# EXP-S1-META-001: 컨테이너 서명 규칙 + CNN(v4) 폴백  (이 블록은 predict_stage1_synth.py 뒤에 이어붙여
# predict_stage1_meta.py 를 만든다 — build_stage1_meta_snippet.py 참고)
# 공개 5쌍/샘플 10영상 실측: ORIGINAL은 전부 MPEG-4 Part2(FMP4/mp4v, Lavf58.12), RERECORDED는 전부
# H.264(avc1, Lavf63.1, 비트레이트 2배). 재녹화(재촬영이든 재인코딩이든)는 반드시 다시 인코딩을 거치므로
# 코덱/먹서 서명이 남는다 — 가이드 §2.3의 "추가 압축 및 화질 저하 / 해상도·fps 변화"와 같은 계열의 단서.
# 영상마다 독립적으로 그 파일의 헤더만 읽는다(파일 간 통계 없음, 가이드 §12.2 준수).
# ---------------------------------------------------------------------------


def _s1_container_signature(path: Path):
    """(codec, lavf) — codec: 'mp4v'|'avc1'|'hev1'|'hvc1'|...|'unknown', lavf: 'Lavf58.12.100' 같은 먹서 태그 또는 None"""
    try:
        data = path.read_bytes()
    except Exception:
        return "unknown", None
    codec = "unknown"
    for tag in (b"mp4v", b"avc1", b"avc3", b"hev1", b"hvc1", b"av01", b"vp09"):
        if data.find(tag) >= 0:
            codec = tag.decode()
            break
    if codec == "unknown":
        try:
            cap = cv2.VideoCapture(str(path))
            f = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            cc = "".join(chr((f >> 8 * i) & 255) for i in range(4)).strip().lower()
            codec = {"fmp4": "mp4v", "mp4v": "mp4v", "h264": "avc1", "avc1": "avc1", "hevc": "hev1", "hvc1": "hvc1"}.get(cc, "unknown")
        except Exception:
            pass
    k = data.find(b"Lavf")
    lavf = data[k : k + 16].split(b"\x00")[0].decode("latin1", errors="replace") if k >= 0 else None
    return codec, lavf


def _s1_rule(codec: str, lavf):
    """확정 판정만 반환. 애매하면 None -> CNN 폴백."""
    if codec == "mp4v":
        return "ORIGINAL"  # 원본 배포 규격(MPEG-4 Part2). 재녹화가 이 코덱으로 다시 인코딩될 가능성은 사실상 없음
    if codec in ("avc1", "avc3", "hev1", "hvc1", "av01", "vp09"):
        return "RERECORDED"  # 원본 규격이 아닌 현대 코덱 = 재인코딩을 거침
    if lavf and not lavf.startswith("Lavf58"):
        return "RERECORDED"
    if lavf and lavf.startswith("Lavf58"):
        return "ORIGINAL"
    return None


def predict_stage1(data_dir, model_dir):
    cnn = _predict_stage1_cnn(data_dir, model_dir)  # 폴백용(모든 영상에 대해 계산, v4와 동일)
    cnn_map = dict(zip(cnn["ID"], cnn["answer"]))
    rows = []
    for path in _video_paths(Path(data_dir) / "videos"):
        codec, lavf = _s1_container_signature(path)
        verdict = _s1_rule(codec, lavf)
        rows.append({"ID": path.stem, "answer": verdict if verdict is not None else cnn_map.get(path.stem, "ORIGINAL")})
    return pd.DataFrame(rows, columns=["ID", "answer"])
