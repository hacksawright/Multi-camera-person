# Onboarding codebase + Kaggle Notebook

Tài liệu này gồm hai phần:

1. **Mentor walkthrough** — đọc codebase từ kiến trúc đến matching.
2. **Kaggle Notebook** — các Markdown/Code cell copy/paste (không `git clone`; unzip ZIP + video của bạn).

Ngôn ngữ: giải thích tiếng Việt. Code và comment tiếng Anh.

> **Lưu ý kiến trúc:** đây **không** phải MTMCT cổ điển (không graph matching đồng thời mọi camera). Hệ thống xử lý **từng video tuần tự**, chia sẻ một `PersonMemory` + SQLite. ID xuyên camera là **Pxxx** (anonymous) hoặc **EMPxxx** (gallery). Body ReID **không** được tự đổi employee ID.

---

# PHẦN A — ĐỌC CODEBASE

## A1. Cấu trúc thư mục

Repo khá gọn: pipeline nằm ở root + `src/`. Không có `configs/*.yaml` kiểu YOLO-project lớn; cấu hình tracker BoT-SORT là YAML, còn identity/ReID là **CLI argparse** trong `src/infer_engine.py`.

```text
person_identity_tracking/
├── infer.py                    # Entry point mỏng: gọi src.infer_engine.main()
├── check.py                    # Kiểm tra CUDA, ORT, third_party, weights, data
├── analyze_run.py              # Tóm tắt thống kê một thư mục runs/
├── build_identity_db.py        # Tiện ích DB danh tính (không phải entry inference)
├── tracker_botsort_reid.yaml   # Profile Ultralytics BoT-SORT + appearance
├── requirements.txt            # Deps; cố ý KHÔNG pin torch (tái sử dụng venv/Kaggle)
├── setup.ps1 / run_video.ps1 / run_epfl.ps1 / download_*.ps1  # Windows local only
├── gallery/                    # Ảnh enroll EMPxxx (gitignored trừ README)
├── models/                     # Weights (gitignored) — YOLO, OSNet, AdaFace, SCRFD
├── data/                       # Video test (gitignored)
├── runs/                       # Output inference (gitignored)
├── third_party/                # AdaFace + deep-person-reid (gitignored, clone lúc setup)
├── tests/test_regressions.py
└── src/
    ├── infer_engine.py         # Orchestrator: track loop, IO, post-fusion, render
    ├── models.py               # Dataclasses: TrackKey, FaceObservation, TrackSummary
    ├── identity_manager.py     # State machine EMPxxx trên 1 local track
    ├── person_memory.py        # Pxxx memory, face authority, short-gap body
    ├── anonymous_face_recognizer.py  # Open-set rank face vs P banks
    ├── gallery.py              # Build/match gallery AdaFace
    ├── embedding_store.py      # NPZ + metadata.json raw embeddings
    ├── identity_database.py    # SQLite observations/assignments/search
    ├── evidence_fusion.py      # Face+body fusion sau khi hết camera
    ├── reid_embedder.py        # OSNet via torchreid FeatureExtractor
    ├── adaface_quality.py      # AdaFace IR50 embedder
    ├── face_detector.py        # Wrapper SCRFD
    ├── scrfd_runtime.py        # ONNXRuntime SCRFD
    ├── head_roi.py             # Face trong person box
    ├── face_quality.py / face_utils.py / face_temporal_tracker.py
    ├── prototypes.py           # L2-normalize, cosine, face/body prototypes
    └── head_roi.py
```

### Ánh xạ module MTMCT “sách giáo khoa” → file thật

| Module sách | Trong repo này | File |
|---|---|---|
| Object Detection | YOLO11s, class `person` (COCO 0) | `ultralytics.YOLO` trong `src/infer_engine.py` (`model.track`, `--yolo`) |
| Single-camera tracking (SCT) | BoT-SORT + ReID appearance (Ultralytics) | `tracker_botsort_reid.yaml` + `YOLO.track(..., tracker=...)` |
| Feature extraction body | OSNet x1.0 MSMT17 | `src/reid_embedder.py` |
| Feature extraction face | AdaFace IR50 WebFace4M | `src/adaface_quality.py` |
| Face detect | SCRFD 10G ONNX | `src/face_detector.py`, `src/scrfd_runtime.py` |
| Cross-camera association | **Không** clustering/graph. Shared `PersonMemory` + cosine + vote + post-hoc DB fusion | `src/person_memory.py`, `src/anonymous_face_recognizer.py`, `src/identity_database.py`, `src/evidence_fusion.py` |
| Named identity (gallery) | Open-set AdaFace vs `gallery/EMPxxx/` | `src/gallery.py`, `src/identity_manager.py` |

`setup.ps1` trên Windows: clone AdaFace + deep-person-reid, tải weights, patch `view` → `reshape`. Trên Kaggle bạn làm tương đương bằng các cell Python ở Phần B.

---

## A2. Entry points và data flow

### Điểm vào

| Muốn làm gì | Chạy gì |
|---|---|
| Inference | `python infer.py --sources ...` → `src.infer_engine.main()` |
| Sanity | `python check.py` |
| Tóm tắt run | `python analyze_run.py <runs/dir>` |
| Local Windows 1 video | `run_video.ps1 -Source path.mp4` |
| Local Windows EPFL | `download_epfl.ps1` rồi `run_epfl.ps1 -CameraCount 1..3` |

`main()` tạo **một** `PersonMemory`, **một** `EmbeddingStore`, **một** `IdentityDatabase`, rồi `for source in args.sources: process_camera(...)`. Camera sau thấy profile P của camera trước.

### Data flow (một camera)

```text
video path
  -> ultralytics YOLO.track (BoT-SORT)  → local track_id L, bbox, conf
  -> PersonMemory.touch_track(key=(cam, L), bbox)
  -> mỗi N frame: crop body → OSNet → PersonMemory.observe_body
  -> mỗi M frame: head ROI → SCRFD → align 112 → AdaFace
        -> match_gallery (nếu --gallery) → IdentityManager (EMPxxx sticky)
        -> PersonMemory.observe_face (Pxxx, face authority)
  -> EmbeddingStore.add + IdentityDatabase.add_observation
sau TẤT CẢ camera:
  -> reconcile_track_faces
  -> TrackletEvidenceFusion (canonical SQLite search)
  -> export CSV/JSON/NPZ/SQLite
  -> render {cam}_identity.mp4
```

`TrackKey = (camera_name, local_track_id)`. Local ID **reset theo từng video**. Global ID là `P001`, `P002`, … trong `PersonMemory.profiles`.

### Output (`--output`, mặc định `runs/face_first_person_memory`)

| Artifact | Nội dung |
|---|---|
| `{cam}_local_identity.mp4` | Overlay online trong lúc track |
| `{cam}_identity.mp4` | Overlay sau reconcile/fusion (dùng cái này để xem) |
| `tracks.csv` | bbox, L, P, EMP, fusion fields từng frame |
| `face_observations.csv`, `face_identity_diagnostics.csv` | AdaFace diagnostics |
| `person_memory.json` + `person_memory/` crops | Profile Pxxx |
| `embedding_store/embeddings.npz` + `metadata.json` | Mọi face/body vector |
| `identity.sqlite` | Source of truth assignments |
| `face_prototypes.npz`, `body_prototypes.npz`, `person_memory_prototypes.npz` | Prototype dumps |
| `identity_events.json`, `track_summaries.json` | Sự kiện / tóm tắt tracklet |

### Tham số quan trọng (không có `configs/*.yaml` identity)

Nguồn: `parse_args()` trong `src/infer_engine.py` + `tracker_botsort_reid.yaml`.

**Paths / device**

- `--sources` video (nhiều path = nhiều camera tuần tự)
- `--output`, `--device`, `--yolo`, `--tracker`
- `--reid-model`, `--reid-name`
- `--adaface-checkpoint`, `--adaface-repo`
- `--scrfd-model` (None = auto `models/scrfd/det_10g.onnx`)
- `--gallery` thư mục enroll
- `--identity-store` restore P từ run cũ
- `--identity-db` / `--canonical-db`

**Detector / SCT**

- `--conf 0.10` (phải thấp để BoT-SORT dùng `track_low_thresh`)
- `--imgsz 640`, `--no-half`
- YAML: `track_high_thresh 0.25`, `track_low_thresh 0.10`, `track_buffer 60`, `with_reid: True`, `gmc_method: none`

**Gallery EMPxxx**

- `--identity-threshold 0.55`, `--identity-margin 0.07`
- `--confirm-frames 3`, `--confirm-weight 1.45`
- `--strong-quality 0.68`

**PersonMemory Pxxx**

- `--person-face-threshold 0.56`, `--person-face-margin 0.04`
- `--person-body-near 0.76`, `--person-body-far 0.84` (cosine **similarity**, không phải distance)
- `--person-short-gap 45` frames
- `--require-face-before-person` (default True)

**Sampling (Kaggle VRAM/RAM)**

- `--face-every 2`, `--reid-sample-every 4`
- `--max-reid-embeddings 128`, `--person-max-body-views 24`

Không có file camera calibration / homography / camera mapping. Tên camera = stem file video (`cam0.mp4` → `cam0`).

---

## A3. Hai module cốt lõi

### Cross-camera / cross-track ID

Ba lớp ID, đừng nhầm:

1. **L (local)** — BoT-SORT, chỉ sống trong một video.
2. **Pxxx** — anonymous person trong `PersonMemory` (đây là “cùng một người giữa camera”).
3. **EMPxxx** — tên gallery; chỉ face gallery + `IdentityManager` mới gắn. Body **không** đổi EMP.

`PersonMemory` invariants (docstring trong `src/person_memory.py`):

- Body/motion **không** chuyển một track đã bind sang P khác.
- Chỉ **trusted face** (tier `support`/`strong`) được rebind / identify gap dài.
- Body chỉ **short-gap continuity** (cùng session, gap nhỏ, kèm motion/geometry).
- Frame crowd/crossing (`IoU`/`IoS` cao) **pause learning**.

Luồng face (`observe_face`):

1. `AnonymousFaceRecognizer.rank`: cosine `q @ prototype`, cần `score >= 0.56` và `margin >= 0.04`.
2. Match P đang bind → verify + học thêm prototype.
3. Match P khác → vote vài frame rồi `face_authority_rebind` (segment mới, **không** sửa lịch sử frame cũ).
4. Unbound + face lạ → vote rồi mint `P_new`.
5. P đã có face-core nhưng face query không match → conflict, **không** nhồi embedding vào bank.

Sau mọi camera:

- `reconcile_track_faces`: vote lại cả tracklet vs mọi face bank.
- `IdentityDatabase.search_identities` trên observation `canonical=1`.
- `TrackletEvidenceFusion.decide`: FACE_ONLY / FACE_BODY / BODY_ONLY_PROVISIONAL; margin thấp → CONFLICT (export có thể xóa P trên overlay).

**Hệ quả thực tế:** hai camera overlapping thời gian thật **không** được xử lý song song. Bạn nối `--sources camA.mp4 camB.mp4`; cam B match P của cam A nếu face (hoặc short-gap body nếu cùng “session” logic) đủ mạnh.

### Feature bank và khoảng cách

Mọi embedding **L2-normalize** (`src/prototypes.py`). “Khoảng cách” thực tế là **cosine similarity = dot product**.

```text
score = q_hat · p_hat
```

Ngưỡng là similarity (0.56 face, 0.76/0.84 body), **không** phải Euclidean distance matrix NxN toàn cục.

**Nơi lưu vector**

| Store | Role |
|---|---|
| `PersonProfile.face_bank` / `body_bank` | Prototype online (EMA nếu sim cao, slot mới nếu novel) |
| `body_bank[tid]` trong `process_camera` | Reservoir 128 vector / local track → `select_body_prototypes` cuối clip |
| `EmbeddingStore` | Mọi observation; `query()` brute-force cosine |
| `identity.sqlite` observations BLOB | Search canonical; `search()` = `query @ vector` |

Không FAISS/Annoy. Brute-force trên bank nhỏ (vài P × vài prototype).

Gallery: max cosine query vs prototypes mỗi EMP; accept nếu score + margin qua threshold (mặt xấu **siết** threshold, không nới).

---

# PHẦN B — KAGGLE NOTEBOOK (copy/paste)

Tạo Notebook mới trên [Kaggle](https://www.kaggle.com/). Mỗi khối `### CELL` = một cell. Loại cell ghi ở dòng đầu.

## B0. Cấu hình Kaggle (làm trước khi Run All)

1. **GPU:** Settings (bên phải) → Accelerator → **GPU T4 x2**.  
   **Không chọn TPU** — pipeline là PyTorch + ONNX CUDA, TPU không chạy được.
2. **Internet:** Settings → **Internet ON** (tải AdaFace, OSNet, YOLO, InsightFace buffalo_l, clone third_party).
3. **Persistence:** Settings → Persistence → **Files only** nếu muốn giữ `/kaggle/working` giữa session.
4. **Add data (bắt buộc với lựa chọn của bạn):**
   - Dataset 1: **ZIP codebase** (zip cả folder `person_identity_tracking`, gồm `src/`, `infer.py`, `requirements.txt`, `tracker_botsort_reid.yaml`). Upload Kaggle Dataset, Add Input.
   - Dataset 2: **video của bạn** (`.mp4`). Nên đặt tên rõ camera, ví dụ `cam0.mp4`, `cam1.mp4`. Add Input.
   - Dataset 3 (tuỳ chọn): `gallery/EMP001/*.jpg` nếu muốn nhãn EMPxxx.
5. Ghi path Input sau khi add: `/kaggle/input/<dataset-slug>/...`

SCRFD 10G nằm trong InsightFace `buffalo_l`. Public weights có điều khoản **non-commercial research**. Cell tải model chỉ dùng khi bạn chấp nhận điều khoản đó cho research/evaluation.

---

### CELL 0 — Markdown

```markdown
# Person Identity Tracking on Kaggle

Face-first pipeline: YOLO11s -> BoT-SORT -> SCRFD + AdaFace -> OSNet -> PersonMemory.

This notebook unzips an uploaded codebase ZIP (no git clone), downloads weights,
and runs inference on your uploaded camera videos.
```

---

### CELL 1 — Code: unzip workspace + inspect inputs

```python
# Unzip uploaded repo into /kaggle/working and locate videos.
from pathlib import Path
import os, zipfile, shutil

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "person_identity_tracking"

print("=== Kaggle input datasets ===")
for p in sorted(INPUT.iterdir()):
    print(p)
    for child in list(p.rglob("*"))[:30]:
        if child.is_file():
            print(" ", child.relative_to(p))

# --- EDIT if your ZIP path differs ---
zip_candidates = list(INPUT.rglob("*.zip"))
print("ZIP candidates:", zip_candidates)

src_zip = None
for z in zip_candidates:
    # Prefer a zip that looks like the repo, not buffalo_l later.
    if "buffalo" in z.name.lower():
        continue
    src_zip = z
    break

if PROJECT.exists() and (PROJECT / "infer.py").is_file():
    print("Project already extracted:", PROJECT)
elif src_zip is not None:
    extract_tmp = WORK / "_extract"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir()
    with zipfile.ZipFile(src_zip) as zf:
        zf.extractall(extract_tmp)
    # Handle either zip-root == repo or zip-root/person_identity_tracking
    infer_hits = list(extract_tmp.rglob("infer.py"))
    if not infer_hits:
        raise FileNotFoundError("infer.py not found inside ZIP")
    repo_root = infer_hits[0].parent
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(repo_root, PROJECT)
    print("Extracted repo to", PROJECT)
else:
    # Fallback: dataset already contains the folder (not zipped).
    infer_hits = list(INPUT.rglob("infer.py"))
    if not infer_hits:
        raise FileNotFoundError("Upload a ZIP of the repo or a dataset containing infer.py")
    repo_root = infer_hits[0].parent
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(repo_root, PROJECT, ignore=shutil.ignore_patterns(".git"))
    print("Copied repo to", PROJECT)

os.chdir(PROJECT)
print("CWD:", Path.cwd())
print("Has tracker yaml:", (PROJECT / "tracker_botsort_reid.yaml").is_file())

# --- EDIT: your video paths after Add Data ---
VIDEO_DIR = None
for d in INPUT.iterdir():
    vids = list(d.rglob("*.mp4")) + list(d.rglob("*.avi"))
    if vids:
        VIDEO_DIR = d
        print("Videos under", d)
        for v in vids:
            print(" ", v)
        break
if VIDEO_DIR is None:
    print("WARNING: no mp4/avi in /kaggle/input yet. Add your video dataset.")
```

---

### CELL 2 — Code: environment (keep Kaggle PyTorch)

```python
# Install project deps WITHOUT replacing Kaggle's CUDA PyTorch.
import sys, subprocess, os
from pathlib import Path

PROJECT = Path("/kaggle/working/person_identity_tracking")
os.chdir(PROJECT)

def pip(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

import torch
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())
else:
    raise RuntimeError("Enable GPU T4 x2 in Notebook settings")

# requirements.txt excludes torch on purpose. Install the rest.
# onnxruntime-gpu on PyPI may mismatch Kaggle CUDA; try GPU wheel, fall back to CPU EP.
pip("-r", "requirements.txt")

try:
    import onnxruntime as ort
    print("ORT providers", ort.get_available_providers())
except Exception as e:
    print("ORT import failed, retrying onnxruntime-gpu:", e)
    pip("--upgrade", "onnxruntime-gpu")
    import onnxruntime as ort
    print("ORT providers", ort.get_available_providers())

# deep-person-reid from source, NOT PyPI torchreid.
import shutil, subprocess
third = PROJECT / "third_party"
third.mkdir(exist_ok=True)
ada = third / "AdaFace"
reid = third / "deep-person-reid"
if not (ada / "net.py").is_file():
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/mk-minchul/AdaFace.git", str(ada)])
if not (reid / "torchreid" / "__init__.py").is_file():
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/KaiyangZhou/deep-person-reid.git", str(reid)])

# Same AdaFace patch as setup.ps1 (PyTorch non-contiguous view).
net_py = ada / "net.py"
text = net_py.read_text(encoding="utf-8")
patched = (
    text.replace("return input.view(input.size(0), -1)", "return input.reshape(input.size(0), -1)")
        .replace("x = x.view(x.shape[0], -1)", "x = x.reshape(x.shape[0], -1)")
)
if patched != text:
    net_py.write_text(patched, encoding="utf-8")
    print("Patched AdaFace view -> reshape")

# Register local torchreid (do not pip install torchreid).
sys.path = [p for p in sys.path if "deep-person-reid" not in p]
sys.path.insert(0, str(reid))
import torchreid
print("torchreid from", torchreid.__file__)

# Cache dirs inside working disk
os.environ["PIT_WORKSPACE_ROOT"] = str(PROJECT.parent)
os.environ["PIT_ASSETS_ROOT"] = str(PROJECT)
os.environ["HF_HOME"] = str(PROJECT.parent / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT.parent / ".cache" / "torch")
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT.parent / ".cache" / "ultralytics")
print("Env ready")
```

Nếu `pip install -r requirements.txt` kéo `onnxruntime-gpu` hỏng CUDA provider: vẫn chạy được với `CPUExecutionProvider` cho SCRFD (chậm hơn, YOLO/AdaFace/OSNet vẫn GPU).

---

### CELL 3 — Code: weights (YOLO, OSNet, AdaFace, SCRFD)

```python
# Download official checkpoints into models/. Accept InsightFace non-commercial research terms for buffalo_l.
from pathlib import Path
import os, sys, zipfile, shutil, urllib.request, subprocess

PROJECT = Path("/kaggle/working/person_identity_tracking")
os.chdir(PROJECT)
for sub in ["adaface", "scrfd", "osnet", "yolo"]:
    (PROJECT / "models" / sub).mkdir(parents=True, exist_ok=True)

ada_dst = PROJECT / "models" / "adaface" / "adaface_ir50_webface4m.ckpt"
osn_dst = PROJECT / "models" / "osnet" / "osnet_x1_0_msmt17.pth"
yolo_dst = PROJECT / "models" / "yolo" / "yolo11s.pt"
scrfd_dst = PROJECT / "models" / "scrfd" / "det_10g.onnx"

# AdaFace IR50 WebFace4M (Google Drive id used by setup.ps1)
if not ada_dst.is_file() or ada_dst.stat().st_size < 500_000_000:
    subprocess.check_call([sys.executable, "-m", "gdown", "1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN", "-O", str(ada_dst)])
print("AdaFace bytes", ada_dst.stat().st_size)

# OSNet x1.0 MSMT17
OSNET_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
if not osn_dst.is_file() or osn_dst.stat().st_size < 10_000_000:
    urllib.request.urlretrieve(OSNET_URL, osn_dst)
print("OSNet bytes", osn_dst.stat().st_size)

# YOLO11s via ultralytics (downloads to CWD of the call)
if not yolo_dst.is_file():
    os.chdir(yolo_dst.parent)
    from ultralytics import YOLO
    YOLO("yolo11s.pt")
    # ultralytics may write yolo11s.pt in cwd
    found = Path("yolo11s.pt")
    if found.is_file() and found.resolve() != yolo_dst.resolve():
        shutil.copy2(found, yolo_dst)
    os.chdir(PROJECT)
print("YOLO", yolo_dst, yolo_dst.is_file())

# SCRFD 10G from InsightFace buffalo_l (research / non-commercial public zoo)
ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH = True
if not ACCEPT_INSIGHTFACE_NONCOMMERCIAL_RESEARCH:
    raise SystemExit("Set True only if you accept InsightFace public-weight terms")
if not scrfd_dst.is_file():
    zip_path = PROJECT.parent / ".cache" / "buffalo_l.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    urllib.request.urlretrieve(url, zip_path)
    extract = PROJECT.parent / ".cache" / "buffalo_l_extract"
    if extract.exists():
        shutil.rmtree(extract)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract)
    found = next(extract.rglob("det_10g.onnx"))
    shutil.copy2(found, scrfd_dst)
print("SCRFD bytes", scrfd_dst.stat().st_size)
```

Nếu `gdown` quota Google Drive: tải sẵn `adaface_ir50_webface4m.ckpt` lên Kaggle Dataset rồi copy:

```python
# Optional fallback if gdown is quota-blocked
# shutil.copy2(Path("/kaggle/input/your-weights-dataset/adaface_ir50_webface4m.ckpt"), ada_dst)
```

---

### CELL 4 — Code: copy videos + optional gallery + smoke check

```python
from pathlib import Path
import os, shutil, sys

PROJECT = Path("/kaggle/working/person_identity_tracking")
os.chdir(PROJECT)
data = PROJECT / "data"
data.mkdir(exist_ok=True)

# Copy every uploaded mp4/avi into data/ (edit glob if you only want 1-2 cams)
videos = []
for ext in ("*.mp4", "*.avi", "*.mkv"):
    videos.extend(Path("/kaggle/input").rglob(ext))
# Skip anything accidentally inside the extracted repo
videos = [v for v in videos if "person_identity_tracking" not in str(v)]
if not videos:
    raise FileNotFoundError("Add a Kaggle Dataset with your camera videos")

# Stable names: keep original stem (used as camera name)
for v in sorted(videos):
    dst = data / v.name
    if not dst.exists():
        shutil.copy2(v, dst)
    print("video", dst, "MB", round(dst.stat().st_size / 1e6, 1))

# Optional employee gallery: copy a dataset folder that contains EMP001/, EMP002/
GALLERY = None
for emp in Path("/kaggle/input").rglob("EMP*"):
    if emp.is_dir():
        gdst = PROJECT / "gallery"
        # parent of EMP* dirs
        src_root = emp.parent
        if (src_root / emp.name).is_dir():
            if gdst.exists():
                shutil.rmtree(gdst)
            shutil.copytree(src_root, gdst)
            GALLERY = gdst
            print("Gallery copied from", src_root)
            break
print("GALLERY", GALLERY)

# Quick CUDA / path check (skip missing EPFL cam1/cam2)
sys.path.insert(0, str(PROJECT / "third_party" / "deep-person-reid"))
import torch, onnxruntime as ort
print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("ORT", ort.get_available_providers())
for rel in [
    "third_party/AdaFace/net.py",
    "third_party/deep-person-reid/torchreid/__init__.py",
    "models/adaface/adaface_ir50_webface4m.ckpt",
    "models/osnet/osnet_x1_0_msmt17.pth",
    "models/scrfd/det_10g.onnx",
    "models/yolo/yolo11s.pt",
]:
    p = PROJECT / rel
    print(("OK" if p.is_file() else "MISSING"), p)
```

---

### CELL 5 — Code: inference (start with 1 video, then 2)

```python
# Sequential multi-camera: one PersonMemory across --sources (not parallel decode).
import os, sys, glob
from pathlib import Path

PROJECT = Path("/kaggle/working/person_identity_tracking")
os.chdir(PROJECT)
os.environ["PIT_WORKSPACE_ROOT"] = str(PROJECT.parent)
os.environ["PIT_ASSETS_ROOT"] = str(PROJECT)

videos = sorted((PROJECT / "data").glob("*.mp4")) + sorted((PROJECT / "data").glob("*.avi"))
# First run: 1 camera to validate VRAM. Then set MAX_CAMS = 2 or 3.
MAX_CAMS = 1
sources = [str(v) for v in videos[:MAX_CAMS]]
print("sources", sources)

out = PROJECT / "runs" / "kaggle_demo"
cmd = [
    sys.executable, "infer.py",
    "--sources", *sources,
    "--device", "0",
    "--output", str(out),
    "--yolo", str(PROJECT / "models" / "yolo" / "yolo11s.pt"),
    "--reid-model", str(PROJECT / "models" / "osnet" / "osnet_x1_0_msmt17.pth"),
    "--adaface-repo", str(PROJECT / "third_party" / "AdaFace"),
    "--adaface-checkpoint", str(PROJECT / "models" / "adaface" / "adaface_ir50_webface4m.ckpt"),
    "--scrfd-model", str(PROJECT / "models" / "scrfd" / "det_10g.onnx"),
    "--tracker", str(PROJECT / "tracker_botsort_reid.yaml"),
    "--imgsz", "640",
    "--conf", "0.10",
    "--face-every", "2",
    "--reid-sample-every", "4",
]
gallery = PROJECT / "gallery"
if gallery.is_dir() and any(gallery.iterdir()):
    cmd += ["--gallery", str(gallery)]

print(" ".join(cmd))
import subprocess
subprocess.check_call(cmd)
```

Chạy 2 camera: `MAX_CAMS = 2` (cùng cell, Run lại). Cùng `PersonMemory` nên Pxxx có thể nối giữa file.

---

### CELL 6 — Code: analyze + preview + zip download

```python
from pathlib import Path
import os, subprocess, sys, shutil
from IPython.display import Video, display, FileLink

PROJECT = Path("/kaggle/working/person_identity_tracking")
out = PROJECT / "runs" / "kaggle_demo"
os.chdir(PROJECT)

subprocess.check_call([sys.executable, "analyze_run.py", str(out)])

# Preview first rendered identity video (Kaggle player needs h264; mp4v may not play in-browser).
mp4s = sorted(out.glob("*_identity.mp4"))
print("outputs", list(out.iterdir())[:20])
print("videos", mp4s)

# Re-encode a short preview for the notebook player
preview = out / "preview_h264.mp4"
if mp4s:
    src = mp4s[0]
    # 20s preview to keep RAM/time down
    ff = [
        "ffmpeg", "-y", "-i", str(src), "-t", "20",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(preview),
    ]
    try:
        subprocess.check_call(ff)
        display(Video(str(preview), embed=True, width=640))
    except Exception as e:
        print("ffmpeg preview failed, download the zip instead:", e)

# Zip artifacts for local download (Output tab / FileLink)
zip_base = Path("/kaggle/working") / "kaggle_demo_output"
if Path(str(zip_base) + ".zip").exists():
    Path(str(zip_base) + ".zip").unlink()
shutil.make_archive(str(zip_base), "zip", out)
zpath = Path(str(zip_base) + ".zip")
print("zip", zpath, "MB", round(zpath.stat().st_size / 1e6, 1))
display(FileLink(str(zpath)))
```

Mở `tracks.csv` nhanh:

```python
import pandas as pd
from pathlib import Path
p = Path("/kaggle/working/person_identity_tracking/runs/kaggle_demo/tracks.csv")
df = pd.read_csv(p)
print(df.columns.tolist())
print(df[["camera", "frame", "track_id", "person_id", "identity", "identity_state"]].head(20))
print(df.groupby(["camera", "person_id"]).size().head(30))
```

---

## B1. Lưu ý Kaggle (RAM / VRAM / session)

**VRAM**

- Camera được decode **tuần tự**, không 3 YOLO song song. Bottleneck là YOLO + AdaFace + OSNet trên **một** GPU.
- Bắt đầu `MAX_CAMS = 1`, `--imgsz 640`. Nếu OOM: `--imgsz 512`, tăng `--face-every 4`, `--reid-sample-every 8`.
- T4 ~16GB thường đủ YOLO11s + IR50 + OSNet. Đừng load hai process `infer.py`.
- `torch.cuda.empty_cache()` không cần nếu chỉ một process.

**RAM**

- `EmbeddingStore` giữ mọi vector trên RAM rồi ghi NPZ. Video dài + nhiều người: tăng `--reid-sample-every`, giảm `--max-reid-embeddings`.
- Đừng `display(Video(embed=True))` cả file 4K; chỉ preview 20s H264.

**Session không rớt**

- Kaggle idle ~15–30 phút có thể disconnect. Chạy cell dài thì giữ tab mở.
- Console trình duyệt (F12) snippet cổ điển (dùng có trách nhiệm, có thể đổi theo UI Kaggle):

```javascript
function keepAlive() {
  console.log("keep-alive", new Date().toISOString());
  document.dispatchEvent(new MouseEvent("mousemove", {bubbles: true}));
}
setInterval(keepAlive, 60000);
```

- Save Version → **Save & Run All** chạy trên server kể cả khi đóng laptop (quota GPU tuần vẫn bị trừ). Output zip nằm trong Version.
- Quota GPU Kaggle ~30h/tuần; T4 x2 không nhân đôi VRAM cho một process PyTorch (vẫn `cuda:0` trừ khi bạn tự split).

**Lỗi thường gặp**

| Triệu chứng | Cách xử |
|---|---|
| `deep-person-reid is not importable` | Clone `third_party/deep-person-reid`, `sys.path.insert`, **không** `pip install torchreid` |
| AdaFace shape / view error | Patch `reshape` như Cell 2 |
| `SCRFD model not found` | Cell 3 buffalo_l / `det_10g.onnx` |
| `CUDAExecutionProvider` missing | SCRFD CPU; YOLO vẫn GPU. Có thể bỏ qua nếu FPS chấp nhận được |
| Overlay P:--- suốt | Mặt nhỏ/mờ; giảm `--min-face-detect-size`, kiểm tra `face_observations.csv` |
| EMP không hiện | Thiếu `--gallery` hoặc ảnh enroll không qua quality |
| Player Kaggle không phát mp4 | Re-encode H264 (Cell 6) hoặc tải zip về |

**Thứ tự đọc code sau khi chạy 1 video**

1. `infer.py` → `src/infer_engine.py` (`process_camera`, `main`)
2. `tracker_botsort_reid.yaml`
3. `src/person_memory.py` (`observe_face`, `observe_body`)
4. `src/anonymous_face_recognizer.py` + `src/gallery.py`
5. `src/embedding_store.py` + `src/identity_database.py` + `src/evidence_fusion.py`

Hết phần onboarding. Chạy Cell 1→6 trên GPU T4, Internet ON, với ZIP repo + video Dataset.
