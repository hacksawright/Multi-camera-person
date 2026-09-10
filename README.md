# Person Identity Tracking — CLEAN

Bản này **không phụ thuộc bất kỳ source v1/v2/v3 cũ nào**. Chỉ tái sử dụng hai thư mục nằm ở cấp cha:

```text
D:\Work\Tinasoft\DetectPerson\
├── .venv\                    # giữ nguyên
├── .cache\                   # giữ nguyên
└── person_identity_tracking\ # giải nén ZIP tại đây
    ├── models\               # setup tự tải model vào đây
    ├── data\                 # video test, nếu cần
    ├── third_party\          # source AdaFace + deep-person-reid
    ├── gallery\
    ├── runs\
    ├── src\
    ├── setup.ps1
    ├── check.py
    └── run_video.ps1
```

Không có `shared`, không dò source cũ, không hardlink sang project cũ. Nếu xóa `person_identity_tracking`, source/model/data/output của project biến mất; `.venv` và `.cache` vẫn còn.

## Pipeline

```text
YOLO11s (COCO baseline)
 -> BoT-SORT + ReID
 -> head ROI -> SCRFD 10G -> quality/pose -> AdaFace IR50
 -> OSNet x1.0 hỗ trợ continuity/reacquire
 -> Identity Manager
 -> EMPxxx hoặc UNKNOWN
```

Face là nguồn xác nhận danh tính. Body ReID không được tự đổi employee ID. Người ngoài gallery giữ `UNKNOWN`.

## Setup

PowerShell:

```powershell
$Base = "D:\Work\Tinasoft\DetectPerson"
$Proj = "$Base\person_identity_tracking"
$Py   = "$Base\.venv\Scripts\python.exe"
cd $Proj

powershell -ExecutionPolicy Bypass -File .\setup.ps1 `
  -AcceptInsightFaceNonCommercial
```

`setup.ps1` giữ nguyên PyTorch/CUDA trong `.venv`, dùng `.cache` ở cấp cha, clone source third-party vào project, tự tải AdaFace IR50, OSNet, YOLO11s và SCRFD 10G. Detector mặc định là `models/yolo/yolo11s.pt`. Public InsightFace pretrained weights có điều khoản riêng/non-commercial research; flag trên chỉ dùng khi bạn chấp nhận điều khoản đó cho research/evaluation.

Kiểm tra:

```powershell
& $Py .\check.py
```

Cần thấy `CHECK OK`, CUDA True, và `deep-person-reid` trỏ vào chính `person_identity_tracking\third_party`.

## Test EPFL

Tải video test một lần:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_epfl.ps1
```

Chạy một camera:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_epfl.ps1 -CameraCount 1
```

Phân tích:

```powershell
& $Py .\analyze_run.py .\runs\epfl
start .\runs\epfl\cam0_identity.mp4
```

## Video thật

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 `
  -Source "D:\video\camera01.mp4"
```

Có gallery nhân viên:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 `
  -Source "D:\video\camera01.mp4" `
  -Gallery ".\gallery"
```

Gallery:

```text
gallery\
├── EMP001\
│   ├── front.jpg
│   ├── left30.jpg
│   ├── right30.jpg
│   ├── left60.jpg
│   └── right60.jpg
└── EMP002\
```

## Lưu ý

- SCRFD 10G chạy head ROI `640x640` để khớp public ONNX metadata.
- AdaFace đã có patch `view -> reshape` khi setup để tương thích PyTorch hiện tại.
- `setup.ps1` tự loại các `.pth` cũ chứa `deep-person-reid` trong venv rồi đăng ký đúng source của project này.
- Không cài lại Torch.
- Threshold hiện là baseline benchmark; production cần calibration trên camera thật.
