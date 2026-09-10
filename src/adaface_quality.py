from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


class AdaFaceQualityEmbedder:
    """AdaFace IR50 extractor returning L2 embeddings and the raw feature norm."""

    def __init__(self, repo_dir: str, checkpoint: str, device: str = "cuda:0") -> None:
        repo = Path(repo_dir)
        ckpt_path = Path(checkpoint)
        net_path = repo / "net.py"
        if not net_path.is_file():
            raise FileNotFoundError(f"AdaFace source not found: {net_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"AdaFace checkpoint not found: {ckpt_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for AdaFace but torch.cuda.is_available() is False")

        spec = importlib.util.spec_from_file_location("adaface_net_identity", str(net_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import AdaFace net.py from {net_path}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo))
        try:
            spec.loader.exec_module(module)
        finally:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass

        self.model = module.build_model("ir_50")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        cleaned = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[AdaFace] warning: {len(missing)} missing keys")
        if unexpected:
            print(f"[AdaFace] warning: {len(unexpected)} unexpected keys")

        self.device = torch.device(device)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode(self, aligned_bgr_faces: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        faces = [f for f in aligned_bgr_faces if f is not None and f.size > 0]
        if not faces:
            return np.empty((0, 512), np.float32), np.empty((0,), np.float32)
        arr = np.stack(faces).astype(np.float32)
        arr = (arr / 255.0 - 0.5) / 0.5
        arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
        tensor = torch.from_numpy(arr).to(self.device, non_blocking=True)
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            raw_feature = output[0]
            raw_norm = output[1] if len(output) > 1 else None
        else:
            raw_feature = output
            raw_norm = None
        raw_feature = raw_feature.float()
        if raw_norm is None:
            norms = torch.linalg.vector_norm(raw_feature, dim=1)
        else:
            norms = torch.as_tensor(raw_norm, device=raw_feature.device).float().reshape(-1)
        feature = torch.nn.functional.normalize(raw_feature, dim=1)
        return feature.cpu().numpy().astype(np.float32), norms.cpu().numpy().astype(np.float32)
