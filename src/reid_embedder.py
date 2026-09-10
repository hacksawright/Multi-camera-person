from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch


class TorchReIDEmbedder:
    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "osnet_x1_0",
        model_path: str | None = None,
    ) -> None:
        try:
            from torchreid.utils import FeatureExtractor
        except Exception as exc:
            raise RuntimeError(
                "deep-person-reid is not importable. "
                "Run setup.ps1 from the project root."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False"
            )

        if not model_path:
            raise FileNotFoundError(
                "OSNet checkpoint path is required."
            )

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"OSNet checkpoint not found: {model_path}"
            )

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(model_path),
            device=device,
            verbose=False,
        )

    @torch.inference_mode()
    def encode(
        self,
        bgr_images: Iterable[np.ndarray],
    ) -> np.ndarray:
        images: List[np.ndarray] = []

        for im in bgr_images:
            if im is None or im.size == 0:
                continue

            if im.ndim != 3 or im.shape[2] != 3:
                raise ValueError(
                    f"Expected HxWx3 BGR image, got shape={im.shape}"
                )

            # OpenCV frame/crop is BGR.
            # TorchReID ndarray input must receive RGB.
            rgb = cv2.cvtColor(
                im,
                cv2.COLOR_BGR2RGB,
            )

            images.append(rgb)

        if not images:
            return np.empty(
                (0, 512),
                dtype=np.float32,
            )

        features = self.extractor(images)

        if isinstance(features, np.ndarray):
            arr = features
        else:
            arr = (
                features
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        arr = arr.astype(
            np.float32,
            copy=False,
        )

        arr /= np.clip(
            np.linalg.norm(
                arr,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )

        return arr
