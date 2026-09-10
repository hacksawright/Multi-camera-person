from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .prototypes import normalize


@dataclass(frozen=True)
class EmbeddingRecord:
    record_id: str
    kind: str
    camera: str
    frame: int
    track_id: int
    embedding: np.ndarray
    quality: float
    person_id: int | None = None
    pose: str = ""


class EmbeddingStore:
    """Persistent raw embedding store used for later identity reconciliation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[EmbeddingRecord] = []

    def add(self, record: EmbeddingRecord) -> None:
        vector = normalize(record.embedding).astype(np.float32)
        self.records.append(EmbeddingRecord(
            record.record_id, record.kind, record.camera, int(record.frame),
            int(record.track_id), vector, float(record.quality), record.person_id,
            record.pose,
        ))

    def query(self, embedding: np.ndarray, kind: str | None = None, limit: int = 10) -> list[tuple[EmbeddingRecord, float]]:
        q = normalize(embedding)
        candidates = [x for x in self.records if kind is None or x.kind == kind]
        ranked = sorted(((x, float(q @ x.embedding)) for x in candidates), key=lambda item: item[1], reverse=True)
        return ranked[:max(int(limit), 0)]

    def save(self) -> None:
        vectors = np.stack([x.embedding for x in self.records]).astype(np.float32) if self.records else np.empty((0, 0), np.float32)
        np.savez_compressed(self.root / "embeddings.npz", vectors=vectors)
        metadata = []
        for i, x in enumerate(self.records):
            metadata.append({
                "index": i, "record_id": x.record_id, "kind": x.kind,
                "camera": x.camera, "frame": x.frame, "track_id": x.track_id,
                "quality": x.quality, "person_id": x.person_id, "pose": x.pose,
            })
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, root: str | Path) -> "EmbeddingStore":
        store = cls(root)
        vectors_path = store.root / "embeddings.npz"
        metadata_path = store.root / "metadata.json"
        if not vectors_path.is_file() or not metadata_path.is_file():
            return store
        vectors = np.load(vectors_path)["vectors"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in metadata:
            store.records.append(EmbeddingRecord(
                item["record_id"], item["kind"], item["camera"], int(item["frame"]),
                int(item["track_id"]), vectors[int(item["index"])], float(item["quality"]),
                item.get("person_id"), item.get("pose", ""),
            ))
        return store
