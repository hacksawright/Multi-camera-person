from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterator
import uuid

import numpy as np

from .prototypes import normalize
from .embedding_store import EmbeddingRecord


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VectorHit:
    observation_id: int
    identity_id: int | None
    tracklet_id: int
    score: float
    quality: float
    camera: str
    frame: int
    pose: str


@dataclass(frozen=True)
class IdentityHit:
    identity_id: int
    score: float
    support: int
    best_score: float


class IdentityDatabase:
    """SQLite source of truth for immutable observations and mutable assignments."""

    def __init__(self, path: str | Path, session_id: str | None = None, readonly: bool = False) -> None:
        self.path = Path(path)
        self.readonly = bool(readonly)
        if not self.readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.readonly:
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
        else:
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not self.readonly:
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.session_id = session_id or uuid.uuid4().hex
        self._search_cache: dict[tuple[str, str, str, int], list[tuple[VectorHit, np.ndarray]]] = {}
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "IdentityDatabase":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def _migrate(self) -> None:
        if self.readonly:
            return
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS schema_info(version INTEGER NOT NULL);
                INSERT INTO schema_info(version)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);
                CREATE TABLE IF NOT EXISTS identities(
                    id INTEGER PRIMARY KEY, employee_id TEXT UNIQUE,
                    state TEXT NOT NULL DEFAULT 'PROVISIONAL', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS tracklets(
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, camera TEXT NOT NULL, local_track_id INTEGER NOT NULL,
                    first_frame INTEGER NOT NULL, last_frame INTEGER NOT NULL,
                    UNIQUE(session_id, camera, local_track_id)
                );
                CREATE TABLE IF NOT EXISTS detections(
                    id INTEGER PRIMARY KEY, tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
                    frame INTEGER NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
                    confidence REAL NOT NULL, ambiguous INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(tracklet_id, frame)
                );
                CREATE TABLE IF NOT EXISTS observations(
                    id INTEGER PRIMARY KEY, tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
                    frame INTEGER NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('face','body')),
                    quality REAL NOT NULL, pose TEXT NOT NULL DEFAULT '', feature_norm REAL,
                    model_name TEXT NOT NULL, model_version TEXT NOT NULL, dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                    canonical INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_observation_kind_model ON observations(kind, model_name, model_version, dimension);
                CREATE INDEX IF NOT EXISTS idx_observation_tracklet ON observations(tracklet_id, frame);
                CREATE TABLE IF NOT EXISTS assignments(
                    id INTEGER PRIMARY KEY, tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
                    identity_id INTEGER REFERENCES identities(id), state TEXT NOT NULL,
                    score REAL, margin REAL, reason TEXT NOT NULL, valid_from INTEGER NOT NULL,
                    valid_to INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_assignment_identity ON assignments(identity_id, state);
                CREATE TABLE IF NOT EXISTS identity_events(
                    id INTEGER PRIMARY KEY, tracklet_id INTEGER REFERENCES tracklets(id),
                    old_identity_id INTEGER, new_identity_id INTEGER, frame INTEGER NOT NULL,
                    event TEXT NOT NULL, evidence_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS fusion_decisions(
                    id INTEGER PRIMARY KEY, tracklet_id INTEGER NOT NULL REFERENCES tracklets(id),
                    frame INTEGER NOT NULL, identity_id INTEGER, state TEXT NOT NULL,
                    reason TEXT NOT NULL, face_score REAL, body_score REAL, motion_score REAL,
                    fusion_score REAL, margin REAL, face_support INTEGER NOT NULL DEFAULT 0,
                    body_support INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_fusion_tracklet ON fusion_decisions(tracklet_id, frame);
            """)
            columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(observations)")}
            if "canonical" not in columns:
                self.connection.execute("ALTER TABLE observations ADD COLUMN canonical INTEGER NOT NULL DEFAULT 0")

    def upsert_tracklet(self, camera: str, local_track_id: int, frame: int) -> int:
        with self.connection:
            self.connection.execute(
                "INSERT INTO tracklets(session_id,camera,local_track_id,first_frame,last_frame) VALUES(?,?,?,?,?) "
                "ON CONFLICT(session_id,camera,local_track_id) DO UPDATE SET "
                "first_frame=MIN(first_frame,excluded.first_frame),last_frame=MAX(last_frame,excluded.last_frame)",
                (self.session_id, camera, int(local_track_id), int(frame), int(frame)),
            )
        row = self.connection.execute(
            "SELECT id FROM tracklets WHERE session_id=? AND camera=? AND local_track_id=?",
            (self.session_id, camera, int(local_track_id))
        ).fetchone()
        return int(row["id"])

    def add_detection(self, tracklet_id: int, frame: int, box: np.ndarray, confidence: float, ambiguous: bool) -> None:
        x1, y1, x2, y2 = [float(x) for x in box]
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO detections(tracklet_id,frame,x1,y1,x2,y2,confidence,ambiguous) VALUES(?,?,?,?,?,?,?,?)",
                (tracklet_id, int(frame), x1, y1, x2, y2, float(confidence), int(ambiguous)),
            )

    def add_observation(
        self, tracklet_id: int, frame: int, kind: str, embedding: np.ndarray, quality: float,
        *, pose: str = "", feature_norm: float | None = None, model_name: str,
        model_version: str, metadata: dict | None = None,
    ) -> int:
        vector = normalize(np.asarray(embedding, dtype=np.float32)).reshape(-1)
        if not np.all(np.isfinite(vector)) or vector.size == 0:
            raise ValueError("Embedding must be a finite non-empty vector")
        if kind not in {"face", "body"}:
            raise ValueError(f"Unsupported observation kind: {kind}")
        with self.connection:
            cur = self.connection.execute(
                "INSERT INTO observations(tracklet_id,frame,kind,quality,pose,feature_norm,model_name,model_version,dimension,embedding,metadata_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (tracklet_id, int(frame), kind, float(quality), pose, feature_norm, model_name, model_version,
                 int(vector.size), vector.tobytes(), json.dumps(metadata or {}, separators=(",", ":"))),
            )
        self._search_cache.clear()
        return int(cur.lastrowid)

    def ensure_identity(self, identity_id: int, employee_id: str | None = None, state: str = "PROVISIONAL") -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO identities(id,employee_id,state) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET employee_id=COALESCE(excluded.employee_id,employee_id),state=excluded.state",
                (int(identity_id), employee_id, state),
            )

    def assign(self, tracklet_id: int, identity_id: int | None, frame: int, state: str, reason: str,
               score: float | None = None, margin: float | None = None) -> None:
        with self.connection:
            current = self.connection.execute(
                "SELECT id,identity_id,state,valid_from FROM assignments WHERE tracklet_id=? AND valid_to IS NULL "
                "ORDER BY id DESC LIMIT 1", (tracklet_id,),
            ).fetchone()
            if current is not None and current["identity_id"] == identity_id and current["state"] == state:
                return
            if current is not None and int(current["valid_from"]) >= int(frame):
                self.connection.execute("DELETE FROM assignments WHERE id=?", (int(current["id"]),))
            self.connection.execute(
                "UPDATE assignments SET valid_to=? WHERE tracklet_id=? AND valid_to IS NULL AND valid_from<?",
                (int(frame) - 1, tracklet_id, int(frame)),
            )
            self.connection.execute(
                "INSERT INTO assignments(tracklet_id,identity_id,state,score,margin,reason,valid_from) VALUES(?,?,?,?,?,?,?)",
                (tracklet_id, identity_id, state, score, margin, reason, int(frame)),
            )
        self._search_cache.clear()

    def search(self, embedding: np.ndarray, kind: str, model_name: str, model_version: str, limit: int = 10) -> list[VectorHit]:
        query = normalize(np.asarray(embedding, dtype=np.float32)).reshape(-1)
        cache_key = (kind, model_name, model_version, int(query.size))
        cached = self._search_cache.get(cache_key)
        if cached is None:
            rows = self.connection.execute(
                "SELECT o.*,t.camera,a.identity_id FROM observations o "
                "JOIN tracklets t ON t.id=o.tracklet_id "
                "LEFT JOIN assignments a ON a.tracklet_id=o.tracklet_id AND a.valid_to IS NULL "
                "WHERE o.kind=? AND o.model_name=? AND o.model_version=? AND o.dimension=? AND o.canonical=1",
                (kind, model_name, model_version, int(query.size)),
            ).fetchall()
            cached = []
            for row in rows:
                vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                cached.append((VectorHit(
                    int(row["id"]), row["identity_id"], int(row["tracklet_id"]), 0.0,
                    float(row["quality"]), row["camera"], int(row["frame"]), row["pose"],
                ), vector))
            self._search_cache[cache_key] = cached
        hits = []
        for hit, vector in cached:
            hits.append(VectorHit(
                hit.observation_id, hit.identity_id, hit.tracklet_id, float(query @ vector),
                hit.quality, hit.camera, hit.frame, hit.pose,
            ))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:max(int(limit), 0)]

    def assigned_embedding_records(self) -> list[EmbeddingRecord]:
        rows = self.connection.execute(
            "SELECT o.*,t.camera,t.local_track_id,a.identity_id FROM observations o "
            "JOIN tracklets t ON t.id=o.tracklet_id "
            "JOIN assignments a ON a.tracklet_id=o.tracklet_id AND a.valid_to IS NULL "
            "WHERE a.identity_id IS NOT NULL AND o.canonical=1"
        ).fetchall()
        return [EmbeddingRecord(
            f"db:{int(row['id'])}", row["kind"], row["camera"], int(row["frame"]),
            int(row["local_track_id"]), np.frombuffer(row["embedding"], dtype=np.float32).copy(),
            float(row["quality"]), int(row["identity_id"]), row["pose"],
        ) for row in rows]

    def promote_tracklet_references(self, tracklet_id: int, identity_id: int) -> None:
        """Promote only observations from a face-anchored final tracklet."""
        with self.connection:
            self.connection.execute(
                "UPDATE observations SET canonical=1 WHERE tracklet_id=? AND "
                "((kind='face' AND quality>=0.45) OR (kind='body' AND quality>=0.35))",
                (int(tracklet_id),),
            )
            self.ensure_identity(identity_id, state="VERIFIED")

    def search_identities(
        self, embeddings: list[np.ndarray], kind: str, model_name: str,
        model_version: str, *, exclude_tracklet_id: int | None = None,
        limit: int = 10,
    ) -> list[IdentityHit]:
        """Rank canonical identities, not individual observations."""
        grouped: dict[int, dict[int, list[tuple[float, float]]]] = {}
        for embedding in embeddings:
            for hit in self.search(embedding, kind, model_name, model_version, limit=200):
                if hit.identity_id is None or hit.tracklet_id == exclude_tracklet_id:
                    continue
                grouped.setdefault(int(hit.identity_id), {}).setdefault(hit.tracklet_id, []).append(
                    (hit.score, max(hit.quality, 0.2))
                )
        ranked = []
        for identity_id, tracklet_scores in grouped.items():
            # Collapse repeated frames from the same local tracklet first.
            scored = []
            for values in tracklet_scores.values():
                values.sort(key=lambda item: item[0], reverse=True)
                scored.append(values[0])
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:min(5, len(scored))]
            score = sum(value * weight for value, weight in top) / sum(weight for _value, weight in top)
            ranked.append(IdentityHit(identity_id, float(score), len(scored), float(scored[0][0])))
        ranked.sort(key=lambda x: (x.score, x.support), reverse=True)
        return ranked[:max(int(limit), 0)]

    def integrity_check(self) -> str:
        return str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])

    def canonical_stats(self) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT COUNT(*) AS observations, COUNT(DISTINCT identity_id) AS identities "
            "FROM (SELECT o.id, a.identity_id FROM observations o JOIN assignments a "
            "ON a.tracklet_id=o.tracklet_id AND a.valid_to IS NULL WHERE o.canonical=1 AND a.identity_id IS NOT NULL)"
        ).fetchone()
        models = self.connection.execute(
            "SELECT COUNT(DISTINCT model_name || ':' || model_version || ':' || dimension) FROM observations WHERE canonical=1"
        ).fetchone()[0]
        return {"observations": int(row["observations"]), "identities": int(row["identities"]), "model_signatures": int(models)}

    def export_canonical(self, destination: str | Path, *, max_frame: int | None = None) -> dict[str, int]:
        frame_clause = " AND o.frame < ?" if max_frame is not None else ""
        frame_args = (int(max_frame),) if max_frame is not None else ()
        stats = self.canonical_stats()
        if stats["observations"] == 0 or stats["identities"] == 0:
            raise ValueError("Source DB has no canonical identity references")
        with IdentityDatabase(destination, session_id="canonical") as target:
            rows = self.connection.execute(
                "SELECT DISTINCT i.id,i.employee_id,i.state FROM identities i JOIN assignments a ON a.identity_id=i.id "
                "JOIN observations o ON o.tracklet_id=a.tracklet_id "
                "WHERE a.valid_to IS NULL AND a.identity_id IS NOT NULL AND o.canonical=1" + frame_clause,
                frame_args,
            ).fetchall()
            for row in rows:
                target.ensure_identity(int(row["id"]), row["employee_id"], "VERIFIED")
            tracklet_map: dict[int, int] = {}
            observations = self.connection.execute(
                "SELECT o.*,t.session_id,t.camera,t.local_track_id,a.identity_id FROM observations o "
                "JOIN tracklets t ON t.id=o.tracklet_id JOIN assignments a ON a.tracklet_id=t.id AND a.valid_to IS NULL "
                "WHERE o.canonical=1 AND a.identity_id IS NOT NULL" + frame_clause,
                frame_args,
            ).fetchall()
            for row in observations:
                old_tid = int(row["tracklet_id"])
                if old_tid not in tracklet_map:
                    tracklet_map[old_tid] = target.upsert_tracklet(row["camera"], int(row["local_track_id"]), int(row["frame"]))
                new_tid = tracklet_map[old_tid]
                target.add_observation(
                    new_tid, int(row["frame"]), row["kind"], np.frombuffer(row["embedding"], dtype=np.float32),
                    float(row["quality"]), pose=row["pose"], feature_norm=row["feature_norm"],
                    model_name=row["model_name"], model_version=row["model_version"],
                    metadata=json.loads(row["metadata_json"]),
                )
                target.connection.execute(
                    "UPDATE observations SET canonical=1 WHERE id=(SELECT MAX(id) FROM observations)"
                )
            for old_tid, new_tid in tracklet_map.items():
                row = self.connection.execute(
                    "SELECT identity_id,valid_from FROM assignments WHERE tracklet_id=? AND valid_to IS NULL",
                    (old_tid,),
                ).fetchone()
                if row:
                    target.assign(new_tid, int(row["identity_id"]), int(row["valid_from"]), "VERIFIED", "canonical_import")
            target.connection.commit()
            if target.integrity_check() != "ok":
                raise RuntimeError("Canonical DB integrity check failed")
        return stats

    def add_fusion_decision(self, tracklet_id: int, frame: int, decision: object) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO fusion_decisions(tracklet_id,frame,identity_id,state,reason,face_score,body_score,motion_score,fusion_score,margin,face_support,body_support) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    int(tracklet_id), int(frame), getattr(decision, "identity_id", None),
                    getattr(decision, "state", "UNKNOWN"), getattr(decision, "reason", ""),
                    getattr(decision, "face_score", None), getattr(decision, "body_score", None),
                    getattr(decision, "motion_score", None), getattr(decision, "fusion_score", None),
                    getattr(decision, "margin", None), int(getattr(decision, "face_support", 0)),
                    int(getattr(decision, "body_support", 0)),
                ),
            )
