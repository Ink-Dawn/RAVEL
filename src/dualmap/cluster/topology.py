from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


ServiceCurve = Tuple[Tuple[int, float], ...]


def _interpolate_curve(
    curve: ServiceCurve,
    active_sequences: int,
    fallback: float,
) -> float:
    """Linearly interpolate a measured service curve without extrapolating."""
    if not curve:
        return fallback
    target = max(0, int(active_sequences))
    if target <= curve[0][0]:
        return curve[0][1]
    if target >= curve[-1][0]:
        return curve[-1][1]
    for (left_n, left_value), (right_n, right_value) in zip(curve, curve[1:]):
        if left_n <= target <= right_n:
            fraction = (target - left_n) / (right_n - left_n)
            return left_value + fraction * (right_value - left_value)
    return fallback


def _parse_curve(
    rows: Sequence[Mapping[str, object]],
    value_keys: Sequence[str],
    scale: float = 1.0,
    allow_zero: bool = False,
) -> ServiceCurve:
    points: Dict[int, float] = {}
    for row in rows:
        active = int(row.get("active_sequences", row.get("background_sequences", 0)))
        value = None
        for key in value_keys:
            if row.get(key) is not None:
                value = float(row[key]) * scale
                break
        if value is not None and (value > 0 or (allow_zero and value == 0)):
            points[active] = value
    return tuple(sorted(points.items()))


@dataclass(frozen=True)
class ClusterSpec:
    cluster_id: str
    replica_ids: Tuple[int, ...]
    rtt_ms_by_client_region: Mapping[str, float]
    prefill_tpot_s: float
    decode_tpot_s: float
    decode_marginal_tpot_s: Optional[float] = None
    prefill_intercept_s: float = 0.0
    prefill_tpot_curve: ServiceCurve = ()
    prefill_intercept_curve: ServiceCurve = ()
    decode_tpot_curve: ServiceCurve = ()
    service_profile_source: str = "inline"

    def rtt_s(self, client_region: str) -> float:
        if client_region in self.rtt_ms_by_client_region:
            return float(self.rtt_ms_by_client_region[client_region]) / 1000.0
        if "default" in self.rtt_ms_by_client_region:
            return float(self.rtt_ms_by_client_region["default"]) / 1000.0
        raise KeyError(
            f"cluster {self.cluster_id!r} has no RTT for client region "
            f"{client_region!r} and no default"
        )

    def prefill_tpot_for(self, active_sequences: int) -> float:
        return _interpolate_curve(
            self.prefill_tpot_curve,
            active_sequences,
            self.prefill_tpot_s,
        )

    def prefill_intercept_for(self, active_sequences: int) -> float:
        return _interpolate_curve(
            self.prefill_intercept_curve,
            active_sequences,
            self.prefill_intercept_s,
        )

    def decode_tpot_for(self, active_sequences: int) -> float:
        return _interpolate_curve(
            self.decode_tpot_curve,
            active_sequences,
            self.decode_tpot_s,
        )


class ClusterTopology:
    def __init__(self, clusters: Iterable[ClusterSpec], default_client_region: str):
        self.clusters = tuple(clusters)
        self.default_client_region = default_client_region
        if len(self.clusters) < 2:
            raise ValueError("cross-cluster RAVEL requires at least two clusters")

        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("cluster ids must be unique")

        replica_ids = [rid for cluster in self.clusters for rid in cluster.replica_ids]
        if not replica_ids:
            raise ValueError("topology has no replicas")
        if len(replica_ids) != len(set(replica_ids)):
            raise ValueError("a replica may belong to only one cluster")
        if min(replica_ids) < 0:
            raise ValueError("replica ids must be non-negative")

        self._by_id: Dict[str, ClusterSpec] = {
            cluster.cluster_id: cluster for cluster in self.clusters
        }
        self._cluster_by_replica = {
            rid: cluster.cluster_id
            for cluster in self.clusters
            for rid in cluster.replica_ids
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        base_dir: Optional[Path] = None,
    ) -> "ClusterTopology":
        default_client_region = str(raw.get("default_client_region", "local"))
        raw_clusters = raw.get("clusters")
        if not isinstance(raw_clusters, list):
            raise ValueError("topology must contain a clusters list")

        clusters = []
        for item in raw_clusters:
            if not isinstance(item, dict):
                raise ValueError("each cluster entry must be an object")
            profile: Mapping[str, object] = {}
            profile_source = "inline"
            profile_path = item.get("service_profile")
            if profile_path:
                resolved = Path(str(profile_path))
                if not resolved.is_absolute() and base_dir is not None:
                    resolved = base_dir / resolved
                with resolved.open("r", encoding="utf-8") as profile_file:
                    loaded = json.load(profile_file)
                if not isinstance(loaded, dict):
                    raise ValueError(f"service profile {resolved} must be an object")
                if int(loaded.get("schema_version", 0)) != 2:
                    raise ValueError(
                        f"service profile {resolved} must use schema_version=2"
                    )
                if not str(loaded.get("engine_fingerprint", "")).strip():
                    raise ValueError(
                        f"service profile {resolved} requires engine_fingerprint"
                    )
                profile = loaded
                profile_source = str(resolved.resolve())

            raw_prefill_rows = profile.get(
                "prefill_curve", item.get("prefill_curve", [])
            )
            prefill_curve = _parse_curve(
                raw_prefill_rows,
                ("tpot_s_median", "seconds_per_token", "prefill_tpot_s"),
            )
            prefill_intercept_curve = _parse_curve(
                raw_prefill_rows,
                ("intercept_s",),
                allow_zero=True,
            )
            raw_decode_rows = profile.get(
                "decode_curve", item.get("decode_curve", [])
            )
            decode_curve = _parse_curve(
                raw_decode_rows,
                ("tpot_s_median", "decode_tpot_s"),
            )
            if not decode_curve:
                decode_curve = _parse_curve(
                    raw_decode_rows,
                    ("tpot_ms_median",),
                    scale=0.001,
                )

            prefill_tpot_s = float(
                profile.get("prefill_tpot_s", item.get("prefill_tpot_s", 0.0))
            )
            prefill_intercept_s = float(
                profile.get(
                    "prefill_intercept_s",
                    prefill_intercept_curve[0][1]
                    if prefill_intercept_curve
                    else item.get("prefill_intercept_s", 0.0),
                )
            )
            decode_tpot_s = float(
                profile.get(
                    "decode_base_tpot_s",
                    decode_curve[0][1]
                    if decode_curve
                    else item.get("decode_tpot_s", 0.0),
                )
            )
            if (
                prefill_tpot_s <= 0
                or prefill_intercept_s < 0
                or decode_tpot_s <= 0
            ):
                raise ValueError(
                    f"cluster {item.get('id')!r} requires positive prefill/decode service rates"
                )
            marginal = profile.get(
                "decode_marginal_tpot_s", item.get("decode_marginal_tpot_s")
            )
            clusters.append(
                ClusterSpec(
                    cluster_id=str(item["id"]),
                    replica_ids=tuple(int(rid) for rid in item["replica_ids"]),
                    rtt_ms_by_client_region={
                        str(region): float(rtt)
                        for region, rtt in item["rtt_ms_by_client_region"].items()
                    },
                    prefill_tpot_s=prefill_tpot_s,
                    decode_tpot_s=decode_tpot_s,
                    decode_marginal_tpot_s=(
                        float(marginal) if marginal is not None else None
                    ),
                    prefill_intercept_s=prefill_intercept_s,
                    prefill_tpot_curve=prefill_curve,
                    prefill_intercept_curve=prefill_intercept_curve,
                    decode_tpot_curve=decode_curve,
                    service_profile_source=profile_source,
                )
            )
        return cls(clusters, default_client_region)

    @classmethod
    def from_json(cls, path: str | Path) -> "ClusterTopology":
        topology_path = Path(path)
        with topology_path.open("r", encoding="utf-8") as topology_file:
            return cls.from_dict(json.load(topology_file), topology_path.parent)

    @property
    def replica_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._cluster_by_replica))

    def cluster(self, cluster_id: str) -> ClusterSpec:
        return self._by_id[cluster_id]

    def cluster_for_replica(self, replica_id: int) -> ClusterSpec:
        return self.cluster(self._cluster_by_replica[replica_id])

    def validate_replica_count(self, replica_count: int) -> None:
        expected = tuple(range(replica_count))
        if self.replica_ids != expected:
            raise ValueError(
                "topology replica ids must exactly match configured endpoints: "
                f"expected {expected}, got {self.replica_ids}"
            )

    def validate_service_profiles(self, allow_inline: bool = False) -> None:
        inline_clusters = [
            cluster.cluster_id
            for cluster in self.clusters
            if cluster.service_profile_source == "inline"
        ]
        if inline_clusters and not allow_inline:
            raise ValueError(
                "formal runs require an external schema-v2 service profile for every "
                f"cluster; inline profile found for {inline_clusters}. Calibrate each "
                "engine or explicitly allow inline profiles only for legacy/smoke runs."
            )
