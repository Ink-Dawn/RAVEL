from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Tuple


LATENCY = 0
THROUGHPUT = 1
COLLECTIVE = 2


@dataclass(frozen=True)
class RequestSLO:
    ttft_s: float
    tbt_s: float
    ttlt_s: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.ttft_s, self.tbt_s, self.ttlt_s


# Values used by the released JITServe paper experiment scripts.
JITSERVE_SLO_PROFILES: Mapping[str, RequestSLO] = {
    "paper_e2e": RequestSLO(0.8, 0.08, 8.0),
    "paper_collective": RequestSLO(0.8, 0.08, 8.0),
}


def build_request_slo(
    profile_name: str,
    collection_id: int,
    request_type: int,
    stage_num: int = 1,
) -> RequestSLO:
    if profile_name not in JITSERVE_SLO_PROFILES:
        raise ValueError(
            f"unknown JITServe SLO profile {profile_name!r}; choose one of "
            f"{sorted(JITSERVE_SLO_PROFILES)}"
        )
    base = JITSERVE_SLO_PROFILES[profile_name]
    multiplier = int(collection_id) % 4 + 1

    # JITServe applies the 1x-4x heterogeneity to every branch request.
    # Its workflow-level deadline is separate: base TTLT times stage count.
    # Multiplying a branch TTLT by stage_num would count that factor twice.
    # RAVEL_TTFT_SCALE optionally tightens the TTFT budget (routing
    # differentiation experiments); default 1.0 keeps the released profile.
    try:
        ttft_scale = float(os.environ.get("RAVEL_TTFT_SCALE", "1.0"))
    except ValueError:
        ttft_scale = 1.0
    return RequestSLO(
        ttft_s=base.ttft_s * multiplier * ttft_scale,
        tbt_s=base.tbt_s * multiplier,
        ttlt_s=base.ttlt_s * multiplier,
    )


def collective_task_deadline_s(
    request_ttlt_s: float,
    collection_id: int,
    stage_num: int,
) -> float:
    """Return the deadline used by JITServe's released task evaluator.

    benchmark_scheduler_deepresearch.py first applies the collection-specific
    1x-4x multiplier to the branch TTLT and then multiplies that branch budget
    by the number of workflow stages. The released client penalty path omits
    the collection multiplier; reporting follows the benchmark evaluator.
    """
    del collection_id  # The multiplier is already present in request_ttlt_s.
    return max(0.0, float(request_ttlt_s)) * max(1, int(stage_num))


def request_meets_slo(
    request_type: int,
    slo: RequestSLO,
    ttft_s: float,
    tbt_values_s: Tuple[float, ...],
    ttlt_s: float,
) -> bool:
    if request_type == LATENCY:
        return ttft_s <= slo.ttft_s and all(tbt <= slo.tbt_s for tbt in tbt_values_s)
    if request_type in (THROUGHPUT, COLLECTIVE):
        return ttlt_s <= slo.ttlt_s
    raise ValueError(f"unsupported JITServe request type: {request_type}")
