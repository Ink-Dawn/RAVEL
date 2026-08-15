import requests
import aiohttp
import asyncio
import time
import json
import sys
import os
import traceback
from contextlib import asynccontextmanager
from typing import List, Optional
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.entities.request import Request
from dualmap.entities.replica import Replica
from dualmap.cluster.slo import RequestSLO, request_meets_slo
from dualmap.logger import init_logger
logger = init_logger(__name__)
last_request_time = Optional[float]

@asynccontextmanager
async def request_session(timeout, shared_session=None):
    if shared_session is not None:
        yield shared_session
        return
    async with aiohttp.ClientSession(timeout=timeout) as session:
        yield session

def update_request_time():
    # logger.debug(f'update_request_time')
    last_request_time = time.perf_counter() 

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def endpoint_url(endpoint: str, path: str) -> str:
    base = str(endpoint).rstrip("/")
    if "://" not in base:
        base = f"http://{base}"
    return f"{base}{path}"


async def async_send_request(metric_store, result_path, model_name, 
        replica_id: int = None, native_session_id: str = None, 
        target_ip_port=None, request: Request = None,
        replica: Replica = None,
        scheduler_callback=None,
        on_prefill_started=None,
        request_timeout_s: float = 3600.0,
        http_session=None,
        _connect_retries_remaining: int = 2):
    if request is None or target_ip_port is None or replica is None:
        logger.error(f"async_send_request:invalid parameters:request={request},target_ip_port={target_ip_port},replica={replica}")
        return None  

    output_token_len = 0
    prompt_token_len = 0
    tbt_token_count = 0
    tbt_event_count = 0
    tbt_measurement_valid = True
    request_start_time = min(time.perf_counter(),request._arrived_at)
    wait_queue_s = max(0.0, time.perf_counter() - request._arrived_at)
    network_rtt_s = max(0.0, float(getattr(request, "_network_rtt_s", 0.0)))
    inject_network_delay = bool(
        getattr(request, "_inject_network_delay", True)
    )
    most_recent_timestamp = request_start_time
    time_to_first_token = 0.0 # ttft
    first_token_flag = False
    interList: List[float] = []
    generator_text = ""
    request_latency = 0.0
    num_request_pending = -1
    # for preble
    start_time = time.time()
    ttft = 0
    output = RequestFuncOutput()
    scheduling_overhead = time.time() - start_time
    output.request_id = str(request._id)
    output.session_id = str(request._session_id)
    output.round_id = str(request._round_id)
    output.prompt_text = request._prompts
    output.prompt_len = len(request._prompts)
    output.runtime_selected = replica_id
    num_request_pending = len(replica.pending_requests)

    replica_url = endpoint_url(
        str(target_ip_port), "/v1/chat/completions"
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": request._prompts}],
        "n": 1,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": request._output_len,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
        "top_logprobs": 0,
    }
    if os.environ.get("RAVEL_ENGINE_PRIORITY", "0") == "1":
        payload["priority"] = int(
            getattr(request, "_vllm_priority", 0)
        )
    headers = {
        'Content-Type': 'application/json'
    }
    api_key = os.environ.get("RAVEL_API_KEY") or os.environ.get(
        "VLLM_API_KEY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    logger.info(f"async_send_request:request._id={request._id},session={native_session_id},{target_ip_port}")

    data = {
        "request_id": str(request._id),
        "dataset_type": request._dataset_type,
        "request_start_time": request_start_time,
        "request_end_time": 0,
        "native_session_id": native_session_id,
        "round_id": request._round_id,
        "replica_id": replica_id,
        "cluster_id": request._primary_cluster,
        "client_region": request._client_region,
        "collection_id": request._collection_id,
        "request_type": request._request_type,
        "stage_id": request._stage_id,
        "stage_num": request._stage_num,
        "task_branch_count": request._task_branch_count,
        "time_to_first_token": 3600000,
        "request_latency": 3600000,
        "max_tbt": 3600000,
        "TPS(tokens/s)": 3600000,
        "tpot(ms)": 3600000,
        "wait_queue_s": round(wait_queue_s, 6),
        "policy_induced_hold_s": round(
            float(getattr(request, "_policy_induced_hold_s", 0.0)), 6
        ),
        "policy_induced_hold_budget_s": round(
            float(
                getattr(request, "_policy_induced_hold_budget_s", 0.0)
            ),
            6,
        ),
        "network_rtt_s": round(network_rtt_s, 6),
        "network_delay_mode": (
            "synthetic" if inject_network_delay else "physical"
        ),
        "num_request_pending": num_request_pending,
        "input_len": request._num_prefill_tokens,
        "actual_prompt_tokens": 0,
        "output_len": request._output_len,
        "actual_output_tokens": 0,
        "routing_output_tokens_hint": request._routing_output_tokens_hint,
        "routing_output_tokens_hint_used": request._routing_output_tokens_hint_used,
        "routing_output_tokens_upper_hint": request._routing_output_tokens_upper_hint,
        "routing_output_tokens_upper_hint_used": request._routing_output_tokens_upper_hint_used,
        "pd_ratio": round(request._num_prefill_tokens/request._output_len,2),
        "actual_num_prefill_tokens": request._actual_num_prefill_tokens,
        "admission_prompt_charge_tokens": request._actual_num_prefill_tokens,
        "prefix_hit_prompt_tokens": request._estimated_prefix_hit_tokens,
        "estimated_prefix_hit_prompt_tokens": request._estimated_prefix_hit_tokens,
        "prefix_hit_source": request._prefix_hit_source,
        "tbt_source": "stream_logprob_token_interarrival",
        "tbt_token_count": 0,
        "tbt_event_count": 0,
        "tbt_measurement_valid": 0,
        "slo_ttft_s": request._slo_constraint[0],
        "slo_tbt_s": request._slo_constraint[1],
        "slo_ttlt_s": request._slo_constraint[2],
        "ttft_slo_met": 0,
        "tbt_slo_met": 0,
        "ttlt_slo_met": 0,
        "request_slo_met": 0,
        "predicted_ttft_s": request._predicted_ttft_s,
        "point_predicted_ttft_s": request._point_predicted_ttft_s,
        "predicted_objective_s": request._predicted_objective_s,
        "point_predicted_objective_s": request._point_predicted_objective_s,
        "ttft_residual_guard_s": request._ttft_residual_guard_s,
        "objective_residual_guard_s": request._objective_residual_guard_s,
        "risk_calibrated": int(request._risk_calibrated),
        "ttft_prediction_residual_s": request._ttft_prediction_residual_s,
        "objective_prediction_residual_s": request._objective_prediction_residual_s,
        "cluster_route_feasible": int(request._cluster_route_feasible),
        "cluster_route_reason": request._cluster_route_reason,
        "rebind_count": int(getattr(request, "_rebind_count", 0)),
        "sidecar_soft_moves": int(getattr(request, "_ravel_soft_moves", 0)),
        "abort_attempts": int(getattr(request, "_abort_attempts", 0)),
        "abort_completed": int(getattr(request, "_abort_completed", 0)),
        "attempt": int(getattr(request, "_attempt", 0)),
        "engine_priority": int(getattr(request, "_vllm_priority", 0)),
        "ravel_prefill_gap_s": round(
            float(getattr(request, "_ravel_prefill_gap_s", 0.0)), 6
        ),
        "ravel_protected_tbt_s": round(
            float(getattr(request, "_ravel_protected_tbt_s", 0.0)), 6
        ),
        "ravel_prefill_chunk_tokens": int(
            getattr(request, "_ravel_prefill_chunk_tokens", 0)
        ),
        "ravel_admission_deferrals": int(
            getattr(request, "_ravel_admission_deferrals", 0)
        ),
        "ravel_max_admission_victims": int(
            getattr(request, "_ravel_max_admission_victims", 0)
        ),
        "ravel_admission_victims": int(
            getattr(request, "_ravel_admission_victims", 0)
        ),
        "ravel_admission_envelope_safe": int(
            bool(getattr(request, "_ravel_admission_envelope_safe", True))
        ),
        "ravel_yield_deferred": int(
            bool(getattr(request, "_ravel_yield_deferred", False))
        ),
        "ravel_soft_admission_active": int(
            bool(getattr(request, "_ravel_soft_admission_active", False))
        ),
        "ravel_soft_admission_protected": int(
            bool(getattr(request, "_ravel_soft_admission_protected", False))
        ),
        "ravel_soft_admission_slack_release": int(
            bool(
                getattr(
                    request,
                    "_ravel_soft_admission_slack_release",
                    False,
                )
            )
        ),
        "ravel_soft_admission_min_slack_s": round(
            float(
                getattr(
                    request,
                    "_ravel_soft_admission_min_slack_s",
                    0.0,
                )
            ),
            6,
        ),
        "ravel_soft_admission_increment_s": round(
            float(
                getattr(
                    request,
                    "_ravel_soft_admission_increment_s",
                    0.0,
                )
            ),
            6,
        ),
        "ravel_soft_admission_victims": int(
            getattr(request, "_ravel_soft_admission_victims", 0)
        ),
        "ravel_soft_admission_hold_budget_s": round(
            max(
                0.0,
                float(
                    getattr(
                        request, "_ravel_soft_admission_release_at", 0.0
                    )
                )
                - float(request._arrived_at),
            ),
            6,
        ),
        "ravel_soft_plan_generation": int(
            getattr(request, "_ravel_soft_plan_generation", -1)
        ),
        "ravel_profile_output_expected": int(
            getattr(request, "_ravel_profile_output_expected", 0)
        ),
        "ravel_profile_output_upper": int(
            getattr(request, "_ravel_profile_output_upper", 0)
        ),
        "request_end_time": time.perf_counter(),
        "req_arrived_at": request._arrived_at,
        "time_interval": request._time_interval,
        "scheduled_arrived_at": round(float(getattr(request, "_scheduled_arrived_at", 0.0)), 6),
        "emitted_at": round(float(getattr(request, "_emitted_at", 0.0)), 6),
        "emission_lag_s": round(float(getattr(request, "_emission_lag_s", 0.0)), 6),
        "rounting_cache_hit_max": request._rounting_cache_hit_max,
        "is_dh_cache_affinity": request._is_dh_cache_affinity,
        "is_dh_least_loaded": request._is_dh_least_loaded,
        "is_dh_cache_affinity_least_loaded":request._is_dh_cache_affinity_least_loaded
    }
    await metric_store.insert_metrics(data)

    timeout = aiohttp.ClientTimeout(total=max(1.0, float(request_timeout_s)))
    response_status = 0
    if inject_network_delay and network_rtt_s > 0:
        await asyncio.sleep(network_rtt_s / 2.0)
    try:
        async with request_session(timeout, http_session) as session:
            try:
                async with session.post(url=replica_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_status = 200
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ").strip()
                            if chunk == "[DONE]":
                                request_latency = time.perf_counter() - request_start_time
                                output.success = True
                                output.request_latency = time.perf_counter() - request_start_time # for preble
                                replica.complete_request_decode(request, generator_text)
                            else:
                                data = json.loads(chunk)
                                timestamp = time.perf_counter()
                                if data.get("usage"):
                                    usage = data["usage"]
                                    prompt_token_len = int(
                                        usage.get("prompt_tokens", prompt_token_len)
                                    )
                                    output_token_len = int(
                                        usage.get("completion_tokens", output_token_len)
                                    )
                                    output.output_len = output_token_len
                                    output.max_new_tokens = output_token_len
                                choices = data.get("choices") or []
                                if not choices:
                                    continue
                                choice = choices[0]
                                delta = choice.get("delta", {})
                                content = delta.get("content") or ""
                                logprobs = choice.get("logprobs") or {}
                                token_entries = logprobs.get("content") or []
                                chunk_token_count = len(token_entries)
                                has_output = chunk_token_count > 0 or bool(content)
                                if has_output:
                                    if chunk_token_count <= 0:
                                        tbt_measurement_valid = False
                                        chunk_token_count = 1
                                    if first_token_flag is False:  # First token
                                        if (
                                            inject_network_delay
                                            and network_rtt_s > 0
                                        ):
                                            await asyncio.sleep(network_rtt_s / 2.0)
                                        timestamp = time.perf_counter()
                                        logger.debug(f'async_send_request: request finish prefill, {request._id}')
                                        time_to_first_token = timestamp - request_start_time
                                        ttft = time_to_first_token  # for preble
                                        output.ttft = ttft  # for preble
                                        first_token_flag = True
                                        request._ravel_ttft_alive = (
                                            time_to_first_token
                                            <= float(request._slo_constraint[0])
                                        )
                                        request._ravel_tbt_alive = True
                                        request._ravel_first_token_at = timestamp
                                        request._ravel_last_token_at = timestamp
                                        extra_tokens = max(0, chunk_token_count - 1)
                                        if extra_tokens:
                                            interList.extend([0.0] * extra_tokens)
                                            output.itl.extend([0.0] * extra_tokens)
                                            tbt_token_count += extra_tokens
                                        logger.info(f"async_send_request: prefill completed: req={request._id},replica_id={replica_id}")
                                        success = await replica.complete_request_prefill(replica_id, request, ttft)
                                        if on_prefill_started is not None:
                                            try:
                                                await on_prefill_started(request._id)
                                            except Exception:
                                                logger.exception(
                                                    "prefill-started callback failed for req %s",
                                                    request._id,
                                                )
                                        if not success:
                                            logger.warning(f"Request {request} not found in replica {replica_id}")
                                        elif scheduler_callback is not None:
                                            try:
                                                await scheduler_callback(None)
                                            except Exception:
                                                logger.exception(
                                                    "scheduler callback failed for req %s",
                                                    request._id,
                                                )
                                    else:  # Decode token(s)
                                        per_token_tbt = (
                                            timestamp - most_recent_timestamp
                                        ) / chunk_token_count
                                        request._ravel_last_token_at = timestamp
                                        if (
                                            request._request_type == 0
                                            and per_token_tbt
                                            > float(request._slo_constraint[1])
                                        ):
                                            request._ravel_tbt_alive = False
                                        interList.extend(
                                            [per_token_tbt] * chunk_token_count
                                        )
                                        output.itl.extend(
                                            [per_token_tbt] * chunk_token_count
                                        )
                                        tbt_token_count += chunk_token_count
                                        tbt_event_count += 1
                                    generator_text += content
                                    most_recent_timestamp = timestamp
                    else:
                        output.error = response.reason or ""
                        output.success = False # for preble
                        logger.debug(f"response.status={response.status}:request._id={request._id},session={native_session_id},{target_ip_port}")
                        logger.debug(f"bad prompts:{target_ip_port}:output.error ={output.error}:request._id={request._id},input={request._num_prefill_tokens},output={request._output_len}")
            except aiohttp.ClientConnectorError:
                if _connect_retries_remaining > 0:
                    retry_index = 3 - _connect_retries_remaining
                    retry_delay_s = 0.1 * (2 ** retry_index)
                    logger.warning(
                        "Connection establishment failed for req %s; "
                        "retrying in %.1fs (%s retries remain)",
                        request._id,
                        retry_delay_s,
                        _connect_retries_remaining,
                    )
                    await asyncio.sleep(retry_delay_s)
                    return await async_send_request(
                        metric_store=metric_store,
                        result_path=result_path,
                        model_name=model_name,
                        replica_id=replica_id,
                        native_session_id=native_session_id,
                        target_ip_port=target_ip_port,
                        request=request,
                        replica=replica,
                        scheduler_callback=scheduler_callback,
                        on_prefill_started=on_prefill_started,
                        request_timeout_s=request_timeout_s,
                        http_session=http_session,
                        _connect_retries_remaining=(
                            _connect_retries_remaining - 1
                        ),
                    )
                raise
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                logger.debug(f'async_send_request:Exception: {target_ip_port}: {output.error}')

    except asyncio.CancelledError as e:
        logger.debug(f"{target_ip_port}:Request {request._id} was cancelled:{str(e)}")
        output.success = False
        output.error = "Request cancelled"
    except Exception as e:
        logger.debug(f"{target_ip_port}:Request {request._id} failed: {str(e)}")
        output.success = False
        if not output.error:
            output.error = str(e)

    update_request_time()
    if request._id % 10 == 0:
        await metric_store.save_cache()
    # preble: throughput as token generated per second
    output.scheduling_overhead = scheduling_overhead
    if output.success:
        await replica.complete_request_objective(request, request_latency)
        # logger.debug(f"response.status={200}:request._id={request._id},session={native_session_id},{target_ip_port}")
        logger.debug(f"good request:{target_ip_port}:good,request._id={request._id},input={request._num_prefill_tokens},output={request._output_len}")
        measured_output_tokens = output_token_len or request._output_len
        output.tpot = (
            (output.request_latency - output.ttft)
            / max(1, measured_output_tokens - 1)
        )
        slo = RequestSLO(*request._slo_constraint)
        expected_tbt_tokens = max(0, measured_output_tokens - 1)
        tbt_measurement_valid = (
            tbt_measurement_valid
            and output_token_len > 0
            and tbt_token_count == expected_tbt_tokens
        )
        tbt_values = tuple(interList)
        max_tbt = max(tbt_values, default=0.0)
        ttft_slo_met = ttft <= slo.ttft_s
        tbt_slo_met = tbt_measurement_valid and all(
            tbt <= slo.tbt_s for tbt in tbt_values
        )
        ttlt_slo_met = request_latency <= slo.ttlt_s
        request_slo_met = request_meets_slo(
            request._request_type,
            slo,
            ttft,
            tbt_values,
            request_latency,
        )
        if request._request_type == 0 and not tbt_measurement_valid:
            request_slo_met = False
        updates = {
            "time_to_first_token": ttft,
            "request_end_time": time.perf_counter(),
            "request_latency": round(request_latency, 4),
            "max_tbt": round(max_tbt, 4),
            "TPS(tokens/s)": round(measured_output_tokens/request_latency, 4),
            "tpot(ms)": round(output.tpot*1000, 4),
            "sidecar_soft_moves": int(getattr(request, "_ravel_soft_moves", 0)),
            "abort_attempts": int(getattr(request, "_abort_attempts", 0)),
            "abort_completed": int(getattr(request, "_abort_completed", 0)),
            "attempt": int(getattr(request, "_attempt", 0)),
            "engine_priority": int(getattr(request, "_vllm_priority", 0)),
            "ravel_yield_deferred": int(
                bool(getattr(request, "_ravel_yield_deferred", False))
            ),
            "ravel_soft_admission_active": int(
                bool(getattr(request, "_ravel_soft_admission_active", False))
            ),
            "ravel_soft_admission_protected": int(
                bool(
                    getattr(
                        request, "_ravel_soft_admission_protected", False
                    )
                )
            ),
            "ravel_soft_admission_slack_release": int(
                bool(
                    getattr(
                        request,
                        "_ravel_soft_admission_slack_release",
                        False,
                    )
                )
            ),
            "ravel_soft_admission_min_slack_s": round(
                float(
                    getattr(
                        request,
                        "_ravel_soft_admission_min_slack_s",
                        0.0,
                    )
                ),
                6,
            ),
            "ravel_soft_admission_increment_s": round(
                float(
                    getattr(
                        request,
                        "_ravel_soft_admission_increment_s",
                        0.0,
                    )
                ),
                6,
            ),
            "ravel_soft_admission_victims": int(
                getattr(request, "_ravel_soft_admission_victims", 0)
            ),
            "ravel_soft_plan_generation": int(
                getattr(request, "_ravel_soft_plan_generation", -1)
            ),
            "ravel_profile_output_expected": int(
                getattr(request, "_ravel_profile_output_expected", 0)
            ),
            "ravel_profile_output_upper": int(
                getattr(request, "_ravel_profile_output_upper", 0)
            ),
            "actual_num_prefill_tokens": request._actual_num_prefill_tokens,
            "admission_prompt_charge_tokens": request._actual_num_prefill_tokens,
            "actual_prompt_tokens": prompt_token_len,
            "actual_output_tokens": output_token_len,
            "ttft_slo_met": int(ttft_slo_met),
            "tbt_slo_met": int(tbt_slo_met),
            "ttlt_slo_met": int(ttlt_slo_met),
            "tbt_token_count": tbt_token_count,
            "tbt_event_count": tbt_event_count,
            "tbt_measurement_valid": int(tbt_measurement_valid),
            "prefix_hit_prompt_tokens": request._estimated_prefix_hit_tokens,
            "estimated_prefix_hit_prompt_tokens": request._estimated_prefix_hit_tokens,
            "prefix_hit_source": request._prefix_hit_source,
            "routing_output_tokens_hint_used": request._routing_output_tokens_hint_used,
            "routing_output_tokens_upper_hint": request._routing_output_tokens_upper_hint,
            "routing_output_tokens_upper_hint_used": request._routing_output_tokens_upper_hint_used,
            "risk_calibrated": int(request._risk_calibrated),
            "request_slo_met": int(request_slo_met),
            "ttft_prediction_residual_s": request._ttft_prediction_residual_s,
            "objective_prediction_residual_s": request._objective_prediction_residual_s,
        }
        await metric_store.update_metrics(str(request._id), updates) 
    else:
        logger.debug(f"failed:request._id={request._id},session={native_session_id},{target_ip_port}")
        logger.debug(f"bad request:{target_ip_port}:good,request._id={request._id},input={request._num_prefill_tokens},output={request._output_len}")
        logger.debug(f"output failed:request._id={request._id},session={native_session_id},{target_ip_port}")
        await replica.abort_request(request)
    return output
