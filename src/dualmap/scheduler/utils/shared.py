import asyncio
import re
import time
import os
import random
from typing import Dict, Optional
from collections import defaultdict, deque
import math
import aiohttp
from dualmap.entities.replica import Replica
from dualmap.entities.request import Request
from dualmap.client.open_ai import async_send_request
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.logger import init_logger
from dualmap.cluster.topology import ClusterTopology
from dualmap.sidecar.waiting_movable import SidecarWaitingMovableRegistry

logger = init_logger(__name__)

class SharedState:
    def set_scheduler_callback(self, scheduler_callback):
        self._scheduler_callback = scheduler_callback

    async def _safe_scheduler_callback(self) -> None:
        callback = getattr(self, "_scheduler_callback", None)
        if callback is None:
            return
        try:
            await callback(None)
        except Exception:
            logger.exception("Control-plane replan callback failed")

    def request_replan(self) -> None:
        """Fire-and-forget control-plane replan; never blocks the data path."""
        if getattr(self, "_scheduler_callback", None) is None:
            return
        asyncio.create_task(self._safe_scheduler_callback())

    def set_prefill_started_callback(self, callback):
        self._prefill_started_callback = callback

    def set_request_terminal_callback(self, callback):
        self._request_terminal_callback = callback

    def __init__(self, metric_store, tokenizer, args):
        self.metric_store = metric_store
        self.tokenizer = tokenizer
        self.runtime_events = {} #  Dict[str, Tuple[asyncio.Event, str]], runtime_events[request._id] = (event, runtime_id)
        self.runtime_request_queue = asyncio.Queue() # Request
        self.finished_requests_queue = asyncio.Queue()
        self.posting_request_tasks = {} # {(req_id, target_ip_port): task}
        self.posting_request_tasks_lock = asyncio.Lock() #{req_id, task}
        self._prefill_started_callback = None
        self._request_terminal_callback = None
        self.replicas_ip_port = args.replicas_ip_port.split(',')
        self.num_replicas = len(self.replicas_ip_port)
        self.sidecar_waiting_movable = SidecarWaitingMovableRegistry(
            self.num_replicas
        )
        self.replica_budgets: Dict[str, Replica] = {}  # {replica_id: Replica}
        self.replica_slo_budget = args.replica_slo_budget
        self._result_path = args.result_path
        self._model_name = args.model_name
        self._network_delay_mode = str(
            getattr(args, "network_delay_mode", "synthetic")
        )
        if self._network_delay_mode not in {"synthetic", "physical"}:
            raise ValueError("network_delay_mode must be synthetic or physical")
        self._request_timeout_s = float(getattr(args, "request_active_timeout", 3600.0))
        self._http_session = None
        self._use_shared_http_session = True
        self._output_quantile = float(getattr(args, "ravel_output_quantile", 0.9))
        self._output_min_samples = int(getattr(args, "ravel_output_min_samples", 20))
        output_window = int(getattr(args, "ravel_output_history_window", 256))
        if not 0.0 < self._output_quantile <= 1.0:
            raise ValueError("ravel_output_quantile must be in (0, 1]")
        if self._output_min_samples <= 0 or output_window <= 0:
            raise ValueError("output history sizes must be positive")
        self._output_history = defaultdict(lambda: deque(maxlen=output_window))
        topology_path = getattr(args, "cluster_topology", "")
        self.cluster_topology = (
            ClusterTopology.from_json(topology_path) if topology_path else None
        )
        self.last_request_time: Optional[float] = None
        for replica_id in range(self.num_replicas):
            self.replica_budgets[replica_id] = Replica(replica_id, self.tokenizer, args)  

    def _get_http_session(self):
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=max(1.0, self._request_timeout_s))
            connector = aiohttp.TCPConnector(
                force_close=False,
                limit=0,
                limit_per_host=64,
                keepalive_timeout=30.0,
                ttl_dns_cache=3600,
            )
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
        return self._http_session

    async def close(self) -> None:
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()

    async def get_min_ttft_replica(self, request: Request):
        chose_replica_id = -1
        target_actual_prefill_len = -1
        replicas_pending_waiting_budget = {}      
        actual_num_prefill_tokens_list = {}
        for rid in range(self.num_replicas):
            replica = self.replica_budgets[rid]
            actual_num_prefill_tokens = replica.get_num_recompute_token_ids(request._input_ids)
            current_budget, last_prefill_completed_at, last_ttft, num_pending_requests, qps = await replica.get_load_states()
            replicas_pending_waiting_budget[rid] = current_budget - actual_num_prefill_tokens
            logger.debug(f"req_id={request._id},replicas_pending_waiting_budget[{rid}]={replicas_pending_waiting_budget[rid]}={current_budget}-{actual_num_prefill_tokens}")
            actual_num_prefill_tokens_list[rid] = actual_num_prefill_tokens
        if replicas_pending_waiting_budget:
            max_pending_waiting_budget = max(replicas_pending_waiting_budget.values())
            candidates = [rid for rid, pending_waiting_budget in replicas_pending_waiting_budget.items() if pending_waiting_budget == max_pending_waiting_budget]
            rnd = random.Random(42)
            chose_replica_id = rnd.choice(candidates)
            target_actual_prefill_len = actual_num_prefill_tokens_list[chose_replica_id]
        else:
            chose_replica_id = -1  # or some other default value
        pending_tokens = self.replica_slo_budget - (replicas_pending_waiting_budget[chose_replica_id] + target_actual_prefill_len)
        logger.debug(f"req_id={request._id},chose replica_id={chose_replica_id},waiting_tokens={pending_tokens},actual_prefill_len={target_actual_prefill_len}")
        return chose_replica_id, target_actual_prefill_len

    async def abort_posting_request_tasks(
        self, replica_id, request: Request, task_attempt: Optional[int] = None
    ):
        target_ip_port = self.replicas_ip_port[replica_id]
        key = (request._id, target_ip_port)
        async with self.posting_request_tasks_lock:
            item = self.posting_request_tasks.get(key)
            if item is None:
                return
            task, stored_attempt = item
            if (
                task_attempt is not None
                and int(stored_attempt) != int(task_attempt)
            ):
                return
            if not task.done():
                task.cancel()
                logger.info(f"Abort:Cancelled task for request {request._id} on {target_ip_port}")
            else:
                task.exception()
            del self.posting_request_tasks[key]
            logger.debug(f"Abort:Removed task for request {request._id} on {target_ip_port}")

    async def del_posting_request_tasks(
        self,
        request: Request,
        target_ip_port: str,
        task_attempt: Optional[int] = None,
    ):
        key = (request._id, target_ip_port)
        async with self.posting_request_tasks_lock:
            item = self.posting_request_tasks.get(key)
            if item is None:
                return False
            task, stored_attempt = item
            if task_attempt is not None and int(stored_attempt) != int(task_attempt):
                logger.debug(
                    f"Stale task cleanup ignored for {key} "
                    f"attempt {stored_attempt} vs {task_attempt}"
                )
                return False
            cancelled = False
            if not task.done():
                task.cancel()
                cancelled = True
                logger.info(f"Cancelled task for request {request._id} on {target_ip_port}")
            else:
                task.exception()
            del self.posting_request_tasks[key]
            logger.debug(f"Removed task for request {request._id} on {target_ip_port}")
            return cancelled

    async def on_request_complete(
        self,
        output: RequestFuncOutput,
        request: Request,
        target_ip_port: str,
        task_attempt: Optional[int] = None,
    ):
        self.last_request_time = time.perf_counter()
        if request is None:
            logger.error(f"Completion error on {target_ip_port}, request is None")
            return
        request_id = getattr(request, "_id", None)
        try:
            if output is not None and output.success and int(output.output_len or 0) > 0:
                self._output_history[request._request_type].append(int(output.output_len))
            await self.del_posting_request_tasks(
                request, target_ip_port, task_attempt
            )
            await self.finished_requests_queue.put((output, request._prompts, request._input_ids))
            self.request_replan()
        except Exception as e:
            logger.error(f"Completion error for {request_id} on {target_ip_port}: {e}")
        finally:
            if (
                hasattr(self, "_request_terminal_callback")
                and self._request_terminal_callback is not None
            ):
                await self._request_terminal_callback(
                    request_id,
                    task_attempt
                    if task_attempt is not None
                    else int(getattr(request, "_attempt", 0)),
                )

    async def on_request_task_failed(
        self,
        request: Request,
        target_ip_port: str,
        replica: Replica,
        task_attempt: Optional[int] = None,
    ) -> None:
        request_id = getattr(request, "_id", None)
        try:
            await replica.abort_request(request)
        finally:
            await self.del_posting_request_tasks(
                request, target_ip_port, task_attempt
            )
        if (
            hasattr(self, "_request_terminal_callback")
            and self._request_terminal_callback is not None
        ):
            await self._request_terminal_callback(
                request_id,
                task_attempt
                if task_attempt is not None
                else int(getattr(request, "_attempt", 0)),
            )
        self.request_replan()

    async def abort_in_flight(
        self,
        request: Request,
        replica_id: int,
        task_attempt: Optional[int] = None,
    ) -> bool:
        """Cancel the matching active HTTP attempt before ledger removal."""

        target_ip_port = self.replicas_ip_port[replica_id]
        cancelled = await self.del_posting_request_tasks(
            request,
            target_ip_port,
            task_attempt,
        )
        if not cancelled:
            return False
        await self.replica_budgets[replica_id].abort_request(request)
        return True


    def get_routing_output_hint(
        self, request_type: int, fallback_tokens: int
    ) -> int:
        """Return a causal empirical output-demand quantile.

        Only completion events already observed by the Router enter the
        history. The current request's configured/trace output length is not
        an input. Before the minimum sample count, the public fallback is
        returned unchanged.
        """
        fallback = max(1, int(fallback_tokens))
        samples = list(self._output_history[int(request_type)])
        if len(samples) < self._output_min_samples:
            return fallback
        rank = max(1, math.ceil(self._output_quantile * len(samples)))
        empirical = max(1, int(sorted(samples)[rank - 1]))
        # Completion order is length-biased under concurrency: short outputs
        # enter history first. Never let that censored sample lower the
        # request-visible conservative budget.
        return max(fallback, empirical)

    def get_max_cache_hit_replica(self, request: Request):
        replica_id = -1
        prefix_cache_hit_len_list = {} #{replica_id:actual_num_prefill_tokens}
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            actual_num_prefill_tokens = replica.get_num_recompute_token_ids(request._input_ids)
            prefix_cache_hit_len_list[replica_id] = max(0, len(request._input_ids) - actual_num_prefill_tokens)
        if prefix_cache_hit_len_list:
            replica_id = max(prefix_cache_hit_len_list.items(), key=lambda x: x[1])[0]
        else:
            replica_id = -1  # or some other default value
        return replica_id

    async def add_posting_request_tasks(self, replica_id, request: Request):
        self.last_request_time = time.perf_counter()
        logger.info(f"Adding task for request {request._id} to replica {replica_id},session_id={request._native_session_id}")
        replica = self.replica_budgets[replica_id]
        if not await replica.add_request(request):
            return False
        target_ip_port = self.replicas_ip_port[replica_id]
        if self.cluster_topology is not None:
            cluster = self.cluster_topology.cluster_for_replica(replica_id)
            if not request._primary_cluster:
                request._primary_cluster = cluster.cluster_id
            request._network_rtt_s = cluster.rtt_s(request._client_region)
            request._inject_network_delay = (
                self._network_delay_mode == "synthetic"
            )
        task_key = (request._id, target_ip_port)
        # logger.info(f"Creating task for {task_key}")
        max_cache_hit_replcia = self.get_max_cache_hit_replica(request)
        if max_cache_hit_replcia == replica_id:
            request._rounting_cache_hit_max = 1
        else:
            request._rounting_cache_hit_max = 0
        async with self.posting_request_tasks_lock:
            try:
                task_attempt = int(getattr(request, "_attempt", 0))
                prefill_cb = getattr(self, "_prefill_started_callback", None)

                async def on_prefill(request_id):
                    if prefill_cb is None:
                        return None
                    return await prefill_cb(request_id, task_attempt)

                output_task = asyncio.create_task(
                    async_send_request(
                        metric_store=self.metric_store,
                        result_path=self._result_path,
                        model_name=self._model_name,
                        replica_id=replica_id,
                        native_session_id=request._native_session_id,
                        target_ip_port=target_ip_port,
                        request=request,
                        replica=replica,
                        scheduler_callback=getattr(self, "_scheduler_callback", None),
                        on_prefill_started=on_prefill,
                        request_timeout_s=self._request_timeout_s,
                        http_session=(
                            self._get_http_session()
                            if self._use_shared_http_session
                            else None
                        ),
                    )
                )

                def handle_task_completion(task):
                    key = (request._id, target_ip_port)
                    try:
                        result = task.result()
                        asyncio.create_task(
                            self.on_request_complete(
                                result, request, target_ip_port, task_attempt
                            )
                        )
                    except asyncio.CancelledError:
                        logger.info(f"Task {key} cancelled before completion")
                        asyncio.create_task(
                            self.on_request_task_failed(
                                request,
                                target_ip_port,
                                replica,
                                task_attempt,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Task {key} failed: {str(e)}")
                        asyncio.create_task(
                            self.on_request_task_failed(
                                request,
                                target_ip_port,
                                replica,
                                task_attempt,
                            )
                        )
                
                output_task.add_done_callback(handle_task_completion)
                self.posting_request_tasks[task_key] = (output_task, task_attempt)
                return True
                # logger.debug(f"Registered task {task_key}")
                
            except Exception as e:
                logger.error(f"Task creation failed for {request._id}: {str(e)}")
                await replica.abort_request(request)
                await self.del_posting_request_tasks(request, target_ip_port)
                raise

    async def schedule_request_to_replica(self, request, runtime_id): 
        event = None
        try:
            event, _ = self.runtime_events[request._id]
            self.runtime_events[request._id] = (event, runtime_id)
        except Exception as e:
            logger.error(f"Selection error: {e}")
            self.runtime_events[request._id] = (event, None)
        finally:
            if event is not None:
                event.set()
                self.runtime_request_queue.task_done()

    def record_replica_num_pending_request(self, base_path, cur_request_id):
        num_request_pending_list = []
        num_request_running_list = []
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            num_request_pending = len(replica.pending_requests)
            num_request_pending_list.append(num_request_pending)
            num_request_running_list.append(replica.get_num_running_req())

        os.makedirs(base_path, exist_ok=True)
        try:
            with open (f'{base_path}/number_pending_requests.log', "a+") as file:
                for replica_id in range(len(num_request_pending_list)):
                    file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_pending_requests,{num_request_pending_list[replica_id]},number_running_requests,{num_request_running_list[replica_id]}\n')
            file.close()
        except Exception as e:
            print(f'error:MetricsConfig:save number_pending_requests failed! {e}')    


    def record_replica_num_pending_tokens(self, base_path, cur_request_id):
        number_pending_tokens_list = []
        for replica_id in range(self.num_replicas):
            replica = self.replica_budgets[replica_id]
            number_pending_tokens = self.replica_slo_budget - replica.current_budget
            number_pending_tokens_list.append(number_pending_tokens)

        os.makedirs(base_path, exist_ok=True)
        try:
            with open (f'{base_path}/number_pending_tokens.log', "a+") as file:
                for replica_id in range(len(number_pending_tokens_list)):
                    file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_pending_tokens,{number_pending_tokens_list[replica_id]}\n')
            file.close()
        except Exception as e:
            print(f'error:MetricsConfig:save number_pending_tokens failed! {e}')

    def record_replica_num_pending_input_tokens(self, base_path, cur_request_id):
            number_input_tokens_list = []
            for replica_id in range(self.num_replicas):
                replica = self.replica_budgets[replica_id]
                total_input_tokens = sum(len(req._input_ids) for req in replica.pending_requests)
                number_input_tokens_list.append(total_input_tokens)

            os.makedirs(base_path, exist_ok=True)
            try:
                with open(f'{base_path}/number_pending_input_tokens.log', "a+") as file:
                    for replica_id in range(len(number_input_tokens_list)):
                        file.write(f'cur_request_id,{cur_request_id},replica_id,{replica_id},number_input_tokens,{number_input_tokens_list[replica_id]}\n')
            except Exception as e:
                print(f'error:MetricsConfig:save number_input_tokens failed! {e}')

    def get_pending_input_tokens_replica(self,replica_id):
        replica = self.replica_budgets[replica_id]
        total_input_tokens = sum(len(req._input_ids) for req in replica.pending_requests)
        return total_input_tokens

    def get_num_actual_pending_tokens_replica(self,replica_id):
        replica = self.replica_budgets[replica_id]
        return replica.get_num_actual_pending_tokens()

    def dump_replica_queue_info(self, replica_id):
        replica = self.replica_budgets[replica_id]
        logger.debug(f"local_info:replica={replica_id},"
        f"num_pending_req={replica.get_num_pending_req()},{replica.get_num_pending_req_info()},"
        f"running_req_blocks_cnt={replica.get_running_req_blocks_cnt()},"
        f"num_running_req={replica.get_num_running_req()},{replica.get_num_running_req_info()},")
