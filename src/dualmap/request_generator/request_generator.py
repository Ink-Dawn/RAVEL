import asyncio
from typing import Optional
import time
import json
import os
import statistics
from datetime import datetime
from dualmap.scheduler.utils.shared import SharedState
from dualmap.scheduler.utils.lazy_prefix_table import LazyPrefixTable,HotPrefixDetector,LazyExpansionController
from itertools import count
from dualmap.entities.request import Request
from dualmap.logger import init_logger
from dualmap.cluster.slo import build_request_slo
from dualmap.request_generator.jitserve_trace import (
    assign_request_types,
    load_jitserve_records,
    parse_request_ratio,
    stable_prefix_block_hashes,
    trace_time_ms,
)

logger = init_logger(__name__)
request_id_counter = count()

class RequestGenerator:
    def __init__(self, shared_state: SharedState, tokenizer, num_replicas, args):
        self.shared_state = shared_state
        self._args = args
        self._num_replicas = num_replicas
        self._max_model_len = args.max_model_len
        self._dataset_type = args.dataset_type
        self._request_dataset_dir = args.request_dataset_dir
        self._dataset_file = args.dataset_file
        self._trace_format = getattr(args, "trace_format", "auto")
        self._slo_profile = getattr(args, "slo_profile", "paper_e2e")
        ratio_value = getattr(args, "jitserve_request_ratio", "")
        self._jitserve_request_ratio = parse_request_ratio(ratio_value) if ratio_value else None
        self._trace_seed = int(getattr(args, "trace_seed", 42))
        self._client_region = getattr(args, "client_region", "local")
        self._routing_output_tokens_hint = int(
            getattr(args, "routing_output_tokens_hint", 256)
        )
        self._request_generate_qps = float(args.request_generate_qps)
        self._arrival_speedup = float(getattr(args, "arrival_speedup", 0.0))
        if self._arrival_speedup < 0:
            raise ValueError("arrival_speedup cannot be negative")
        self._num_request = args.request_num
        self._warm_up_requests_num = args.warm_up_requests_num
        self._warm_up_qps = float(getattr(args, "warm_up_qps", 0.5) or 0.5)
        self._requests_num_dataset_start = args.requests_num_dataset_start
        self._cur_num_request = 0
        self._active_timeout = args.request_active_timeout  #s
        self._is_finished = False
        self.prefix_table = LazyPrefixTable()
        self.hot_prefix_detector = HotPrefixDetector()
        self.prefix_expansion_ctrl = LazyExpansionController(self.prefix_table, self.hot_prefix_detector)   

    def is_request_active(self) -> bool:
        last_request_time = self.shared_state.last_request_time
        if last_request_time is None:
            return True
        if self._is_finished:
            return False
        if not last_request_time:
            last_request_time = time.perf_counter()
            self.shared_state.last_request_time = time.perf_counter() 
        return time.perf_counter() - last_request_time < self._active_timeout

    async def _generate_request_helper(self, request: Request, native_session_id: str):
        # logger.info(f"_generate_request_helper: {request._id}, {request._hash_session_id}")
        if request is None:
            return
        start = time.perf_counter()
        event = asyncio.Event()
        self.shared_state.runtime_events[request._id] = (event, None)
        await self.shared_state.runtime_request_queue.put((request))

        start = time.perf_counter()
        await self.shared_state.runtime_events[request._id][0].wait()
        replica_id = self.shared_state.runtime_events[request._id][1]
        self.shared_state.runtime_events.pop(request._id)

        if replica_id is None:
            raise RuntimeError("Runtime selection failed")
        if replica_id < 0:
           return 
        if replica_id >= 0 and replica_id < self._num_replicas:
            await self.shared_state.add_posting_request_tasks(replica_id, request)

    async def generate_request_offline(self, record: dict, time_interval, dataset_type) -> dict:
        prompts = record["prompts"]
        input_ids = record["input_ids"]
        output_len = record["output_len"]
        g_session_id = record["g_session_id"]
        hash_session_id = record["hash_session_id"]
        hash_ids = record["hash_ids"]
        if g_session_id == "" or hash_session_id == ""\
            or prompts == "" or len(input_ids) > self._max_model_len or len(input_ids) == 0\
            or output_len <= 0 or hash_ids is None or len(hash_ids) < 1:
            return

        parts_hash_session_id = hash_session_id.split("@")
        if len(parts_hash_session_id) < 2:
            return
        session_id = parts_hash_session_id[1]

        if len(input_ids) >= self._max_model_len:
            return

        parts_g_session_id = g_session_id.split("@")
        if len(parts_g_session_id) < 3:
            return
        hash_prefix_len = self.prefix_table.lookup(hash_ids)
        self.prefix_expansion_ctrl.process(hash_ids)
        hash_prefix_str = "".join(map(str, hash_ids[:hash_prefix_len]))
        request_id = next(request_id_counter)
        new_g_session_id = f"{parts_g_session_id[0]}@{parts_g_session_id[1]}@{request_id}"
        hash_session_id = f"{parts_hash_session_id[0]}@{hash_prefix_str}"
        logger.info(f"Generating request: {request_id}, {new_g_session_id}, {hash_session_id}, {hash_prefix_len}")
        request = Request(
            request_id = int(request_id),
            dataset_type = dataset_type,
            native_session_id = int(session_id),
            session_id = new_g_session_id,
            hash_session_id = hash_session_id,
            round_id = 0, 
            prompts = prompts,
            input_ids = input_ids,
            num_prefill_tokens = len(input_ids),
            actual_num_prefill_tokens = len(input_ids),
            output_len = int(output_len),
            over_flow = False,
            n = 1,
            temperature = 0,
            top_p = 1,
            max_tokens = int(output_len),
            stream = True,
            arrived_at = time.perf_counter(),
            time_interval = time_interval,
            hash_prefix_len = hash_prefix_len
        )
        self._cur_num_request += 1
        await self._generate_request_helper(request, session_id)
    def build_jitserve_request(
        self, record: dict, time_interval: float
    ) -> Optional[Request]:
        prompt = str(record.get("prompt", ""))
        output_len = int(record.get("output_len", 0))
        if not prompt or output_len <= 0:
            return None

        tokenizer = self.shared_state.tokenizer
        routing_input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError, TypeError):
            input_ids = routing_input_ids
        if not input_ids or len(input_ids) >= self._max_model_len:
            return None
        output_len = min(output_len, self._max_model_len - len(input_ids))
        if output_len <= 0:
            return None

        # The engine sees chat-formatted tokens, but routing affinity must not
        # collapse onto the universal chat-template block shared by every
        # request. APC locality still uses request._input_ids below.
        routing_block_hashes = stable_prefix_block_hashes(
            routing_input_ids, self.shared_state.replica_budgets[0].block_size
        )
        collection_id = int(record.get("collection_id", self._cur_num_request))
        request_type = int(record.get("request_type", 0))
        stage_id = int(record.get("stage_id", 0))
        stage_num = int(record.get("stage_num", 1))
        slo = build_request_slo(
            self._slo_profile,
            collection_id,
            request_type,
            stage_num=stage_num,
        )
        request_id = next(request_id_counter)
        prefix_key = (
            routing_block_hashes[0]
            if routing_block_hashes
            else f"request-{request_id}"
        )

        request = Request(
            request_id=request_id,
            dataset_type=self._dataset_type,
            native_session_id=collection_id,
            session_id=f"jitserve@{collection_id}@{request_id}",
            hash_session_id=prefix_key,
            round_id=stage_id,
            prompts=prompt,
            input_ids=input_ids,
            num_prefill_tokens=len(input_ids),
            actual_num_prefill_tokens=len(input_ids),
            output_len=output_len,
            over_flow=False,
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=self._routing_output_tokens_hint,
            stream=True,
            arrived_at=time.perf_counter(),
            time_interval=time_interval,
            hash_prefix_len=0,
            request_type=request_type,
            collection_id=collection_id,
            slo_constraint=slo.as_tuple(),
            client_region=str(record.get("client_region", self._client_region)),
            routing_output_tokens_hint=self._routing_output_tokens_hint,
            stage_id=stage_id,
            stage_num=stage_num,
            task_branch_count=int(record.get("task_branch_count", 1)),
        )
        self._cur_num_request += 1
        return request

    async def generate_jitserve_request(
        self, record: dict, time_interval: float
    ) -> None:
        request = self.build_jitserve_request(record, time_interval)
        if request is not None:
            await self._generate_request_helper(
                request, str(request._collection_id)
            )

    def valid_record_offline(self, record):
        valid = True
        if record["timestamp"] < 0:
            valid = False
        return valid

    async def generate_from_file(self):   
        if self._trace_format == "jitserve" or (
            self._trace_format == "auto" and self._dataset_file.endswith(".json")
        ):
            await self.generate_from_jitserve_file()
            return
        dataset_file = self._dataset_file
        request_native_qps = 0
        if "conversation" in self._dataset_type:
            request_native_qps = 3.34
            self._warm_up_qps = 0.3 
        elif "toolagent" in self._dataset_type:
            request_native_qps = 6.5 / 2 
            self._warm_up_qps = 0.5 

        time_interval = 1
        logger.info(f"Generating requests from file: {dataset_file}")
        try:    
            with open(dataset_file, 'r') as file:
                logger.info(f"Reading file: {dataset_file}")
                current_group = []
                prev_timestamp = None

                cur_line_num = 0
                while True:
                    if self._cur_num_request >= self._num_request:
                        break    

                    line = file.readline()
                    cur_line_num += 1
                    if self._requests_num_dataset_start >= cur_line_num:
                        continue

                    if not line:  
                        for record_in_group in current_group:
                            start = time.perf_counter()
                            await self.generate_request_offline(record_in_group, time_interval, self._dataset_type)
                            delay = time.perf_counter() - start
                            await asyncio.sleep(max(0, time_interval - delay))
                            await asyncio.sleep(0)                    
                        break
                    
                    record = json.loads(line)
                    if not self.valid_record_offline(record):
                        continue

                    current_timestamp = record["timestamp"]
                    
                    if prev_timestamp is None:
                        prev_timestamp = current_timestamp               
                    
                    if current_timestamp == prev_timestamp:
                        current_group.append(record)
                    else:
                        if current_group:
                            native_time_interval = round((current_timestamp - prev_timestamp) / len(current_group) / 1000, 3)
                            if self._warm_up_requests_num > self._cur_num_request:
                                if self._warm_up_qps > 0:
                                    time_interval = 1/self._warm_up_qps # warm up tool-agent:qps=0.5, mooncake-conversation qps=0.2
                                else:
                                    time_interval = 2
                            else:
                                time_interval = round(native_time_interval * request_native_qps / self._request_generate_qps, 3)
                            for record_in_group in current_group:
                                start = time.perf_counter()
                                await self.generate_request_offline(record_in_group, time_interval, self._dataset_type)
                                delay = time.perf_counter() - start
                                await asyncio.sleep(max(0, time_interval - delay)) 
                                await asyncio.sleep(0) 
                        current_group = [record]
                        prev_timestamp = current_timestamp
        
        except Exception as e:
            logger.error(f"Error generating requests from file: {dataset_file}, error: {e}")
            return

    async def generate_from_jitserve_file(self) -> None:
        records = load_jitserve_records(self._dataset_file)
        records = assign_request_types(
            records[: self._num_request],
            self._jitserve_request_ratio,
            self._trace_seed,
        )
        if not records:
            return

        arrival_ms = [trace_time_ms(record, index) for index, record in enumerate(records)]
        positive_gaps_s = [
            (arrival_ms[index] - arrival_ms[index - 1]) / 1000.0
            for index in range(1, len(arrival_ms))
            if arrival_ms[index] > arrival_ms[index - 1]
        ]
        native_mean_gap_s = statistics.mean(positive_gaps_s) if positive_gaps_s else 1.0
        if self._arrival_speedup > 0:
            time_scale = 1.0 / self._arrival_speedup
        else:
            target_mean_gap_s = 1.0 / self._request_generate_qps
            time_scale = target_mean_gap_s / native_mean_gap_s

        logger.info(
            "Generating %s JITServe requests with SLO profile=%s, ratio=%s, "
            "native_mean_gap=%.4fs, time_scale=%.4f, arrival_speedup=%.4f",
            len(records),
            self._slo_profile,
            self._jitserve_request_ratio,
            native_mean_gap_s,
            time_scale,
            self._arrival_speedup,
        )
        # Pre-tokenize all requests before the timing loop so CPU-bound
        # tokenization cannot distort the arrival timeline.
        base_arrival_ms = arrival_ms[0]
        prepared = []
        previous_offset_s = 0.0
        for index, record in enumerate(records):
            if self._cur_num_request >= self._num_request:
                break
            offset_s = (
                max(0.0, arrival_ms[index] - base_arrival_ms)
                / 1000.0
                * time_scale
            )
            gap_s = max(0.0, offset_s - previous_offset_s)
            previous_offset_s = offset_s
            request = self.build_jitserve_request(record, gap_s)
            if request is not None:
                prepared.append((offset_s, request))
        if not prepared:
            self._is_finished = True
            return
        # Open-loop emitter: fire arrivals at absolute deadlines derived from
        # the trace. Router scheduling and beam search share this event loop
        # but must never throttle the next arrival.
        start = time.perf_counter()
        tasks = []
        for offset_s, request in prepared:
            deadline = start + offset_s
            now = time.perf_counter()
            if deadline > now:
                await asyncio.sleep(deadline - now)
            # Record the planned arrival deadline, not wake jitter, so the
            # measured arrival timeline equals the trace for every policy.
            emitted_at = time.perf_counter()
            request._arrived_at = deadline
            request._scheduled_arrived_at = deadline
            request._emitted_at = emitted_at
            request._emission_lag_s = max(0.0, emitted_at - deadline)
            tasks.append(
                asyncio.create_task(
                    self._generate_request_helper(
                        request, str(request._collection_id)
                    )
                )
            )
        await asyncio.gather(*tasks)
        self._is_finished = True
