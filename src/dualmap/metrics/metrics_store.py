import os
import aiofiles
import csv
import asyncio
from collections import defaultdict
import io
from dualmap.logger import init_logger

logger = init_logger(__name__)

class RooundInfo():

    def __init__(self):
        self.session_num = 0
        self.request_num = 0
        self.request_latency_list = []
        self.ttft_list = []
        self.decode_token_time_list = []
        self.output_decode_token_num_list = []       

    def save_info(self,
                  request_latency: float,
                  time_to_first_token: float,
                  decode_token_time: float,
                  output_token_len: int):
        self.request_num += 1
        self.request_latency_list.append(request_latency)
        self.ttft_list.append(time_to_first_token)
        self.decode_token_time_list.append(decode_token_time)
        self.output_decode_token_num_list.append(output_token_len)

class MetricsStore:
    def __init__(self, args):
        self.csv_dir = args.result_path
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_filename = f"{self.csv_dir}/request_metrics.csv"
        self.lock = asyncio.Lock()
        self.data_cache = defaultdict(dict)
        self.dirty = False 

    async def load_cache(self):
        try:
            async with aiofiles.open(self.csv_filename, mode="r", encoding="utf-8") as csvfile:
                content = await csvfile.read()
                lines = content.splitlines()
                reader = csv.DictReader(lines)
                for row in reader:
                    self.data_cache[row['request_id']] = row
        except FileNotFoundError:
            pass

    async def save_cache(self):
        async with self.lock:
            if not self.dirty:
                return
            if self.dirty:
                try:
                    output = io.StringIO()
                    fieldnames = [
                        'request_id', 'dataset_type', 'request_start_time', 'request_end_time',
                        'native_session_id', 'round_id', 'replica_id', 'cluster_id',
                        'client_region', 'collection_id', 'request_type',
                        'stage_id', 'stage_num', 'task_branch_count',
                        'time_to_first_token', 'request_latency', 'max_tbt',
                        'TPS(tokens/s)', 'tpot(ms)', 'wait_queue_s',
                        'policy_induced_hold_s',
                        'policy_induced_hold_budget_s',
                        'network_rtt_s', 'network_delay_mode',
                        'num_request_pending', 'input_len', 'actual_prompt_tokens',
                        'output_len', 'actual_output_tokens', 'routing_output_tokens_hint',
                        'routing_output_tokens_hint_used',
                        'routing_output_tokens_upper_hint',
                        'routing_output_tokens_upper_hint_used', 'pd_ratio',
                        'actual_num_prefill_tokens', 'admission_prompt_charge_tokens',
                        'prefix_hit_prompt_tokens',
                        'estimated_prefix_hit_prompt_tokens', 'prefix_hit_source', 'tbt_source',
                        'tbt_token_count', 'tbt_event_count', 'tbt_measurement_valid',
                        'req_arrived_at', 'time_interval',
                        'scheduled_arrived_at', 'emitted_at', 'emission_lag_s',
                        'slo_ttft_s', 'slo_tbt_s', 'slo_ttlt_s', 'ttft_slo_met',
                        'tbt_slo_met', 'ttlt_slo_met', 'request_slo_met',
                        'predicted_ttft_s', 'point_predicted_ttft_s',
                        'predicted_objective_s', 'point_predicted_objective_s',
                        'quote_handoff_replica_id', 'quote_handoff_timestamp_s',
                        'quote_handoff_age_s', 'quote_handoff_predicted_ttft_s',
                        'quote_handoff_predicted_tbt_s',
                        'quote_handoff_point_completion_s',
                        'quote_handoff_risk_completion_s',
                        'quote_handoff_feasible', 'quote_handoff_valid',
                        'ttft_residual_guard_s', 'objective_residual_guard_s',
                        'risk_calibrated', 'ttft_prediction_residual_s',
                        'objective_prediction_residual_s',
                        'cluster_route_feasible', 'cluster_route_reason', 'rebind_count',
                        'sidecar_soft_moves', 'abort_attempts', 'abort_completed', 'attempt',
                        'engine_priority', 'ravel_yield_deferred',
                        'ravel_soft_admission_active',
                        'ravel_soft_admission_protected',
                        'ravel_soft_admission_slack_release',
                        'ravel_soft_admission_min_slack_s',
                        'ravel_soft_admission_increment_s',
                        'ravel_soft_admission_victims',
                        'ravel_soft_admission_hold_budget_s',
                        'ravel_soft_admission_first_deferred_at',
                        'ravel_soft_admission_deferral_s',
                        'ravel_soft_admission_deadline_release',
                        'ravel_soft_admission_dispatch_at',
                        'ravel_soft_plan_generation',
                        'ravel_profile_output_expected',
                        'ravel_profile_output_upper',
                        'ravel_prefill_gap_s', 'ravel_protected_tbt_s',
                        'ravel_prefill_chunk_tokens',
                        'ravel_admission_envelope_safe',
                        'ravel_admission_victims',
                        'ravel_admission_deferrals',
                        'ravel_max_admission_victims',
                        'rounting_cache_hit_max', 'is_dh_cache_affinity',
                        'is_dh_least_loaded', 'is_dh_cache_affinity_least_loaded'
                    ]
                    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
                    writer.writeheader()
                    for row in self.data_cache.values():
                        writer.writerow(row)
                    csv_content = output.getvalue()
                    output.close()

                    async with aiofiles.open(self.csv_filename, mode="w", encoding="utf-8") as csvfile:
                        await csvfile.write(csv_content)
                    self.dirty = False
                except Exception as e:
                    logger.error(f"Failed to save cache to CSV: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())

    async def insert_metrics(self, data):
        async with self.lock:
            self.data_cache[data['request_id']] = data
            self.dirty = True

    async def update_metrics(self, request_id, updates):
        async with self.lock:
            if request_id in self.data_cache:
                self.data_cache[request_id].update(updates)
                logger.debug(f"update_metrics:request_id={request_id}")
                self.dirty = True

    async def sync_cache(self):
        await self.save_cache()
