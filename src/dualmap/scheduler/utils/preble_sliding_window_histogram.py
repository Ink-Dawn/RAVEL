from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from dualmap.cache_manager.global_lru_cache_preble import TreeNode
from dualmap.logger import init_logger
logger = init_logger(__name__)

class SlidingWindowHistogram:
    # window_duration : second
    def __init__(self, args, window_duration:timedelta, gpu_allocations, num_gpus=2, enable_miss_rate=True,topt_queues_for_gpus=None):
        self.prefill_tpot = args.prefill_tpot
        self.window_duration = window_duration 
        self.gpu_allocations = gpu_allocations 
        self.histogram = defaultdict(int) 
        self.num_requests_for_node_maps = defaultdict(int) 
        self.timestamps: List[Tuple[datetime, TreeNode, TreeNode]] = []
        self.num_gpus = num_gpus
        self.enable_miss_rate = enable_miss_rate
        self.unshare_prefix_ratio_for_node_maps = {}
        self.sum_share_tokens_for_node_maps = defaultdict(int) 
        self.sum_prompt_tokens_for_node_maps = defaultdict(int)
        self.cur_num_decode_for_node_maps = defaultdict(int) 
        self.topt_queues_for_gpus = topt_queues_for_gpus
        self.cur_actual_prefill_cost_for_gpu = [0 for i in range(self.num_gpus)]
        self.current_decode_lengths_per_gpu = [0 for i in range(self.num_gpus)] 
        self.actual_prefill_cost_per_gpu_for_node_maps = defaultdict(int) 
        self.total_decode_length_for_node_maps = defaultdict(int) 

    @property
    def current_prefill_decode_load_cost_per_gpu(self):
        costs = []
        for i in range(self.num_gpus):
            topt = np.median(self.topt_queues_for_gpus[i])
            costs.append(self.cur_actual_prefill_cost_for_gpu[i] + self.current_decode_lengths_per_gpu[i] * topt) 
        return costs
    
    # node: nearest_important_parent
    def add_request(self, timestamp, ancestor_node_low_share_prefix_ratio: TreeNode, child_node_contain_request: TreeNode, runtime_idx, decoding_length):
        self.timestamps.append((timestamp, ancestor_node_low_share_prefix_ratio, child_node_contain_request))
        self.histogram[ancestor_node_low_share_prefix_ratio] += 1 * child_node_contain_request.context_length
        self.num_requests_for_node_maps[ancestor_node_low_share_prefix_ratio] += 1
        self.cur_num_decode_for_node_maps[ancestor_node_low_share_prefix_ratio] = decoding_length
        self.total_decode_length_for_node_maps[ancestor_node_low_share_prefix_ratio] += decoding_length
        self.sum_share_tokens_for_node_maps[ancestor_node_low_share_prefix_ratio] += child_node_contain_request.context_length - child_node_contain_request.num_tokens
        self.sum_prompt_tokens_for_node_maps[ancestor_node_low_share_prefix_ratio] += child_node_contain_request.context_length
        self.unshare_prefix_ratio_for_node_maps[ancestor_node_low_share_prefix_ratio] = 1 - (self.sum_share_tokens_for_node_maps[ancestor_node_low_share_prefix_ratio] / self.sum_prompt_tokens_for_node_maps[ancestor_node_low_share_prefix_ratio])
        self._remove_old_entries(timestamp)
        self.update_prefill_cost_for_gpu_with_node(ancestor_node_low_share_prefix_ratio, runtime_idx)
        self.current_decode_lengths_per_gpu[runtime_idx] += decoding_length

    def update_prefill_cost_for_gpu_with_node(self, node, runtime_idx):
        if runtime_idx not in self.gpu_allocations.get(node) or node not in self.unshare_prefix_ratio_for_node_maps or node not in self.num_requests_for_node_maps:
            return # only update gpu allocation for allocated nodes
        new_cost = self.get_actual_prefill_cost_per_gpu_for_node(node)
        old_cost = self.actual_prefill_cost_per_gpu_for_node_maps[node]
        self.cur_actual_prefill_cost_for_gpu[runtime_idx] -= old_cost
        self.cur_actual_prefill_cost_for_gpu[runtime_idx] += new_cost
        self.actual_prefill_cost_per_gpu_for_node_maps[node] = new_cost

    def _remove_old_entries(self, current_timestamp):
        window_start = current_timestamp - self.window_duration
        while self.timestamps and self.timestamps[0][0] < window_start:
            timestamp, node, leaf_node = self.timestamps.pop(0)
            self.histogram[node] -= 1 * leaf_node.context_length
            self.num_requests_for_node_maps[node] -= 1
            self.sum_share_tokens_for_node_maps[node] -= leaf_node.context_length - leaf_node.num_tokens
            self.sum_prompt_tokens_for_node_maps[node] -= leaf_node.context_length
            for gpu in range(self.num_gpus):
                self.update_prefill_cost_for_gpu_with_node(node, gpu)
                
            if self.histogram[node] <= 0:
                del self.histogram[node]
                del self.num_requests_for_node_maps[node]
                del self.unshare_prefix_ratio_for_node_maps[node]
                del self.sum_share_tokens_for_node_maps[node]
                del self.sum_prompt_tokens_for_node_maps[node]
                del self.cur_num_decode_for_node_maps[node]
                self.gpu_allocations[node] = set() # Reset the gpu allocation outside the time window

    def replace_old_node_with_new(self, old_child_node, new_parent_node):
        if old_child_node in self.histogram:
            self.histogram[new_parent_node] = self.histogram.pop(old_child_node)
            self.num_requests_for_node_maps[new_parent_node] = self.num_requests_for_node_maps.pop(old_child_node)
            self.unshare_prefix_ratio_for_node_maps[new_parent_node] = self.unshare_prefix_ratio_for_node_maps.pop(old_child_node)
            self.sum_share_tokens_for_node_maps[new_parent_node] = self.sum_share_tokens_for_node_maps.pop(old_child_node)
            self.sum_prompt_tokens_for_node_maps[new_parent_node] = self.sum_prompt_tokens_for_node_maps.pop(old_child_node)
            self.cur_num_decode_for_node_maps[new_parent_node] = self.cur_num_decode_for_node_maps.pop(old_child_node)
            self.total_decode_length_for_node_maps[new_parent_node] = self.total_decode_length_for_node_maps.pop(old_child_node)
            for gpu in range(self.num_gpus):
                self.actual_prefill_cost_per_gpu_for_node_maps[new_parent_node] = self.actual_prefill_cost_per_gpu_for_node_maps[old_child_node]
                self.actual_prefill_cost_per_gpu_for_node_maps[old_child_node] = 0
                self.update_prefill_cost_for_gpu_with_node(new_parent_node, gpu)
            timestamps = []
            for timestamp, important_node, leaf_node in self.timestamps:
                if important_node == old_child_node:
                    important_node = new_parent_node
                timestamps.append((timestamp, important_node, leaf_node))
            self.timestamps = timestamps

    def query(self):
        return dict(self.histogram)

    # prefill_cost + decode_cost
    def get_total_prefill_decode_cost_for_gpus(self):
        cost_for_gpus_list = [0 for _ in range(self.num_gpus)]
        topts = []
        for i in range(self.num_gpus):
            topts.append(np.median(self.topt_queues_for_gpus[i]))
        node: TreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                cost_for_gpus_list[gpu] += self.get_node_prefill_decode_cost_in_target_gpu(node, gpu, topts[gpu])
        return cost_for_gpus_list

    def get_prefill_decode_cost_for_gpus_list_with_min_num_request(self, min_num_request=2):
        prefill_decode_cost_for_gpu_list = [0 for _ in range(self.num_gpus)]
        median_topts_for_gpus_list = []
        for i in range(self.num_gpus):
            median_topts_for_gpus_list.append(np.median(self.topt_queues_for_gpus[i]))

        node: TreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                if self.num_requests_for_node_maps[node] < min_num_request: 
                    continue
                prefill_decode_cost_for_gpu_list[gpu] += self.get_node_prefill_decode_cost_in_target_gpu(node, gpu, median_topts_for_gpus_list[gpu])
        return prefill_decode_cost_for_gpu_list

    def prefill_time(self, num_prefill_tokens):
        P1TT = self.prefill_tpot
        simple_prefill_time = num_prefill_tokens * P1TT
        return simple_prefill_time

    def get_actual_prefill_cost_per_gpu_for_node(self, node: TreeNode):
        return self.unshare_prefix_ratio_for_node_maps[node] * self.prefill_time(node.context_length) * self.num_requests_for_node_maps[node] / len(self.gpu_allocations.get(node)) # potentionally divide by length of node.cached_gpus here
    
    def get_node_prefill_decode_cost_in_target_gpu(self, node: TreeNode, gpu, tpot):
        prefill_cost = self.get_actual_prefill_cost_per_gpu_for_node(node)
        output_len = self.cur_num_decode_for_node_maps[node]
        if node.decode_length_queue:
            output_len = np.median(node.decode_length_queue)
        num_uncompleted_request = node.num_uncompleted_request_for_gpu_map[gpu] 
        decode_cost = num_uncompleted_request * output_len * tpot
        return prefill_cost + decode_cost

    def get_eviction_prefill_cost_for_node(self, node: TreeNode, gpu, is_node_unshare_prefix_over_50_percent: bool):
        if not is_node_unshare_prefix_over_50_percent:
            return 0 
        return self.unshare_prefix_ratio_for_node_maps.get(node, 1.0) * \
                self.num_requests_for_node_maps.get(node, node.num_uncompleted_request_for_gpu_map[gpu]) * \
                self.prefill_time(node.context_length) 
    
    def update_gpus_cost_with_migrate_node(self, node: TreeNode, from_gpu, to_gpu):
        cost = self.actual_prefill_cost_per_gpu_for_node_maps[node]
        self.cur_actual_prefill_cost_for_gpu[from_gpu] -= cost
        self.cur_actual_prefill_cost_for_gpu[to_gpu] += cost
        total_decoding_length = self.total_decode_length_for_node_maps[node]
        self.current_decode_lengths_per_gpu[from_gpu] -= total_decoding_length
        self.current_decode_lengths_per_gpu[to_gpu] += total_decoding_length