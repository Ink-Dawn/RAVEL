import time
from datetime import datetime, timedelta
import os
import numpy as np
import threading
import heapq
from collections import deque
from typing import Dict
from dualmap.entities.benchmark_utils_preble import RequestFuncOutput
from dualmap.cache_manager.global_lru_cache_preble import LPRadixCache, TreeNode
from dualmap.scheduler.utils.ttft_overload_detector_preble import TTFTWindowedOverloadedDetector
from dualmap.scheduler.utils.preble_sliding_window_histogram import SlidingWindowHistogram
from dualmap.logger import init_logger

logger = init_logger(__name__)

class PrebleGlobalSchedulerUtils():
    def __init__(self, num_replicas, window_duration, args):
        self._max_dram_tokens = args.cache_capacity / max(1, args.kv_cache_size_per_token)
        self._request_counter = 0    
        self.num_gpus = num_replicas
        self.gpu_allocations = {} # {node:gpu_ids},gpu_ids=set[gpu_1,gpu2,...]，
        self.counter = 0 
        self.enable_eviction = True
        self.num_requests_for_gpus_list = {i: 0 for i in range(num_replicas)}
        self.all_gpus = set(range(num_replicas))
        self.mem_cost = [0 for _ in range(num_replicas)] 
        self.lock = threading.Lock()
        self.enable_miss_rate = True
        self.topt_queues_for_gpus = [deque(maxlen=200) for _ in range(num_replicas)] 
        for i in range(num_replicas):
            self.topt_queues_for_gpus[i].append(.15)
        self.histogram = SlidingWindowHistogram(
            args,
            window_duration=timedelta(minutes=int(window_duration)), 
            gpu_allocations=self.gpu_allocations, 
            num_gpus=self.num_gpus, 
            enable_miss_rate=self.enable_miss_rate,
            topt_queues_for_gpus=self.topt_queues_for_gpus
        )
        self.cache = LPRadixCache(histogram=self.histogram, num_gpus=self.num_gpus, lock=self.lock)
        self.max_tokens_for_gpu_list = [self._max_dram_tokens for _ in range(num_replicas)] 
        self.HIGH_LOAD_THRESHOLD = 1.5
        self.REBALANCING_CHAIN_LENGTH = 3 # Max rebalancing length for chain. For long chains, don't rebalance to allow infercept to take over
        self.overload_detector = TTFTWindowedOverloadedDetector(window_duration=timedelta(minutes=int(window_duration)))
        self.enable_rebalancing = True

    # Consider Split nodes
    def update_gpu_allocations_with_split_nodes(self, split_nodes, gpu_allocations):
        for child_node, new_node in split_nodes.items():
            # FIXME: this should be a deepcopy
            gpu_allocations[new_node] = gpu_allocations[child_node].copy()

    def update_histogram_with_split_node(self, split_nodes: Dict[TreeNode, TreeNode]):
        for child, parent_node in split_nodes.items():
            if self.is_unshare_prefix_over_50_percent_for_node(parent_node) and not self.is_unshare_prefix_over_50_percent_for_node(child): # new node is parent is now larger
                self.histogram.replace_old_node_with_new(child, parent_node)
                parent_node.decode_length_queue.extend(child.decode_length_queue) 
                child.decode_length_queue = [] 
                for gpu in self.gpu_allocations.get(child, {}):
                    self.overload_detector.replace_old_node_with_new(child, parent_node, gpu)
            
            if self.is_unshare_prefix_over_50_percent_for_node(child):
                for gpu in self.gpu_allocations.get(child, {}):
                    self.histogram.update_prefill_cost_for_gpu_with_node(child, gpu)

    # Recursively get parent gpu allocation
    def get_parent_gpu_allocation(self, node: TreeNode):
        if not node:
            return self.all_gpus
        if node == self.cache.root_node:
            logger.debug(f'{time.time()}:{os.getpid()}:{threading.get_ident()}:node == self.cache.root_node')
        if self.gpu_allocations.get(node):
            return self.gpu_allocations.get(node)
        return self.get_parent_gpu_allocation(node.parent)
    
    # get actual_num_prefill_tokens for target input_ids in all gpus
    def get_actual_num_prefill_tokens_for_gpus(self, request):
        leaf_node = self.cache.find_node(request._input_ids)       
        actual_num_prefill_tokens_for_gpus = {} #{gpu_id, actual_num_prefill_tokens}
        current_node = leaf_node
        while current_node is not None and current_node != self.cache.root_node:
            selected_gpus = self.gpu_allocations.get(current_node)
            if selected_gpus is not None:
                actual_num_prefill_tokens = current_node.num_tokens
                for gpu in selected_gpus:
                    if gpu not in actual_num_prefill_tokens_for_gpus:
                        actual_num_prefill_tokens_for_gpus[gpu] = actual_num_prefill_tokens
                        logger.debug(f'get_actual_num_prefill_tokens_for_gpus: request-{request._id} in replica-{gpu}, actual_num_prefill_tokens_for_gpus = {actual_num_prefill_tokens_for_gpus}')
            if selected_gpus is not None and len(selected_gpus) == self.num_gpus:
                break
            current_node = current_node.parent
        return actual_num_prefill_tokens_for_gpus

    # get actual_num_prefill_tokens for target input_ids in target gpus
    def get_actual_num_prefill_tokens_for_target_gpu(self, request, gpu_id):
        leaf_node = self.cache.find_node(request._input_ids)       
        actual_num_prefill_tokens = -1
        current_node = leaf_node
        while current_node is not None and current_node != self.cache.root_node:
            selected_gpus = self.gpu_allocations.get(current_node)
            if selected_gpus is not None and gpu_id in selected_gpus:
                actual_num_prefill_tokens = len(request._input_ids) - current_node.context_so_far
                if actual_num_prefill_tokens < 128: 
                    actual_num_prefill_tokens = len(request._input_ids)
                return actual_num_prefill_tokens
            current_node = current_node.parent
        if actual_num_prefill_tokens == -1:
            actual_num_prefill_tokens = len(request._input_ids)
        return actual_num_prefill_tokens


    def add_gpu_allocation_for_parent(self, node: TreeNode, gpu_id):
        if not node:
            return
        self.gpu_allocations[node] = self.gpu_allocations.get(node, set()).union(gpu_id)
        self.add_gpu_allocation_for_parent(node.parent, gpu_id)

    def remove_gpu_allocation_from_node(self, node: TreeNode, gpu_id):
        if not node:
            return
        if node in self.gpu_allocations:
            self.gpu_allocations[node].discard(gpu_id)  
            if not self.gpu_allocations[node]:
                del self.gpu_allocations[node]
        else:
            return

    #ratio of shared_prefix_in_prompt >= 0.5
    def is_more_share_prefix_node(self, node: TreeNode):
        return not self.is_unshare_prefix_over_50_percent_for_node(node)
    
    # num_share_prefix_tokens < num_unshare_prefix_tokens
    # large:ratio of shared_prefix_in_prompt < 0.5
    def is_unshare_prefix_over_50_percent_for_node(self, node: TreeNode):
        return node.num_tokens > node.context_so_far

    # ratio of shared_prefix_in_prompt < 0.5
    def get_nearest_ancestor_node_with_low_share_prefix_ratio(self, node: TreeNode):
        if node is None:
            return
        if self.is_unshare_prefix_over_50_percent_for_node(node):
            return node
        return self.get_nearest_ancestor_node_with_low_share_prefix_ratio(node.parent)
    
    def evict_callback(self, node: TreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        # Maybe update the parent if child has no parent
        node.evicted_gpus.add(runtime_selected)
        node.cached_gpus.remove(runtime_selected)
        return len(node.value)
    
    def handle_cache_eviction_for_gpu(self, runtime_selected):
        current_max_tokens = self.max_tokens_for_gpu_list[runtime_selected]
        assert self.cache.allocated_size_for_gpu(runtime_selected) >= 0
        if self.cache.allocated_size_for_gpu(runtime_selected) > current_max_tokens:
            num_tokens_to_evict = self.cache.allocated_size_for_gpu(runtime_selected) - current_max_tokens
            self.cache.evict_with_runtime_id_without_removing(num_tokens_to_evict, lambda node: self.evict_callback(node, runtime_selected), runtime_selected)
    
    # NOTE: simple heuristic used: assume GPU memory is always full
    #       -> evict size is the leaf node size
    def get_virtual_eviction_prefill_cost_for_routing(self, leaf_node: TreeNode, runtime_selected: int):
        num_to_evict = leaf_node.num_tokens
        virtual_evict_tree_nodes = self.cache.virtual_lru_eviction(num_to_evict, runtime_selected)
        eviction_cost = 0
        victim: TreeNode
        for victim in virtual_evict_tree_nodes:
            eviction_cost += self.histogram.get_eviction_prefill_cost_for_node(victim, runtime_selected, self.is_unshare_prefix_over_50_percent_for_node(victim))
        return eviction_cost
    
    #  L_i = prefill_decode_cost_for_gpus
    #  M = virtual_evict_for_routing(leaf_node, gpu_id)
    def calculate_min_load_cost(self, leaf_node, selected_gpus):
        prefill_decode_cost_for_gpus = self.histogram.get_total_prefill_decode_cost_for_gpus()
        costs = []
        for gpu_id in selected_gpus:
            cost = prefill_decode_cost_for_gpus[gpu_id]
            if self.enable_eviction:
                cost += self.get_virtual_eviction_prefill_cost_for_routing(leaf_node, gpu_id)
            costs.append(cost)
        gpu_selected = int(np.argmin(costs))
        return gpu_selected

    def handle_important_node_stealing(self, scheduled_idx):
        if sum(self.num_requests_for_gpus_list.values()) < 50 * self.num_gpus: 
            return
        prefill_decode_cost_for_gpus_list = self.histogram.get_prefill_decode_cost_for_gpus_list_with_min_num_request(2)
        prefill_decode_cost_for_gpus_with_indices = [(gpu_id, prefill_decode_cost_for_gpus_list[gpu_id]) for gpu_id in range(len(prefill_decode_cost_for_gpus_list))]
        # logger.info(allocations_with_indices)
        shorted_prefill_decode_cost_for_gpus_with_indices = list(sorted(prefill_decode_cost_for_gpus_with_indices, key=lambda x: -x[1]))
        self.handle_important_node_stealing_recursive(shorted_prefill_decode_cost_for_gpus_with_indices)

    def handle_important_node_stealing_recursive(self, shorted_prefill_decode_cost_for_gpus_with_indices):
        if len(shorted_prefill_decode_cost_for_gpus_with_indices) <= 1:
            return
        heavy_gpu, larger_prefill_decode_cost = shorted_prefill_decode_cost_for_gpus_with_indices[0]
        light_gpu, smaller_prefill_decode_cost = shorted_prefill_decode_cost_for_gpus_with_indices[-1] # Last element is the smallest
        if larger_prefill_decode_cost <= self.HIGH_LOAD_THRESHOLD * smaller_prefill_decode_cost:
            return
        median_tpot_of_heavy_gpu = np.median(self.topt_queues_for_gpus[heavy_gpu])
        rebalance_cost_for_node_in_heavy_gpu = []
        all_rebalancing_cost = [] 
        for node, prefill_decode_cost in self.histogram.histogram.items():
            if heavy_gpu in self.gpu_allocations.get(node) and self.is_unshare_prefix_over_50_percent_for_node(node) and self.histogram.num_requests_for_node_maps[node] > 1:
                prefill_decode_cost_for_node = self.histogram.get_node_prefill_decode_cost_in_target_gpu(node, heavy_gpu, median_tpot_of_heavy_gpu)
                heapq.heappush(rebalance_cost_for_node_in_heavy_gpu, (prefill_decode_cost_for_node, node))
                all_rebalancing_cost.append(prefill_decode_cost_for_node)
        
        if len(rebalance_cost_for_node_in_heavy_gpu) == 1:
            # Handle load splitting a single node in two
            prefill_decode_cost, node = rebalance_cost_for_node_in_heavy_gpu[0] 
            prefill_decode_cost /= 2 # load is now split into two
            if light_gpu not in self.gpu_allocations[node] and self.overload_detector.is_node_overloaded(node, heavy_gpu): 
                # Copying the node to the smallest device will not change the larger allocation
                larger_prefill_decode_cost -= prefill_decode_cost
                smaller_prefill_decode_cost += prefill_decode_cost
                self.gpu_allocations[node].add(light_gpu)
                self.overload_detector.delete_node_in_target_gpu(node, heavy_gpu)
        else:
            steal_n = 0
            while rebalance_cost_for_node_in_heavy_gpu:
                node: TreeNode
                prefill_decode_cost, node = heapq.heappop(rebalance_cost_for_node_in_heavy_gpu)
                assert self.is_unshare_prefix_over_50_percent_for_node(node)
                if larger_prefill_decode_cost - prefill_decode_cost < smaller_prefill_decode_cost + prefill_decode_cost:
                    break
                larger_prefill_decode_cost -= prefill_decode_cost
                smaller_prefill_decode_cost += prefill_decode_cost
                self.gpu_allocations[node] = {light_gpu} 
                self.histogram.update_gpus_cost_with_migrate_node(node, heavy_gpu, light_gpu)
                self.update_gpu_allocations_for_children(node, light_gpu) 
                steal_n += 1
                if larger_prefill_decode_cost < self.HIGH_LOAD_THRESHOLD * smaller_prefill_decode_cost:
                    break
            if steal_n != 0:
                logger.debug(f"handle_important_node_stealing_recursive: Steal {steal_n} nodes from {heavy_gpu} to {light_gpu}")
        shorted_prefill_decode_cost_for_gpus_with_indices[0] = (heavy_gpu, larger_prefill_decode_cost)
        shorted_prefill_decode_cost_for_gpus_with_indices[-1] = (light_gpu, smaller_prefill_decode_cost)
        self.handle_important_node_stealing_recursive(shorted_prefill_decode_cost_for_gpus_with_indices[1:])

    def migrate_request_between_gpus(self, request, source_gpu_id, target_gpu_id):
        node = self.cache.find_node(request._input_ids)
        if node is not None:
            remove_from_source = True
            for child in node.children.values():
                child_gpu_allocations = self.gpu_allocations.get(child)
                if source_gpu_id in child_gpu_allocations:
                    remove_from_source = False
                    break
            if remove_from_source is True:
                self.remove_gpu_allocation_from_node(node, source_gpu_id)
        self.add_gpu_allocation_for_parent(node, {target_gpu_id})
        return

    def update_gpu_allocations_for_children(self, node: TreeNode, gpu_id):
        for child in node.children.values():
            self.gpu_allocations[child] = {gpu_id}
            self.update_gpu_allocations_for_children(child, gpu_id)

    def update_overload_detector(self, input_ids, runtime_idx, func_output: RequestFuncOutput):
        # Overload detector based on the current ttft
        leaf_node = self.cache.find_node(input_ids)
        important_node = self.get_nearest_ancestor_node_with_low_share_prefix_ratio(leaf_node)
        self.overload_detector.add_data_point(datetime.now(), important_node, runtime_idx, func_output.ttft)