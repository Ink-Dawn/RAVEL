import json
import os
import time
import threading
import random
from collections import OrderedDict
from pympler import asizeof
import matplotlib.pyplot as plt
from collections import OrderedDict,deque, defaultdict
from dualmap.logger import init_logger

logger = init_logger(__name__)

class LRUCapBucket:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.data = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                self.data[key] = value
            else:
                self.data[key] = value
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)

    def delete(self, key):
        with self.lock:
            if key in self.data:
                del self.data[key]

class LazyPrefixTable:
    def __init__(self, block_size=512, cap_per_level=10000):
        self.block_size = block_size
        self.cap_per_level = cap_per_level
        self.level_buckets = {}
        self.level_locks = {}
        self.stats = {}
        self.hash_base = 1315423911  

    def _get_bucket(self, depth):
        if depth not in self.level_buckets:
            self.level_buckets[depth] = LRUCapBucket(self.cap_per_level)
            self.level_locks[depth] = threading.Lock()
            self.stats[depth] = 0
        return self.level_buckets[depth]

    def rolling_hash(self, prefix_blocks, depth):
        h = 0
        for b in prefix_blocks[:depth]:
            h = (h * self.hash_base + b) & 0xFFFFFFFFFFFFFFFF
        return h

    def lookup(self, prefix_blocks):
        depth = 1
        max_depth = len(prefix_blocks)
        while depth <= max_depth:  
            h = self.rolling_hash(prefix_blocks, depth)
            bucket = self._get_bucket(depth)
            decision = bucket.get(h)
            self.stats[depth] += 1
            if decision is None:
                return depth
            else:
                depth = decision
        return max_depth  

    def mark_expand(self, prefix_blocks, new_depth):
        h = self.rolling_hash(prefix_blocks, new_depth - 1)
        bucket = self._get_bucket(new_depth - 1)
        bucket.put(h, new_depth)

    def mark_shrink(self, prefix_blocks, old_depth):
        h = self.rolling_hash(prefix_blocks, old_depth)
        bucket = self._get_bucket(old_depth)
        bucket.delete(h)

class HotPrefixDetector:
    def __init__(self, window_size: int = 200, hot_ratio: float = 0.0612, min_samples: int = 20):
        self.window_size = int(window_size)
        self.hot_ratio = float(hot_ratio)
        self.min_samples = int(min_samples)  
        self.window = deque()                 
        self.counts = defaultdict(int)       
        self.lock = threading.Lock()          

    def observe(self, prefix):
        with self.lock:
            self.window.append(prefix)
            self.counts[prefix] += 1

            if len(self.window) > self.window_size:
                old = self.window.popleft()
                self.counts[old] -= 1
                if self.counts[old] <= 0:
                    del self.counts[old]

            total = len(self.window)
            if total < self.min_samples:
                return False, 0.0

            cnt = self.counts.get(prefix, 0)
            ratio = cnt / total if total > 20 else 0.0
            return (ratio >= self.hot_ratio), ratio

    def snapshot_counts(self):
        with self.lock:
            return dict(self.counts), len(self.window)

    def set_params(self, window_size: int = None, hot_ratio: float = None, min_samples: int = None):
        with self.lock:
            if window_size is not None:
                self.window_size = int(window_size)
            if hot_ratio is not None:
                self.hot_ratio = float(hot_ratio)
            if min_samples is not None:
                self.min_samples = int(min_samples)

class LazyExpansionController:
    def __init__(self, table, detector, cooldown_sec=10):
        self.table = table
        self.detector = detector
        self.cooldown_sec = cooldown_sec
        self.cnt = 0

    def process(self, prefix_blocks):
        self.cnt += 1
        depth = self.table.lookup(prefix_blocks)
        prefix_key = tuple(prefix_blocks[:depth])
        is_hot, ratio = self.detector.observe(prefix_key)

        if is_hot and depth < len(prefix_blocks):
            self.table.mark_expand(prefix_blocks, depth + 1)
            if depth + 1 > 2:
                logger.debug(f"[HOT] req_cnt:{self.cnt},prefix={prefix_blocks[:depth + 1]}, ratio={ratio:.2%}, expand to {depth+1}")
        
        if depth > 1:
            p_prefix_key = tuple(prefix_blocks[:depth-1])
            _, p_ratio = self.detector.observe(p_prefix_key)

            if p_ratio < 0.02:
                self.table.mark_shrink(prefix_blocks, depth)
                logger.debug(f"mark_shrink:req_cnt:{self.cnt},p_prefix={prefix_blocks[:depth - 1]}, p_ratio={p_ratio:.2%}, p_depth:{depth-1};ratio={ratio:.2%}")

# ------------------------------
# Benchmark harness
# ------------------------------
def benchmark_single_thread(table, n_ops=100000, depth=4):
    prefix = [random.randint(0, 10000) for _ in range(depth)]
    t0 = time.time()
    for _ in range(n_ops):
        table.lookup(prefix)
    elapsed = time.time() - t0
    throughput = n_ops / elapsed
    print(f"[Single-thread] ops={n_ops}, elapsed={elapsed:.4f}s, "
          f"throughput={throughput:.2f} ops/s, avg={1e6*elapsed/n_ops:.2f} µs/op")


def benchmark_multi_thread(table, n_ops=100000, depth=4, n_threads=8):
    prefix = [random.randint(0, 10000) for _ in range(depth)]

    def worker(iters):
        for _ in range(iters):
            table.lookup(prefix)

    iters = n_ops // n_threads
    threads = []
    t0 = time.time()
    for _ in range(n_threads):
        th = threading.Thread(target=worker, args=(iters,))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    elapsed = time.time() - t0
    throughput = n_ops / elapsed
    print(f"[Multi-thread] threads={n_threads}, ops={n_ops}, elapsed={elapsed:.4f}s, "
          f"throughput={throughput:.2f} ops/s, avg={1e6*elapsed/n_ops:.2f} µs/op")

def measure_memory(table):
    total = 0
    for depth, bucket in table.level_buckets.items():
        total += asizeof.asizeof(bucket)
    return total

if __name__ == "__main__":
    prefix_table = LazyPrefixTable()
    hot_prefix_detector = HotPrefixDetector()
    prefix_expansion_ctrl = LazyExpansionController(prefix_table, hot_prefix_detector)

    data_type = "toolagent"
    dataset_file = r".\mooncake\toolagent_trace.jsonl"

    n_limit = 1000_000  
    cnt = 0
    lookup_time = 0.0
    expand_time = 0.0

    cnt_step = 20
    depth_trace = []  # (cnt, max_depth)

    with open(dataset_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if "hash_ids" not in record:
                continue

            cnt += 1
            hash_ids = record["hash_ids"]
            t0 = time.perf_counter()
            depth_before = prefix_table.lookup(hash_ids)
            lookup_time += (time.perf_counter() - t0)

            # expand/shrink 
            t1 = time.perf_counter()
            prefix_expansion_ctrl.process(hash_ids)
            expand_time += (time.perf_counter() - t1)

            # max_depth
            if cnt % cnt_step == 0:
                max_depth = max(prefix_table.level_buckets.keys(), default=0)
                depth_trace.append((cnt, max_depth))

            if cnt % 10000 == 0:
                print(f"Processed {cnt} requests")

            if cnt >= n_limit:
                break

    avg_lookup_us = 1e6 * lookup_time / cnt
    avg_expand_us = 1e6 * expand_time / cnt
    mem_usage_kb = measure_memory(prefix_table) / 1024

    print(f"\n==== {data_type} Benchmark Result ====")
    print(f"Total requests: {cnt}")
    print(f"Avg lookup time: {avg_lookup_us:.2f} µs")
    print(f"Avg expand/shrink time: {avg_expand_us:.2f} µs")
    print(f"Approx memory usage: {mem_usage_kb:.2f} KB")

    print("\n==== Max Depth Trace ====")
    plot_data = [(cnt, d) for cnt, d in depth_trace if cnt <= 500]
    if plot_data:
        step_cnts, max_depths = zip(*plot_data)

        pics_dir = os.path.join(os.getcwd(), "pics")
        os.makedirs(pics_dir, exist_ok=True)
        pdf_path = os.path.join(pics_dir, f"{data_type}_max_depth_trace.pdf")

        plt.figure(figsize=(4, 3))
        plt.plot(step_cnts, max_depths, marker='o', linestyle='-')
        plt.xlabel("Request count")
        plt.ylabel("Max depth")
        plt.title(f"{data_type} Max Depth Trace")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(pdf_path)
        plt.close()
        print(f"Max depth trace plot saved to: {pdf_path}")
    else:
        print("No data to plot for step_cnt <= 200")