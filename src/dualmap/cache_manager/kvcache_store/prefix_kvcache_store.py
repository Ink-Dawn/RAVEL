from typing import List
from dualmap.cache_manager.kvcache_store.kvcache_engine import KvCacheEngine
from dualmap.cache_manager.kvcache_store.prefix_kvcache_mgr import PrefixKVCacheMgr
from dualmap.cache_manager.kvcache_store.logger import init_logger

logger = init_logger(__name__)

class PrefixKvCacheStore:
    _MAX_CAPACITY = 1024 * 1024 * 1024 * 1024

    def __init__(self, args):
        self.cache_engine = KvCacheEngine(args.cache_capacity, args.block_size, args.kv_cache_size_per_token)
        self.cache_mgr = PrefixKVCacheMgr(self.cache_engine)

    def get_num_cached_blocks(self, prompt_token_ids: List[int]) -> int:
        return self.cache_mgr.get_num_cached_blocks(prompt_token_ids)

    def save(self, prompt_token_ids: List[int]) -> int:
        return self.cache_mgr.save(prompt_token_ids)

    def save_prefill(self, request_id, prompt_token_ids: List[int]) -> int:
        return self.cache_mgr.save_prefill(request_id, prompt_token_ids)

    def get_size(self):
        return self.cache_mgr.get_size()