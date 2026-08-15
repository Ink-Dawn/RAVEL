from dualmap.cache_manager.kvcache_store.logger import init_logger
logger = init_logger(__name__)

class KvCacheEngine:
    def __init__(self, capacity: int,
                 block_size: int,
                 kv_cache_per_token: int):
        self.block_size = block_size
        self.kv_cache_per_token = kv_cache_per_token
        self.num_blocks = capacity // self.cal_block_capacity()

    def get_num_blocks(self) -> int:
        return self.num_blocks

    def cal_block_capacity(self) -> int:
        return self.kv_cache_per_token * self.block_size