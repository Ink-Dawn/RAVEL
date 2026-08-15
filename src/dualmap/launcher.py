from transformers import AutoTokenizer
from dualmap.scheduler.utils.shared import SharedState
from dualmap.metrics.metrics_store import MetricsStore
from dualmap.scheduler.global_scheduler.request_router_proxy import RequestRouterProxy
from dualmap.request_generator.request_generator import RequestGenerator
from dualmap.scheduler.global_scheduler.ravel_unified_global_scheduler import (
    RavelUnifiedGlobalScheduler,
)
from dualmap.logger import init_logger

logger = init_logger(__name__)

class SystemLauncher:
    def __init__(self, args):
        self._num_replicas = len(args.replicas_ip_port.split(','))
        self._global_scheduler_type = args.global_scheduler_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.metric_store = MetricsStore(args)
        self.shared_state = SharedState(self.metric_store, self.tokenizer, args)
        self._request_router_proxy = RequestRouterProxy(self.shared_state, self._global_scheduler_type, args.balance_type, args)
        self._request_generator = RequestGenerator(self.shared_state, self.tokenizer, self._num_replicas, args)

    async def is_request_active(self) -> bool:
        if self._request_generator is None:
            return True
        active = self._request_generator.is_request_active()
        if not active:
            await self.metric_store.sync_cache()
        return active

    async def initialize(self, args):
        self._init_router(args)
        
    def _init_router(self, args):
        if self._global_scheduler_type != "ravel_unified":
            raise ValueError(
                "this RAVEL-only release exposes only scheduler ravel_unified"
            )
        logger.info("init_router: ravel_unified")
        self._request_router_proxy.global_scheduler = RavelUnifiedGlobalScheduler(
            num_replicas=self._num_replicas,
            shared_state=self.shared_state,
            args=args,
        )

    async def start_client(self):
        await self._request_generator.generate_from_file()
    
    async def start_global_scheduler(self):
        await self._request_router_proxy.start()

    async def run(self):
        await self._request_router_proxy.start()
        await self._request_generator.generate_from_file()

    async def stop(self):
        try:
            await self._request_router_proxy.stop()
        finally:
            await self.shared_state.close()
