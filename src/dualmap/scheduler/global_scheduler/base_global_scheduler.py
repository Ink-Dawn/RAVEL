from abc import ABC, abstractmethod

from dualmap.entities.request import Request
class BaseGlobalScheduler(ABC):
    def __init__(self, num_replicas):
        self._num_replicas = num_replicas

    @abstractmethod
    async def schedule(self, request: Request) -> int:
        pass

    @abstractmethod
    def finish_request(
        self, func_output=None, text: str = None, input_ids=None
    ):
        pass