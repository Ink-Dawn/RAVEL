import asyncio
import time
from dualmap.scheduler.utils.shared import SharedState
from dualmap.logger import init_logger

logger = init_logger(__name__)

class RequestRouterProxy:
    def __init__(self, shared_state: SharedState, global_scheduler_type, balance_type, args):
        self.shared_state = shared_state
        self.global_scheduler = None
        self._running = False
        self._global_scheduler_type = global_scheduler_type
        self._result_path = args.result_path
        self.replica_selection_task = None
        self.cleanup_selection_task = None
        self.inflight_selection_tasks = set()
        self._binding_condition = asyncio.Condition()
        self._next_binding_ticket = 0
        self._serving_binding_ticket = 0

    async def _schedule_and_bind(self, request):
        runtime_id = await self.global_scheduler.schedule(request)
        await self.shared_state.schedule_request_to_replica(
            request, runtime_id
        )

    async def _run_binding_ticket(self, request, ticket):
        async with self._binding_condition:
            await self._binding_condition.wait_for(
                lambda: ticket == self._serving_binding_ticket
            )
            try:
                await self._schedule_and_bind(request)
            finally:
                self._serving_binding_ticket += 1
                self._binding_condition.notify_all()

    async def _skip_binding_ticket(self, ticket):
        async with self._binding_condition:
            await self._binding_condition.wait_for(
                lambda: ticket == self._serving_binding_ticket
            )
            self._serving_binding_ticket += 1
            self._binding_condition.notify_all()

    async def _select_and_bind(self, request, ticket=None):
        try:
            wait_before_schedule = getattr(
                self.global_scheduler, "wait_before_schedule", None
            )
            if wait_before_schedule is not None:
                await wait_before_schedule(request)
            if ticket is None:
                await self._schedule_and_bind(request)
            else:
                await self._run_binding_ticket(request, ticket)
        except Exception:
            if (
                ticket is not None
                and ticket >= self._serving_binding_ticket
            ):
                await self._skip_binding_ticket(ticket)
            logger.exception(
                "Router schedule failed for request %s; dropping it to keep the worker alive",
                getattr(request, "_id", None),
            )
            try:
                event, _ = self.shared_state.runtime_events[request._id]
                self.shared_state.runtime_events[request._id] = (event, -1)
                event.set()
            except Exception:
                logger.exception(
                    "Failed to fail request %s after schedule error",
                    getattr(request, "_id", None),
                )
            try:
                self.shared_state.runtime_request_queue.task_done()
            except Exception:
                pass

    async def _process_replica_selection(self):
        if self.global_scheduler is None:
            return
        while self._running:
            request = await self.shared_state.runtime_request_queue.get()
            if request is None:
                continue
            if request._id % 10 == 0:
                await self.shared_state.metric_store.sync_cache()
                if self._global_scheduler_type != "dualmap":
                    self.shared_state.record_replica_num_pending_request(self._result_path, request._id)
                    self.shared_state.record_replica_num_pending_tokens(self._result_path, request._id)
                    self.shared_state.record_replica_num_pending_input_tokens(self._result_path, request._id)   
            if getattr(self.global_scheduler, "concurrent_schedule", False):
                ticket = None
                if getattr(
                    self.global_scheduler, "serialize_after_delay", False
                ):
                    ticket = self._next_binding_ticket
                    self._next_binding_ticket += 1
                task = asyncio.create_task(
                    self._select_and_bind(request, ticket)
                )
                self.inflight_selection_tasks.add(task)
                task.add_done_callback(self.inflight_selection_tasks.discard)
                continue
            await self._select_and_bind(request)

    # for preble
    async def _process_cleanup_selection(self):
        while self._running:
            output, text, input_ids = await self.shared_state.finished_requests_queue.get()
            if output and output.success:
                self.global_scheduler.finish_request(func_output=output, text=text, input_ids=input_ids)

    async def start(self):
        self._running = True
        self.replica_selection_task = asyncio.create_task(self._process_replica_selection())
        self.cleanup_selection_task = asyncio.create_task(self._process_cleanup_selection()) #

    async def stop(self):
        self._running = False
        tasks = [
            task
            for task in (
                self.replica_selection_task,
                self.cleanup_selection_task,
                *tuple(self.inflight_selection_tasks),
            )
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        close = getattr(self.global_scheduler, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
