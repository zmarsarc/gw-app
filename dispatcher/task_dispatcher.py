from asyncio.locks import Lock
from datetime import datetime
from typing import List, Optional
import subprocess


class DispatchError(Exception):
    pass


class InferenceBusyError(Exception):
    pass


class Runner:
    def __init__(self, model_id: str, api_port: int):
        pass

    @property
    def model_id(self) -> str:
        pass

    @property
    def last_use_time(self) -> datetime:
        pass

    @property
    def create_time(self) -> datetime:
        pass

    @property
    def api_port(self) -> int:
        pass

    def run(self):
        pass

    def run_inference(self, task_id: str, image_url: str, post_process: str):
        pass

    def is_busy(self) -> bool:
        pass

    def kill(self):
        pass


class Dispatcher:

    def __init__(self, slot_num: int = 10):
        self._lock: Lock = Lock()
        self._slot_num: int = slot_num
        self._slot: List[Runner] = [None for _ in range(slot_num)]

    async def dispatch(self, model_id: str, task_id: str, image_url: str, post_process: str):
        if not self.is_model_available(model_id):
            raise DispatchError("invalid model id.")
        if not self.is_post_process_available(post_process):
            raise DispatchError("invalid post process name.")

        async with self._lock:
            # If have free loaded model, we just use it.
            runner = self._find_a_loaded_free_model(model_id)
            if runner is not None:
                runner.run_inference(task_id, image_url, post_process)
                return

            # Or find a free slot and start a new runner.
            idx = self._find_a_free_slot_index()
            if idx is not None:
                runner = Runner(model_id)
                self._slot[idx] = runner
                runner.run_inference(task_id, image_url, post_process)
                return

            # No available slot, find if any slot can free to reuse slot.
            idx = self._find_a_reusable_slot_index()
            if idx is not None:
                self._slot[idx].kill()
                runner = Runner(model_id)
                self._slot[idx] = runner
                runner.run_inference(task_id, image_url, post_process)
                return

            # No any resource to run inference, error.
            raise InferenceBusyError("no resource to run inference task.")

    def is_model_available(self, model_id: str) -> bool:
        # TODO: search model directory to check if model available.
        return True

    def is_post_process_available(self, post_process: str) -> bool:
        # TODO: search all post process to check if post process available.
        return True

    def _find_a_loaded_free_model(self, model_id) -> Optional[Runner]:
        for r in self._slot:
            if r is None:
                continue
            if r.model_id == model_id and not r.is_busy():
                return r
        return None

    def _find_a_free_slot_index(self) -> Optional[int]:
        for i in range(self._slot_num):
            if self._slot[i] is None:
                return i
        return None

    def _find_a_reusable_slot_index(self) -> Optional[int]:
        last_time = datetime.max
        idx = None

        for i in range(self._slot_num):
            if self._slot[i] is None:
                continue

            if self._slot[i].is_busy():
                continue

            if self._slot[i].last_use_time < last_time:
                idx = i
                last_time = self._slot[i].last_use_time

        return idx
