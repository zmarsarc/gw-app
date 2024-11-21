from threading import RLock, Thread
import time
from loguru import logger


class ModelRunner:

    def __init__(self):
        self._busy: bool = False
        self._lock: RLock = RLock()
        self._t: Thread = None

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def run_innference(self):
        with self._lock:
            self._lock = True
            self._t = Thread(target=self.inference, kwargs={})
            self._t.start()

    def inference(self):
        logger.info("run inference")
        
        time.sleep(30)  # TODO: do any actual model inference things.

        # Keep cleanup code.
        with self._lock:
            self._t = None
            self._busy = False
