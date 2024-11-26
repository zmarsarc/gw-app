from datetime import datetime

from redis import ConnectionPool, Redis

from .runner import RunnerPool
from .task import Task


class Dispatcher:

    key_runner_num = "gw::dispatcher::runner_num"

    def __init__(
        self,
        runner_pool: RunnerPool = None,
        rdb: Redis = None,
        connection_pool: ConnectionPool = None,
        max_runner: int = 10,
    ) -> None:
        if rdb is not None:
            self._rdb = rdb
        elif connection_pool is not None:
            self._rdb = Redis(connection_pool=connection_pool)
        else:
            raise TypeError("must have at least one redis client")

        if runner_pool is None:
            raise TypeError("must have a runner pool to manage runners.")

        self._runnerpool = runner_pool
        self.runner_num = max_runner

    @property
    def redis_client(self) -> Redis:
        return self._rdb

    @property
    def runner_num(self) -> int:
        return int(self.redis_client.get(self.key_runner_num))

    @runner_num.setter
    def runner_num(self, num: int):
        self.redis_client.set(self.key_runner_num, num)

    def dispatch(self, task: Task):
        # If have loaded worker which runing model, use this.
        for r in self._runnerpool.runners():
            if r.model_id == task.model_id and not r.is_busy:
                r.run_task(task.task_id)
                return

        # If no worker, if have free slot then start a new runner.
        if self._runnerpool.count() < self.runner_num:
            runner = self._runnerpool.new(task.model_id)
            runner.run_task(task.task_id)
            return

        # If runner count reach the max number, try unload a runner.
        name = None
        last_utime = datetime.max
        for runner in self._runnerpool.runners():
            if not runner.is_busy:
                if runner.utime < last_utime:
                    last_utime = runner.utime
                    name = runner.name

        if name is not None:
            self._runnerpool.delete(name)
            runner = self._runnerpool.new(task.model_id)
            runner.run_task(task.task_id)
            return

        raise Exception("too busy.")
