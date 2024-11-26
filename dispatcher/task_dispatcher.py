import subprocess
from threading import Event
from uuid import uuid1

import redis
from loguru import logger

from gw.dispatcher import Dispatcher
from gw.runner import RunnerPool, WorkerStarter
from gw.settings import get_app_settings
from gw.streams import Streams
from gw.task import TaskPool


class SubprocessStarter(WorkerStarter):
    def start_runner(self, name, model_id):
        subprocess.Popen(["python", "task_runner.py", name, model_id])


def main():
    logger.info("start dispatcher...")

    settings = get_app_settings()

    rdb = redis.Redis(host=settings.redis_host,
                      port=settings.redis_port,
                      db=settings.redis_db)
    logger.info(
        f"connect to redis {settings.redis_port}:{settings.redis_port}, use db {settings.redis_db}")

    taskpool = TaskPool(connection_pool=rdb.connection_pool,
                        ttl=settings.task_lifetime_s)
    logger.info(
        f"connect task pool, task lifetime set to {settings.task_lifetime_s} second(s)")

    starter = SubprocessStarter()
    runnerpool = RunnerPool(
        connection_pool=rdb.connection_pool, starter=starter)
    logger.info(f"connect runner pool, use {type(starter)} as starter.")

    dispatcher = Dispatcher(connection_pool=rdb.connection_pool,
                            runner_pool=runnerpool, max_runner=settings.runner_slot_num)
    logger.info(f"init dispatcher, max runner number {dispatcher.runner_num}")

    task_create_stream = Streams(
        connection_pool=rdb.connection_pool).task_create
    consumer = str(uuid1())
    logger.info(
        f"use task create stream, stream name {task_create_stream.stream}, readgroup {task_create_stream.readgroup}, consumer {consumer}")

    stop_evt = Event()
    while not stop_evt.is_set():
        messages = task_create_stream.pull(consumer, count=1, block=1 * 1000)
        if len(messages) == 0:
            continue

        tid = messages[0].data["task_id"].decode()
        task = taskpool.get(task_id=tid)
        if task is None:
            logger.warning(f"task not exists, id {tid}")
            messages[0].ack()
            continue

        try:
            dispatcher.dispatch(task)
            logger.info(
                f"task dispatch, id {task.task_id}, use model {task.model_id}")
            messages[0].ack()
        except:
            pass


if __name__ == "__main__":
    main()
