import signal
from datetime import datetime
from threading import Event, Thread

import redis
from loguru import logger

from gw.runner import Command, Message, Runner
from gw.task import TaskPool

exit_flag = Event()


def signal_handler(signum, frame):
    if not exit_flag.is_set():
        exit_flag.set()


signal.signal(signal.SIGTERM, signal_handler)


def start_heartbeat(runner: Runner) -> Event:
    stop_flag = Event()

    def func():
        while True:
            runner.utime = datetime.now()
            if stop_flag.wait(30):
                return

    Thread(target=func).start()
    return stop_flag


def main(runner: Runner, taskpool: TaskPool):
    try:
        while not stop_flag.is_set():
            messages = runner.stream.pull(f"consumer-{runner.name}",count=1, block=100)
            if len(messages) == 0:
                continue
            messages[0].ack()

            msg = Message.model_validate(messages[0].data)
            if msg.cmd == Command.task:
                logger.info("receive run task command.")

                tid = msg.data.decode()
                task = taskpool.get(task_id=tid)
                logger.info(
                    f"task id {task.task_id}, image url {task.image_url}")

            if msg.cmd == Command.stop:
                logger.info("recive stop command, exit...")
                return

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    from argparse import ArgumentParser

    from gw.settings import get_app_settings

    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("model_id")
    cli = parser.parse_args()

    logger.info(f"start runner {cli.name}, use model {cli.model_id}")

    settings = get_app_settings()
    rdb = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db)

    runner = Runner(rdb=rdb, name=cli.name)
    taskpool = TaskPool(connection_pool=rdb.connection_pool)

    stop_flag = start_heartbeat(runner)
    main(runner, taskpool)

    stop_flag.set()
