import signal
import threading
import time
from datetime import datetime

import redis
from loguru import logger

from gw.runner import Command, Message, Runner
from gw.settings import get_app_settings
from gw.streams import Streams
from gw.task import TaskPool


def read_name_and_model_id_from_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("model_id")

    cli = parser.parse_args()
    return (cli.name, cli.model_id)


def make_signal_handler(evt: threading.Event):
    def handler(signum, frame):
        evt.set()

    return handler


def runner_heartbeat(
    runner: Runner, period: float, ttl: float, stop_flag: threading.Event
):
    runner.update_heartbeat(datetime.now(), ttl)

    while not stop_flag.wait(period):
        runner.update_heartbeat(datetime.now(), ttl)


def main(name: str, model_id: str):

    settings = get_app_settings()

    # Connect redis.
    rdb = redis.Redis(
        host=settings.redis_host, port=settings.redis_port, db=settings.redis_db
    )
    logger.info(
        f"connect redis {settings.redis_host}:{settings.redis_port}, "
        + f"use db {settings.redis_db}"
    )

    # Try connect runner data.
    # We assume runner data already exists in redis and it always exists.
    # it promised by dispatcher.
    runner = Runner(rdb=rdb, name=name)
    runner.is_alive = True

    # Connect stream, which use to notify that inference complete.
    complete_stream = Streams(rdb=rdb).task_inference_complete
    logger.info(f"notify complete message via {complete_stream.stream}")

    # Make a message consumer name which use to receive commands.
    consumer = f"{name}::runner::consumer"
    logger.info(f"receive runner command use name {consumer}")

    # Connect task pool.
    taskpool = TaskPool(rdb=rdb)

    # A event to flag if it need to exit.
    stop_flag = threading.Event()

    # Register signal handler, start runner heartbeat.
    signal.signal(signal.SIGTERM, make_signal_handler(stop_flag))
    signal.signal(signal.SIGINT, make_signal_handler(stop_flag))
    threading.Thread(
        target=runner_heartbeat,
        args=(
            runner,
            settings.runner_heartbeat_ttl_s,
            settings.runner_heartbeat_update_period_s,
            stop_flag,
        ),
    ).start()
    logger.info(
        "start runner heartbeat, "
        + f"heartbeat ttl {settings.runner_heartbeat_ttl_s} second(s), "
        + f"update period {settings.runner_heartbeat_update_period_s} second(s)."
    )

    # TODO: load model here...

    logger.info("start message loop.")
    while not stop_flag.is_set():

        # Pull one message form command, block 1000 ms.
        # Give a chance to check if stop flag was set.
        messages = runner.stream.pull(consumer, count=1, block=1 * 1000)

        # Ignore when no message receive.
        if len(messages) == 0:
            continue

        msg = messages[0]
        cmd = Message.model_validate(msg.data)
        logger.info(f"receive message {msg.id}")

        # Command is task, do inference task.
        if cmd.cmd == Command.task:
            tid = cmd.data.decode()
            task = taskpool.get(tid)

            # Ignore invalid task, but consume this message.
            if task is None:
                msg.ack()
                continue

            # Update task status to let dispatcher know inference running.
            logger.info(
                f"run new inference, task id {task.task_id}, "
                + f"image url {task.image_url}"
            )
            runner.is_busy = True
            runner.utime = datetime.now()
            runner.task = task.task_id

            # TODO: do actual inference things.
            # We here just assume inference need 30 seconds.
            # Then notify inference complete.
            # NOTE: For test we blocking time
            import os

            blocking_time = int(os.environ.get("TEST_BLOCK_TIME", "30"))
            logger.debug(f"blocking time {blocking_time} secnods.")
            time.sleep(blocking_time)
            # FIXME: put inference result into redis.

            # Notify post process that inference complete.
            complete_stream.publish({"task_id": tid})
            msg.ack()
            logger.info(f"task {tid} infernece complete, notified.")

            # Update runner status make dispatcher know I'm available.
            runner.task = None
            runner.is_busy = False

            continue

        # Command is stop, set stop flag.
        if cmd.cmd == Command.stop:
            logger.info("recieve stop command, stop message loop.")
            stop_flag.set()
            msg.ack()
            break

    # Do cleanup.
    # Delete heartbeat to notify runner exit.
    # Set exit flag
    logger.info("message loop stopped, cleanup...")
    runner.clean_heartbeat()
    runner.is_alive = False
    rdb.close()


if __name__ == "__main__":
    from multiprocessing import current_process

    from gw.utils import initlize_logger

    (name, model_id) = read_name_and_model_id_from_cli()

    initlize_logger(f"runner-{name}")

    logger.info(f"start new runner by name [{name}], " +
                f"model_id [{model_id}], " +
                f"pid [{current_process().pid}]")
    main(name, model_id)
    logger.info(f"runner {name} shutdown.")
