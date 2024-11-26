import multiprocessing
import signal
import threading
from uuid import uuid1

import redis
from loguru import logger

from gw.settings import get_app_settings
from gw.streams import RedisStream, Streams
from gw.task import Task, TaskPool


def make_signal_handler(evt: threading.Event):
    def handler(signum, frame):
        evt.set()
    return handler


def postprocess_worker(task: Task, stream: RedisStream):
    # TODO: do post process, set result in task then send notification to stream.

    # Let's just assume it need 10 seconds to do post process.
    # Then set result and notify next.
    import json
    import time

    time.sleep(10)
    task.result = json.dumps({
        "image": task.image_url,
        "postprocess": task.post_process,
        "result": "postprocess ok, task complete."
    }).encode()
    stream.publish({"task_id": task.task_id})


def main():

    # Connect to redis.
    settings = get_app_settings()
    rdb = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db
    )
    logger.info(f"connect redis {settings.redis_host}:{settings.redis_port}, " +
                f"use db {settings.redis_db}")

    # Connect message streams, and make a consumer name.
    # in stream use to pull message from runner to notify that inference complete.
    # out stream use to send postprocess complete message to next step,
    consumer = f"{uuid1}::postprocess::consumer"
    streams_maker = Streams(connection_pool=rdb.connection_pool)
    in_stream = streams_maker.task_inference_complete
    out_stream = streams_maker.task_finish
    logger.info(f"use input stream {in_stream.stream}, readgroup {in_stream.stream}, " +
                f"consumer name {consumer}. " +
                f"output stream {out_stream.stream}, readgroup {out_stream.readgroup}.")

    # Connect to task pool.
    taskpool = TaskPool(connection_pool=rdb.connection_pool)

    # Make process pool to run postprocess, assume postprocess need lot of CPU.
    workerpool = multiprocessing.Pool()

    # Make stop flag and register signal handler.
    stop_flag = threading.Event()
    signal.signal(signal.SIGTERM, make_signal_handler(stop_flag))

    logger.info("register signal handler and start message loop.")
    while not stop_flag.is_set():

        # Pull a message from input stream, block wait 1000 ms.
        # If no message we just continue.
        # This is in order to check if stop flag was set.
        # and if set we will stop loop.
        messages = in_stream.pull(consumer, count=1, block=1 * 1000)
        if len(messages) == 0:
            continue

        msg = messages[0]
        mid = msg.id
        tid = msg.data["task_id"].decode()
        logger.info(f"message received, message id {mid}, task id {tid}")

        # Read task data from task pool,
        # if task invalid, ignore and consume message.
        task = taskpool.get(tid)
        if task is None:
            msg.ack()
            continue

        # Put task into postprocess worker pool
        # Argument stream use to send notification when postprocess down.
        # No return value.
        workerpool.apply_async(func=postprocess_worker,
                               args=(task, out_stream))
        msg.ack()

    # Clean worker pool, terminate all working process.
    # Any unfinished post process will be drop.
    logger.info("recieve stop signal, cleanup...")
    workerpool.terminate()
    workerpool.join()
    rdb.close()


if __name__ == "__main__":
    logger.info("start post process app...")
    main()
    logger.info("post process app shutdown.")
