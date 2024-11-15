import asyncio
import signal
from uuid import uuid1

import redis.asyncio as redis
from loguru import logger

import gw
import gw.redis_utils

exit_signal_trigged = False


def signal_handler(signum, frame):
    global exit_signal_trigged
    exit_signal_trigged = True


signal.signal(signal.SIGTERM, signal_handler)


async def main():
    # Init config.
    conf = gw.get_global_config()

    # Connect redis.
    rdb = redis.Redis(host=conf.redis_host, port=conf.redis_port)

    # Create Consume Group.
    try:
        await rdb.xgroup_create(
            conf.redis_keys.stream_task_finish,
            conf.redis_keys.readgroup_task_finish_notifier,
            mkstream=True)
    except redis.ResponseError:
        pass

    consumer_name = f"task-finish-notifier-{uuid1()}"
    logger.info(
        f"inference task finish notifier online, consumer name {consumer_name}")

    # We claim all other pending messages and try to re-process.
    # Because it will make a new consumer name every time when this app run,
    # pending messages in another consumer wihch offline will never have a chance
    # to process, so claim those messages.
    await rdb.xautoclaim(conf.redis_keys.stream_task_finish,
                         conf.redis_keys.readgroup_task_finish_notifier,
                         consumer_name,
                         1 * 1000,  # 60 seconds
                         justid=True)

    # Handle pending message.
    resp = await rdb.xreadgroup(conf.redis_keys.readgroup_task_finish_notifier,
                                consumer_name,
                                {conf.redis_keys.stream_task_finish: 0})
    pending_finish_messages = gw.redis_utils.readgroup_response_to_dict(
        resp)[conf.redis_keys.stream_task_finish]
    if len(pending_finish_messages) != 0:
        logger.info("have pending finish message, process...")
        for msg in pending_finish_messages:
            # TODO: do actual things to handle message.

            await rdb.xack(conf.redis_keys.stream_task_finish,
                           conf.redis_keys.readgroup_task_finish_notifier,
                           msg.id)
            logger.info(f"message {msg.id} handled and ack.")

    # Listen new message.
    while not exit_signal_trigged:
        try:
            resp = await rdb.xreadgroup(conf.redis_keys.readgroup_task_finish_notifier,
                                        consumer_name,
                                        {conf.redis_keys.stream_task_finish: ">"}, block=1000)
            finsih_messages = gw.redis_utils.readgroup_response_to_dict(resp)
            if conf.redis_keys.stream_task_finish not in finsih_messages:
                logger.debug("no finish message in last 1 seconds")
                continue

            logger.info("have new task finish messages")
            for msg in finsih_messages[conf.redis_keys.stream_task_finish]:
                # TODO: do actual things to handle message.

                await rdb.xack(conf.redis_keys.stream_task_finish,
                               conf.redis_keys.readgroup_task_finish_notifier,
                               msg.id)
                logger.info(f"message {msg.id} handled and ack.")

        except asyncio.CancelledError:
            break

    await rdb.aclose()
    logger.info("redis disconnect, quit...")


if __name__ == '__main__':
    asyncio.run(main())
