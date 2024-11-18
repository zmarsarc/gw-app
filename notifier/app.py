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
    conf = gw.get_app_settings()
    stream_name = conf.redis.s_task_finish
    readgroup_name = conf.redis.rg_task_finish
    consumer_name = f"task-finish-notifier-{uuid1()}"
    message_pending_time_ms = conf.pending_message_claim_time_ms

    # Connect redis.
    rdb = redis.Redis(host=conf.redis.host, port=conf.redis.port)

    # Create Consume Group.
    try:
        await rdb.xgroup_create(stream_name, readgroup_name, mkstream=True)
    except redis.ResponseError:
        pass

    logger.info(f"task notifier online, consumer name {consumer_name}")

    # We claim all other pending messages and try to re-process.
    # Because it will make a new consumer name every time when this app run,
    # pending messages in another consumer wihch offline will never have a chance
    # to process, so claim those messages.
    await rdb.xautoclaim(stream_name, readgroup_name, consumer_name,
                         message_pending_time_ms, justid=True)

    # Handle pending message.
    resp = await rdb.xreadgroup(readgroup_name, consumer_name, {stream_name: 0})
    pending_messages = gw.redis_utils.readgroup_response_to_dict(resp)[
        stream_name]
    if len(pending_messages) != 0:
        logger.info("have pending finish message, process...")
        for msg in pending_messages:
            # TODO: do actual things to handle message.

            await rdb.xack(stream_name, readgroup_name, msg.id)
            logger.info(f"message {msg.id} handled and ack.")

    # Listen new message.
    while not exit_signal_trigged:
        try:
            resp = await rdb.xreadgroup(readgroup_name, consumer_name, {stream_name: ">"}, block=1000)
            finsih_messages = gw.redis_utils.readgroup_response_to_dict(resp)
            if stream_name not in finsih_messages:
                logger.debug("no finish message in last 1 seconds")
                continue

            logger.info("have new task finish messages")
            for msg in finsih_messages[stream_name]:
                # TODO: do actual things to handle message.

                await rdb.xack(stream_name, readgroup_name, msg.id)
                logger.info(f"message {msg.id} handled and ack.")

        except asyncio.CancelledError:
            break

    await rdb.aclose()
    logger.info("redis disconnect, quit...")


if __name__ == '__main__':
    asyncio.run(main())
