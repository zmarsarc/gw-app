import asyncio
import signal
import traceback

import redis.asyncio as redis
from loguru import logger

from . import redis_utils


async def connect_redis_and_make_sure_it_ok(host: str, port: int):
    logger.info(f"try connect redis {host}:{port} ...")
    try:
        rdb = redis.Redis(host=host, port=port)
        await rdb.ping()
        logger.info("redis connection ok.")
        return rdb
    except redis.ConnectionError as e:
        logger.error(f"connect redis error, {e}")
        raise


async def make_read_group(rdb: redis.Redis, stream_name: str, group_name: str):
    try:
        await rdb.xgroup_create(stream_name, group_name, "$", mkstream=True)
        logger.info(
            f"make read gronp {group_name} on stream {stream_name}")
    except redis.ResponseError:
        return
    except redis.ConnectionError as e:
        logger.error(f"redis connection broken, {e}")
        raise


async def message_handler_wrapper(func,
                                  host: str, port: int,
                                  stream_name: str, group_name: str, consumer_name: str,
                                  min_idle_time: int, block: int):

    try:
        rdb = await connect_redis_and_make_sure_it_ok(host, port)
        await make_read_group(rdb, stream_name, group_name)
        await rdb.xautoclaim(stream_name, group_name, consumer_name, min_idle_time, justid=True)
        logger.info("claim pending messsages from other comsumers.")

        resp = await rdb.xreadgroup(group_name, consumer_name, {stream_name: "0"})

        messages = redis_utils.readgroup_response_to_dict(resp)
        messages = messages[stream_name] if stream_name in messages else []

        for m in messages:
            logger.info(f"process pending message {m.id}")
            if await func(rdb, m):
                await rdb.xack(stream_name, group_name, m.id)
                logger.info(f"handle message {m.id} ok, ack.")
            else:
                logger.warning(f"handle message {m.id} not ok, keep pending.")

    except redis.ConnectionError as e:
        logger.error(f"connection broken, {e}")
        return
    except Exception as e:
        logger.error(f"error from message handler, {e}")
        for line in traceback.format_stack():
            logger.debug(line)
        await rdb.aclose()
        return

    exit_flag_set = False

    def signal_handler(signum, frame):
        nonlocal exit_flag_set
        exit_flag_set = True

    signal.signal(signal.SIGTERM, signal_handler)

    while not exit_flag_set:
        try:
            resp = await rdb.xreadgroup(group_name, consumer_name, {stream_name: ">"}, block=block)
            messages = redis_utils.readgroup_response_to_dict(resp)
            messages = messages[stream_name] if stream_name in messages else []

            if len(messages) == 0:
                logger.trace(
                    f"there are no message in last {block / 1000} seconds")
                continue

            for m in messages:
                logger.info(f"receive message {m.model_dump_json()}")
                if await func(rdb, m):
                    await rdb.xack(stream_name, group_name, m.id)
                    logger.info(f"handle message {m.id} ok, ack.")
                else:
                    logger.warning(
                        f"handle message {m.id} not ok, message pending.")

        except asyncio.CancelledError:
            break

        except redis.ConnectionError as e:
            logger.error(f"connection broken, {e}")
            return

        except Exception as e:
            logger.error(f"error from message handler, {e}")
            for line in traceback.format_stack():
                logger.debug(line)

    logger.info("disconnect redis ...")
    await rdb.aclose()


def messagehandler(host: str, port: int,
                   stream_name: str, group_name: str, consumer_name: str,
                   min_idle_time: int, block: int = 1 * 1000):

    def decorator(func):
        def wrapper():
            return message_handler_wrapper(
                func,
                host, port,
                stream_name, group_name, consumer_name,
                min_idle_time, block)
        return wrapper
    return decorator
