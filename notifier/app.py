from uuid import uuid1

import redis.asyncio as redis
from loguru import logger
import asyncio
import gw

settings = gw.get_app_settings()

consumer_name = f"task-finish-notifier-{uuid1()}"


@gw.messagehandler(
    host=settings.redis.host, port=settings.redis.port,
    stream_name=settings.redis.s_task_create, group_name=settings.redis.rg_task_create, consumer_name=consumer_name,
    min_idle_time=settings.pending_message_claim_time_ms
)
async def task_finish_message_handler(rdb: redis.Redis, msg: gw.StreamMessage) -> bool:
    return True


if __name__ == '__main__':
    asyncio.run(task_finish_message_handler())
