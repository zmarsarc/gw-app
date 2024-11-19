from uuid import uuid1

import redis.asyncio as redis

import gw

settings = gw.get_app_settings()

consumer_name = f"gw-dispatcher-{uuid1()}"


@gw.messagehandler(
    host=settings.redis.host, port=settings.redis.port,
    stream_name=settings.redis.s_task_finish, group_name=settings.redis.rg_task_finish, consumer_name=consumer_name,
    min_idle_time=settings.pending_message_claim_time_ms
)
async def task_create_message_handler(rdb: redis.Redis, msg: gw.StreamMessage) -> bool:
    return True


if __name__ == '__main__':
    task_create_message_handler()
