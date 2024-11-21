import asyncio
from uuid import uuid1

import pydantic
import redis.asyncio as redis
from loguru import logger
from task_dispatcher import Dispatcher, DispatchError, InferenceBusyError

import gw

settings = gw.get_app_settings()

consumer_name = f"gw-dispatcher-{uuid1()}"

dispatcher = Dispatcher(settings.runner_slot_num)


@gw.messagehandler(
    host=settings.redis.host, port=settings.redis.port,
    stream_name=settings.redis.s_task_create, group_name=settings.redis.rg_task_create, consumer_name=consumer_name,
    min_idle_time=settings.pending_message_claim_time_ms
)
async def task_create_message_handler(rdb: redis.Redis, msg: gw.StreamMessage) -> bool:
    try:
        msg = gw.TaskIdMessage.model_validate(msg.message)
        task_id = msg.id
        task_key = gw.make_task_key(task_id)

        resp = await rdb.get(task_key)
        task = gw.InferenceTask.model_validate_json(resp)
    except pydantic.ValidationError:
        logger.warning(
            f"invalid message or task data, msg [{msg}] data [{resp}], abort.")
        return True
    except redis.ConnectionError:
        logger.error("redis connection broken")
        return False

    try:
        dispatcher.dispatch(task.inference_model_id, task.image_url)
    except DispatchError as e:
        logger.error(f"dispatch task error, {e}")
        return True
    except InferenceBusyError as e:
        logger.warning(f"too many inference task running.")
        return False

    return True

if __name__ == '__main__':
    asyncio.run(task_create_message_handler())
