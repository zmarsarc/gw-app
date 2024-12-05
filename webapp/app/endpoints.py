from uuid import uuid4

import redis.asyncio as redis
from fastapi import APIRouter, Request, UploadFile
from loguru import logger

from gw.settings import AppSettings
from gw.streams import RedisStream
from gw.tasks import TaskPool

from . import models

router = APIRouter()


def get_global_config(req: Request) -> AppSettings:
    return req.app.state.app_settings


def get_task_pool(req: Request) -> TaskPool:
    return req.app.state.taskpool


def get_task_create_stream(req: Request) -> RedisStream:
    return req.app.state.stream


@router.post("/task")
async def create_task(task: models.CreateInferenceTaskRequest, req: Request):

    try:
        # FIXME
        t = get_task_pool(req).new(
            model_id=task.mid,
            image_url=task.image_url,
            post_process=task.post_process,
            callback=task.callback
        )
        logger.info(f"create new task {t.task_id}, data {task.model_dump()}")

        get_task_create_stream(req).publish({"task_id": t.task_id})
        logger.info(f"send task id {t.task_id} to task create stream")

    except redis.ConnectionError as e:
        logger.error(f"create new task error, {str(e)}")
        return models.APIResponse(ok=models.REQUEST_ERR, message="redis connection error")

    return models.CreateInferenceTaskResponse(message="inference task created", task_id=t.task_id)
