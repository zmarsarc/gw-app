from uuid import uuid4

import redis.asyncio as redis
from fastapi import APIRouter, Request, UploadFile
from loguru import logger

from gw.settings import AppSettings
from gw.streams import RedisStream
from gw.task import TaskPool

from . import models

router = APIRouter()


def get_global_config(req: Request) -> AppSettings:
    return req.app.state.app_settings


def get_task_pool(req: Request) -> TaskPool:
    return req.app.state.taskpool


def get_task_create_stream(req: Request) -> RedisStream:
    return req.app.state.stream

# def get_filename_extension(filename: str) -> str:
#     names = filename.split(".")
#     if len(names) < 2:
#         return ''
#     return names[-1]


# def check_image_format_by_filename(name: str, conf: gw.AppSettings) -> bool:
#     extension = get_filename_extension(name)
#     if extension == '' or extension not in conf.allow_format:
#         return False
#     return True


# @router.post("/image")
# async def upload_image(image: UploadFile, req: Request):
#     rdb = get_redis_connection(req)
#     conf = get_global_config(req)

#     if not check_image_format_by_filename(image.filename, conf):
#         logger.warning(
#             f"image format not support, 'filename' {image.filename}, support {conf.allow_format}")
#         return models.APIResponse(ok=models.REQUEST_ERR,
#                                   message=f"image format not support, should be one of {conf.allow_format}")

#     extension = get_filename_extension(image.filename)
#     uid = str(uuid4())
#     key = gw.make_image_key(uid, extension)
#     image_url = gw.make_image_url(uid, extension)

#     data = await image.read()
#     logger.info(f"upload image '{image.filename}', size {len(data)} bytes.")
#     await image.close()

#     try:
#         await rdb.set(key, data, conf.image_lifetime_s)
#         logger.info(f"write image '{image.filename}' into redis as key {key}, " +
#                     f"set ttl {conf.image_lifetime_s} seconds, " +
#                     f"url: {image_url}")
#     except redis.ConnectionError as e:
#         logger.error(f"redis connection error, {str(e)}")
#         return models.APIResponse(ok=models.REQUEST_ERR, message={"message": "redis connect error"})

#     return models.UploadImageResponse(message="image uploaded", url=image_url)


@router.post("/task")
async def create_task(task: models.CreateInferenceTaskRequest, req: Request):

    try:
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
