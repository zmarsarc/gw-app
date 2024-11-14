from uuid import uuid4

import redis.asyncio as redis
from fastapi import APIRouter, Request, UploadFile
from loguru import logger

from . import config, models

router = APIRouter()


def get_redis_connection(req: Request) -> redis.Redis:
    return req.app.state.redis_connection


def get_global_config(req: Request) -> config.Config:
    return req.app.state.conf


def get_filename_extension(filename: str) -> str:
    names = filename.split(".")
    if len(names) < 2:
        return ''
    return names[-1]


def check_image_format_by_filename(name: str, conf: config.Config) -> bool:
    extension = get_filename_extension(name)
    if extension == '' or extension not in conf.allow_format:
        return False
    return True


@router.post("/image")
async def upload_image(image: UploadFile, req: Request):
    rdb = get_redis_connection(req)
    conf = get_global_config(req)

    if not check_image_format_by_filename(image.filename, conf):
        logger.warning(
            f"image format not support, 'filename' {image.filename}, support {conf.allow_format}")
        return models.APIResponse(ok=models.REQUEST_ERR,
                                  message=f"image format not support, should be one of {conf.allow_format}")

    extension = get_filename_extension(image.filename)
    uid = str(uuid4())
    key = f"{conf.image_prefix}::{extension}::{uid}"
    image_url = f"redis://{conf.redis_host}:{conf.redis_port}/{key}"

    data = await image.read()
    logger.info(f"upload image '{image.filename}', size {len(data)} bytes.")
    await image.close()

    try:
        await rdb.set(key, data, conf.image_lifetime_s)
        logger.info(f"write image '{image.filename}' into redis as key {key}, " +
                    f"set ttl {conf.image_lifetime_s} seconds, " +
                    f"url: {image_url}")
    except redis.ConnectionError as e:
        logger.error(f"redis connection error, {str(e)}")
        return models.APIResponse(ok=models.REQUEST_ERR, message={"message": "redis connect error"})

    return models.UploadImageResponse(message="image uploaded", url=image_url)


@router.post("/task")
async def create_task(task: models.CreateInferenceTaskRequest, req: Request):
    rdb = get_redis_connection(req)
    conf = get_global_config(req)

    task_id = str(uuid4())
    new_task = models.InferenceTask(
        task_id=task_id, mid=task.mid, image_url=task.image_url, callback=task.callback)
    try:
        await rdb.xadd(conf.task_stream_name, new_task.model_dump())
        logger.info(
            f"send create new task message into stream, task: {new_task.model_dump()}")
    except redis.ConnectionError as e:
        logger.error(f"send message to stream error, {str(e)}")
        return models.APIResponse(ok=models.REQUEST_ERR, message="redis connection error")

    return models.CreateInferenceTaskResponse(message="inference task created", task_id=task_id)
