from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI
from loguru import logger

from gw.settings import get_app_settings
from gw.streams import Streams
from gw.tasks import TaskPool

from . import endpoints


@asynccontextmanager
async def lifespan(app: FastAPI):
    conf = get_app_settings()
    logger.info(f"load app settings: {conf.model_dump()}")
    app.state.app_settings = conf

    rdb = redis.Redis(host=conf.redis_host,
                      port=conf.redis_port,
                      db=conf.redis_db)
    logger.info("connect redis")
    app.state.redis_connection = rdb

    taskpool = TaskPool(connection_pool=rdb.connection_pool,
                        task_ttl=conf.task_lifetime_s)
    app.state.taskpool = taskpool

    stream = Streams(connection_pool=rdb.connection_pool).task_create
    app.state.stream = stream

    yield

    rdb.close()


app = FastAPI(lifespan=lifespan)
app.include_router(endpoints.router)
