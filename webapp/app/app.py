from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from loguru import logger

import gw

from . import endpoints


@asynccontextmanager
async def lifespan(app: FastAPI):
    conf = gw.get_app_settings()
    logger.info(f"load app settings: {conf.model_dump()}")
    app.state.app_settings = conf

    rdb = redis.Redis(host=conf.redis.host,
                      port=conf.redis.port, decode_responses=False)
    logger.info("connect redis")
    app.state.redis_connection = rdb

    yield

    await rdb.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(endpoints.router)
