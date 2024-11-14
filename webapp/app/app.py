from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from loguru import logger

from . import config, endpoints


@asynccontextmanager
async def lifespan(app: FastAPI):
    conf = config.get_global_config()
    logger.info(f"load global config: {conf.model_dump()}")
    app.state.conf = conf

    rdb = redis.Redis(host=conf.redis_host, port=conf.redis_port, decode_responses=False)
    logger.info("connect redis")
    app.state.redis_connection = rdb

    yield

    await rdb.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(endpoints.router)
