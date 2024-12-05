import os
from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0

    log_level: str = "DEBUG"

    app_root: str = "/app"

    allow_format: Set[str] = {"png", "jpg", "jpeg"}

    pending_message_claim_time_ms: int = 30 * 1000

    image_lifetime_s: int = 24 * 60 * 60
    task_lifetime_s: int = 24 * 60 * 60

    runner_slot_num: int = 10
    runner_heartbeat_ttl_s: int = 10
    runner_heartbeat_update_period_s: int = 9

    @property
    def pt_model_root(self) -> str:
        return os.path.join(self.app_root, "models")


@lru_cache
def get_app_settings():
    return AppSettings()
