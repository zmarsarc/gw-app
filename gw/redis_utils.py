from typing import Any, Dict, List

from .models import StreamMessage
from .settings import get_app_settings


def readgroup_response_to_dict(response: List[Any]) -> Dict[str, List[StreamMessage]]:
    result = {}
    for items in response:
        result[bytes(items[0]).decode()] = [StreamMessage(
            id=m[0], message=m[1]) for m in items[1]]
    return result


def make_image_key(uid: str, extension: str) -> str:
    conf = get_app_settings()
    return f"{conf.redis.image_prefix}::{extension}::{uid}"


def make_image_url(uid: str, extension: str) -> str:
    conf = get_app_settings()
    return f"redis://{conf.redis.host}:{conf.redis.port}/{make_image_key(uid, extension)}"


def make_task_key(tid: str) -> str:
    conf = get_app_settings()
    return f"{conf.redis.task_prefix}::{tid}"
