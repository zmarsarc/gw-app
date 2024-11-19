from .message_handler import messagehandler
from .models import InferenceTask, StreamMessage, TaskIdMessage
from .settings import AppSettings, get_app_settings


def make_image_key(uid: str, extension: str) -> str:
    conf = get_app_settings()
    return f"{conf.redis.image_prefix}::{extension}::{uid}"


def make_image_url(uid: str, extension: str) -> str:
    conf = get_app_settings()
    return f"redis://{conf.redis.host}:{conf.redis.port}/{make_image_key(uid, extension)}"
