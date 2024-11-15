from .config import Config, get_global_config
from .models import InferenceTask


def make_image_key(uid: str, extension: str) -> str:
    conf = get_global_config()
    return f"{conf.image_prefix}::{extension}::{uid}"


def make_image_url(uid: str, extension: str) -> str:
    conf = get_global_config()
    return f"redis://{conf.redis_host}:{conf.redis_port}/{make_image_key(uid, extension)}"
