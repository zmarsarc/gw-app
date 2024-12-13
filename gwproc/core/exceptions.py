import binascii
import time
import traceback
from functools import wraps

import PIL

from utils.log_config import get_logger

logger = get_logger()


# 定义装饰器error_handler
def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            result = {}
            result["code"] = 400
            result["time"] = round(time.time() * 1000)
            result["data"] = []
            if isinstance(e, binascii.Error):
                result["message"] = '图片解析出错，请检查图片格式是否正确。'
                logger.error(traceback.format_exc())
                return result
            if isinstance(e, PIL.UnidentifiedImageError):
                result["message"] = '图片解析出错，请检查图片格式是否正确。'
                logger.error(traceback.format_exc())
                return result
            else:
                result["message"] = e.__repr__()
                logger.error(traceback.format_exc())
                return result

    return wrapper
