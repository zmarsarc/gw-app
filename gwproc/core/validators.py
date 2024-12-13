import json
import time
import traceback
from typing import List, Literal

import jsonpath
from pydantic import BaseModel, Field, field_validator

from utils.commons import get_dict_all_keys
from utils.log_config import get_logger

logger = get_logger()


class CustomValueError(Exception):
    pass


# 图像param基本类型
class CvExtraArgsParamBaseModel(BaseModel):
    conf: float = Field(
        default=None,
        ge=0,
        le=1
    )
    iou: float = Field(
        default=None,
        ge=0,
        le=1
    )


class CvExtraArgsBaseModel(BaseModel):
    model: str
    param: CvExtraArgsParamBaseModel

    class Config:
        str_min_length = 1


class CvFullModel(BaseModel):
    task_tag: str
    image_type: Literal['base64']
    images: List[str]
    extra_args: List[CvExtraArgsBaseModel]

    # class_: str = Field(
    #     alias='class'
    # )

    @field_validator('images', mode='after')
    def validate_images(cls, value):
        if len(value) == 0:
            raise CustomValueError("没有检测到照片")
        for i, image in enumerate(value):
            value[i] = image.split(';base64,')[-1]
        return value

    class Config:
        str_min_length = 1


class CheckBase():

    def get_all_keys(self, *args, **kwargs):
        raise 'not implement'

    def check_request_data_common(self, request_data, **kwargs):
        task_tag = request_data.get("task_tag")

        if not task_tag or (task_tag not in self.model_list_string) or (task_tag != self.task_tag):
            return False, {'code': 400, "message": "task_tag参数有误", 'time': round(time.time() * 1000),
                           'data': []}
        if not request_data.get("extra_args"):
            return False, {'code': 400, "message": "extra_args不能为空", 'time': round(time.time() * 1000),
                           'data': []}
        req_model_list = jsonpath.findall('$..model', request_data)
        req_param_list = jsonpath.findall('$..param', request_data)
        if len(req_model_list) != len(req_param_list):
            return False, {'code': 400, "message": "model或param参数不完整", 'time': round(time.time() * 1000),
                           'data': []}
        if len(req_model_list) and not all(req_model_list):
            return False, {'code': 400, "message": "model参数不能为空", 'time': round(time.time() * 1000),
                           'data': []}
        for req_model in req_model_list:
            if req_model not in self.model_list_string:
                return False, {'code': 400, "message": "model参数有误", 'time': round(time.time() * 1000),
                               'data': []}

        return True, request_data

    def check_all_keys(self, request_data):
        request_keys = get_dict_all_keys(request_data)
        # request_keys = jsonpath.findall('$..~', request_data)

        # 检测param的key是否正确
        if not set(request_keys).issubset(set(self.all_proj_keys)):
            return False, {'code': 400, "message": f"输入的json中key有误，参考值: {self.all_proj_keys}",
                           'time': round(time.time() * 1000),
                           'data': []}
        return True, request_data

    @property
    def _CurrentModel(self):
        return self.CurrentModel

    def request_data_adaptor(self, request_data):
        '''
        经过pydantic校验后的request_data
        :param request_data:
        :return: 需要返回check表示符
        '''
        try:
            valid_param = self._CurrentModel(**request_data)
            request_data = valid_param.model_dump()
        except Exception as e:
            logger.error(traceback.format_exc())
            if isinstance(e, CustomValueError):
                err = e.__str__()
            else:
                err = [i.strip() for i in e.__repr__().split('\n') if
                       not ('validation error' in i or 'For further information' in i)]
                err = f"参数有误：{', '.join(err)}"
            return False, {'code': 400, "message": err, 'time': round(time.time() * 1000),
                           'data': []}

        return True, request_data

    def check_entrance(self, request_data):
        # 先检查所有的key是否合法
        check_pass, check_result = self.check_all_keys(request_data)
        if not check_pass:
            return check_pass, check_result

        # 检查req入参类型，并对数据强转，失败则报错
        check_pass, check_result = self.check_request_data_common(request_data)

        if not check_pass:
            return check_pass, check_result

        # 检查req入参类型，并对数据强转，失败则报错
        check_pass, request_data_valid = self.request_data_adaptor(request_data)

        if not check_pass:
            return check_pass, request_data_valid

        return True, request_data_valid


if __name__ == '__main__':

    payload = {
        "task_tag": "leak_detect",
        "image_type": "base64",
        "images": ["asdfasdfsd"],
        "class": 'sdf',
        "extra_args": [
            {
                "model": "svg",
                "param": {"conf": "0.5", "iou": 0.5},
                "tmp": 0

            },
            {
                "model": "battery",
                "param": {"conf": 0.5, "iou": 0.5}
            }
        ]
    }

    try:
        ret = CvFullModel(**payload)
        print(json.dumps(ret.model_dump(), indent=4, ensure_ascii=Field()))
    except Exception as e:
        if isinstance(e, CustomValueError):
            err = e.__str__()
        else:
            err = [i.strip() for i in e.__repr__().split('\n') if
                   not ('validation error' in i or 'For further information' in i)]
            err = ', '.join(err)
            err = f"参数有误：{err}"
        print(err)
    exit()
