from datetime import datetime
from typing import Optional

from pydantic import BaseModel

GET_STATUS_URL = "/status"
GET_INFO_URL = "/info"

POST_LOAD_INFERENCE_ENV = "/inference/env"
POST_RUN_INFERENCE = "/inference/run"


class GetStatusResponse(BaseModel):
    ready: bool
    busy: bool


class GetInfoResponse(BaseModel):
    ctime: datetime
    utime: datetime
    mid: Optional[str] = None
    postprocess: Optional[str] = None


class LoadInferenceEnvRequest(BaseModel):
    mid: str
    postprocess: str


class LoadInferenceEnvResponse(BaseModel):
    ok: bool
    reason: Optional[str] = None


class RunInferenceRequest(BaseModel):
    task_id: str
    image_url: str
    callback: str


class RunInferenceResponse(BaseModel):
    ok: bool
    reason: Optional[str] = None
