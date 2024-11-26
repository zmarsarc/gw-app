from pydantic import BaseModel

REQUEST_OK = 1
REQUEST_ERR = 0


class APIResponse(BaseModel):
    ok: int = REQUEST_OK
    message: str


class UploadImageResponse(APIResponse):
    url: str


class CreateInferenceTaskRequest(BaseModel):
    mid: str  # means model_id, where prefix 'model_' is used by pydantic
    image_url: str
    post_process: str
    callback: str


class CreateInferenceTaskResponse(APIResponse):
    task_id: str
