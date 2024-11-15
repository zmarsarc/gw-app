from pydantic import BaseModel


class InferenceTask(BaseModel):
    task_id: str
    inference_model_id: str
    image_url: str
    callback: str

    @staticmethod
    def new(tid: str, mid: str, url: str, cb: str) -> 'InferenceTask':
        return InferenceTask(task_id=tid,
                             inference_model_id=mid,
                             image_url=url,
                             callback=cb)
