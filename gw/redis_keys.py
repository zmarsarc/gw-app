from enum import StrEnum

class RedisKeys(StrEnum):

    max_runner_num = "runner_num::dispatcher::gw"

    inference_result = lambda x: f"{x}::inference::result::task::gw"
    postprocess_result = lambda x: f"{x}::postprocess:result::task::gw"
    task = lambda x: f"{x}::task::gw"