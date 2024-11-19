from typing import Any, Dict, List

from .models import StreamMessage


def readgroup_response_to_dict(response: List[Any]) -> Dict[str, List[StreamMessage]]:
    result = {}
    for items in response:
        result[bytes(items[0]).decode()] = [StreamMessage(
            id=m[0], message=m[1]) for m in items[1]]
    return result
