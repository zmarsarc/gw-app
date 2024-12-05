from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event, Thread
from typing import Callable, Dict, List

import redis
import redis.typing

from .redis_keys import RedisKeys


class StreamMessage:
    def __init__(
        self,
        id: str,
        data: Dict[bytes, bytes],
        rdb: redis.Redis,
        stream: str,
        readgroup: str,
    ) -> None:
        self._id = id
        self._data = {x.decode(): data[x] for x in data.keys()}
        self._rdb = rdb
        self._stream = stream
        self._readgroup = readgroup

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> Dict[str, bytes]:
        return self._data

    def ack(self):
        self._rdb.xack(self._stream, self._readgroup, self.id)


MessageCallback = Callable[[redis.Redis, StreamMessage], None]


class RedisStream:

    def __init__(
        self,
        stream: str,
        readgroup: str,
        rdb: redis.Redis = None,
        connection_pool: redis.ConnectionPool = None,
    ) -> None:

        if rdb is not None:
            self._rdb = rdb
        elif connection_pool is not None:
            self._rdb = redis.Redis(connection_pool=connection_pool)
        else:
            raise TypeError("must have a valid redis client.")

        self._stream = stream
        self._readgroup = readgroup

        try:
            self.redis_client.xgroup_create(
                self.stream, self.readgroup, mkstream=True)
        except redis.ResponseError:
            pass

    @property
    def redis_client(self) -> redis.Redis:
        return self._rdb

    @property
    def stream(self) -> str:
        return self._stream

    @property
    def readgroup(self) -> str:
        return self._readgroup

    def _scan_message(self, resp):
        # response like: [[b'test', [(b'1732536686488-0', {b'message': b'ok'})]]]
        if len(resp) == 0:
            return []

        streams = {
            x[0].decode(): [
                StreamMessage(
                    m[0].decode(),
                    m[1],
                    redis.Redis(
                        connection_pool=self.redis_client.connection_pool),
                    self.stream,
                    self.readgroup,
                )
                for m in x[1]
            ]
            for x in resp
        }

        return streams[self.stream] if self.stream in streams else []

    def publish(self, message: Dict[redis.typing.FieldT, redis.typing.EncodableT]):
        self.redis_client.xadd(self.stream, message)

    def pull(self, consumer: str, count: int = None, block: int = None) -> List[StreamMessage]:
        # move pending message belongs to another consumer in this readgroup to myself.
        self.redis_client.xautoclaim(
            self.stream, self.readgroup, consumer, min_idle_time=6000, justid=True
        )

        # read pending messages.
        resp = self.redis_client.xreadgroup(
            self.readgroup, consumer, {self.stream: "0"}, count=count, block=block
        )
        messages = self._scan_message(resp)
        if len(messages) != 0:
            return messages

        # read new messages.
        resp = self.redis_client.xreadgroup(
            self.readgroup, consumer, {self.stream: ">"}, count=count, block=block
        )
        return self._scan_message(resp)

    def subscribe(self, consumer: str, cb: MessageCallback) -> Event:
        # Use to stop notifier thread.
        evt = Event()

        def notifier():
            pool = ThreadPoolExecutor(max_workers=10)
            while not evt.is_set():
                messages = self.pull(consumer, count=1, block=100)
                if len(messages) == 0:
                    continue
                pool.submit(cb, self.redis_client, messages[0])

        t = Thread(target=notifier)
        t.start()

        return evt


class Streams:

    def __init__(self, rdb: redis.Redis = None,
                 connection_pool: redis.ConnectionPool = None):
        if rdb is not None:
            self._rdb = rdb
        elif connection_pool is not None:
            self._rdb = redis.Redis(connection_pool=connection_pool)
        else:
            raise TypeError("stream must have a redis connection.")

    @property
    def task_create(self) -> RedisStream:
        return RedisStream(RedisKeys.stream_task_create, RedisKeys.stream_readgroup_task_create, connection_pool=self._rdb.connection_pool)

    @property
    def task_inference_complete(self) -> RedisStream:
        return RedisStream(
            stream=RedisKeys.stream_inference_complete,
            readgroup=RedisKeys.stream_readgroup_inference_complete,
            connection_pool=self._rdb.connection_pool
        )

    @property
    def task_finish(self) -> RedisStream:
        return RedisStream(
            stream=RedisKeys.stream_postprocess_complete,
            readgroup=RedisKeys.stream_readgroup_postprocess_complete,
            connection_pool=self._rdb.connection_pool
        )
