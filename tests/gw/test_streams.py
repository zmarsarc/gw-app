from threading import Event

from gw.streams import RedisStream, StreamMessage


def test_publish_message(fake_redis_client):
    stream = RedisStream("test", "test-reader", rdb=fake_redis_client)

    stream.publish({"message": "ok"})

    resp = fake_redis_client.xreadgroup("test-reader", "consumer", {"test": ">"})
    assert resp[0][0].decode() == "test"
    assert resp[0][1][0][1]["message".encode()] == "ok".encode()


def test_pull_message(fake_redis_client):
    stream = RedisStream("test", "test-reader", rdb=fake_redis_client)

    for i in range(10):
        stream.publish({"message": i})

    messages = stream.pull("consumer", 10)
    assert len(messages) == 10
    assert int(messages[0].data["message"]) == 0
    assert int(messages[9].data["message"]) == 9

    for m in messages:
        m.ack()

    messages = stream.pull("consumer", block=1)
    assert len(messages) == 0

    stream.publish({"message": "test"})
    messages = stream.pull("consumer")
    assert len(messages) == 1
    assert messages[0].data["message"] == "test".encode()

    messages[0].ack()
    stream.publish({"msg": "a"})
    stream.publish({"msg": "b"})

    messages = stream.pull("consumer", 1)
    assert len(messages) == 1
    assert messages[0].data["msg"] == "a".encode()
    messages[0].ack()

    messages = stream.pull("consumer", 1)
    assert len(messages) == 1
    assert messages[0].data["msg"] == "b".encode()
    messages[0].ack()


def test_subscribe(fake_redis_client):
    stream = RedisStream("abc", "def", rdb=fake_redis_client)
    evt = Event()
    answer = ""

    def callback(rdb, msg: StreamMessage):
        nonlocal answer
        answer = msg.data["message"].decode()
        msg.ack()
        evt.set()

    stopper = stream.subscribe("consumer", callback)

    stream.publish({"message": "test"})
    evt.wait()

    assert len(stream.pull("consumer", block=1)) == 0
    assert answer == "test"

    stopper.set()
