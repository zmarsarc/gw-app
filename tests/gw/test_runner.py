from datetime import datetime

import pytest

from gw.redis_keys import RedisKeys
from gw.runner import Command, Runner, RunnerPool, WorkerStarter


@pytest.fixture
def fake_runner_pool(fake_redis_client):

    class Starter(WorkerStarter):
        def start_runner(self, name, model_id):
            r = Runner(name=name, rdb=fake_redis_client)
            r.is_alive = True
            r.update_heartbeat(datetime.now(), ttl=60)

    yield RunnerPool(connection_pool=fake_redis_client.connection_pool,
                     starter=Starter())


def test_new_runner(fake_runner_pool):
    ctime = datetime.now()
    expect_name = "abc"
    expect_model_id = "def"

    runner = fake_runner_pool.new(
        expect_model_id, name=expect_name, ctime=ctime)

    assert runner.name == expect_name
    assert runner.model_id == expect_model_id
    assert runner.ctime == ctime
    assert runner.utime == ctime
    assert runner.is_busy == False
    assert runner.task == None
    assert runner.heartbeat is not None
    assert runner.is_alive == True


def test_get_runner(fake_runner_pool):
    runner = fake_runner_pool.get("test_runner")
    assert runner is None

    fake_runner_pool.new("abc", name="test_runner")
    runner = fake_runner_pool.get("test_runner")
    assert runner is not None
    assert runner.name == "test_runner"


def test_delete_runner(fake_runner_pool):

    fake_runner_pool.new("abc", name="def")
    assert fake_runner_pool.get("def") is not None

    fake_runner_pool.delete("def")
    assert fake_runner_pool.get("def") is None


def test_runner_starter(fake_runner_pool):
    model_id = "abc"
    name = "xyz"

    runner = fake_runner_pool.new(model_id=model_id, name=name)
    assert runner.heartbeat is not None

    fake_runner_pool.delete(name=name)
    assert runner.heartbeat is None


def test_write_runner_utime(fake_runner_pool):
    dt = datetime.now()

    r = fake_runner_pool.new("model_id", ctime=dt)
    assert r.utime == dt

    new_time = datetime.max
    r.utime = new_time
    assert r.utime == new_time


def test_write_runner_busy(fake_runner_pool):

    r = fake_runner_pool.new("model_id")
    assert r.is_busy == False

    r.is_busy = True
    assert r.is_busy == True


def test_write_runner_task(fake_runner_pool):

    r = fake_runner_pool.new("model_id")
    assert r.task is None

    r.task = "abc"
    assert r.task == "abc"


def test_count_runner(fake_runner_pool):
    r1 = fake_runner_pool.new("abc")
    r2 = fake_runner_pool.new("def")

    assert fake_runner_pool.count() == 2

    fake_runner_pool.delete(r2.name)
    assert fake_runner_pool.count() == 1


def test_get_all_runners(fake_runner_pool):
    fake_runner_pool.new("abc", "r1")
    fake_runner_pool.new("def", "r2")

    runners = fake_runner_pool.runners()
    assert len(runners) == 2

    names = [r.name for r in runners]
    names.sort()
    assert names == ["r1", "r2"]


def test_runner_stream_and_readgroup(fake_runner_pool):

    runner = fake_runner_pool.new("abc")
    assert int(fake_runner_pool.exists(
        RedisKeys.runner_stream(runner.name))) == 1

    fake_runner_pool.xadd(RedisKeys.runner_stream(
        runner.name), {"message": "ok"})
    resp = fake_runner_pool.xreadgroup(
        RedisKeys.runner_stream_readgroup(runner.name), runner.name, {
            RedisKeys.runner_stream(runner.name): ">"}
    )

    assert resp[0][0].decode() == RedisKeys.runner_stream(runner.name)
    assert dict(resp[0][1][0][1])["message".encode()] == "ok".encode()

    fake_runner_pool.delete(runner.name)
    assert int(fake_runner_pool.exists(
        RedisKeys.runner_stream(runner.name))) == 0


def test_runner_alive_status(fake_runner_pool):
    r = fake_runner_pool.new("model")

    assert r.is_alive == True

    r.is_alive = True
    assert int(fake_runner_pool.hget(
        RedisKeys.runner(r.name), "is_alive")) == 1
    assert r.is_alive == True

    r.is_alive = False
    assert int(fake_runner_pool.hget(
        RedisKeys.runner(r.name), "is_alive")) == 0
    assert r.is_alive == False


def test_clean_dead_runner(fake_runner_pool):

    r = fake_runner_pool.new("test_clean_runner")
    r.clean_heartbeat()
    fake_runner_pool.clean_dead_runners()

    assert int(fake_runner_pool.exists(RedisKeys.runner(r.name))) == 0
    assert int(fake_runner_pool.exists(
        RedisKeys.runner_heartbeat(r.name))) == 0
    assert int(fake_runner_pool.exists(RedisKeys.runner_stream(r.name))) == 0


def test_update_and_clean_heartbeat(fake_runner_pool):
    r = fake_runner_pool.new("test_model")

    assert r.heartbeat is not None

    dt = datetime.now()
    r.update_heartbeat(dt, ttl=100)
    assert r.heartbeat == dt

    r.clean_heartbeat()
    assert r.heartbeat == None
