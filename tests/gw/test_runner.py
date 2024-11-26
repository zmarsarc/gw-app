from datetime import datetime

from gw.runner import Command, Runner, RunnerPool, WorkerStarter


class DumbWorkerStarter(WorkerStarter):
    def start_runner(self, name, model_id):
        pass


class HeartbeatWorkerStarter(WorkerStarter):

    def __init__(self, rdb):
        self._rdb = rdb

    def start_runner(self, name, model_id):
        r = Runner(self._rdb, name)
        self._rdb.set(r.keys.heartbeat, datetime.now().isoformat())


def test_new_runner(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())
    ctime = datetime.now()
    expect_name = "abc"
    expect_model_id = "def"

    runner = pool.new(expect_model_id, name=expect_name, ctime=ctime)

    assert runner.name == expect_name
    assert runner.model_id == expect_model_id
    assert runner.ctime == ctime
    assert runner.utime == ctime
    assert runner.is_busy == False
    assert runner.task == None
    assert runner.heartbeat == None


def test_get_runner(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())

    runner = pool.get("test_runner")
    assert runner is None

    pool.new("abc", name="test_runner")
    runner = pool.get("test_runner")
    assert runner is not None
    assert runner.name == "test_runner"


def test_delete_runner(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())

    pool.new("abc", name="def")
    assert pool.get("def") is not None

    pool.delete("def")
    assert pool.get("def") is None


def test_runner_starter(fake_redis_client):
    model_id = "abc"
    name = "xyz"

    starter = HeartbeatWorkerStarter(fake_redis_client)
    pool = RunnerPool(rdb=fake_redis_client, starter=starter)
    runner = pool.new(model_id=model_id, name=name)
    assert runner.heartbeat is not None

    pool.delete(name=name)
    assert runner.heartbeat is None


def test_write_runner_utime(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())
    dt = datetime.now()

    r = pool.new("model_id", ctime=dt)
    assert r.utime == dt

    new_time = datetime.max
    r.utime = new_time
    assert r.utime == new_time


def test_write_runner_busy(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())

    r = pool.new("model_id")
    assert r.is_busy == False

    r.is_busy = True
    assert r.is_busy == True


def test_write_runner_task(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())

    r = pool.new("model_id")
    assert r.task is None

    r.task = "abc"
    assert r.task == "abc"


def test_count_runner(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())
    r1 = pool.new("abc")
    r2 = pool.new("def")

    assert pool.count() == 2

    pool.delete(r2.name)
    assert pool.count() == 1


def test_get_all_runners(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())
    pool.new("abc", "r1")
    pool.new("def", "r2")

    runners = pool.runners()
    assert len(runners) == 2

    names = [r.name for r in runners]
    names.sort()
    assert names == ["r1", "r2"]


def test_runner_stream_and_readgroup(fake_redis_client):
    pool = RunnerPool(rdb=fake_redis_client, starter=DumbWorkerStarter())

    runner = pool.new("abc")
    assert int(fake_redis_client.exists(runner.keys.stream)) == 1

    fake_redis_client.xadd(runner.keys.stream, {"message": "ok"})
    resp = fake_redis_client.xreadgroup(
        runner.keys.readgroup, runner.name, {runner.keys.stream: ">"}
    )

    assert resp[0][0].decode() == runner.keys.stream
    assert dict(resp[0][1][0][1])["message".encode()] == "ok".encode()

    pool.delete(runner.name)
    assert int(fake_redis_client.exists(runner.keys.stream)) == 0
