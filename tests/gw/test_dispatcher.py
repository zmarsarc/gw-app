from datetime import datetime
from threading import Event, Thread

from redis import Redis

from gw.tasks import TaskPool
from gw.dispatcher import Dispatcher
from gw.runner import Command, Message, Runner, RunnerPool, WorkerStarter


class FakeStarter(WorkerStarter):
    def __init__(self, rdb, evt: Event):
        self._rdb: Redis = rdb
        self._evt = evt
        self._stop_evt = Event()

    def __del__(self):
        self._stop_evt.set()

    def stop(self):
        if not self._stop_evt.is_set():
            self._stop_evt.set()

    def start_runner(self, name, model_id):
        runner = Runner(rdb=self._rdb, name=name)

        def func():
            while not self._stop_evt.is_set():
                messages = runner.stream.pull("consumer", count=1, block=100)
                if len(messages) == 0:
                    continue

                msg = Message.model_validate(messages[0].data)
                messages[0].ack()

                if msg.cmd == Command.task:
                    runner.is_busy = True
                    runner.utime = datetime.now()
                    runner.task = msg.data.decode()
                    self._evt.set()

                if msg.cmd == Command.stop:
                    return

        t = Thread(target=func)
        t.start()


def test_dispatch_task_on_model_already_exists(fake_redis_client):
    model_id = "test-model-id"
    runner_name = "fake-task-runner"
    fake_image_url = "fakeimageurl"
    fake_post_process = "fakepostprocess"
    fake_callback = "fakecallback"
    evt = Event()
    starter = FakeStarter(fake_redis_client, evt)

    runnerpool = RunnerPool(connection_pool=fake_redis_client.connection_pool, starter=starter)
    runnerpool.new(model_id=model_id, name=runner_name)

    taskpool = TaskPool(connection_pool=fake_redis_client.connection_pool)
    task = taskpool.new(
        model_id=model_id,
        image_url=fake_image_url,
        post_process=fake_post_process,
        callback=fake_callback,
    )
    dispatcher = Dispatcher(rdb=fake_redis_client, runner_pool=runnerpool)
    dispatcher.dispatch(task)

    evt.wait()

    runner = runnerpool.get(runner_name)
    assert runner.is_busy == True
    assert runner.task == task.task_id

    starter.stop()


def test_dispatch_task_wtih_no_running_worker(fake_redis_client):
    model_id = "test-model-id"
    fake_image_url = "fakeimageurl"
    fake_post_process = "fakepostprocess"
    fake_callback = "fakecallback"
    evt = Event()
    starter = FakeStarter(fake_redis_client, evt)

    runnerpool = RunnerPool(connection_pool=fake_redis_client.connection_pool, starter=starter)
    assert runnerpool.count() == 0

    taskpool = TaskPool(connection_pool=fake_redis_client.connection_pool)
    task = taskpool.new(
        model_id=model_id,
        image_url=fake_image_url,
        post_process=fake_post_process,
        callback=fake_callback,
    )
    dispatcher = Dispatcher(rdb=fake_redis_client, runner_pool=runnerpool)
    dispatcher.dispatch(task)

    evt.wait()

    assert runnerpool.count() == 1

    runner = runnerpool.runners()[0]
    assert runner.is_busy == True
    assert runner.task == task.task_id

    starter.stop()


def test_dispatch_task_no_free_slot(fake_redis_client):
    model_id = "test-model-id"
    runner_name = "fake-task-runner"
    fake_image_url = "fakeimageurl"
    fake_post_process = "fakepostprocess"
    fake_callback = "fakecallback"
    evt = Event()
    starter = FakeStarter(fake_redis_client, evt)

    runnerpool = RunnerPool(connection_pool=fake_redis_client.connection_pool, starter=starter)
    runnerpool.new(model_id=model_id, name=runner_name)

    taskpool = TaskPool(connection_pool=fake_redis_client.connection_pool)
    task = taskpool.new(
        model_id="another_model",
        image_url=fake_image_url,
        post_process=fake_post_process,
        callback=fake_callback,
    )
    dispatcher = Dispatcher(rdb=fake_redis_client, runner_pool=runnerpool, max_runner=1)
    dispatcher.dispatch(task)

    evt.wait()

    assert runnerpool.get(runner_name) is None
    assert runnerpool.count() == 1

    runner = runnerpool.runners()[0]
    assert runner.is_busy == True
    assert runner.task == task.task_id
    assert runner.model_id == "another_model"

    starter.stop()
