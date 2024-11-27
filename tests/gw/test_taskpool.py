from gw.task import Task, TaskPool


def test_new_task(fake_redis_client):
    pool = TaskPool(rdb=fake_redis_client)
    task = pool.new(model_id="model", image_url="image", post_process="postprocess", callback="callback", task_id="task")

    assert task.task_id == "task"
    assert task.model_id == "model"
    assert task.image_url == "image"
    assert task.post_process == "postprocess"
    assert task.callback == "callback"


def test_get_task(fake_redis_client):
    pool = TaskPool(rdb=fake_redis_client)
    tid = "abc"

    assert pool.get(tid) == None

    pool.new("def", "xyz", "foo", "bar", task_id=tid)
    task = pool.get(tid)
    
    assert task is not None
    assert task.task_id == tid
    assert task.model_id == "def"
    assert task.image_url == "xyz"
    assert task.post_process == "foo"
    assert task.callback == "bar"


def test_del_task(fake_redis_client):
    pool = TaskPool(rdb = fake_redis_client)
    
    pool.new("model", "image", "postprocess", "callback", "task_id")
    assert pool.get("task_id") is not None

    pool.delete("task_id")
    assert pool.get("task_id") is None


def test_task_result(fake_redis_client):
    pool = TaskPool(rdb=fake_redis_client)
    task = pool.new("model", "image", "postprocess", "callback", "task_id")

    assert task.result is None
    
    task.result = "result".encode()
    assert task.result.decode()  == "result"