import pytest

@pytest.fixture
def fake_redis_client():
    import fakeredis
    return fakeredis.FakeStrictRedis()