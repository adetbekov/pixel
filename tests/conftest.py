import pytest

from backend import db
from backend.brain.engine import set_engine
from backend.brain.skill import seed_db

from .fakes import FakeEngine


@pytest.fixture()
def conn(tmp_path):
    """A fresh database per test — never the developer's ./pixel.db."""
    connection = db.init(str(tmp_path / "pixel.db"))
    yield connection
    db.close()


@pytest.fixture()
def seeded(conn):
    """The four starter skills, loaded into the test database."""
    seed_db(conn)
    return conn


@pytest.fixture()
def engine():
    """Install a fake engine on the request path, then take it back out."""
    fake = FakeEngine()
    set_engine(fake)
    yield fake
    set_engine(None)
