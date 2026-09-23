import pytest

from backend import db
from backend.brain.engine import set_engine
from backend.brain.skill import seed_db
from backend.teacher import GeminiTeacher, set_teacher

from .fakes import FakeEngine, FakeGeminiClient


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


@pytest.fixture()
def teacher():
    """Install a teacher whose Gemini client is scripted per test.

    Off by default everywhere else, which is what keeps the stage-2 tests — and
    a key-less deployment — on the Laya-only path.
    """

    def install(*script) -> FakeGeminiClient:
        """Install a teacher on the request path and hand back its fake client."""
        fake = FakeGeminiClient(*script)
        set_teacher(GeminiTeacher(client=fake, model="fake-model"))
        return fake

    yield install
    set_teacher(None)
