import pytest

from backend import db
from backend.brain.engine import set_engine
from backend.brain.skill import seed_db
from backend.miner import set_generator, stop_worker
from backend.miner.generate import GeminiSkillGenerator
from backend.ratelimit import chat_limiter, mine_limiter
from backend.teacher import GeminiTeacher, set_teacher

from .fakes import FakeEngine, FakeGeminiClient


@pytest.fixture(autouse=True)
def rate_limits():
    """Every test starts with an empty rate-limit window.

    The limiters are module-level by design — the window has to outlive a single
    request — and `httpx.ASGITransport` gives every test the same client address,
    so without this one test's chat requests count against the next one's. That
    failure would be a suite-order-dependent 429 in a test about something else.
    """
    chat_limiter.reset()
    mine_limiter.reset()
    yield
    chat_limiter.reset()
    mine_limiter.reset()


@pytest.fixture()
def conn(tmp_path):
    """A fresh database per test — never the developer's ./pixel.db."""
    connection = db.init(str(tmp_path / "pixel.db"))
    yield connection
    # Before the close, not after: `/api/chat` hands the automatic mining run to
    # a background thread, and a run still in flight reads this very connection.
    stop_worker()
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


@pytest.fixture()
def generator():
    """Install the real miner generator behind a scripted Gemini client.

    The real one, not a stand-in for it: the draft schema and every validation
    layer between Gemini's JSON and a `Skill` are exactly what stage 4 has to get
    right, and a hand-rolled fake generator would skip all of them.
    """

    def install(*script, grouping=None) -> FakeGeminiClient:
        fake = FakeGeminiClient(*script, grouping=grouping)
        set_generator(GeminiSkillGenerator(client=fake, model="fake-model"))
        return fake

    yield install
    set_generator(None)
