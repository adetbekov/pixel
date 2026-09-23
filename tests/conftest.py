import pytest

from backend import db


@pytest.fixture()
def conn(tmp_path):
    """A fresh database per test — never the developer's ./pixel.db."""
    connection = db.init(str(tmp_path / "pixel.db"))
    yield connection
    db.close()
