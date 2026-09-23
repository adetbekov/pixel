"""Scaffold smoke test — keeps the suite non-empty until stage 1 lands real tests."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_example_lists_required_keys():
    keys = {
        line.split("=", 1)[0]
        for line in (REPO_ROOT / ".env.example").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {"GEMINI_API_KEY", "PIXEL_DB_PATH", "LAYA_MODEL", "ROUTER_THRESHOLD"} <= keys
