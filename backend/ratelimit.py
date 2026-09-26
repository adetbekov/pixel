"""A per-IP request ceiling, in the process that serves the request.

`https://pixel.yeldos.dev` answers without authentication (JEB-1623): the Nginx
Proxy Manager route carries no Access List, and the Access List is the owner's to
add. This module is the half we own, and it stays useful after the proxy is
fixed — `POST /api/chat` spends a 20-requests-a-day Gemini bucket and takes the
engine's single lock, so twenty requests from a stranger are enough to switch
learning off until the next morning.

Deliberately in memory and deliberately not Redis: there is exactly one worker —
`backend/db.py` is built on one connection behind a module lock — so a second
copy of this counter cannot exist. The window is a plain deque of arrival
instants per key; the cost is one `popleft` per expired request.

This is a brake on accidental hammering, not authentication. The key comes from
`X-Forwarded-For`, which the client controls; a real boundary is the Access List.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

#: The window every limit below is counted over. A minute is short enough that a
#: user who tripped the limit is let back in while still looking at the screen.
WINDOW_S = 60.0

#: `POST /api/chat` — a human types a handful of commands a minute; 20 leaves
#: room for a fast tester and still caps a scripted caller at a rate the daily
#: teacher budget survives.
DEFAULT_CHAT_RATE = 20

#: `POST /api/mine` — the most expensive endpoint there is (one grouping call
#: plus a draft per cluster) and the one a visitor never needs. Two a minute is
#: enough for a developer pressing the button.
DEFAULT_MINE_RATE = 2


def chat_rate() -> int:
    return int(os.environ.get("CHAT_RATE_LIMIT", DEFAULT_CHAT_RATE))


def mine_rate() -> int:
    return int(os.environ.get("MINE_RATE_LIMIT", DEFAULT_MINE_RATE))


def client_key(request) -> str:
    """Who is asking — the first hop of `X-Forwarded-For`, or the socket.

    Behind Nginx Proxy Manager the socket is always the proxy, so without the
    header every visitor shares one bucket and the first of them locks out the
    rest. The header may be a chain (`client, proxy1, proxy2`) and the client's
    own address is the first entry.

    Never trusted as identity — see the module docstring.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


class RateLimiter:
    """A sliding window per key. ``take`` is the whole surface."""

    #: How many keys the map may hold before `_sweep` starts collecting.
    _SWEEP_AT = 256

    def __init__(
        self,
        limit: Callable[[], int],
        window_s: float = WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # A callable rather than a number: the limit is read from the environment
        # per request, the same way `ROUTER_THRESHOLD` is, so a test can set it
        # without rebuilding the app.
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def take(self, key: str) -> int | None:
        """``None`` when the request may proceed, else seconds to wait.

        The returned number is what `Retry-After` carries: how long until the
        oldest hit in the window expires and a slot opens.
        """
        limit = self._limit()
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self._window_s
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if limit <= 0 or len(hits) >= limit:
                if not hits:
                    # A limit of 0 switches the endpoint off; there is no oldest
                    # hit to wait for, so quote the whole window.
                    return math.ceil(self._window_s)
                return max(1, math.ceil(hits[0] + self._window_s - now))
            hits.append(now)
            self._sweep(cutoff)
            return None

    def _sweep(self, cutoff: float) -> None:
        """Drop keys whose whole window has expired. Caller holds the lock.

        Without it the dict is a per-IP leak in a long-running process. Only run
        once the map is big enough to be worth walking — a handful of keys is the
        normal state and walking them on every request would be the bigger cost.
        """
        if len(self._hits) <= self._SWEEP_AT:
            return
        for key in [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]

    def reset(self) -> None:
        """Forget every window. Used between tests, never in the request path."""
        with self._lock:
            self._hits.clear()


#: The two live limiters. Module-level on purpose: the window has to outlive the
#: request, and there is one worker to hold it.
chat_limiter = RateLimiter(chat_rate)
mine_limiter = RateLimiter(mine_rate)
