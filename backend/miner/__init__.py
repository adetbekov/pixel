"""The skill miner: the teacher's misses turned into a skill Laya can route.

This is where Pixel actually learns. Gemini answering a command is not learning
— it costs money every single time. Learning is the moment a *pattern* in those
answers becomes a skill in the library, and the same command stops reaching
Gemini at all.

The pipeline, once per run:

``teacher_log (mined=0)`` -> cluster -> generate -> backtest -> proposal

and it stops there. **A mined skill is never activated by this package.** Only
``POST /api/proposals/{id}/accept`` puts one in the library — that is the
project's standing invariant, not a policy this module may relax.
"""

from .generate import build_generator, get_generator, set_generator
from .run import MineResult, mine_once, mining_due

__all__ = [
    "MineResult",
    "build_generator",
    "get_generator",
    "mine_once",
    "mining_due",
    "set_generator",
]
