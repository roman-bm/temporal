"""Multi-model orchestration council.

One model you choose chairs a panel of frontier models from other providers.
The panel proposes, cross-examines, and negotiates over a shared claim ledger;
the chair synthesises the result into analysis plus an executable plan, with
unresolved disagreement reported rather than averaged away.
"""

from .consensus import Ledger
from .orchestrator import Council, CouncilConfig
from .registry import Registry
from .report import to_markdown

__version__ = "0.1.0"

__all__ = ["Council", "CouncilConfig", "Ledger", "Registry", "to_markdown"]
