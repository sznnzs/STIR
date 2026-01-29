from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskExample:
    """
    A unified representation for all tasks used in the paper.
    """

    task: str
    id: str
    question: str
    answer: str
    meta: dict[str, Any]


