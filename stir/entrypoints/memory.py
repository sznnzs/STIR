from __future__ import annotations

import logging

from stir.config import STIRConfig

logger = logging.getLogger(__name__)


def run_memory(cfg: STIRConfig) -> None:
    from stir.offline.stage3 import build_memory

    logger.info("Stage III memory: start")
    build_memory(cfg)
    logger.info("Stage III memory: done")
