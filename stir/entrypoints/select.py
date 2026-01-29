from __future__ import annotations

import logging

from stir.config import STIRConfig

logger = logging.getLogger(__name__)


def run_select(cfg: STIRConfig) -> None:
    from stir.offline.stage2 import select_library

    logger.info("Stage II selection: start")
    select_library(cfg)
    logger.info("Stage II selection: done")

