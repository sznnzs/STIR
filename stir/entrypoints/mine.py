from __future__ import annotations

import logging

from stir.config import STIRConfig

logger = logging.getLogger(__name__)


def run_mine(cfg: STIRConfig) -> None:
    from stir.offline.stage1 import mine_candidates

    logger.info("Stage I mining: start")
    mine_candidates(cfg)
    logger.info("Stage I mining: done")

