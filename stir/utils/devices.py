from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_embed_device(embed_device: str, *, tensor_parallel_size: int = 1) -> str:
    """
    Resolve the device string for the HF embedder.

    Supported values:
    - "cuda", "cuda:0", "cuda:1", "cpu", ...
    - "auto" / "cuda:auto": pick "cuda" if available else "cpu".

    Note: this repo's default multi-GPU usage is "one process per GPU" (not a single
    process spanning multiple GPUs). So "auto" intentionally stays on the current
    visible CUDA device rather than opportunistically using another GPU.
    """
    d = str(embed_device or "").strip()
    if d == "":
        d = "cuda"

    key = d.lower()
    if key not in {"auto", "cuda:auto", "cuda_auto"}:
        return d

    try:
        import torch  # type: ignore

        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        resolved = "cpu"

    logger.info("Resolved embed_device=%r -> %s", embed_device, resolved)
    return resolved
