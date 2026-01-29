from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def _as_torch_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16"}:
        return torch.float16
    if d in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _extract_hidden_from_layer_output(out):
    # HF decoder layers often return tuple(hidden_states, *extras)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    if hasattr(out, "hidden_states") and isinstance(out.hidden_states, torch.Tensor):
        return out.hidden_states
    raise TypeError(f"Unsupported layer output type: {type(out)}")


@dataclass
class HFLastTokenStateExtractor:
    """
    Efficiently capture last-token residual states from selected decoder layers
    using forward hooks (avoids returning all hidden states).

    This is used in offline mining stages to compute diffmean directions.
    """

    model_name_or_path: str
    dtype: str = "bfloat16"
    device: str = "cuda"

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        torch_dtype = _as_torch_dtype(self.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            # For many decoder-only models, pad defaults to EOS.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map={"": self.device},
        )
        self.model.eval()

        # Best-effort: locate decoder layers.
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.layers = self.model.transformer.h
        else:  # pragma: no cover
            raise RuntimeError("Unable to locate decoder layers (expected model.model.layers or model.transformer.h).")

        self.num_layers = len(self.layers)
        self.hidden_size = int(self.model.config.hidden_size)
        logger.info(
            "HF extractor ready. layers=%d hidden=%d model=%s",
            self.num_layers,
            self.hidden_size,
            self.model_name_or_path,
        )

    def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # fallback: minimal concatenation
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}:\n{m['content']}\n")
        parts.append("ASSISTANT:\n")
        return "\n".join(parts)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> tuple[list[list[int]], list[str], list[int]]:
        """
        Generate multiple sampled continuations from a single prompt.

        Returns:
          - gen_token_ids_list: list[K][T_gen]
          - gen_text_list: list[K]
          - prompt_token_ids: list[int]
        """
        if seed is not None:
            torch.manual_seed(int(seed))

        tok = self.tokenizer
        model = self.model
        device = self.device

        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = int(enc["input_ids"].shape[1])
        prompt_token_ids = enc["input_ids"][0].tolist()

        out = model.generate(
            **enc,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_beams=1,
            num_return_sequences=int(num_return_sequences),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=int(tok.pad_token_id),
            eos_token_id=int(tok.eos_token_id) if tok.eos_token_id is not None else None,
            use_cache=True,
        )
        # out: (K, prompt_len + gen_len)
        gen_token_ids_list: list[list[int]] = []
        gen_text_list: list[str] = []
        for seq in out:
            gen_ids = seq[prompt_len:].tolist()
            gen_token_ids_list.append(gen_ids)
            gen_text_list.append(tok.decode(gen_ids, skip_special_tokens=True))
        return gen_token_ids_list, gen_text_list, prompt_token_ids

    def select_top_third_layers(self) -> list[int]:
        start = (2 * self.num_layers) // 3
        return list(range(start, self.num_layers))

    @torch.no_grad()
    def last_token_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_layers: list[int],
    ) -> torch.Tensor:
        """
        Capture last-token states for each target layer.

        Args:
          input_ids: (B, T)
          attention_mask: (B, T)
          target_layers: list of layer indices (0-based)

        Returns:
          states: (B, L, H) where L=len(target_layers)
        """
        # Storage dict: layer -> (B, H)
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_id: int) -> Callable:
            def _hook(_module, _inp, out):
                hs = _extract_hidden_from_layer_output(out)
                # hs: (B, T, H)
                captured[layer_id] = hs[:, -1, :].detach()
                return out

            return _hook

        for lid in target_layers:
            hooks.append(self.layers[lid].register_forward_hook(make_hook(lid)))

        try:
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # Stack in the same order as target_layers.
        out_states = [captured[lid] for lid in target_layers]
        return torch.stack(out_states, dim=1)

    def batch_to_tensors(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a batch of token-id sequences and move to device.
        """
        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(x) for x in sequences)
        bsz = len(sequences)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attn[i, :n] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        return input_ids, attn

