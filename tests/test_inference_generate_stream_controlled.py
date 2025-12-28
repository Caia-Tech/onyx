from dataclasses import dataclass

import torch
import torch.nn as nn

from onyx.inference import generate_stream


@dataclass
class _Out:
    logits: torch.Tensor
    memory_states: list
    kv_cache: object


class ControlledModel(nn.Module):
    """
    Minimal model stub to deterministically drive `generate_stream()` branches.
    """

    def __init__(self, vocab_size: int, preferred_token: int, eos_token: int, prefer_eos: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token
        self.eos_token = eos_token
        self.prefer_eos = prefer_eos
        self.p = nn.Parameter(torch.zeros(()))

    def init_memory_states(self, batch_size: int, device, dtype):
        return [{"count": torch.zeros((batch_size,), device=device, dtype=dtype)}]

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states=None,
        update_memories: bool = True,
        inference_mode: bool = True,
        kv_cache=None,
        position_offset: int = 0,
        **_kwargs,
    ):
        _ = (inference_mode, position_offset)
        B, S = input_ids.shape
        logits = torch.full((B, S, self.vocab_size), -10.0, dtype=torch.float32, device=input_ids.device)

        if self.prefer_eos:
            # Strongly prefer EOS; generate_stream may suppress it early.
            logits[:, :, self.eos_token] = 10.0
            logits[:, :, self.preferred_token] = 9.0
        else:
            # Prefer the non-eos token so stop_tokens behavior is exercised.
            logits[:, :, self.preferred_token] = 10.0
            logits[:, :, self.eos_token] = 9.0

        if memory_states is None:
            memory_states = self.init_memory_states(B, input_ids.device, torch.float32)

        if update_memories:
            memory_states = [{"count": memory_states[0]["count"] + S}]

        return {"logits": logits, "memory_states": memory_states, "kv_cache": kv_cache}


def test_generate_stream_min_tokens_before_eos_blocks_eos_then_allows():
    model = ControlledModel(vocab_size=8, preferred_token=3, eos_token=2, prefer_eos=True).eval()
    prompt = torch.tensor([[1, 1, 1]], dtype=torch.long)

    toks = []
    for tok, _mem in generate_stream(
        model=model,
        input_ids=prompt,
        tokenizer=None,
        max_new_tokens=3,
        temperature=0.0,  # deterministic argmax
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        update_memory=False,
        eos_token_id=2,
        stop_tokens=[2],
        min_tokens_before_eos=2,
        stop_on_eos=True,
        use_kv_cache=True,
    ):
        toks.append(int(tok.item()))

    # First two tokens must be the non-eos preferred token; third can be eos (and stop).
    assert toks[:2] == [3, 3]
    assert toks[-1] == 2


def test_generate_stream_stop_tokens_stops_after_yielding_stop_token():
    model = ControlledModel(vocab_size=8, preferred_token=5, eos_token=2, prefer_eos=False).eval()
    prompt = torch.tensor([[1, 1]], dtype=torch.long)

    out = list(
        generate_stream(
            model=model,
            input_ids=prompt,
            tokenizer=None,
            max_new_tokens=10,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            update_memory=False,
            eos_token_id=None,
            stop_tokens=[5],
            min_tokens_before_eos=0,
            stop_on_eos=False,
            use_kv_cache=False,
        )
    )
    assert len(out) == 1
    assert int(out[0][0].item()) == 5
