import builtins
import json
import sys
from pathlib import Path

import torch

import onyx_inference


class TinyTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return 32

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        _ = (add_special_tokens,)
        toks = [3] * max(1, min(len(text), 8))
        if return_tensors == "pt":
            return torch.tensor([toks], dtype=torch.long)
        return toks

    def decode(self, token_ids, skip_special_tokens=True):
        _ = skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(97 + (int(t) % 3)) for t in token_ids)


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.saved = 0
        self.loaded = 0

    def get_num_params(self, non_embedding: bool = False):
        _ = non_embedding
        return 1234

    def init_memory_states(self, batch_size, device, dtype):
        return [{"count": torch.zeros((batch_size,), device=device, dtype=dtype)}]

    def save_memory_states(self, memory_states, path: str):
        torch.save({"memory_states": memory_states}, path)
        self.saved += 1

    def load_memory_states(self, path: str, device, dtype=None):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.loaded += 1
        out = []
        for layer in ck["memory_states"]:
            out.append({k: v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v for k, v in layer.items()})
        return out

    def forward(
        self,
        input_ids,
        memory_states=None,
        update_memories=True,
        inference_mode=True,
        kv_cache=None,
        position_offset=0,
        **_kw,
    ):
        _ = (inference_mode, kv_cache, position_offset)
        B, S = input_ids.shape
        logits = torch.full((B, S, self.vocab_size), -5.0, dtype=torch.float32, device=input_ids.device)

        # Prefer token 4, but also include eos as a strong alternative to exercise skipping.
        logits[:, :, 4] = 5.0
        logits[:, :, 2] = 4.0

        if memory_states is None:
            memory_states = self.init_memory_states(B, input_ids.device, torch.float32)
        if update_memories:
            memory_states = [{"count": memory_states[0]["count"] + S}]
        return {"logits": logits, "memory_states": memory_states, "kv_cache": kv_cache}


def test_inference_main_prompt_path_offline(monkeypatch, tmp_path: Path):
    tok = TinyTokenizer()
    model = TinyModel(vocab_size=len(tok)).eval()

    monkeypatch.setattr(onyx_inference, "TRANSFORMERS_AVAILABLE", True, raising=False)

    class _AT:
        @staticmethod
        def from_pretrained(_name, trust_remote_code=True):
            _ = trust_remote_code
            return tok

    monkeypatch.setattr(onyx_inference, "AutoTokenizer", _AT, raising=False)

    def _load_model(_ckpt, _tokenizer, device, dtype, _model_config):
        _ = (_ckpt, _tokenizer)
        return model.to(device=device, dtype=dtype), None

    monkeypatch.setattr(onyx_inference, "load_model", _load_model, raising=True)

    ckpt = tmp_path / "dummy.pt"
    torch.save({"model_state_dict": {}}, ckpt)

    argv = [
        "onyx_inference.py",
        "--checkpoint",
        str(ckpt),
        "--tokenizer",
        "dummy",
        "--prompt",
        "hello",
        "--no_stream",
        "--max_tokens",
        "2",
        "--temperature",
        "0",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=True)
    onyx_inference.main()


def test_chat_commands_and_persistent_memory_offline(monkeypatch, tmp_path: Path):
    tok = TinyTokenizer()
    model = TinyModel(vocab_size=len(tok)).eval()

    mem_path = tmp_path / "mem.pt"
    model.save_memory_states(model.init_memory_states(1, torch.device("cpu"), torch.float32), str(mem_path))

    # Drive the chat loop: toggle learning/temp/system, do one turn, save, exit.
    inputs = iter(
        [
            "/learning on",
            "/temp 0.2",
            "/system be nice",
            "hi",
            "/save",
            "/exit",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    # Make generation deterministic and short, and include EOS so skip branch is hit.
    def _fake_generate_stream(**_kw):
        yield torch.tensor([[4]]), [{"count": torch.tensor([1.0])}]
        yield torch.tensor([[tok.eos_token_id]]), [{"count": torch.tensor([2.0])}]

    monkeypatch.setattr(onyx_inference, "generate_stream", _fake_generate_stream, raising=True)

    onyx_inference.chat(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        dtype=torch.float32,
        memory_mode="persistent",
        memory_path=str(mem_path),
        learning=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        min_p=0.0,
        repetition_penalty=1.0,
        max_tokens=3,
        stream=True,
        system_prompt=None,
    )

    assert model.loaded >= 1
    assert model.saved >= 2  # initial + /save or learning persistence


def test_load_model_model_config_path_branch_offline(tmp_path: Path):
    # Covers explicit model_config_path parsing + flattening.
    cfg = {
        "architecture": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 1,
            "n_kv_heads": 1,
            "head_dim": 32,
            "d_ff": 64,
            "vocab_size": 17,
            "max_seq_len": 16,
            "train_seq_len": 16,
        }
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Minimal checkpoint with matching tensors.
    import onyx_model

    m = onyx_model.Onyx(onyx_model.OnyxConfig(**cfg["architecture"]))
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": m.state_dict(), "config": {}}, ckpt_path)

    loaded, loaded_cfg = onyx_inference.load_model(
        str(ckpt_path),
        tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_config_path=str(cfg_path),
    )
    assert loaded_cfg.vocab_size == 17
    assert loaded.embed.weight.shape[0] == 17
