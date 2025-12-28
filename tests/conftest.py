import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onyx.model import Onyx, OnyxConfig


class DummyTokenizer:
    """Offline tokenizer stub for dataset tests."""

    def __init__(self, eos_token_id: int = 2, pad_token_id: int = 2):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return 64  # arbitrary small vocab for tests

    def encode(self, text: str, add_special_tokens: bool = False):
        _ = add_special_tokens
        n = max(1, min(len(text), 16))
        return list(range(10, 10 + n))


@pytest.fixture
def tiny_config():
    return OnyxConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,
        d_ff=128,
        vocab_size=101,
        max_seq_len=64,
        train_seq_len=64,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return Onyx(tiny_config).cpu().eval()


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


@pytest.fixture
def tiny_jsonl(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    rows = [
        {"text": "User: 2+2?\nAssistant: 4"},
        {"text": "User: hello\nAssistant: hi"},
        {"text": "User: python list\nAssistant: [1,2,3]"},
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)
