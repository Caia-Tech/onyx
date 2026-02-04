import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(()))

    def init_memory_states(self, batch_size, device, dtype):
        # memory is a single scalar counter
        return torch.zeros((batch_size,), device=device, dtype=torch.float32)

    def detach_memory_states(self, ms):
        return ms.detach()

    def forward(self, input_ids, labels=None, memory_states=None, update_memories=True, **kwargs):
        if memory_states is None:
            memory_states = self.init_memory_states(input_ids.size(0), input_ids.device, torch.float32)
        if update_memories:
            memory_states = memory_states + 1.0
        logits = torch.zeros((input_ids.size(0), input_ids.size(1), 10), device=input_ids.device)
        loss = torch.tensor(0.0, device=input_ids.device)
        return {"logits": logits, "loss": loss, "memory_states": memory_states}


def test_memory_persists_across_steps_when_enabled():
    device = torch.device("cpu")
    model = DummyModel().to(device)

    B, S = 2, 4
    input_ids = torch.zeros((B, S), dtype=torch.long, device=device)

    ms = model.init_memory_states(B, device, torch.float32)
    out1 = model(input_ids, memory_states=ms)
    ms1 = out1["memory_states"]
    out2 = model(input_ids, memory_states=ms1)
    ms2 = out2["memory_states"]

    # counter increments across steps if passed through
    assert torch.allclose(ms2, torch.full((B,), 2.0))


def test_memory_resets_when_not_carried():
    device = torch.device("cpu")
    model = DummyModel().to(device)

    B, S = 2, 4
    input_ids = torch.zeros((B, S), dtype=torch.long, device=device)

    out1 = model(input_ids, memory_states=None)
    out2 = model(input_ids, memory_states=None)

    assert torch.allclose(out1["memory_states"], torch.full((B,), 1.0))
    assert torch.allclose(out2["memory_states"], torch.full((B,), 1.0))
