import torch


def _clone_mem(mem):
    cloned = []
    for layer in mem:
        layer_c = {}
        for k, v in layer.items():
            if isinstance(v, dict):
                layer_c[k] = {kk: vv.clone() for kk, vv in v.items()}
            elif torch.is_tensor(v):
                layer_c[k] = v.clone()
            else:
                layer_c[k] = v
        cloned.append(layer_c)
    return cloned


def _mem_allclose(a, b, atol=0.0, rtol=0.0):
    for la, lb in zip(a, b):
        for k in la.keys():
            va, vb = la[k], lb[k]
            if isinstance(va, dict):
                for kk in va.keys():
                    if not torch.allclose(va[kk], vb[kk], atol=atol, rtol=rtol):
                        return False
            else:
                if not torch.allclose(va, vb, atol=atol, rtol=rtol):
                    return False
    return True


def test_memory_does_not_change_when_update_false(tiny_model, tiny_config):
    B, S = 2, 16
    x = torch.randint(0, tiny_config.vocab_size, (B, S))
    mem0 = tiny_model.init_memory_states(B, device=x.device, dtype=torch.float32)
    mem0_clone = _clone_mem(mem0)

    out = tiny_model(
        input_ids=x,
        memory_states=mem0,
        update_memories=False,
        inference_mode=True,
    )
    mem1 = out["memory_states"]

    assert _mem_allclose(mem0_clone, mem1, atol=0.0, rtol=0.0)


def test_memory_changes_when_update_true(tiny_model, tiny_config):
    B, S = 2, 16
    x = torch.randint(0, tiny_config.vocab_size, (B, S))
    mem0 = tiny_model.init_memory_states(B, device=x.device, dtype=torch.float32)
    mem0_clone = _clone_mem(mem0)

    out = tiny_model(
        input_ids=x,
        memory_states=mem0,
        update_memories=True,
        inference_mode=True,
    )
    mem1 = out["memory_states"]

    assert not _mem_allclose(mem0_clone, mem1, atol=0.0, rtol=0.0)
