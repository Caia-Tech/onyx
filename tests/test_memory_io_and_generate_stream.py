import torch

from onyx_inference import generate_stream


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


def test_save_and_load_memory_states_roundtrip(tiny_model, tiny_config, tmp_path):
    B, S = 1, 8
    x = torch.randint(0, tiny_config.vocab_size, (B, S))
    mem = tiny_model.init_memory_states(B, device=x.device, dtype=torch.float32)

    out = tiny_model(input_ids=x, memory_states=mem, update_memories=True, inference_mode=True)
    mem1 = out["memory_states"]

    p = tmp_path / "mem.pt"
    tiny_model.save_memory_states(mem1, str(p))

    # Load into a fresh model instance.
    from onyx_model import Onyx

    m2 = Onyx(tiny_model.config).cpu().eval()
    loaded = m2.load_memory_states(str(p), device=torch.device("cpu"), dtype=torch.float32)
    assert _mem_allclose(mem1, loaded, atol=0.0, rtol=0.0)


def test_generate_stream_advances_memory_update_count(tiny_model, dummy_tokenizer, tiny_config):
    B, S = 1, 5
    x = torch.randint(0, tiny_config.vocab_size, (B, S))

    # update_memory=True should increment internal counter by S + max_new_tokens.
    _ = list(
        generate_stream(
            model=tiny_model,
            input_ids=x,
            tokenizer=dummy_tokenizer,
            max_new_tokens=3,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            update_memory=True,
            use_kv_cache=True,
            stop_on_eos=False,
        )
    )
    assert getattr(tiny_model, "_memory_update_count", 0) == S + 3


def test_generate_stream_does_not_advance_memory_when_disabled(tiny_model, dummy_tokenizer, tiny_config):
    B, S = 1, 5
    x = torch.randint(0, tiny_config.vocab_size, (B, S))

    _ = list(
        generate_stream(
            model=tiny_model,
            input_ids=x,
            tokenizer=dummy_tokenizer,
            max_new_tokens=3,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            update_memory=False,
            use_kv_cache=True,
            stop_on_eos=False,
        )
    )
    assert getattr(tiny_model, "_memory_update_count", 0) == 0
