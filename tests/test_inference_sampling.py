import torch

from onyx.inference import sample_token


def test_sample_token_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 0.2, -0.5]], dtype=torch.float32)
    tok = sample_token(logits.clone(), temperature=0.0)
    assert tok.shape == (1, 1)
    assert int(tok.item()) == 1


def test_sample_token_top_k_one_is_deterministic():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 0.0, -1.0, -2.0]], dtype=torch.float32)
    tok = sample_token(logits.clone(), temperature=1.0, top_k=1, top_p=1.0)
    assert int(tok.item()) == 0


def test_sample_token_top_p_keeps_only_top_token_when_peaked():
    torch.manual_seed(0)
    logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    tok = sample_token(logits.clone(), temperature=1.0, top_k=0, top_p=0.5)
    assert int(tok.item()) == 0


def test_sample_token_min_p_filters_small_probs():
    torch.manual_seed(0)
    logits = torch.tensor([[12.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    tok = sample_token(logits.clone(), temperature=1.0, top_k=0, top_p=1.0, min_p=0.9)
    assert int(tok.item()) == 0


def test_sample_token_repetition_penalty_changes_argmax_under_temperature_zero():
    # With temperature=0, sample_token returns argmax after applying repetition penalty.
    # Make token 0 slightly better than token 1, then penalize token 0 to flip the argmax.
    logits = torch.tensor([[1.0, 0.99, -5.0]], dtype=torch.float32)
    generated = torch.tensor([0], dtype=torch.long)
    tok = sample_token(
        logits.clone(),
        temperature=0.0,
        repetition_penalty=2.0,
        generated_tokens=generated,
    )
    assert int(tok.item()) == 1


def test_sample_token_accepts_python_set_for_repetition_penalty():
    logits = torch.tensor([[1.0, 0.99, -5.0]], dtype=torch.float32)
    tok = sample_token(
        logits.clone(),
        temperature=0.0,
        repetition_penalty=2.0,
        generated_tokens={0},
    )
    assert int(tok.item()) == 1
