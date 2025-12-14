✅ Tokenizer & Decoding Sanity TODO (Onyx / Llama‑3)
A. Verify tokenizer correctness (one‑time)

Confirm tokenizer = Llama‑3

Assert len(tokenizer) == 128_258

Verify config.vocab_size == 128_258

Confirm no vocab trimming or ID remapping in preprocessing

Confirm lm_head.weight is tied to embed.weight

➡️ If all true: no tokenizer mismatch

B. Fix generation termination (mandatory)

Always pass eos_token_id into generate()

eos_token_id = tokenizer.eos_token_id


Verify generation loop breaks on EOS

⚠️ Missing this = infinite drift that looks like tokenizer corruption

C. Fix temperature=0 behavior (critical)

Implement deterministic decoding path

if temperature == 0:
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
else:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)


⚠️ Sampling at temperature=0 will destroy small models

D. Align Llama‑3 special tokens end‑to‑end

Ensure training and inference handle these identically:

<|begin_of_text|>

<|end_of_text|>

<|eot_id|>

Specifically:

No stripping of control tokens

No remapping of reserved IDs

BOS/EOS insertion logic matches inference

Instruction / chat formatting is consistent

⚠️ Misalignment here causes semantic nonsense, not crashes

E. Isolate tokenizer correctness (debug mode)

For sanity checks, run inference with:

memory_mode="stateless"

update_memory=False

temperature=0

top_p=None

top_k=None

➡️ This removes Hope memory + sampling noise
➡️ Confirms tokens → embeddings → logits are correct

F. Understand Hope memory side‑effects (not a bug)

Acknowledge outputs can differ for identical prompts

Accept decoding path is stateful

Do not debug tokenizer with memory enabled

This is attention state mutation, not nondeterminism

G. Optional hard checks (recommended)

Tokenize → detokenize round‑trip test

Single‑token prompt test ("2", "hello")

Fixed seed + deterministic decode comparison

Verify EOS appears in logits distribution

Final verdict checklist

Tokenizer mismatch ❌

Generation semantics issues ✅

Fixable without retraining ✅