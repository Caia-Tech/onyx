# Residual Matrix Transformer (RMT)

RMT changes the residual stream from vectors to matrices.

- Baseline residual: `x[b, t, d_model]`
- RMT residual: `X[b, t, Dk, Dv]`

Enable with:

```json
{
  "use_rmt": true,
  "rmt_dk": 16,
  "rmt_dv": 16,
  "rmt_heads": 2,
  "rmt_ff_multiplier": 4.0,
  "rmt_init": "normal",
  "rmt_readwrite_dtype": "float32"
}
```

## Read/Write equations

- Read per-head vectors:
  - `Q = einsum("hk,btkv->bthv", r_q, X)`
  - `K = einsum("hk,btkv->bthv", r_k, X)`
  - `V = einsum("hk,btkv->bthv", r_v, X)`
- Write attention output:
  - `X <- X + einsum("hk,bthv->btkv", w_o, O)`
- FFN read/write:
  - Read `R` vectors with `r_ff`, flatten to `R*Dv`, run FF core, unflatten, write with `w_ff`.

## Norm handling

RMT blocks use a token-view readout `view[b,t,r,dv]` derived from fixed read keys.
RMSNorm is applied on flattened `view[b,t,r*dv]`. The normalized vectors are written
back into `X` with learned write keys before attention and before FFN.

## HoPE and memory updates

When `use_hope_attention=true`, RMT attention reuses rotary positional embedding on
`Q/K` and can route `K/V` through existing self-referential memory modules. Memory
states remain FP32 in the memory modules as in the baseline path.

## mHC + Sinkhorn

For `experimental_mhc`, RMT uses per-lane matrix streams:

- `X_streams[b,t,n,dk,dv]`
- mix only across lane axis `n`:
  - `X_new[..., m, :, :] = sum_n P[n,m] * X[..., n, :, :]`

`P` is produced by the existing mHC mixer. Sinkhorn projection runs in FP32
(`sinkhorn_project` uses float32 normalization regardless of model dtype).
