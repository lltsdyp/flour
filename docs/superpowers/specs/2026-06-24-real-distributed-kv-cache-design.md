# Real Distributed KV Cache Reuse — Design

Status: implemented (branch `feat/real-distributed-kv`)
Date: 2026-06-24

## Goal

Reuse **real** paged KV blocks across nodes. After node A prefills a prompt prefix, it exports
the actual K/V tensors as a self-describing `KvBundle`, stores the bytes through the existing
DistKV object path, and node B — sharing only DistKV, with an empty local `Cache` — imports the
bundle, runs prefill over only the unmatched suffix, and produces logits identical to a cold
full prefill.

Non-goals (this version): GPU RDMA, compression, cross-model migration, multi-replica
consistency. DistKV Master/Worker stay a byte-object store; Master keeps its dirty-read /
generation / two-phase PUT semantics unchanged.

## Key semantics: why a block-aligned prompt reuses N−1 blocks

`PrefixKeyBuilder::reusable_token_count(prompt_len)` is the **largest multiple of `BLOCK_SIZE`
strictly less than `prompt_len`**:

```
prompt_len = 15, BLOCK_SIZE = 16 -> 0
prompt_len = 16, BLOCK_SIZE = 16 -> 0
prompt_len = 17, BLOCK_SIZE = 16 -> 16
prompt_len = 32, BLOCK_SIZE = 16 -> 16
prompt_len = 40, BLOCK_SIZE = 16 -> 32
```

KV cache only provides attention *history*; it cannot produce the logits for the last prompt
token. At least one suffix token must pass through `model.forward` to obtain the logits needed to
sample the first generated token. So a perfectly block-aligned prompt (e.g. 32 tokens) reuses
only its first block (16 tokens) and recomputes the rest as suffix.

## Bundle format (`src/kv_cache/bundle.rs`)

Binary frame, little-endian:

```
[magic: 4 bytes "FLKV"]
[version: u16]
[header_len: u32]
[header_json: UTF-8 JSON of KvBundleMeta]
[payload: raw tensor bytes]
```

`KvBundleMeta` carries `model_id`, `token_count`, `token_ids`, `block_size`, `num_layers`,
`num_kv_heads`, `head_dim`, `dtype`. Payload order, logical block by logical block, then layer by
layer: K bytes then V bytes, each shaped `(num_kv_heads, block_size, head_dim)`. First version
supports CPU Candle `f32`/`f16`/`bf16` raw little-endian bytes; an unsupported dtype is a safe
remote miss (`BundleDType::from_candle` returns `None`).

`KvBundleCodec::decode` rejects: bad magic, unsupported version, truncated header, non-block-
aligned `token_count`, `token_ids` length mismatch, and payload length not matching the declared
shape. `Cache::import_prefix_bundle` additionally rejects a model/cache dimension or dtype
mismatch and a prompt that does not start with the bundle's `token_ids`; `KvSession` rejects a
`model_id` mismatch before importing.

## Cache APIs

* `PagedKvPool::write_block(layer, block_id, k, v)` — overwrite one full physical block from
  `(1, kv_heads, block_size, head_dim)` (or the squeezed 3-D shape) via one `slice_set`.
* `Cache::export_prefix_bundle(model_id, token_ids, token_count)` — read real K/V for the first
  `token_count` (block-aligned, ≤ live len) tokens into a `KvBundle`.
* `Cache::import_prefix_bundle(bundle, prompt_tokens)` — reset the sequence, allocate one physical
  block per bundle block, `write_block` every layer, push into the `BlockTable`, advance the live
  length, and register each block in the `PrefixRegistry` with the same chained `block_hash` and
  extra refcount as local prefill. Rolls back all allocations on a mid-write failure.

## Model API

`CausalLM::prefill_suffix(input_ids, reused_prefix_tokens, cache)` runs `forward` over only
`input_ids[reused_prefix_tokens..]` at `index_pos = reused_prefix_tokens`. It does **not** reset
the sequence, match a local prefix, or register — the caller owns that lifecycle.
`prefill_cached` is now: reset → local `match_prefix` → `prefill_suffix` → `register_prefix`.

## Session flow (`KvSession::prefill`)

1. Reset the sequence; take the local `match_prefix`.
2. If the remote cache is enabled and a key exists, GET the object. The fetch determines
   `remote_cache_hit`.
3. Import only when the local match is shorter than the reusable prefix. If the local paged cache
   already covers the whole reusable prefix, the remote object still counts as a hit, but
   importing it would add nothing. (This keeps the same-engine "second request" hit metric while
   enabling true cross-node import.)
4. Run `prefill_suffix` over the unmatched suffix; `register_prefix`.

## Safety / fallback behavior

Correctness never depends on the remote cache:

* A failed GET, a corrupt/incompatible bundle, or a failed import is logged, counted as a
  **miss** (`remote_cache_hit = Some(false)`), recorded in `remote_cache_error`, and the request
  falls back to local/cold prefill (re-establishing the local match after import rollback).
* Publishing is best-effort: the bundle is exported + encoded inside `KvSession::finish` while the
  cache is still borrowed, then PUT after the engine releases its cache lock. A PUT failure is
  logged and ignored.
* Publish policy: at least two prompt blocks, a non-empty reusable prefix, and
  `reused_prefix_tokens < reusable_tokens` (i.e. this request produced reusable prefix blocks the
  remote did not already have). Overwrite is safe — Master generations protect stale reads.

## Observability

`GenerationStats` exposes `reused_prefix_tokens`, `remote_cache_hit`, `remote_key`,
`remote_cache_imported_tokens` (`Some(0)` on an enabled miss, `None` when disabled), and
`remote_cache_error`. The OpenAI response keeps the standard `usage` shape; remote-KV fields are
internal metrics (logged), not part of the public API envelope.
