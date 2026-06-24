use candle_core::{DType, Result, Tensor, D};

/// Expand `(b, n_kv_heads, seq, head_dim)` to `(b, n_kv_heads * n_rep, seq, head_dim)`
/// by repeating each KV head `n_rep` times, for grouped-query attention.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(mask.shape().dims())?;
    mask.where_cond(&on_true, on_false)
}

/// Scaled dot-product attention with an optional additive causal mask.
/// `q`/`k`/`v`: `(b, heads, seq, head_dim)`. `mask`: `(seq_q, seq_kv)`, nonzero = masked out.
pub fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let att = match mask {
        Some(mask) => {
            let mask = mask.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        }
        None => att,
    };
    let att =
        candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(att.dtype())?;
    att.matmul(&v.contiguous()?)
}

/// PagedAttention kernel.
///
/// Computes causal scaled-dot-product attention by streaming over the paged KV blocks
/// instead of materializing the whole key/value sequence at once. Each `blocks` entry is
/// one physical KV block gathered from the pool, shaped `(1, kv_heads, block_n, head_dim)`
/// for K and V respectively, in logical (ascending kv-position) order. Partial results are
/// folded together with a flash-attention-style online softmax, so only a single block of
/// scores is ever resident.
///
/// `q`: `(b, num_heads, seq_q, head_dim)` (already RoPE-encoded). `n_rep` repeats each KV
/// head to match the query head count for grouped-query attention. Causal masking places the
/// `seq_q` query positions at the tail of the KV sequence: query `i` attends kv position `j`
/// iff `j <= (kv_len - seq_q) + i`.
pub fn paged_attention(
    q: &Tensor,
    blocks: &[(Tensor, Tensor)],
    n_rep: usize,
    scale: f64,
) -> Result<Tensor> {
    let (b_sz, num_heads, seq_q, head_dim) = q.dims4()?;
    let device = q.device();
    let kv_len: usize = blocks.iter().map(|(k, _)| k.dims()[2]).sum();
    // New query positions occupy the trailing `seq_q` kv columns; everything earlier is
    // already-cached context that every query can see.
    let offset = kv_len - seq_q;

    // Running flash-attention state, one scalar per (batch, head, query) row.
    let mut running_max = Tensor::full(f32::NEG_INFINITY, (b_sz, num_heads, seq_q, 1), device)?;
    let mut running_sum = Tensor::zeros((b_sz, num_heads, seq_q, 1), DType::F32, device)?;
    let mut acc = Tensor::zeros((b_sz, num_heads, seq_q, head_dim), DType::F32, device)?;

    let mut kv_start = 0usize;
    for (k, v) in blocks {
        let block_n = k.dims()[2];
        let k = repeat_kv(k.clone(), n_rep)?;
        let v = repeat_kv(v.clone(), n_rep)?;

        // (b, heads, seq_q, block_n). The q@k^T matmul runs in the model dtype (fast for
        // bf16/f16), but the online-softmax accumulation below is done in f32 for numerical
        // stability, so cast the scores up front.
        let scores =
            (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?.to_dtype(DType::F32)?;

        // Per-block causal mask: kv position `kv_start + j` is in the future of query `i`
        // (global position `offset + i`) when `kv_start + j > offset + i`.
        let mask: Vec<u8> = (0..seq_q)
            .flat_map(|i| (0..block_n).map(move |j| u8::from(kv_start + j > offset + i)))
            .collect();
        let mask =
            Tensor::from_slice(&mask, (seq_q, block_n), device)?.broadcast_as(scores.shape())?;
        let scores = masked_fill(&scores, &mask, f32::NEG_INFINITY)?;

        let block_max = scores.max_keepdim(D::Minus1)?; // (b, heads, seq_q, 1)
        let new_max = running_max.maximum(&block_max)?;
        // Rescale the prior accumulation onto the new running maximum.
        let correction = running_max.broadcast_sub(&new_max)?.exp()?; // (b, heads, seq_q, 1)
        let probs = scores.broadcast_sub(&new_max)?.exp()?; // (b, heads, seq_q, block_n)

        running_sum = (running_sum.broadcast_mul(&correction)? + probs.sum_keepdim(D::Minus1)?)?;
        acc = (acc.broadcast_mul(&correction)?
            + probs.matmul(&v.contiguous()?.to_dtype(DType::F32)?)?)?;
        running_max = new_max;
        kv_start += block_n;
    }

    // Accumulation happened in f32; return in the query's (model) dtype so downstream layers
    // see a consistent dtype.
    acc.broadcast_div(&running_sum)?.to_dtype(q.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn repeat_kv_is_noop_for_n_rep_1() {
        let x = Tensor::arange(0f32, 24f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3, 4))
            .unwrap();
        let y = repeat_kv(x.clone(), 1).unwrap();
        assert_eq!(y.dims(), x.dims());
    }

    #[test]
    fn repeat_kv_expands_heads() {
        let x = Tensor::arange(0f32, 24f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3, 4))
            .unwrap();
        let y = repeat_kv(x, 3).unwrap();
        assert_eq!(y.dims(), &[1, 6, 3, 4]);
    }

    fn rand_tensor(shape: &[usize], seed: u64) -> Tensor {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let numel: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// Dense reference: full causal attention over a contiguous KV tensor.
    fn dense_reference(q: &Tensor, k: &Tensor, v: &Tensor, n_rep: usize, scale: f64) -> Vec<f32> {
        let seq_q = q.dims()[2];
        let kv_len = k.dims()[2];
        let offset = kv_len - seq_q;
        let mask: Vec<u8> = (0..seq_q)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > i + offset)))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_q, kv_len), &Device::Cpu).unwrap();
        let k = repeat_kv(k.clone(), n_rep).unwrap();
        let v = repeat_kv(v.clone(), n_rep).unwrap();
        causal_attention(q, &k, &v, Some(&mask), scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// Split a `(1, kv_heads, kv_len, head_dim)` tensor into `block_size`-wide blocks along seq.
    fn into_blocks(k: &Tensor, v: &Tensor, block_size: usize) -> Vec<(Tensor, Tensor)> {
        let kv_len = k.dims()[2];
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < kv_len {
            let n = block_size.min(kv_len - start);
            blocks.push((
                k.narrow(2, start, n).unwrap().contiguous().unwrap(),
                v.narrow(2, start, n).unwrap().contiguous().unwrap(),
            ));
            start += n;
        }
        blocks
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn paged_attention_matches_dense_prefill() {
        let scale = 1.0 / (4f64).sqrt();
        let q = rand_tensor(&[1, 2, 5, 4], 1); // 2 heads, seq 5, head_dim 4
        let k = rand_tensor(&[1, 2, 5, 4], 2);
        let v = rand_tensor(&[1, 2, 5, 4], 3);
        let reference = dense_reference(&q, &k, &v, 1, scale);
        // Blocks of 2 -> last block partially filled (5 = 2 + 2 + 1).
        let blocks = into_blocks(&k, &v, 2);
        let out = paged_attention(&q, &blocks, 1, scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(&out, &reference);
    }

    #[test]
    fn paged_attention_matches_dense_decode_step() {
        let scale = 1.0 / (4f64).sqrt();
        // Single query attending to 7 cached kv positions (decode step).
        let q = rand_tensor(&[1, 2, 1, 4], 10);
        let k = rand_tensor(&[1, 2, 7, 4], 11);
        let v = rand_tensor(&[1, 2, 7, 4], 12);
        let reference = dense_reference(&q, &k, &v, 1, scale);
        let blocks = into_blocks(&k, &v, 3);
        let out = paged_attention(&q, &blocks, 1, scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(&out, &reference);
    }

    #[test]
    fn paged_attention_matches_dense_with_gqa() {
        let scale = 1.0 / (4f64).sqrt();
        // 4 query heads, 2 kv heads -> n_rep = 2.
        let q = rand_tensor(&[1, 4, 6, 4], 20);
        let k = rand_tensor(&[1, 2, 6, 4], 21);
        let v = rand_tensor(&[1, 2, 6, 4], 22);
        let reference = dense_reference(&q, &k, &v, 2, scale);
        let blocks = into_blocks(&k, &v, 4);
        let out = paged_attention(&q, &blocks, 2, scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(&out, &reference);
    }

    #[test]
    fn paged_attention_bf16_matches_f32_reference() {
        let scale = 1.0 / (4f64).sqrt();
        let q = rand_tensor(&[1, 2, 5, 4], 1);
        let k = rand_tensor(&[1, 2, 5, 4], 2);
        let v = rand_tensor(&[1, 2, 5, 4], 3);
        // f32 dense result is the ground truth.
        let reference = dense_reference(&q, &k, &v, 1, scale);
        // Run the kernel in bf16 and confirm the output stays close after the f32 accumulation.
        let blocks: Vec<(Tensor, Tensor)> = into_blocks(&k, &v, 2)
            .into_iter()
            .map(|(k, v)| {
                (
                    k.to_dtype(DType::BF16).unwrap(),
                    v.to_dtype(DType::BF16).unwrap(),
                )
            })
            .collect();
        let q_bf16 = q.to_dtype(DType::BF16).unwrap();
        let out = paged_attention(&q_bf16, &blocks, 1, scale).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        let out: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // bf16 has ~3 decimal digits of precision, so allow a loose tolerance.
        for (x, y) in out.iter().zip(reference.iter()) {
            assert!((x - y).abs() < 5e-2, "mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn causal_attention_with_identity_value_returns_weighted_average() {
        // 1 batch, 1 head, 2 query positions, head_dim=1, attending to themselves only (causal).
        let q = Tensor::from_vec(vec![1f32, 1f32], (1, 1, 2, 1), &Device::Cpu).unwrap();
        let k = Tensor::from_vec(vec![1f32, 1f32], (1, 1, 2, 1), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(vec![10f32, 20f32], (1, 1, 2, 1), &Device::Cpu).unwrap();
        // mask[i][j] = 1 (masked) when j > i
        let mask = Tensor::from_vec(vec![0u8, 1u8, 0u8, 0u8], (2, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        let out = causal_attention(&q, &k, &v, Some(&mask), 1.0).unwrap();
        let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // position 0 can only see itself -> exactly 10.0
        assert!((out[0] - 10.0).abs() < 1e-4);
        // position 1 sees both equally-scored positions -> average of 10 and 20
        assert!((out[1] - 15.0).abs() < 1e-4);
    }
}
