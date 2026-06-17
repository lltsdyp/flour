use candle_core::{DType, Result, Tensor};

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
