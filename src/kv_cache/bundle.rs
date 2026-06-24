//! Self-describing, portable encoding of real paged KV blocks for cross-node reuse.
//!
//! A [`KvBundle`] carries the exact K/V tensors for a block-aligned prompt prefix plus the
//! metadata needed to validate it against the importing node's model and cache config. The wire
//! format is a small binary frame:
//!
//! ```text
//! [magic: 4 bytes "FLKV"]
//! [version: u16 little endian]
//! [header_len: u32 little endian]
//! [header_json: UTF-8 JSON of KvBundleMeta]
//! [payload: raw little-endian tensor bytes]
//! ```
//!
//! Payload order, logical block by logical block:
//!
//! ```text
//! for block in 0..num_blocks:
//!   for layer in 0..num_layers:
//!     K block bytes, shape (num_kv_heads, block_size, head_dim)
//!     V block bytes, shape (num_kv_heads, block_size, head_dim)
//! ```

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// Magic prefixing every encoded bundle.
const MAGIC: &[u8; 4] = b"FLKV";
/// Current wire-format version.
const VERSION: u16 = 1;

/// Element dtype of the stored K/V tensors. Only CPU Candle dtypes with a fixed little-endian
/// byte layout are representable; anything else is a safe remote miss (see [`BundleDType::from_candle`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BundleDType {
    F32,
    F16,
    BF16,
}

impl BundleDType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            BundleDType::F32 => 4,
            BundleDType::F16 | BundleDType::BF16 => 2,
        }
    }

    pub fn to_candle(self) -> DType {
        match self {
            BundleDType::F32 => DType::F32,
            BundleDType::F16 => DType::F16,
            BundleDType::BF16 => DType::BF16,
        }
    }

    /// `None` for dtypes this codec cannot serialize, so callers can treat them as a remote miss
    /// rather than failing.
    pub fn from_candle(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(BundleDType::F32),
            DType::F16 => Some(BundleDType::F16),
            DType::BF16 => Some(BundleDType::BF16),
            _ => None,
        }
    }
}

/// Everything an importer needs to validate a bundle against its own model/cache before trusting
/// the payload. Serialized as the JSON header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvBundleMeta {
    pub model_id: String,
    pub token_count: usize,
    pub token_ids: Vec<u32>,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: BundleDType,
}

/// One logical block's K/V for a single layer.
#[derive(Debug)]
pub struct KvLayerBlock {
    pub k: Tensor,
    pub v: Tensor,
}

/// One logical block across all layers, in layer order.
#[derive(Debug)]
pub struct KvBundleBlock {
    pub logical_block_idx: usize,
    pub layers: Vec<KvLayerBlock>,
}

/// A decoded/constructed bundle: metadata plus per-block, per-layer K/V tensors.
#[derive(Debug)]
pub struct KvBundle {
    pub meta: KvBundleMeta,
    pub blocks: Vec<KvBundleBlock>,
}

/// Stateless encoder/decoder for the binary bundle frame.
pub struct KvBundleCodec;

impl KvBundleCodec {
    pub fn encode(bundle: &KvBundle) -> anyhow::Result<Vec<u8>> {
        let meta = &bundle.meta;
        let num_blocks = meta.token_count / meta.block_size.max(1);
        if meta.block_size == 0 || meta.token_count % meta.block_size != 0 {
            anyhow::bail!("bundle token_count {} not block aligned", meta.token_count);
        }
        if bundle.blocks.len() != num_blocks {
            anyhow::bail!(
                "bundle has {} blocks, expected {num_blocks}",
                bundle.blocks.len()
            );
        }

        let header = serde_json::to_vec(meta)?;
        let mut out = Vec::with_capacity(4 + 2 + 4 + header.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);

        for block in &bundle.blocks {
            if block.layers.len() != meta.num_layers {
                anyhow::bail!(
                    "block {} has {} layers, expected {}",
                    block.logical_block_idx,
                    block.layers.len(),
                    meta.num_layers
                );
            }
            for layer in &block.layers {
                out.extend_from_slice(&tensor_to_le_bytes(&layer.k, meta.dtype)?);
                out.extend_from_slice(&tensor_to_le_bytes(&layer.v, meta.dtype)?);
            }
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> anyhow::Result<KvBundle> {
        const PREFIX: usize = 4 + 2 + 4;
        if bytes.len() < PREFIX {
            anyhow::bail!("bundle too short: {} bytes", bytes.len());
        }
        if &bytes[..4] != MAGIC {
            anyhow::bail!("bad bundle magic");
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            anyhow::bail!("unsupported bundle version {version}");
        }
        let header_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        let header_end = PREFIX + header_len;
        if bytes.len() < header_end {
            anyhow::bail!("bundle header truncated");
        }
        let meta: KvBundleMeta = serde_json::from_slice(&bytes[PREFIX..header_end])?;

        if meta.block_size == 0 {
            anyhow::bail!("bundle block_size is zero");
        }
        if meta.token_count == 0 || meta.token_count % meta.block_size != 0 {
            anyhow::bail!(
                "bundle token_count {} not block aligned to {}",
                meta.token_count,
                meta.block_size
            );
        }
        if meta.token_ids.len() != meta.token_count {
            anyhow::bail!(
                "bundle token_ids len {} != token_count {}",
                meta.token_ids.len(),
                meta.token_count
            );
        }

        let num_blocks = meta.token_count / meta.block_size;
        let elems_per_tensor = meta.num_kv_heads * meta.block_size * meta.head_dim;
        let bytes_per_tensor = elems_per_tensor * meta.dtype.size_in_bytes();
        let expected_payload = num_blocks * meta.num_layers * 2 * bytes_per_tensor;
        let payload = &bytes[header_end..];
        if payload.len() != expected_payload {
            anyhow::bail!(
                "bundle payload length {} != expected {expected_payload}",
                payload.len()
            );
        }

        let shape = (meta.num_kv_heads, meta.block_size, meta.head_dim);
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut off = 0usize;
        for b in 0..num_blocks {
            let mut layers = Vec::with_capacity(meta.num_layers);
            for _ in 0..meta.num_layers {
                let k = le_bytes_to_tensor(&payload[off..off + bytes_per_tensor], meta.dtype, shape)?;
                off += bytes_per_tensor;
                let v = le_bytes_to_tensor(&payload[off..off + bytes_per_tensor], meta.dtype, shape)?;
                off += bytes_per_tensor;
                layers.push(KvLayerBlock { k, v });
            }
            blocks.push(KvBundleBlock {
                logical_block_idx: b,
                layers,
            });
        }
        Ok(KvBundle { meta, blocks })
    }
}

/// Flatten a CPU tensor to raw little-endian bytes for the declared dtype.
fn tensor_to_le_bytes(t: &Tensor, dtype: BundleDType) -> anyhow::Result<Vec<u8>> {
    let t = t.flatten_all()?;
    let bytes = match dtype {
        BundleDType::F32 => t
            .to_vec1::<f32>()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        BundleDType::F16 => t
            .to_vec1::<half::f16>()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        BundleDType::BF16 => t
            .to_vec1::<half::bf16>()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    };
    Ok(bytes)
}

/// Rebuild a CPU tensor of `shape` from raw little-endian bytes.
fn le_bytes_to_tensor(
    bytes: &[u8],
    dtype: BundleDType,
    shape: (usize, usize, usize),
) -> anyhow::Result<Tensor> {
    let dev = Device::Cpu;
    let t = match dtype {
        BundleDType::F32 => {
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_vec(vals, shape, &dev)?
        }
        BundleDType::F16 => {
            let vals: Vec<half::f16> = bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                .collect();
            Tensor::from_vec(vals, shape, &dev)?
        }
        BundleDType::BF16 => {
            let vals: Vec<half::bf16> = bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect();
            Tensor::from_vec(vals, shape, &dev)?
        }
    };
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small but structurally valid bundle: 2 blocks, 2 layers, kv_heads 1, block_size 2,
    /// head_dim 2, deterministic values so round-trips are checkable.
    fn sample_bundle() -> KvBundle {
        let block_size = 2usize;
        let num_layers = 2usize;
        let kv_heads = 1usize;
        let head_dim = 2usize;
        let num_blocks = 2usize;
        let token_count = num_blocks * block_size;

        let mut blocks = Vec::new();
        for b in 0..num_blocks {
            let mut layers = Vec::new();
            for l in 0..num_layers {
                let base = (b * 100 + l * 10) as f32;
                let kvals: Vec<f32> = (0..(kv_heads * block_size * head_dim))
                    .map(|i| base + i as f32)
                    .collect();
                let vvals: Vec<f32> = kvals.iter().map(|x| x + 0.5).collect();
                let k = Tensor::from_vec(kvals, (kv_heads, block_size, head_dim), &Device::Cpu)
                    .unwrap();
                let v = Tensor::from_vec(vvals, (kv_heads, block_size, head_dim), &Device::Cpu)
                    .unwrap();
                layers.push(KvLayerBlock { k, v });
            }
            blocks.push(KvBundleBlock {
                logical_block_idx: b,
                layers,
            });
        }

        KvBundle {
            meta: KvBundleMeta {
                model_id: "model-a".into(),
                token_count,
                token_ids: (0..token_count as u32).collect(),
                block_size,
                num_layers,
                num_kv_heads: kv_heads,
                head_dim,
                dtype: BundleDType::F32,
            },
            blocks,
        }
    }

    fn tensor_vals(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn encode_decode_preserves_metadata_and_tensors() {
        let bundle = sample_bundle();
        let bytes = KvBundleCodec::encode(&bundle).unwrap();
        let decoded = KvBundleCodec::decode(&bytes).unwrap();

        assert_eq!(decoded.meta, bundle.meta);
        assert_eq!(decoded.blocks.len(), bundle.blocks.len());
        for (db, sb) in decoded.blocks.iter().zip(bundle.blocks.iter()) {
            assert_eq!(db.logical_block_idx, sb.logical_block_idx);
            for (dl, sl) in db.layers.iter().zip(sb.layers.iter()) {
                assert_eq!(tensor_vals(&dl.k), tensor_vals(&sl.k));
                assert_eq!(tensor_vals(&dl.v), tensor_vals(&sl.v));
            }
        }
    }

    #[test]
    fn decode_rejects_corrupt_magic() {
        let mut bytes = KvBundleCodec::encode(&sample_bundle()).unwrap();
        bytes[0] = b'X';
        assert!(KvBundleCodec::decode(&bytes).is_err());
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let mut bytes = KvBundleCodec::encode(&sample_bundle()).unwrap();
        // Bump the version field (bytes 4..6) to an unknown value.
        bytes[4] = 9;
        bytes[5] = 0;
        assert!(KvBundleCodec::decode(&bytes).is_err());
    }

    #[test]
    fn decode_rejects_wrong_payload_length() {
        let mut bytes = KvBundleCodec::encode(&sample_bundle()).unwrap();
        // Drop the last payload byte: payload length no longer matches the header's shape.
        bytes.pop();
        assert!(KvBundleCodec::decode(&bytes).is_err());
    }

    #[test]
    fn decode_rejects_truncated_header() {
        let bytes = KvBundleCodec::encode(&sample_bundle()).unwrap();
        // Keep only the prefix; the declared header_len now exceeds the buffer.
        let truncated = &bytes[..8];
        assert!(KvBundleCodec::decode(truncated).is_err());
    }

    #[test]
    fn f16_round_trips_exact_bits() {
        // f16 values chosen to be exactly representable so round-trip equality is meaningful.
        let k = Tensor::from_vec(
            vec![half::f16::from_f32(1.5), half::f16::from_f32(-2.0)],
            (1, 2, 1),
            &Device::Cpu,
        )
        .unwrap();
        let bundle = KvBundle {
            meta: KvBundleMeta {
                model_id: "m".into(),
                token_count: 2,
                token_ids: vec![0, 1],
                block_size: 2,
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 1,
                dtype: BundleDType::F16,
            },
            blocks: vec![KvBundleBlock {
                logical_block_idx: 0,
                layers: vec![KvLayerBlock {
                    k: k.clone(),
                    v: k.clone(),
                }],
            }],
        };
        let bytes = KvBundleCodec::encode(&bundle).unwrap();
        let decoded = KvBundleCodec::decode(&bytes).unwrap();
        let got: Vec<half::f16> = decoded.blocks[0].layers[0]
            .k
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got, vec![half::f16::from_f32(1.5), half::f16::from_f32(-2.0)]);
    }
}
