/// KV-cache paging block size. Keep this aligned with the current paged cache block size.
pub const BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub struct PrefixKeyBuilder {
    model_id: String,
    block_size: usize,
}

impl PrefixKeyBuilder {
    pub fn new(model_id: impl Into<String>, block_size: usize) -> Self {
        Self {
            model_id: model_id.into(),
            block_size: block_size.max(1),
        }
    }

    pub fn aligned_token_count(&self, token_count: usize) -> usize {
        (token_count / self.block_size) * self.block_size
    }

    pub fn key_for_tokens(&self, token_ids: &[u32]) -> Option<String> {
        let token_count = self.aligned_token_count(token_ids.len());
        if token_count == 0 {
            return None;
        }
        let hash = self.hash(&token_ids[..token_count]);
        Some(format!(
            "kv://v1/model/{}/prefix/{hash:016x}/tokens/{token_count}",
            self.model_id
        ))
    }

    fn hash(&self, token_ids: &[u32]) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(PRIME);
            }
        };
        feed(self.model_id.as_bytes());
        feed(&(self.block_size as u64).to_le_bytes());
        for &t in token_ids {
            feed(&t.to_le_bytes());
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_key_is_deterministic() {
        let builder = PrefixKeyBuilder::new("model-a", BLOCK_SIZE);
        let tokens: Vec<u32> = (0..40).collect();
        let a = builder.key_for_tokens(&tokens).unwrap();
        let b = builder.key_for_tokens(&tokens).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("kv://v1/model/model-a/prefix/"));
        assert!(a.ends_with("/tokens/32"));
    }

    #[test]
    fn prefix_key_is_model_specific() {
        let tokens: Vec<u32> = (0..40).collect();
        let a = PrefixKeyBuilder::new("model-a", BLOCK_SIZE)
            .key_for_tokens(&tokens)
            .unwrap();
        let b = PrefixKeyBuilder::new("model-b", BLOCK_SIZE)
            .key_for_tokens(&tokens)
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn prefix_key_ignores_trailing_partial_block() {
        let builder = PrefixKeyBuilder::new("m", BLOCK_SIZE);
        let mut a: Vec<u32> = (0..BLOCK_SIZE as u32).collect();
        let mut b = a.clone();
        a.extend([100, 101, 102]);
        b.extend([200, 201]);
        assert_eq!(builder.key_for_tokens(&a), builder.key_for_tokens(&b));
    }

    #[test]
    fn prefix_key_returns_none_when_no_full_block_exists() {
        let builder = PrefixKeyBuilder::new("m", BLOCK_SIZE);
        let tokens: Vec<u32> = (0..(BLOCK_SIZE as u32 - 1)).collect();
        assert_eq!(builder.key_for_tokens(&tokens), None);
    }
}
