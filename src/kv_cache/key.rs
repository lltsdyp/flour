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

    /// Number of leading prompt tokens whose KV can be reused: the largest multiple of the
    /// block size strictly less than `prompt_len`. Strictly-less guarantees at least one suffix
    /// token survives, since cached KV cannot produce the last prompt token's logits.
    ///
    /// ```text
    /// prompt_len = 15, BLOCK_SIZE = 16 -> 0
    /// prompt_len = 16, BLOCK_SIZE = 16 -> 0
    /// prompt_len = 17, BLOCK_SIZE = 16 -> 16
    /// prompt_len = 32, BLOCK_SIZE = 16 -> 16
    /// prompt_len = 40, BLOCK_SIZE = 16 -> 32
    /// ```
    pub fn reusable_token_count(&self, prompt_len: usize) -> usize {
        if prompt_len == 0 {
            return 0;
        }
        ((prompt_len - 1) / self.block_size) * self.block_size
    }

    /// Object key for the reusable block-aligned prefix of `token_ids`, paired with the reusable
    /// token count. `None` when the prompt has no full reusable block.
    pub fn key_for_reusable_prefix(&self, token_ids: &[u32]) -> Option<(String, usize)> {
        let token_count = self.reusable_token_count(token_ids.len());
        if token_count == 0 {
            return None;
        }
        let hash = self.hash(&token_ids[..token_count]);
        let key = format!(
            "kv://v1/model/{}/prefix/{hash:016x}/tokens/{token_count}",
            self.model_id
        );
        Some((key, token_count))
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
    fn reusable_token_count_is_largest_block_multiple_strictly_below_len() {
        let builder = PrefixKeyBuilder::new("m", BLOCK_SIZE);
        assert_eq!(builder.reusable_token_count(0), 0);
        assert_eq!(builder.reusable_token_count(15), 0);
        assert_eq!(builder.reusable_token_count(16), 0);
        assert_eq!(builder.reusable_token_count(17), 16);
        assert_eq!(builder.reusable_token_count(32), 16);
        assert_eq!(builder.reusable_token_count(40), 32);
    }

    #[test]
    fn key_for_reusable_prefix_returns_none_below_one_reusable_block() {
        let builder = PrefixKeyBuilder::new("m", BLOCK_SIZE);
        let toks: Vec<u32> = (0..15).collect();
        assert_eq!(builder.key_for_reusable_prefix(&toks), None);
        let aligned: Vec<u32> = (0..16).collect();
        // Exactly one full block: still no reuse, the final block must stay as suffix.
        assert_eq!(builder.key_for_reusable_prefix(&aligned), None);
    }

    #[test]
    fn key_for_reusable_prefix_counts_match_spec() {
        let builder = PrefixKeyBuilder::new("m", BLOCK_SIZE);
        let toks17: Vec<u32> = (0..17).collect();
        let (k17, n17) = builder.key_for_reusable_prefix(&toks17).unwrap();
        assert_eq!(n17, 16);
        assert!(k17.ends_with("/tokens/16"));

        let toks32: Vec<u32> = (0..32).collect();
        let (_k32, n32) = builder.key_for_reusable_prefix(&toks32).unwrap();
        assert_eq!(n32, 16);

        let toks40: Vec<u32> = (0..40).collect();
        let (k40, n40) = builder.key_for_reusable_prefix(&toks40).unwrap();
        assert_eq!(n40, 32);
        assert!(k40.ends_with("/tokens/32"));
    }

    #[test]
    fn prefix_key_is_deterministic() {
        let builder = PrefixKeyBuilder::new("model-a", BLOCK_SIZE);
        let tokens: Vec<u32> = (0..40).collect();
        let (a, _) = builder.key_for_reusable_prefix(&tokens).unwrap();
        let (b, _) = builder.key_for_reusable_prefix(&tokens).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("kv://v1/model/model-a/prefix/"));
        assert!(a.ends_with("/tokens/32"));
    }

    #[test]
    fn prefix_key_is_model_specific() {
        let tokens: Vec<u32> = (0..40).collect();
        let (a, _) = PrefixKeyBuilder::new("model-a", BLOCK_SIZE)
            .key_for_reusable_prefix(&tokens)
            .unwrap();
        let (b, _) = PrefixKeyBuilder::new("model-b", BLOCK_SIZE)
            .key_for_reusable_prefix(&tokens)
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn prefix_key_ignores_trailing_partial_block() {
        let builder = PrefixKeyBuilder::new("m", BLOCK_SIZE);
        // Both prompts share the same two full reusable blocks; the partial tail differs.
        let mut a: Vec<u32> = (0..(2 * BLOCK_SIZE) as u32).collect();
        let mut b = a.clone();
        a.extend([100, 101, 102]);
        b.extend([200, 201]);
        assert_eq!(
            builder.key_for_reusable_prefix(&a),
            builder.key_for_reusable_prefix(&b)
        );
    }
}
