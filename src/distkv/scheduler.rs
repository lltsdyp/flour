//! Cache-aware route selection helpers for the Engine/Scheduler.
//!
//! This module is pure metadata: it derives the stable object key under which a
//! prefix KV bundle is stored, and decides whether a freshly-computed bundle is
//! worth storing remotely at all. No I/O happens here.

/// KV-cache paging block size (mirrors `models::common::cache::BLOCK_SIZE`).
///
/// The store policy and the prefix key are both block-aligned so that the same
/// prefix always hashes to the same key regardless of how far decoding ran.
pub const BLOCK_SIZE: usize = 16;

pub struct CacheScheduler;

impl CacheScheduler {
    /// Stable object key for a prefix KV bundle.
    ///
    /// The key is `kv://v1/model/{model_id}/prefix/{hash}/tokens/{token_count}`
    /// where `hash` folds in the model id, the block size, and the
    /// block-aligned prefix token ids. Aligning to `block_size` keeps the key
    /// stable across requests that share a prefix but diverge mid-block, and
    /// matches the granularity at which KV blocks are actually reusable.
    pub fn prefix_key(model_id: &str, token_ids: &[u32], block_size: usize) -> String {
        let aligned = block_size.max(1);
        let token_count = (token_ids.len() / aligned) * aligned;
        let hash = Self::hash(model_id, &token_ids[..token_count], aligned);
        format!("kv://v1/model/{model_id}/prefix/{hash:016x}/tokens/{token_count}")
    }

    /// Whether a freshly-computed bundle is worth storing remotely.
    ///
    /// Policy (plan §Task 5): only store prompts of at least two blocks, and skip
    /// storing when the whole prefix was already reused (nothing new to publish).
    pub fn should_store(prompt_tokens: usize, reused_prefix_tokens: usize) -> bool {
        prompt_tokens >= 2 * BLOCK_SIZE && reused_prefix_tokens < prompt_tokens
    }

    /// Deterministic FNV-1a hash over the key-defining inputs.
    ///
    /// FNV-1a is used (rather than `DefaultHasher`) so the key is stable across
    /// processes and Rust versions, which matters for cross-node cache hits.
    fn hash(model_id: &str, token_ids: &[u32], block_size: usize) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(PRIME);
            }
        };
        feed(model_id.as_bytes());
        feed(&(block_size as u64).to_le_bytes());
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
        let tokens: Vec<u32> = (0..40).collect();
        let a = CacheScheduler::prefix_key("model-a", &tokens, BLOCK_SIZE);
        let b = CacheScheduler::prefix_key("model-a", &tokens, BLOCK_SIZE);
        assert_eq!(a, b);
        assert!(a.starts_with("kv://v1/model/model-a/prefix/"));
    }

    #[test]
    fn different_model_ids_produce_different_keys() {
        let tokens: Vec<u32> = (0..40).collect();
        let a = CacheScheduler::prefix_key("model-a", &tokens, BLOCK_SIZE);
        let b = CacheScheduler::prefix_key("model-b", &tokens, BLOCK_SIZE);
        assert_ne!(a, b);
    }

    #[test]
    fn different_tokens_produce_different_keys() {
        let t1: Vec<u32> = (0..40).collect();
        let t2: Vec<u32> = (0..40).map(|i| i + 1).collect();
        assert_ne!(
            CacheScheduler::prefix_key("m", &t1, BLOCK_SIZE),
            CacheScheduler::prefix_key("m", &t2, BLOCK_SIZE)
        );
    }

    #[test]
    fn key_is_block_aligned_and_ignores_trailing_partial_block() {
        // Two prefixes that agree on the first full block but differ only in a
        // trailing partial block must hash to the same key (block alignment).
        let mut a: Vec<u32> = (0..BLOCK_SIZE as u32).collect();
        let mut b = a.clone();
        a.extend([100, 101, 102]);
        b.extend([200, 201]);
        assert_eq!(
            CacheScheduler::prefix_key("m", &a, BLOCK_SIZE),
            CacheScheduler::prefix_key("m", &b, BLOCK_SIZE)
        );
    }

    #[test]
    fn should_store_requires_two_blocks_and_new_content() {
        // Below two blocks: never store.
        assert!(!CacheScheduler::should_store(BLOCK_SIZE, 0));
        assert!(!CacheScheduler::should_store(2 * BLOCK_SIZE - 1, 0));
        // At least two blocks with new content: store.
        assert!(CacheScheduler::should_store(2 * BLOCK_SIZE, 0));
        // Fully reused prefix: nothing new, skip.
        assert!(!CacheScheduler::should_store(
            2 * BLOCK_SIZE,
            2 * BLOCK_SIZE
        ));
    }
}
