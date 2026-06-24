#[derive(Debug, Clone)]
pub struct PublishPolicy {
    min_blocks: usize,
    block_size: usize,
}

impl PublishPolicy {
    pub fn new(min_blocks: usize, block_size: usize) -> Self {
        Self {
            min_blocks,
            block_size: block_size.max(1),
        }
    }

    /// Whether to publish this request's reusable prefix bundle. Requires a prompt of at least
    /// `min_blocks` blocks, a non-empty reusable prefix, and — crucially — that this request
    /// actually produced reusable prefix blocks the remote did not already have
    /// (`reused_prefix_tokens < reusable_tokens`). If the whole reusable prefix was already served
    /// from cache (local or remote), there is nothing new to publish.
    pub fn should_store(
        &self,
        prompt_tokens: usize,
        reusable_tokens: usize,
        reused_prefix_tokens: usize,
    ) -> bool {
        prompt_tokens >= self.min_blocks * self.block_size
            && reusable_tokens > 0
            && reused_prefix_tokens < reusable_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::key::BLOCK_SIZE;

    #[test]
    fn should_store_requires_two_blocks_and_new_reusable_prefix() {
        let policy = PublishPolicy::new(2, BLOCK_SIZE);
        // Too short: under two blocks.
        assert!(!policy.should_store(BLOCK_SIZE, 0, 0));
        assert!(!policy.should_store(2 * BLOCK_SIZE - 1, BLOCK_SIZE, 0));
        // Two-block prompt with a fresh, non-empty reusable prefix -> publish.
        assert!(policy.should_store(2 * BLOCK_SIZE, BLOCK_SIZE, 0));
        // No reusable prefix at all -> nothing to publish.
        assert!(!policy.should_store(2 * BLOCK_SIZE, 0, 0));
        // The whole reusable prefix was already cached -> nothing new to publish.
        assert!(!policy.should_store(3 * BLOCK_SIZE, 2 * BLOCK_SIZE, 2 * BLOCK_SIZE));
    }
}
