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

    pub fn should_store(&self, prompt_tokens: usize, reused_prefix_tokens: usize) -> bool {
        prompt_tokens >= self.min_blocks * self.block_size && reused_prefix_tokens < prompt_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::key::BLOCK_SIZE;

    #[test]
    fn should_store_requires_two_blocks_and_new_content() {
        let policy = PublishPolicy::new(2, BLOCK_SIZE);
        assert!(!policy.should_store(BLOCK_SIZE, 0));
        assert!(!policy.should_store(2 * BLOCK_SIZE - 1, 0));
        assert!(policy.should_store(2 * BLOCK_SIZE, 0));
        assert!(!policy.should_store(2 * BLOCK_SIZE, 2 * BLOCK_SIZE));
    }
}
