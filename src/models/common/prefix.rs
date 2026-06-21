use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Parent hash for the first block in a sequence; chains every subsequent block hash.
pub const PREFIX_HASH_SEED: u64 = 0;

/// Chained content hash: folds the parent block's hash together with this block's token ids,
/// so the result identifies a full prefix path (positions `0..end`), not just one block's
/// contents. Two sequences share a block hash iff every token from position 0 matches.
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    parent.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

/// Maps a full block's chained hash to the physical block holding its KV. The block's token
/// ids are retained so lookups can verify a match and reject hash collisions.
#[derive(Debug, Default)]
pub struct PrefixRegistry {
    entries: HashMap<u64, (Vec<u32>, usize)>,
}

impl PrefixRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Physical block id stored under `hash`, but only when the stored tokens equal `tokens`
    /// (guards against hash collisions returning the wrong KV).
    pub fn get(&self, hash: u64, tokens: &[u32]) -> Option<usize> {
        self.entries
            .get(&hash)
            .and_then(|(stored, id)| (stored.as_slice() == tokens).then_some(*id))
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn insert(&mut self, hash: u64, tokens: Vec<u32>, block_id: usize) {
        self.entries.insert(hash, (tokens, block_id));
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Empty the registry, returning the physical block ids it had been holding so the caller
    /// can release their registry references.
    pub fn drain_block_ids(&mut self) -> Vec<usize> {
        self.entries.drain().map(|(_, (_, id))| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_hash_is_deterministic_and_parent_sensitive() {
        let toks = [1u32, 2, 3];
        assert_eq!(
            block_hash(PREFIX_HASH_SEED, &toks),
            block_hash(PREFIX_HASH_SEED, &toks)
        );
        // Different parent => different child hash (chaining).
        assert_ne!(block_hash(PREFIX_HASH_SEED, &toks), block_hash(42, &toks));
        // Different tokens => different hash.
        assert_ne!(
            block_hash(PREFIX_HASH_SEED, &toks),
            block_hash(PREFIX_HASH_SEED, &[1, 2, 4])
        );
    }

    #[test]
    fn registry_get_verifies_tokens_against_collisions() {
        let mut reg = PrefixRegistry::new();
        let h = block_hash(PREFIX_HASH_SEED, &[1, 2]);
        reg.insert(h, vec![1, 2], 7);

        // Correct tokens at the stored hash -> hit.
        assert_eq!(reg.get(h, &[1, 2]), Some(7));
        assert!(reg.contains(h));
        // Same hash but mismatched tokens (simulated collision) -> miss, not a wrong reuse.
        assert_eq!(reg.get(h, &[9, 9]), None);
        // Unknown hash -> miss.
        assert_eq!(
            reg.get(block_hash(PREFIX_HASH_SEED, &[3, 4]), &[3, 4]),
            None
        );

        reg.clear();
        assert!(!reg.contains(h));
    }
}
