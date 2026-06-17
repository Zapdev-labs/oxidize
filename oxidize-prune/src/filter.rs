#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneFilter {
    keep_contains: Vec<String>,
    drop_contains: Vec<String>,
}

impl PruneFilter {
    pub fn new(keep_contains: Vec<String>, drop_contains: Vec<String>) -> Self {
        Self {
            keep_contains,
            drop_contains,
        }
    }

    pub fn keeps(&self, tensor_name: &str) -> bool {
        let passes_keep = self.keep_contains.is_empty()
            || self
                .keep_contains
                .iter()
                .any(|needle| tensor_name.contains(needle));
        let passes_drop = !self
            .drop_contains
            .iter()
            .any(|needle| tensor_name.contains(needle));
        passes_keep && passes_drop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_all_without_patterns() {
        let filter = PruneFilter::new(Vec::new(), Vec::new());
        assert!(filter.keeps("blk.0.attn_q.weight"));
    }

    #[test]
    fn keep_patterns_are_allow_listed_before_drop_patterns() {
        let filter = PruneFilter::new(vec!["blk.0".to_owned()], vec!["ffn".to_owned()]);
        assert!(filter.keeps("blk.0.attn_q.weight"));
        assert!(!filter.keeps("blk.1.attn_q.weight"));
        assert!(!filter.keeps("blk.0.ffn_gate.weight"));
    }
}
