use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    Attention,
    MlpExpert,
    Other,
}

#[derive(Debug, Clone)]
pub struct MergeRecipe {
    pub attention_t: f32,
    pub mlp_t: f32,
    pub other_t: f32,
    pub default_t: Option<f32>,
}

impl MergeRecipe {
    pub fn kimi_k275() -> Self {
        Self {
            attention_t: 0.3,
            mlp_t: 0.5,
            other_t: 0.4,
            default_t: None,
        }
    }

    pub fn uniform(t: f32) -> Self {
        Self {
            attention_t: t,
            mlp_t: t,
            other_t: t,
            default_t: Some(t),
        }
    }

    pub fn t_for_tensor(&self, name: &str) -> f32 {
        if let Some(t) = self.default_t {
            return t;
        }
        match classify_tensor(name) {
            TensorCategory::Attention => self.attention_t,
            TensorCategory::MlpExpert => self.mlp_t,
            TensorCategory::Other => self.other_t,
        }
    }
}

pub fn classify_tensor(name: &str) -> TensorCategory {
    let lower = name.to_ascii_lowercase();
    if lower.contains("self_attn")
        || lower.contains(".attn.")
        || lower.contains("attention")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
        || lower.contains("qkv")
        // Use the projection-suffixed forms rather than bare "query"/"key"/
        // "value": the latter match unrelated tensors (e.g. routing tables or
        // KV-cache buffers named "...key_cache") and misclassify them as
        // attention weights.
        || lower.contains("query_proj")
        || lower.contains("key_proj")
        || lower.contains("value_proj")
    {
        return TensorCategory::Attention;
    }
    if lower.contains("mlp")
        || lower.contains("ffn")
        || lower.contains("feed_forward")
        || lower.contains("expert")
        || lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains("w1")
        || lower.contains("w2")
        || lower.contains("w3")
    {
        return TensorCategory::MlpExpert;
    }
    TensorCategory::Other
}

pub fn recipe_metadata(recipe: &MergeRecipe, method: &str) -> BTreeMap<String, String> {
    let mut meta = BTreeMap::new();
    meta.insert("oxidize-merge.method".to_owned(), method.to_owned());
    meta.insert(
        "oxidize-merge.attention_t".to_owned(),
        recipe.attention_t.to_string(),
    );
    meta.insert("oxidize-merge.mlp_t".to_owned(), recipe.mlp_t.to_string());
    meta.insert(
        "oxidize-merge.other_t".to_owned(),
        recipe.other_t.to_string(),
    );
    if let Some(default_t) = recipe.default_t {
        meta.insert("oxidize-merge.default_t".to_owned(), default_t.to_string());
    }
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_attention_and_mlp() {
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.q_proj.weight"),
            TensorCategory::Attention
        );
        assert_eq!(
            classify_tensor("model.layers.3.mlp.experts.0.gate_proj.weight"),
            TensorCategory::MlpExpert
        );
        assert_eq!(
            classify_tensor("model.embed_tokens.weight"),
            TensorCategory::Other
        );
    }

    #[test]
    fn kimi_recipe_weights() {
        let recipe = MergeRecipe::kimi_k275();
        assert!((recipe.t_for_tensor("layers.0.self_attn.k_proj.weight") - 0.3).abs() < 1e-6);
        assert!((recipe.t_for_tensor("layers.0.mlp.gate_proj.weight") - 0.5).abs() < 1e-6);
        assert!((recipe.t_for_tensor("model.norm.weight") - 0.4).abs() < 1e-6);
    }
}
