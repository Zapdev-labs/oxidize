use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct XtcSamplerConfig {
    pub probability: f32,
    pub threshold: f32,
}

impl Default for XtcSamplerConfig {
    fn default() -> Self {
        Self {
            probability: 0.0,
            threshold: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DrySamplerConfig {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub penalty_last_n: usize,
    pub sequence_breakers: Vec<u32>,
}

impl Default for DrySamplerConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 256,
            sequence_breakers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DynamicTemperatureConfig {
    pub min: f32,
    pub max: f32,
    pub exponent: f32,
}

impl DynamicTemperatureConfig {
    pub fn temperature_for_entropy(&self, entropy_ratio: f32) -> f32 {
        let clamped = entropy_ratio.clamp(0.0, 1.0).powf(self.exponent.max(0.001));
        self.min + (self.max - self.min) * clamped
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerStep {
    TopK,
    TopP,
    MinP,
    Typical,
    TailFree,
    Xtc,
    Dry,
    Grammar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplerChain {
    pub steps: Vec<SamplerStep>,
    pub grammar_first: bool,
}

impl SamplerChain {
    pub fn from_names(names: &[&str]) -> Result<Self, String> {
        let mut steps = Vec::with_capacity(names.len());
        for name in names {
            steps.push(match name.to_ascii_lowercase().as_str() {
                "top-k" | "top_k" | "k" => SamplerStep::TopK,
                "top-p" | "top_p" | "p" => SamplerStep::TopP,
                "min-p" | "min_p" => SamplerStep::MinP,
                "typical" => SamplerStep::Typical,
                "tail-free" | "tfs" => SamplerStep::TailFree,
                "xtc" => SamplerStep::Xtc,
                "dry" => SamplerStep::Dry,
                "grammar" => SamplerStep::Grammar,
                other => return Err(format!("unknown sampler step: {other}")),
            });
        }
        Ok(Self {
            grammar_first: steps.first() == Some(&SamplerStep::Grammar),
            steps,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters_json_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function_name: String,
    pub arguments: serde_json::Value,
}

pub fn render_tool_call_json(call: &ToolCall) -> String {
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function_name,
            "arguments": serde_json::to_string(&call.arguments)
                .expect("serde_json::Value serialization cannot fail"),
        }
    })
    .to_string()
}

pub fn render_jinja_like_template(template: &str, values: &[(&str, &str)]) -> String {
    let mut rendered = template.to_string();
    for (key, value) in values {
        rendered = rendered.replace(&format!("{{{{ {key} }}}}"), value);
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), value);
    }
    rendered
}

pub fn json_schema_to_simple_grammar(schema: &serde_json::Value) -> String {
    if schema.get("type").and_then(|v| v.as_str()) == Some("object") {
        "root ::= \"{\" .* \"}\"".to_string()
    } else if schema.get("type").and_then(|v| v.as_str()) == Some("array") {
        "root ::= \"[\" .* \"]\"".to_string()
    } else {
        "root ::= .*".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_chain_parses_advanced_steps() {
        let chain = SamplerChain::from_names(&["grammar", "xtc", "dry"]).unwrap();
        assert!(chain.grammar_first);
        assert_eq!(chain.steps.len(), 3);
    }

    #[test]
    fn function_call_renders_openai_shape() {
        let call = ToolCall {
            id: "call_1".into(),
            function_name: "lookup".into(),
            arguments: serde_json::json!({"q":"rust"}),
        };
        let rendered: serde_json::Value =
            serde_json::from_str(&render_tool_call_json(&call)).unwrap();
        assert_eq!(rendered["type"], "function");
        assert_eq!(rendered["function"]["name"], "lookup");
        assert_eq!(rendered["function"]["arguments"], r#"{"q":"rust"}"#);
    }
}
