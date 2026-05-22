use super::{MeshChatPrompt, MeshCommand, NodeCapabilities, ShardPlan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshValidationReport {
    pub valid: bool,
    pub issues: Vec<String>,
}

impl MeshValidationReport {
    pub fn ok() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
        }
    }

    fn push(&mut self, issue: impl Into<String>) {
        self.valid = false;
        self.issues.push(issue.into());
    }
}

pub fn validate_mesh_prompt(prompt: &MeshChatPrompt) -> MeshValidationReport {
    let mut report = MeshValidationReport::ok();
    if prompt.request_id.trim().is_empty() {
        report.push("request_id is empty");
    }
    if prompt.max_tokens == 0 {
        report.push("max_tokens must be greater than zero");
    }
    if !prompt.temperature.is_finite() || prompt.temperature <= 0.0 {
        report.push("temperature must be finite and positive");
    }
    if !prompt.top_p.is_finite() || !(0.0..=1.0).contains(&prompt.top_p) || prompt.top_p == 0.0 {
        report.push("top_p must be in (0, 1]");
    }
    report
}

pub fn validate_mesh_command(command: &MeshCommand) -> MeshValidationReport {
    match command {
        MeshCommand::ChatPrompt(prompt) => validate_mesh_prompt(prompt),
        MeshCommand::Shutdown(_) | MeshCommand::ShardPlan(_) => MeshValidationReport::ok(),
    }
}

pub fn validate_shard_plan(plan: &ShardPlan) -> MeshValidationReport {
    let mut report = MeshValidationReport::ok();
    if plan.assignments.is_empty() {
        report.push("shard plan has no assignments");
    }
    report
}

pub fn validate_node_capabilities(capabilities: &NodeCapabilities) -> MeshValidationReport {
    let mut report = MeshValidationReport::ok();
    if capabilities.device_type.trim().is_empty() {
        report.push("device_type is empty");
    }
    if capabilities.memory_bytes == 0 {
        report.push("memory_bytes must be greater than zero");
    }
    if capabilities.cpu_threads == 0 {
        report.push("cpu_threads must be greater than zero");
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrutiny_rejects_invalid_mesh_prompt() {
        let prompt = MeshChatPrompt {
            request_id: String::new(),
            prompt: "hello".into(),
            max_tokens: 0,
            temperature: 0.0,
            top_p: 2.0,
        };
        let report = validate_mesh_prompt(&prompt);
        assert!(!report.valid);
        assert!(report.issues.len() >= 3);
    }
}
