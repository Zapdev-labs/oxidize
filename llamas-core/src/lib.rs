use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceHealth {
    pub status: &'static str,
}

pub fn workspace_health() -> WorkspaceHealth {
    WorkspaceHealth { status: "ready" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_health_is_ready() {
        assert_eq!(workspace_health().status, "ready");
    }
}
