//! Auto-detection and auto-tuning for oxidize inference.
//!
//! The `autotune` module produces a `TuningPlan` for the user's
//! hardware + model. The CLI and server consume the plan via
//! `PlanOverrides` and apply only the fields the user didn't set
//! themselves.
//!
//! See `plans/auto-detect-and-tune-inference.md` for the design and
//! `AGENTS.md` "WHERE TO LOOK" → autotune for usage.

pub mod apply;
pub mod detect;
pub mod fingerprint;
pub mod rules;

pub use apply::{PlanOverrides, overrides_from_plan};
pub use detect::{HardwareInventory, OsKind, detect};
pub use fingerprint::{
    ModelFingerprint, fingerprint, fingerprint_from_parts, kv_bytes_per_token,
    per_layer_weight_bytes, summary as model_summary,
};
pub use rules::{OxkIsa, OxkTile, PipelineMode, SpeculativeSpec, TuningPlan, plan};
