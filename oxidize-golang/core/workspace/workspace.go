// Package workspace mirrors the small public surface of oxidize_core's
// `workspace_health`, `benchmark_input`, and `wasm_workspace_status` exports.
package workspace

// WorkspaceHealth is the readiness signal returned by health probes.
type WorkspaceHealth struct {
	Status string
}

const readyStatus = "ready"

// Health returns a static, ready-to-serve workspace health record.
func Health() WorkspaceHealth { return WorkspaceHealth{Status: readyStatus} }

// BenchmarkInput returns the health record used by benchmark harnesses.
func BenchmarkInput() WorkspaceHealth { return Health() }

// WasmStatus returns the workspace status string for WASM consumers.
func WasmStatus() string { return readyStatus }
