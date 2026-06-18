// Package validation provides cross-validation harness and parity tests.
package validation

import (
	"errors"
	"sort"
	"sync"
	"time"
)

// Suite mirrors ValidationSuite.
type Suite string

// Recognised suites.
const (
	SuiteForward     Suite = "forward"
	SuiteSampling    Suite = "sampling"
	SuiteTokenizer   Suite = "tokenizer"
	SuiteQuantization Suite = "quantization"
	SuiteMesh        Suite = "mesh"
	SuitePaged       Suite = "paged"
)

// Result mirrors ValidationResult.
type Result struct {
	Suite    Suite
	Passed   bool
	Elapsed  time.Duration
	Output   string
}

// ParityError mirrors ParityError.
type ParityError struct{ Message string }

func (e *ParityError) Error() string { return "parity: " + e.Message }

// ParityReport mirrors ParityReport.
type ParityReport struct {
	RunAt     time.Time
	Total     int
	Passed    int
	Failed    int
	Failures  []string
}

// Runner mirrors ValidationRunner.
type Runner struct {
	mu      sync.Mutex
	suites  map[Suite]bool
	results []Result
}

// NewRunner constructs a runner with no enabled suites.
func NewRunner() *Runner { return &Runner{suites: map[Suite]bool{}} }

// Enable enables a suite.
func (r *Runner) Enable(s Suite) { r.mu.Lock(); r.suites[s] = true; r.mu.Unlock() }

// Disable disables a suite.
func (r *Runner) Disable(s Suite) { r.mu.Lock(); r.suites[s] = false; r.mu.Unlock() }

// Run executes enabled suites using registered probes. Suites without probes fail.
func (r *Runner) Run() ParityReport {
	r.mu.Lock()
	enabled := make([]Suite, 0, len(r.suites))
	for s, on := range r.suites {
		if on {
			enabled = append(enabled, s)
		}
	}
	r.mu.Unlock()
	sort.Slice(enabled, func(i, j int) bool { return enabled[i] < enabled[j] })
	now := time.Now()
	var results []Result
	var failures []string
	for _, s := range enabled {
		start := time.Now()
		if err := RunProbe(s); err != nil {
			msg := string(s) + ": " + err.Error()
			failures = append(failures, msg)
			results = append(results, Result{Suite: s, Passed: false, Elapsed: time.Since(start), Output: msg})
			continue
		}
		results = append(results, Result{Suite: s, Passed: true, Elapsed: time.Since(start), Output: "ok"})
	}
	r.mu.Lock()
	r.results = results
	r.mu.Unlock()
	rep := ParityReport{RunAt: now, Total: len(results), Passed: 0, Failures: failures}
	for _, res := range results {
		if res.Passed {
			rep.Passed++
		}
	}
	rep.Failed = rep.Total - rep.Passed
	return rep
}

// ImplementedSuites returns the list of suites supported by this package.
func ImplementedSuites() []Suite {
	return []Suite{SuiteForward, SuiteSampling, SuiteTokenizer, SuiteQuantization, SuiteMesh, SuitePaged}
}

// RegisterProbe registers a custom probe for a given suite. This is a stub
// to allow the Go port to evolve parity checks without forcing all suites
// to be implemented at once.
type Probe func() error

var (
	probesMu sync.RWMutex
	probes   = map[Suite]Probe{}
)

// RegisterProbe stores a probe.
func RegisterProbe(s Suite, p Probe) {
	probesMu.Lock()
	probes[s] = p
	probesMu.Unlock()
}

// RunProbe runs a registered probe.
func RunProbe(s Suite) error {
	probesMu.RLock()
	p, ok := probes[s]
	probesMu.RUnlock()
	if !ok {
		return errors.New("no probe registered for suite " + string(s))
	}
	return p()
}
