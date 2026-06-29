package mesh

import (
	"context"
	"fmt"
	"time"
)

// DefaultCollectiveTimeout mirrors DEFAULT_COLLECTIVE_TIMEOUT.
const DefaultCollectiveTimeout = 60 * time.Second

// RunnerStatusKind mirrors the RunnerStatus enum discriminant.
type RunnerStatusKind string

const (
	RunnerHealthy      RunnerStatusKind = "healthy"
	RunnerFailed       RunnerStatusKind = "runner_failed"
	RunnerShuttingDown RunnerStatusKind = "shutting_down"
	RunnerOffline      RunnerStatusKind = "offline"
)

// RunnerStatus mirrors RunnerStatus.
type RunnerStatus struct {
	Kind   RunnerStatusKind `json:"kind"`
	Reason string           `json:"reason,omitempty"`
}

// RunnerStatusUpdated mirrors RunnerStatusUpdated.
type RunnerStatusUpdated struct {
	PeerID string       `json:"peer_id"`
	Status RunnerStatus `json:"status"`
	Clock  uint64       `json:"clock"`
}

// ShutdownTask mirrors ShutdownTask.
type ShutdownTask struct {
	InstanceID string `json:"instance_id"`
	Reason     string `json:"reason"`
	Clock      uint64 `json:"clock"`
}

// TimedResultKind mirrors the TimedResult enum discriminant.
type TimedResultKind int

const (
	// TimedOk: operation completed within the deadline.
	TimedOk TimedResultKind = iota
	// TimedOut: operation exceeded the deadline and was cancelled.
	TimedOut
	// TimedErr: operation returned an error.
	TimedErr
)

// TimedResult mirrors TimedResult<T>. Value is valid only when Kind == TimedOk.
type TimedResult[T any] struct {
	Kind  TimedResultKind
	Value T
	Err   string
}

// IsOk reports success.
func (r TimedResult[T]) IsOk() bool { return r.Kind == TimedOk }

// EvalWithTimeout evaluates fut with a hard deadline. If fut does not complete
// within deadline, its context is cancelled and TimedOut is returned. Mirrors
// eval_with_timeout.
func EvalWithTimeout[T any](ctx context.Context, deadline time.Duration, fut func(context.Context) (T, error)) TimedResult[T] {
	cctx, cancel := context.WithTimeout(ctx, deadline)
	defer cancel()

	type result struct {
		val T
		err error
	}
	done := make(chan result, 1)
	go func() {
		val, err := fut(cctx)
		done <- result{val, err}
	}()

	select {
	case res := <-done:
		if res.err != nil {
			return TimedResult[T]{Kind: TimedErr, Err: res.err.Error()}
		}
		return TimedResult[T]{Kind: TimedOk, Value: res.val}
	case <-cctx.Done():
		// Deadline exceeded (or parent cancelled): report a timeout.
		return TimedResult[T]{Kind: TimedOut}
	}
}

// EvalWithTimeoutAndNotify wraps EvalWithTimeout and emits a RunnerStatusUpdated
// when the operation times out. Mirrors eval_with_timeout_and_notify.
func EvalWithTimeoutAndNotify[T any](
	ctx context.Context,
	deadline time.Duration,
	peerID string,
	clock uint64,
	onStatus func(RunnerStatusUpdated),
	fut func(context.Context) (T, error),
) TimedResult[T] {
	res := EvalWithTimeout(ctx, deadline, fut)
	if res.Kind == TimedOut && onStatus != nil {
		onStatus(RunnerStatusUpdated{
			PeerID: peerID,
			Status: RunnerStatus{
				Kind:   RunnerFailed,
				Reason: fmt.Sprintf("collective timed out after %ds", int(deadline.Seconds())),
			},
			Clock: clock,
		})
	}
	return res
}
