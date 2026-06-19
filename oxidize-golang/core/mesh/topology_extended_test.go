package mesh

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

func capsFor(device string, mem uint64, shard bool) NodeCapabilities {
	return NodeCapabilities{DeviceType: device, MemoryBytes: mem, CPUThreads: 8, CanShard: shard}
}

func TestCapGraphAggregate(t *testing.T) {
	g := NewCapTopologyGraph()
	g.AddOrUpdateNode("peer-a", capsFor("cpu", 8_000_000_000, true))
	g.AddOrUpdateNode("peer-b", capsFor("mlx", 4_000_000_000, true))
	agg := g.AggregateCapabilities()
	if agg.NodeCount != 2 {
		t.Fatalf("node count %d", agg.NodeCount)
	}
	if agg.TotalMemoryBytes != 12_000_000_000 {
		t.Fatalf("mem %d", agg.TotalMemoryBytes)
	}
	if agg.TotalCPUThreads != 16 {
		t.Fatalf("threads %d", agg.TotalCPUThreads)
	}
	if agg.CanShardNodes != 2 {
		t.Fatalf("shard nodes %d", agg.CanShardNodes)
	}
	if !reflect.DeepEqual(agg.DeviceTypes, []string{"cpu", "mlx"}) {
		t.Fatalf("device types %v", agg.DeviceTypes)
	}
}

func TestCapGraphEvictStale(t *testing.T) {
	g := NewCapTopologyGraph()
	g.AddOrUpdateNode("peer-a", capsFor("cpu", 1, true))
	g.nodes["peer-a"].LastSeen = time.Now().Add(-100 * time.Second)
	evicted := g.EvictStale(30 * time.Second)
	if !reflect.DeepEqual(evicted, []string{"peer-a"}) {
		t.Fatalf("evicted %v", evicted)
	}
	if g.PeerCount() != 0 {
		t.Fatalf("expected empty, got %d", g.PeerCount())
	}
}

func TestCapGraphShardableFilter(t *testing.T) {
	g := NewCapTopologyGraph()
	g.AddOrUpdateNode("a", capsFor("cpu", 1, true))
	g.AddOrUpdateNode("b", capsFor("cpu", 1, false))
	g.AddOrUpdateNode("c", capsFor("cpu", 1, true))
	if got := g.ShardableNodes(); !reflect.DeepEqual(got, []string{"a", "c"}) {
		t.Fatalf("shardable %v", got)
	}
}

func TestCapGraphPeerIDsExcludesLocal(t *testing.T) {
	g := NewCapTopologyGraph()
	g.LocalPeerID = "me"
	g.AddOrUpdateNode("me", capsFor("cpu", 1, true))
	g.AddOrUpdateNode("peer-a", capsFor("cpu", 1, true))
	if got := g.PeerIDs(); !reflect.DeepEqual(got, []string{"peer-a"}) {
		t.Fatalf("peer ids %v", got)
	}
}

func TestPriorityOrdering(t *testing.T) {
	p1 := NewPriority(1, 0, 0, "a")
	p2 := NewPriority(2, 0, 0, "a")
	p3 := NewPriority(1, 1, 0, "a")
	p4 := NewPriority(1, 0, 1, "a")
	p5 := NewPriority(1, 0, 0, "b")
	if !p1.Less(p2) || !p1.Less(p3) || !p1.Less(p4) || !p1.Less(p5) {
		t.Fatal("p1 should be lowest")
	}
	if !p3.Less(p2) || !p4.Less(p3) || !p5.Less(p4) {
		t.Fatal("tuple ordering broken")
	}
}

func TestRunElectionRound(t *testing.T) {
	g := NewCapTopologyGraph()
	g.AddOrUpdateNode("peer-a", capsFor("cpu", 1, true))
	g.AddOrUpdateNode("peer-b", capsFor("cpu", 1, true))
	g.SetSeniority("peer-a", 3)
	g.SetSeniority("peer-b", 7)
	e := NewBullyElectionEngine("local", 1)
	master, ok := RunElectionRound(e, g)
	if !ok || master != "peer-b" {
		t.Fatalf("expected peer-b, got %q ok=%v", master, ok)
	}
}

func TestRunElectionRoundLocalWins(t *testing.T) {
	e := NewBullyElectionEngine("solo", 1)
	master, ok := RunElectionRound(e, NewCapTopologyGraph())
	if !ok || master != "solo" {
		t.Fatalf("expected solo, got %q", master)
	}
	if !e.IsMaster() {
		t.Fatal("solo should be master")
	}
}

func TestEvalWithTimeoutSucceeds(t *testing.T) {
	res := EvalWithTimeout(context.Background(), 5*time.Second, func(ctx context.Context) (int, error) {
		return 42, nil
	})
	if !res.IsOk() || res.Value != 42 {
		t.Fatalf("expected ok 42, got %+v", res)
	}
}

func TestEvalWithTimeoutTimesOut(t *testing.T) {
	res := EvalWithTimeout(context.Background(), 30*time.Millisecond, func(ctx context.Context) (int, error) {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(time.Hour):
			return 1, nil
		}
	})
	if res.Kind != TimedOut {
		t.Fatalf("expected timeout, got %+v", res)
	}
}

func TestEvalWithTimeoutPropagatesError(t *testing.T) {
	res := EvalWithTimeout(context.Background(), 5*time.Second, func(ctx context.Context) (int, error) {
		return 0, errors.New("boom")
	})
	if res.Kind != TimedErr || res.Err != "boom" {
		t.Fatalf("expected err, got %+v", res)
	}
}

func TestEvalWithTimeoutNotifies(t *testing.T) {
	var ev *RunnerStatusUpdated
	res := EvalWithTimeoutAndNotify(context.Background(), 30*time.Millisecond, "peer-a", 7,
		func(u RunnerStatusUpdated) { ev = &u },
		func(ctx context.Context) (int, error) {
			<-ctx.Done()
			return 0, ctx.Err()
		})
	if res.Kind != TimedOut {
		t.Fatalf("expected timeout, got %+v", res)
	}
	if ev == nil || ev.PeerID != "peer-a" || ev.Clock != 7 || ev.Status.Kind != RunnerFailed {
		t.Fatalf("unexpected status event %+v", ev)
	}
}
