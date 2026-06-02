package mesh

import (
	"testing"
	"time"
)

func TestGossipRouter(t *testing.T) {
	r := NewGossipRouter()
	r.Update(MeshNode{ID: "a", Addr: "1.2.3.4:5", Role: "primary"})
	r.Update(MeshNode{ID: "b", Addr: "1.2.3.4:6", Role: "secondary"})
	peers := r.Peers()
	if len(peers) != 2 {
		t.Fatalf("expected 2 peers, got %d", len(peers))
	}
}

func TestRingTransport(t *testing.T) {
	r := NewRingTransport([]string{"a", "b", "c"})
	if r.Next("a") != "b" {
		t.Fatal("a -> b expected")
	}
	if r.Next("c") != "a" {
		t.Fatal("c -> a expected")
	}
}

func TestChannelTransport(t *testing.T) {
	c := NewChannelTransport()
	c.Send([]byte("hi"))
	if msg := c.Recv(); string(msg) != "hi" {
		t.Fatalf("expected 'hi', got %q", msg)
	}
}

func TestShardPlanValidate(t *testing.T) {
	plan := &ShardPlan{LayerRange: [2]int{0, 32}, Shards: []MeshShard{{NodeID: "a", LayerStart: 0, LayerEnd: 16, Backend: "cuda"}, {NodeID: "b", LayerStart: 16, LayerEnd: 32, Backend: "cpu"}}}
	if err := plan.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestBullyElection(t *testing.T) {
	b := NewBullyElection([]string{"a", "b", "c"})
	if b.Leader() != "a" {
		t.Fatalf("expected a, got %s", b.Leader())
	}
	b.SetLeader("c")
	if b.Leader() != "c" {
		t.Fatal("expected c")
	}
}

func TestDiscoveryService(t *testing.T) {
	r := NewGossipRouter()
	d := NewDiscoveryService(r)
	d.Announce(MeshNode{ID: "x", Addr: "0.0.0.0:1", Healthy: true, LastSeen: time.Now()})
	if _, ok := d.router.PeerByID("x"); !ok {
		t.Fatal("expected x")
	}
}

func TestTopologyGraph(t *testing.T) {
	g := NewTopologyGraph()
	g.AddNode(MeshNode{ID: "a"})
	g.AddNode(MeshNode{ID: "b"})
	g.AddEdge("a", "b")
	ns := g.Neighbors("a")
	if len(ns) != 1 || ns[0] != "b" {
		t.Fatalf("expected [b], got %v", ns)
	}
}

func TestMeshChatEngine(t *testing.T) {
	e := NewMeshChatEngine(MeshNode{ID: "a"})
	plan := &ShardPlan{LayerRange: [2]int{0, 16}, Shards: []MeshShard{{NodeID: "a", LayerStart: 0, LayerEnd: 16}}}
	e.RegisterShard("model", plan)
	if _, ok := e.LookupShard("model"); !ok {
		t.Fatal("expected shard")
	}
}

func TestValidateMesh(t *testing.T) {
	plan := &ShardPlan{LayerRange: [2]int{0, 16}, Shards: []MeshShard{{NodeID: "a", LayerStart: 0, LayerEnd: 16}}}
	rep := Validate(plan, []MeshNode{{ID: "a", Healthy: true}, {ID: "b", Healthy: false}})
	if rep.HealthyNodes != 1 {
		t.Fatalf("expected 1 healthy, got %d", rep.HealthyNodes)
	}
	if rep.LayersCovered != 16 {
		t.Fatalf("expected 16 layers, got %d", rep.LayersCovered)
	}
}
