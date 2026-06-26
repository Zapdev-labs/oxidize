package mesh

import (
	"sort"
	"sync"
	"time"
)

// NodeCapabilities mirrors NodeCapabilities advertised during discovery.
type NodeCapabilities struct {
	DeviceType  string
	MemoryBytes uint64
	CPUThreads  int
	CanShard    bool
	Tags        map[string]string
}

// AggregateCapabilities mirrors AggregateCapabilities: a summary across the
// whole mesh.
type AggregateCapabilities struct {
	NodeCount        int
	TotalMemoryBytes uint64
	TotalCPUThreads  int
	CanShardNodes    int
	DeviceTypes      []string
}

// CapTopologyNode mirrors TopologyNode: a node tracked with capabilities and
// liveness metadata used for eviction and election tie-breaking.
type CapTopologyNode struct {
	PeerID       string
	Capabilities NodeCapabilities
	CommandsSeen uint64
	Seniority    uint64
	LastSeen     time.Time
	JoinedAt     time.Time
}

// IsStale reports whether the node has not been seen within timeout.
func (n *CapTopologyNode) IsStale(timeout time.Duration) bool {
	if n.LastSeen.IsZero() {
		return true
	}
	return time.Since(n.LastSeen) > timeout
}

// CapTopologyGraph mirrors the capability-aware TopologyGraph. It is additive to
// the existing TopologyGraph (which tracks plain MeshNode adjacency) and tracks
// per-peer NodeCapabilities for aggregation and stale eviction.
type CapTopologyGraph struct {
	mu          sync.Mutex
	nodes       map[string]*CapTopologyNode
	LocalPeerID string
}

// NewCapTopologyGraph constructs an empty capability graph.
func NewCapTopologyGraph() *CapTopologyGraph {
	return &CapTopologyGraph{nodes: map[string]*CapTopologyNode{}}
}

// AddOrUpdateNode registers a peer or refreshes its capabilities and heartbeat.
func (g *CapTopologyGraph) AddOrUpdateNode(peerID string, caps NodeCapabilities) {
	g.mu.Lock()
	defer g.mu.Unlock()
	now := time.Now()
	if existing, ok := g.nodes[peerID]; ok {
		existing.Capabilities = caps
		existing.LastSeen = now
		return
	}
	g.nodes[peerID] = &CapTopologyNode{
		PeerID:       peerID,
		Capabilities: caps,
		LastSeen:     now,
		JoinedAt:     now,
	}
}

// RemoveNode drops a peer.
func (g *CapTopologyGraph) RemoveNode(peerID string) {
	g.mu.Lock()
	delete(g.nodes, peerID)
	g.mu.Unlock()
}

// Heartbeat refreshes the last-seen timestamp for a peer.
func (g *CapTopologyGraph) Heartbeat(peerID string) {
	g.mu.Lock()
	if n, ok := g.nodes[peerID]; ok {
		n.LastSeen = time.Now()
	}
	g.mu.Unlock()
}

// EvictStale removes nodes not seen within timeout and returns their IDs sorted.
// Mirrors evict_stale.
func (g *CapTopologyGraph) EvictStale(timeout time.Duration) []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	var stale []string
	for id, n := range g.nodes {
		if n.IsStale(timeout) {
			stale = append(stale, id)
		}
	}
	for _, id := range stale {
		delete(g.nodes, id)
	}
	sort.Strings(stale)
	return stale
}

// PeerCount returns the number of known peers.
func (g *CapTopologyGraph) PeerCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.nodes)
}

// PeerIDs returns all known peer IDs (excluding the local peer if set), sorted.
func (g *CapTopologyGraph) PeerIDs() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]string, 0, len(g.nodes))
	for id := range g.nodes {
		if g.LocalPeerID != "" && id == g.LocalPeerID {
			continue
		}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// CapabilitiesOf returns a peer's capabilities, if known.
func (g *CapTopologyGraph) CapabilitiesOf(peerID string) (NodeCapabilities, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	n, ok := g.nodes[peerID]
	if !ok {
		return NodeCapabilities{}, false
	}
	return n.Capabilities, true
}

// AggregateCapabilities summarizes capabilities across all peers. Mirrors
// aggregate_capabilities.
func (g *CapTopologyGraph) AggregateCapabilities() AggregateCapabilities {
	g.mu.Lock()
	defer g.mu.Unlock()
	agg := AggregateCapabilities{NodeCount: len(g.nodes)}
	seen := map[string]struct{}{}
	for _, n := range g.nodes {
		agg.TotalMemoryBytes += n.Capabilities.MemoryBytes
		agg.TotalCPUThreads += n.Capabilities.CPUThreads
		if n.Capabilities.CanShard {
			agg.CanShardNodes++
		}
		if _, ok := seen[n.Capabilities.DeviceType]; !ok {
			seen[n.Capabilities.DeviceType] = struct{}{}
			agg.DeviceTypes = append(agg.DeviceTypes, n.Capabilities.DeviceType)
		}
	}
	sort.Strings(agg.DeviceTypes)
	return agg
}

// ShardableNodes returns the IDs of peers that can act as shard workers,
// sorted deterministically. Implements capability filtering for can_shard.
func (g *CapTopologyGraph) ShardableNodes() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	var out []string
	for id, n := range g.nodes {
		if n.Capabilities.CanShard {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// SetSeniority assigns a seniority score used for election tie-breaking.
func (g *CapTopologyGraph) SetSeniority(peerID string, seniority uint64) {
	g.mu.Lock()
	if n, ok := g.nodes[peerID]; ok {
		n.Seniority = seniority
	}
	g.mu.Unlock()
}
