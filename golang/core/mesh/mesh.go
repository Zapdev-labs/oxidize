// Package mesh implements the distributed inference mesh.
package mesh

import (
	"errors"
	"sort"
	"sync"
	"time"
)

// MeshConfig mirrors MeshConfig.
type MeshConfig struct {
	NodeID          string
	HeartbeatPeriod time.Duration
	ElectionTimeout time.Duration
	DiscoveryURL    string
	BindAddr        string
}

// DefaultMeshConfig returns sensible defaults.
func DefaultMeshConfig() MeshConfig {
	return MeshConfig{HeartbeatPeriod: 250 * time.Millisecond, ElectionTimeout: 1 * time.Second}
}

// MeshNode mirrors MeshNode.
type MeshNode struct {
	ID        string
	Addr      string
	Role      string
	LastSeen  time.Time
	Healthy   bool
	LayerHint int
}

// GossipRouter mirrors GossipRouter.
type GossipRouter struct {
	mu        sync.RWMutex
	peers     map[string]MeshNode
	callbacks []func(MeshNode)
}

// NewGossipRouter constructs an empty router.
func NewGossipRouter() *GossipRouter { return &GossipRouter{peers: map[string]MeshNode{}} }

// On registers a callback for peer changes.
func (r *GossipRouter) On(cb func(MeshNode)) { r.mu.Lock(); r.callbacks = append(r.callbacks, cb); r.mu.Unlock() }

// Update inserts or refreshes a peer.
func (r *GossipRouter) Update(n MeshNode) {
	r.mu.Lock()
	n.LastSeen = time.Now()
	r.peers[n.ID] = n
	cbs := r.callbacks
	r.mu.Unlock()
	for _, cb := range cbs {
		cb(n)
	}
}

// Peers returns a copy of the current peer list.
func (r *GossipRouter) Peers() []MeshNode {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]MeshNode, 0, len(r.peers))
	for _, p := range r.peers {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// PeerByID returns the named peer.
func (r *GossipRouter) PeerByID(id string) (MeshNode, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n, ok := r.peers[id]
	return n, ok
}

// Remove removes a peer.
func (r *GossipRouter) Remove(id string) {
	r.mu.Lock()
	delete(r.peers, id)
	r.mu.Unlock()
}

// RingTransport mirrors RingTransport.
type RingTransport struct {
	mu    sync.Mutex
	peers []string
}

// NewRingTransport constructs a ring with the given peer addresses.
func NewRingTransport(peers []string) *RingTransport { return &RingTransport{peers: peers} }

// Next returns the next peer in the ring.
func (r *RingTransport) Next(self string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.peers) == 0 {
		return ""
	}
	for i, p := range r.peers {
		if p == self {
			return r.peers[(i+1)%len(r.peers)]
		}
	}
	return r.peers[0]
}

// ChannelTransport mirrors ChannelTransport. It uses Go channels for tests.
type ChannelTransport struct {
	In  chan []byte
	Out chan []byte
}

// NewChannelTransport constructs an in-memory transport.
func NewChannelTransport() *ChannelTransport { return &ChannelTransport{In: make(chan []byte, 64), Out: make(chan []byte, 64)} }

// Send writes a message.
func (c *ChannelTransport) Send(msg []byte) { c.Out <- msg }

// Recv returns a message or nil.
func (c *ChannelTransport) Recv() []byte {
	select {
	case m := <-c.In:
		return m
	default:
		return nil
	}
}

// TcpTransport mirrors TcpTransport. It is a thin shell that records
// configuration but does not actually open TCP connections.
type TcpTransport struct {
	Addr string
}

// NewTcpTransport constructs a transport that will bind to `addr`.
func NewTcpTransport(addr string) *TcpTransport { return &TcpTransport{Addr: addr} }

// ShardPlan mirrors ShardPlan.
type ShardPlan struct {
	Shards       []MeshShard
	LayerRange   [2]int
	ReplicaIndex int
}

// MeshShard mirrors MeshShard.
type MeshShard struct {
	NodeID     string
	LayerStart int
	LayerEnd   int
	DeviceID   int
	Backend    string
}

// Validate ensures the shard plan is consistent.
func (p *ShardPlan) Validate() error {
	if len(p.Shards) == 0 {
		return errors.New("mesh: empty shard plan")
	}
	if p.LayerRange[0] >= p.LayerRange[1] {
		return errors.New("mesh: invalid layer range")
	}
	for _, s := range p.Shards {
		if s.LayerStart >= s.LayerEnd {
			return errors.New("mesh: invalid shard range")
		}
	}
	return nil
}

// DiscoveryService mirrors DiscoveryService.
type DiscoveryService struct {
	mu      sync.Mutex
	router  *GossipRouter
	known   map[string]struct{}
}

// NewDiscoveryService constructs a service bound to a router.
func NewDiscoveryService(r *GossipRouter) *DiscoveryService { return &DiscoveryService{router: r, known: map[string]struct{}{}} }

// Announce registers a node with the service.
func (d *DiscoveryService) Announce(n MeshNode) {
	d.mu.Lock()
	d.known[n.ID] = struct{}{}
	d.mu.Unlock()
	d.router.Update(n)
}

// Forget removes a node.
func (d *DiscoveryService) Forget(id string) {
	d.mu.Lock()
	delete(d.known, id)
	d.mu.Unlock()
	d.router.Remove(id)
}

// BullyElection mirrors BullyElection.
type BullyElection struct {
	mu     sync.Mutex
	peers  []string
	leader string
}

// NewBullyElection constructs a bully election for the given peer ids.
func NewBullyElection(peers []string) *BullyElection {
	return &BullyElection{peers: peers, leader: peers[0]}
}

// Leader returns the current leader id.
func (b *BullyElection) Leader() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.leader
}

// SetLeader manually assigns a leader.
func (b *BullyElection) SetLeader(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.leader = id
}

// TopologyGraph mirrors TopologyGraph.
type TopologyGraph struct {
	mu    sync.Mutex
	nodes map[string]MeshNode
	edges map[string][]string
}

// NewTopologyGraph constructs an empty graph.
func NewTopologyGraph() *TopologyGraph { return &TopologyGraph{nodes: map[string]MeshNode{}, edges: map[string][]string{}} }

// AddNode inserts a node.
func (g *TopologyGraph) AddNode(n MeshNode) {
	g.mu.Lock()
	g.nodes[n.ID] = n
	if _, ok := g.edges[n.ID]; !ok {
		g.edges[n.ID] = []string{}
	}
	g.mu.Unlock()
}

// AddEdge records a link from a -> b.
func (g *TopologyGraph) AddEdge(a, b string) {
	g.mu.Lock()
	g.edges[a] = append(g.edges[a], b)
	g.mu.Unlock()
}

// Neighbors returns the neighbors of a node.
func (g *TopologyGraph) Neighbors(id string) []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := append([]string(nil), g.edges[id]...)
	sort.Strings(out)
	return out
}

// Nodes returns all nodes.
func (g *TopologyGraph) Nodes() []MeshNode {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]MeshNode, 0, len(g.nodes))
	for _, n := range g.nodes {
		out = append(out, n)
	}
	return out
}

// MeshChatEngine mirrors MeshChatEngine.
type MeshChatEngine struct {
	mu        sync.Mutex
	Router    *GossipRouter
	Shards    map[string]*ShardPlan
	LocalNode MeshNode
}

// NewMeshChatEngine constructs a new engine.
func NewMeshChatEngine(local MeshNode) *MeshChatEngine {
	return &MeshChatEngine{Router: NewGossipRouter(), Shards: map[string]*ShardPlan{}, LocalNode: local}
}

// RegisterShard registers a shard plan under a key.
func (e *MeshChatEngine) RegisterShard(key string, plan *ShardPlan) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.Shards[key] = plan
}

// LookupShard returns the named shard plan.
func (e *MeshChatEngine) LookupShard(key string) (*ShardPlan, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	s, ok := e.Shards[key]
	return s, ok
}

// LoadProgressReport mirrors LoadProgressReport.
type LoadProgressReport struct {
	NodeID        string
	BytesLoaded   int64
	BytesTotal    int64
	PercentDone   float32
	LayersReady   int
	LayersTotal   int
	ElapsedMillis int64
}

// MeshValidationReport mirrors MeshValidationReport.
type MeshValidationReport struct {
	TotalNodes   int
	HealthyNodes int
	LayersCovered int
	MissingShards []string
}

// Validate checks a shard plan against a list of healthy nodes.
func Validate(plan *ShardPlan, healthy []MeshNode) MeshValidationReport {
	rep := MeshValidationReport{TotalNodes: len(healthy), HealthyNodes: 0}
	healthyIDs := map[string]struct{}{}
	for _, n := range healthy {
		if n.Healthy {
			rep.HealthyNodes++
		}
		healthyIDs[n.ID] = struct{}{}
	}
	if plan == nil {
		return rep
	}
	for _, s := range plan.Shards {
		if _, ok := healthyIDs[s.NodeID]; ok {
			rep.LayersCovered += s.LayerEnd - s.LayerStart
		} else {
			rep.MissingShards = append(rep.MissingShards, s.NodeID)
		}
	}
	return rep
}
