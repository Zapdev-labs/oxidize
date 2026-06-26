package mesh

// Priority mirrors the deterministic Priority tuple used to rank nodes.
// Ordering: higher Clock wins; if equal, higher Seniority; if equal, higher
// CommandsSeen; if equal, lexicographically larger PeerID.
type Priority struct {
	Clock        uint64
	Seniority    uint64
	CommandsSeen uint64
	PeerID       string
}

// NewPriority constructs a priority tuple.
func NewPriority(clock, seniority, commandsSeen uint64, peerID string) Priority {
	return Priority{Clock: clock, Seniority: seniority, CommandsSeen: commandsSeen, PeerID: peerID}
}

// Less reports whether p ranks below other under the deterministic ordering.
func (p Priority) Less(other Priority) bool {
	if p.Clock != other.Clock {
		return p.Clock < other.Clock
	}
	if p.Seniority != other.Seniority {
		return p.Seniority < other.Seniority
	}
	if p.CommandsSeen != other.CommandsSeen {
		return p.CommandsSeen < other.CommandsSeen
	}
	return p.PeerID < other.PeerID
}

// electionDeclare is a recorded candidacy declaration for the active round.
type electionDeclare struct {
	priority Priority
}

// BullyElectionEngine mirrors the Rust BullyElection state machine with a
// deterministic Priority-based winner. It is additive alongside the simpler
// existing BullyElection type.
type BullyElectionEngine struct {
	LocalPeerID    string
	LocalSeniority uint64
	LocalCommands  uint64
	Clock          uint64
	electing       bool
	elected        bool
	master         string
	declares       map[string]Priority
	Completed      uint64
}

// NewBullyElectionEngine constructs an idle election engine.
func NewBullyElectionEngine(localPeerID string, localSeniority uint64) *BullyElectionEngine {
	return &BullyElectionEngine{
		LocalPeerID:    localPeerID,
		LocalSeniority: localSeniority,
		declares:       map[string]Priority{},
	}
}

// StartElection begins a new round with an incremented clock.
func (e *BullyElectionEngine) StartElection() {
	e.Clock++
	e.declares = map[string]Priority{}
	e.electing = true
	e.elected = false
}

// RecordDeclare records a remote candidacy if it belongs to the current round.
// Stale or future-clock declares are ignored.
func (e *BullyElectionEngine) RecordDeclare(clock, seniority, commandsSeen uint64, peerID string) {
	if !e.electing || clock != e.Clock {
		return
	}
	e.declares[peerID] = NewPriority(clock, seniority, commandsSeen, peerID)
}

// FinalizeElection computes the deterministic winner from all declares plus the
// local priority, transitions to elected, and returns the master peer ID.
func (e *BullyElectionEngine) FinalizeElection() (string, bool) {
	if !e.electing {
		return "", false
	}
	best := NewPriority(e.Clock, e.LocalSeniority, e.LocalCommands, e.LocalPeerID)
	for _, p := range e.declares {
		if best.Less(p) {
			best = p
		}
	}
	e.electing = false
	e.elected = true
	e.master = best.PeerID
	e.Completed++
	return best.PeerID, true
}

// IsMaster reports whether the local node is the elected master.
func (e *BullyElectionEngine) IsMaster() bool {
	return e.elected && e.master == e.LocalPeerID
}

// CurrentMaster returns the elected master peer ID, if any.
func (e *BullyElectionEngine) CurrentMaster() (string, bool) {
	if e.elected {
		return e.master, true
	}
	return "", false
}

// AcceptResult adopts a result broadcast if its clock is current or newer.
func (e *BullyElectionEngine) AcceptResult(clock uint64, master string) {
	if clock >= e.Clock {
		e.Clock = clock
		e.electing = false
		e.elected = true
		e.master = master
	}
}

// IncLocalCommands bumps the local commands-seen counter.
func (e *BullyElectionEngine) IncLocalCommands() { e.LocalCommands++ }

// RunElectionRound runs a full deterministic election using a capability graph:
// it starts the round, injects virtual declares for every peer in the graph,
// and finalizes. Mirrors run_election_round.
func RunElectionRound(e *BullyElectionEngine, graph *CapTopologyGraph) (string, bool) {
	e.StartElection()
	if graph != nil {
		graph.mu.Lock()
		for peerID, node := range graph.nodes {
			if peerID == e.LocalPeerID {
				continue
			}
			e.RecordDeclare(e.Clock, node.Seniority, node.CommandsSeen, peerID)
		}
		graph.mu.Unlock()
	}
	return e.FinalizeElection()
}
