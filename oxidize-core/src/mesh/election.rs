//! Bully-style leader election for the mesh.
//!
//! The election protocol is deterministic: the winner is the node with the
//! highest `(clock, seniority, commands_seen, node_id)` tuple.  All nodes
//! broadcast [`ElectionMessage`]s on the `ELECTION_MESSAGES` topic; after a
//! short timeout every node computes the same winner independently.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

use super::node::NodeCapabilities;
use super::topology::TopologyGraph;

/// Monotonic election clock — incremented every time a new election starts.
/// Events from older clocks are discarded (session invalidation).
pub type ElectionClock = u64;

/// Messages exchanged during the Bully election protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "payload")]
pub enum ElectionMessage {
    /// A node declares its candidacy with its current priority tuple.
    Declare {
        clock: ElectionClock,
        peer_id: String,
        seniority: u64,
        commands_seen: u64,
        capabilities: NodeCapabilities,
    },
    /// A node acknowledges a higher-priority peer and concedes.
    Concede {
        clock: ElectionClock,
        peer_id: String,
        master_peer_id: String,
    },
    /// Final result broadcast once the election converges.
    Result {
        clock: ElectionClock,
        master_peer_id: String,
    },
}

/// Deterministic priority tuple used to rank nodes.
///
/// Ordering: higher `clock` wins; if equal, higher `seniority`; if equal,
/// higher `commands_seen`; if equal, lexicographically larger `peer_id`
/// (strings are totally ordered and deterministic).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Priority {
    pub clock: ElectionClock,
    pub seniority: u64,
    pub commands_seen: u64,
    pub peer_id: String,
}

impl Priority {
    pub fn new(clock: ElectionClock, seniority: u64, commands_seen: u64, peer_id: String) -> Self {
        Self {
            clock,
            seniority,
            commands_seen,
            peer_id,
        }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.clock
            .cmp(&other.clock)
            .then_with(|| self.seniority.cmp(&other.seniority))
            .then_with(|| self.commands_seen.cmp(&other.commands_seen))
            .then_with(|| self.peer_id.cmp(&other.peer_id))
    }
}

/// State machine for the Bully election on a single node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElectionState {
    /// No election in progress.
    Idle,
    /// Election is running; we are collecting `Declare` messages.
    Electing {
        clock: ElectionClock,
        deadline: std::time::Instant,
    },
    /// Election finished; `master` is the winner for this `clock`.
    Elected {
        clock: ElectionClock,
        master: String,
    },
}

/// Bully election engine.
///
/// Holds local node state, tracks remote declares, and produces the
/// deterministic winner after the election timeout expires.
#[derive(Debug)]
pub struct BullyElection {
    pub local_peer_id: String,
    pub local_seniority: u64,
    pub local_commands: u64,
    pub local_capabilities: NodeCapabilities,
    pub state: ElectionState,
    /// Current election clock (monotonically increasing).
    pub clock: ElectionClock,
    /// All declares received during the current election round.
    pub declares: HashMap<String, Priority>,
    /// Duration to wait for declares before computing the winner.
    pub timeout: std::time::Duration,
    /// Number of completed elections (for metrics).
    pub elections_completed: u64,
}

impl BullyElection {
    pub fn new(
        local_peer_id: String,
        local_seniority: u64,
        local_capabilities: NodeCapabilities,
        timeout: std::time::Duration,
    ) -> Self {
        Self {
            local_peer_id,
            local_seniority,
            local_commands: 0,
            local_capabilities,
            state: ElectionState::Idle,
            clock: 0,
            declares: HashMap::new(),
            timeout,
            elections_completed: 0,
        }
    }

    /// Start a new election round with an incremented clock.
    pub fn start_election(&mut self) -> ElectionMessage {
        self.clock += 1;
        self.declares.clear();
        let deadline = std::time::Instant::now() + self.timeout;
        self.state = ElectionState::Electing {
            clock: self.clock,
            deadline,
        };
        ElectionMessage::Declare {
            clock: self.clock,
            peer_id: self.local_peer_id.clone(),
            seniority: self.local_seniority,
            commands_seen: self.local_commands,
            capabilities: self.local_capabilities.clone(),
        }
    }

    /// Record a remote `Declare` if it belongs to the current election.
    pub fn record_declare(&mut self, msg: &ElectionMessage) {
        if let ElectionMessage::Declare {
            clock,
            peer_id,
            seniority,
            commands_seen,
            ..
        } = msg
            && let ElectionState::Electing {
                clock: active_clock,
                ..
            } = &self.state
        {
            if *clock != *active_clock {
                // Stale declare from an older or future election — ignore.
                return;
            }
            let priority = Priority::new(*clock, *seniority, *commands_seen, peer_id.clone());
            self.declares.insert(peer_id.clone(), priority);
        }
    }

    /// Record a remote `Concede` (used for metrics / logging; does not affect
    /// the deterministic result).
    pub fn record_concede(&mut self, _msg: &ElectionMessage) {
        // Currently a no-op; concession messages do not affect the deterministic
        // result. Kept as a hook for future metrics / logging.
    }

    /// True if the election timeout has expired.
    pub fn is_timed_out(&self) -> bool {
        if let ElectionState::Electing { deadline, .. } = &self.state {
            std::time::Instant::now() >= *deadline
        } else {
            false
        }
    }

    /// Compute the winner from all recorded declares plus our own priority.
    ///
    /// Returns an `ElectionMessage::Result` and transitions to `Elected`.
    pub fn finalize_election(&mut self) -> Option<ElectionMessage> {
        if !matches!(self.state, ElectionState::Electing { .. }) {
            return None;
        }

        // Build priority list over references to avoid cloning every declare.
        let local = Priority::new(
            self.clock,
            self.local_seniority,
            self.local_commands,
            self.local_peer_id.clone(),
        );
        let winner = self
            .declares
            .values()
            .chain(std::iter::once(&local))
            .max()?
            .clone();

        self.state = ElectionState::Elected {
            clock: self.clock,
            master: winner.peer_id.clone(),
        };
        self.elections_completed += 1;

        Some(ElectionMessage::Result {
            clock: self.clock,
            master_peer_id: winner.peer_id,
        })
    }

    /// Process an incoming `ElectionMessage`.
    ///
    /// - `Declare` → recorded for the current round.
    /// - `Concede` → recorded (no-op on result).
    /// - `Result` → if clock matches current or is newer, accept it and
    ///   transition to `Elected`.
    pub fn handle_message(&mut self, msg: &ElectionMessage) -> Option<ElectionMessage> {
        match msg {
            ElectionMessage::Declare { .. } => {
                self.record_declare(msg);
                None
            }
            ElectionMessage::Concede { .. } => {
                self.record_concede(msg);
                None
            }
            ElectionMessage::Result {
                clock,
                master_peer_id,
            } => {
                // Accept result if it matches our current election or a newer one.
                if *clock >= self.clock {
                    self.clock = *clock;
                    self.state = ElectionState::Elected {
                        clock: *clock,
                        master: master_peer_id.clone(),
                    };
                }
                None
            }
        }
    }

    /// Whether the local node is currently the elected master.
    pub fn is_master(&self) -> bool {
        if let ElectionState::Elected { master, .. } = &self.state {
            master == &self.local_peer_id
        } else {
            false
        }
    }

    /// Current master peer ID, if any.
    pub fn current_master(&self) -> Option<&str> {
        if let ElectionState::Elected { master, .. } = &self.state {
            Some(master.as_str())
        } else {
            None
        }
    }

    /// True if the current election clock matches `clock`.
    pub fn clock_valid(&self, clock: ElectionClock) -> bool {
        self.clock == clock
    }

    /// Return the remaining time until the election deadline, if electing.
    /// Returns `Some(Duration::ZERO)` when the deadline has already passed.
    pub fn time_remaining(&self) -> Option<std::time::Duration> {
        if let ElectionState::Electing { deadline, .. } = &self.state {
            let now = std::time::Instant::now();
            Some(if *deadline > now {
                *deadline - now
            } else {
                std::time::Duration::ZERO
            })
        } else {
            None
        }
    }

    /// Increment local commands-seen counter.
    pub fn inc_local_commands(&mut self) {
        self.local_commands += 1;
    }
}

/// Convenience: run a full election round using a [`TopologyGraph`].
///
/// 1. Starts election on the local node.
/// 2. Adds every peer in `topology` as a virtual declare.
/// 3. Finalizes and returns the result message.
pub fn run_election_round(
    election: &mut BullyElection,
    topology: &TopologyGraph,
) -> Option<ElectionMessage> {
    let _declare = election.start_election();

    // Inject virtual declares from all known topology nodes.
    for (peer_id, node) in &topology.nodes {
        if *peer_id == election.local_peer_id {
            continue;
        }
        election.record_declare(&ElectionMessage::Declare {
            clock: election.clock,
            peer_id: peer_id.clone(),
            seniority: node.seniority,
            commands_seen: node.commands_seen,
            capabilities: NodeCapabilities::default(),
        });
    }

    // Pretend timeout expired by mutating deadline in-place.
    if let ElectionState::Electing {
        ref mut deadline, ..
    } = election.state
    {
        *deadline = std::time::Instant::now();
    }

    election.finalize_election()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    fn dummy_caps() -> NodeCapabilities {
        NodeCapabilities {
            device_type: "cpu".to_string(),
            memory_bytes: 8_000_000_000,
            cpu_threads: 8,
            can_shard: true,
            tags: HashMap::new(),
        }
    }

    fn make_election(peer_id: &str, seniority: u64) -> BullyElection {
        BullyElection::new(
            peer_id.to_string(),
            seniority,
            dummy_caps(),
            Duration::from_millis(100),
        )
    }

    #[test]
    fn election_starts_with_clock_increment() {
        let mut e = make_election("a", 1);
        assert_eq!(e.clock, 0);
        let msg = e.start_election();
        assert_eq!(e.clock, 1);
        assert!(
            matches!(msg, ElectionMessage::Declare { clock: 1, peer_id, .. } if peer_id == "a")
        );
        assert!(matches!(e.state, ElectionState::Electing { clock: 1, .. }));
    }

    #[test]
    fn priority_ordering_is_total() {
        let p1 = Priority::new(1, 0, 0, "a".to_string());
        let p2 = Priority::new(2, 0, 0, "a".to_string());
        let p3 = Priority::new(1, 1, 0, "a".to_string());
        let p4 = Priority::new(1, 0, 1, "a".to_string());
        let p5 = Priority::new(1, 0, 0, "b".to_string());

        assert!(p1 < p2);
        assert!(p1 < p3);
        assert!(p1 < p4);
        assert!(p1 < p5);
        assert!(p3 < p2);
        assert!(p4 < p3);
        assert!(p5 < p4); // peer_id "b" > "a"
    }

    #[test]
    fn finalize_selects_highest_clock() {
        let mut e = make_election("a", 1);
        e.start_election();
        e.record_declare(&ElectionMessage::Declare {
            clock: 1,
            peer_id: "b".to_string(),
            seniority: 5,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        // Same clock, higher seniority wins.
        let result = e.finalize_election();
        assert_eq!(
            result,
            Some(ElectionMessage::Result {
                clock: 1,
                master_peer_id: "b".to_string(),
            })
        );
        assert_eq!(e.current_master(), Some("b"));
        assert!(!e.is_master());
    }

    #[test]
    fn finalize_selects_local_when_highest() {
        let mut e = make_election("a", 10);
        e.start_election();
        e.record_declare(&ElectionMessage::Declare {
            clock: 1,
            peer_id: "b".to_string(),
            seniority: 5,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        let result = e.finalize_election();
        assert_eq!(
            result,
            Some(ElectionMessage::Result {
                clock: 1,
                master_peer_id: "a".to_string(),
            })
        );
        assert!(e.is_master());
    }

    #[test]
    fn stale_declare_ignored() {
        let mut e = make_election("a", 1);
        e.start_election(); // clock = 1
        e.record_declare(&ElectionMessage::Declare {
            clock: 0, // old clock
            peer_id: "b".to_string(),
            seniority: 100,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        let result = e.finalize_election().unwrap();
        // Local node wins because stale declare was ignored.
        assert_eq!(
            result,
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "a".to_string(),
            }
        );
    }

    #[test]
    fn future_declare_ignored() {
        let mut e = make_election("a", 1);
        e.start_election(); // clock = 1
        e.record_declare(&ElectionMessage::Declare {
            clock: 5, // future clock
            peer_id: "b".to_string(),
            seniority: 100,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        let result = e.finalize_election().unwrap();
        // Local node wins because future declare was ignored.
        assert_eq!(
            result,
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "a".to_string(),
            }
        );
    }

    #[test]
    fn result_message_adopts_new_master() {
        let mut e = make_election("a", 1);
        e.start_election(); // clock = 1
        let out = e.handle_message(&ElectionMessage::Result {
            clock: 1,
            master_peer_id: "c".to_string(),
        });
        assert!(out.is_none());
        assert_eq!(e.current_master(), Some("c"));
    }

    #[test]
    fn result_with_newer_clock_updates_state() {
        let mut e = make_election("a", 1);
        e.start_election(); // clock = 1
        e.handle_message(&ElectionMessage::Result {
            clock: 3,
            master_peer_id: "d".to_string(),
        });
        assert_eq!(e.clock, 3);
        assert_eq!(e.current_master(), Some("d"));
    }

    #[test]
    fn session_invalidation_on_new_election() {
        let mut e = make_election("a", 1);
        e.start_election(); // clock = 1
        e.finalize_election(); // elected a
        assert!(e.is_master());

        // New node joins, triggers re-election
        let msg = e.start_election(); // clock = 2
        assert_eq!(e.clock, 2);
        assert!(matches!(e.state, ElectionState::Electing { .. }));
        assert!(matches!(msg, ElectionMessage::Declare { clock: 2, .. }));
    }

    #[test]
    fn three_node_convergence() {
        let mut e = make_election("low", 1);
        e.start_election();
        e.record_declare(&ElectionMessage::Declare {
            clock: 1,
            peer_id: "mid".to_string(),
            seniority: 5,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        e.record_declare(&ElectionMessage::Declare {
            clock: 1,
            peer_id: "high".to_string(),
            seniority: 10,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        let result = e.finalize_election().unwrap();
        assert_eq!(
            result,
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "high".to_string(),
            }
        );
    }

    #[test]
    fn commands_seen_tiebreaker() {
        let mut e = make_election("a", 5);
        e.start_election();
        e.record_declare(&ElectionMessage::Declare {
            clock: 1,
            peer_id: "b".to_string(),
            seniority: 5,
            commands_seen: 3,
            capabilities: dummy_caps(),
        });
        e.local_commands = 7;
        let result = e.finalize_election().unwrap();
        // Same clock and seniority; higher commands_seen wins.
        assert_eq!(
            result,
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "a".to_string(),
            }
        );
    }

    #[test]
    fn peer_id_tiebreaker() {
        let mut e = make_election("a", 5);
        e.start_election();
        e.record_declare(&ElectionMessage::Declare {
            clock: 1,
            peer_id: "z".to_string(),
            seniority: 5,
            commands_seen: 0,
            capabilities: dummy_caps(),
        });
        e.local_commands = 0;
        let result = e.finalize_election().unwrap();
        // Same clock, seniority, commands_seen; lexicographically larger peer_id wins.
        assert_eq!(
            result,
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "z".to_string(),
            }
        );
    }

    #[test]
    fn clock_valid_matches_current() {
        let mut e = make_election("a", 1);
        e.start_election();
        assert!(e.clock_valid(1));
        assert!(!e.clock_valid(0));
        assert!(!e.clock_valid(2));
    }

    #[test]
    fn topology_election_round() {
        let mut topo = TopologyGraph::new();
        topo.add_or_update_node("peer-a", dummy_caps());
        topo.add_or_update_node("peer-b", dummy_caps());
        topo.nodes.get_mut("peer-a").unwrap().seniority = 3;
        topo.nodes.get_mut("peer-b").unwrap().seniority = 7;

        let mut e = make_election("local", 1);
        let result = run_election_round(&mut e, &topo);
        assert_eq!(
            result.unwrap(),
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "peer-b".to_string(),
            }
        );
    }

    #[test]
    fn empty_election_local_wins() {
        let mut e = make_election("solo", 1);
        let result = run_election_round(&mut e, &TopologyGraph::new());
        assert_eq!(
            result.unwrap(),
            ElectionMessage::Result {
                clock: 1,
                master_peer_id: "solo".to_string(),
            }
        );
    }

    #[test]
    fn election_message_serialize_roundtrip() {
        let msg = ElectionMessage::Declare {
            clock: 7,
            peer_id: "p".to_string(),
            seniority: 3,
            commands_seen: 12,
            capabilities: dummy_caps(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ElectionMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn inc_local_commands_updates_priority() {
        let mut e = make_election("a", 1);
        e.inc_local_commands();
        e.inc_local_commands();
        assert_eq!(e.local_commands, 2);
    }

    #[test]
    fn time_remaining_when_electing() {
        let mut e = make_election("a", 1);
        assert!(e.time_remaining().is_none());
        e.start_election();
        assert!(e.time_remaining().is_some());
        let rem = e.time_remaining().unwrap();
        assert!(rem > Duration::from_secs(0));
        assert!(rem <= Duration::from_millis(100));
    }

    #[test]
    fn time_remaining_zero_after_timeout() {
        let mut e = make_election("a", 1);
        e.start_election();
        // Manually set deadline to the past.
        if let ElectionState::Electing {
            ref mut deadline, ..
        } = e.state
        {
            *deadline = std::time::Instant::now() - Duration::from_secs(1);
        }
        let rem = e.time_remaining().unwrap();
        assert_eq!(rem, Duration::from_secs(0));
    }
}
