//! Integration tests for distributed mesh leader election convergence,
//! session invalidation, and graceful shutdown.

use std::collections::HashMap;
use std::time::Duration;

use oxidize_core::mesh::{BullyElection, ElectionMessage, ElectionState, GossipRouter, MeshEnvelope, NodeCapabilities};

fn dummy_caps() -> NodeCapabilities {
    NodeCapabilities {
        device_type: "cpu".to_string(),
        memory_bytes: 8_000_000_000,
        cpu_threads: 8,
        can_shard: true,
        tags: HashMap::new(),
    }
}

/// Simulate three nodes exchanging Declare messages and converging on a
/// unanimous leader within a single election round.
#[test]
fn three_node_election_converges() {
    let timeout = Duration::from_secs(3);

    let mut node_a = BullyElection::new("node-a".to_string(), 5, dummy_caps(), timeout);
    let mut node_b = BullyElection::new("node-b".to_string(), 10, dummy_caps(), timeout);
    let mut node_c = BullyElection::new("node-c".to_string(), 7, dummy_caps(), timeout);

    // Every node starts its own election round (same clock because all start at 0).
    let declare_a = node_a.start_election();
    let declare_b = node_b.start_election();
    let declare_c = node_c.start_election();

    // All nodes broadcast their declares to each other.
    for msg in [&declare_a, &declare_b, &declare_c] {
        node_a.handle_message(msg);
        node_b.handle_message(msg);
        node_c.handle_message(msg);
    }

    // Pretend timeout expired.
    if let ElectionState::Electing { ref mut deadline, .. } = node_a.state {
        *deadline = std::time::Instant::now();
    }
    if let ElectionState::Electing { ref mut deadline, .. } = node_b.state {
        *deadline = std::time::Instant::now();
    }
    if let ElectionState::Electing { ref mut deadline, .. } = node_c.state {
        *deadline = std::time::Instant::now();
    }

    let result_a = node_a.finalize_election();
    let result_b = node_b.finalize_election();
    let result_c = node_c.finalize_election();

    // All three nodes must produce the same Result.
    assert_eq!(result_a, result_b);
    assert_eq!(result_b, result_c);

    let master = result_a.as_ref().unwrap();
    assert_eq!(
        master,
        &ElectionMessage::Result {
            clock: 1,
            master_peer_id: "node-b".to_string(),
        }
    );

    // Everyone must accept the result.
    node_a.handle_message(master);
    node_b.handle_message(master);
    node_c.handle_message(master);

    assert_eq!(node_a.current_master(), Some("node-b"));
    assert_eq!(node_b.current_master(), Some("node-b"));
    assert_eq!(node_c.current_master(), Some("node-b"));
}

/// When a new leader is elected (higher clock), the router must invalidate
/// prior sessions so stale events are dropped.
#[test]
fn session_invalidation_on_new_election() {
    let mut router = GossipRouter::new("test".to_string());
    router.register_all_topics();

    // First election at clock=1.
    router.invalidate_session(1);
    assert!(router.accept(1));

    // Simulate an old message from clock=1 still in flight.
    let old_env = MeshEnvelope {
        election_clock: 1,
        payload: vec![1, 2, 3],
    };
    let (clock_old, _) = MeshEnvelope::unpack(&serde_json::to_vec(&old_env).unwrap()).unwrap();
    assert!(router.accept(clock_old));

    // New election increments clock to 2.
    router.invalidate_session(2);
    assert!(router.accept(2));
    // Stale clock=1 messages must now be rejected.
    assert!(!router.accept(clock_old));
    // Clock=0 (untagged) is still accepted.
    assert!(router.accept(0));
}

/// Simulates a re-election triggered by node departure. The remaining nodes
/// must increment the clock and drop events tagged with the old clock.
#[test]
fn re_election_invalidates_old_session() {
    let timeout = Duration::from_millis(100);

    let mut node_a = BullyElection::new("node-a".to_string(), 5, dummy_caps(), timeout);
    let mut node_b = BullyElection::new("node-b".to_string(), 10, dummy_caps(), timeout);
    let mut node_c = BullyElection::new("node-c".to_string(), 7, dummy_caps(), timeout);

    // First election.
    let d_a = node_a.start_election();
    let d_b = node_b.start_election();
    let d_c = node_c.start_election();

    for d in [&d_a, &d_b, &d_c] {
        node_a.handle_message(d);
        node_b.handle_message(d);
        node_c.handle_message(d);
    }

    // Force timeout and finalize on all nodes.
    if let ElectionState::Electing { ref mut deadline, .. } = node_a.state {
        *deadline = std::time::Instant::now();
    }
    if let ElectionState::Electing { ref mut deadline, .. } = node_b.state {
        *deadline = std::time::Instant::now();
    }
    if let ElectionState::Electing { ref mut deadline, .. } = node_c.state {
        *deadline = std::time::Instant::now();
    }

    let r1 = node_a.finalize_election().unwrap();
    node_b.finalize_election();
    node_c.finalize_election();

    // All adopt first result.
    node_a.handle_message(&r1);
    node_b.handle_message(&r1);
    node_c.handle_message(&r1);

    let old_clock = node_a.clock;
    assert_eq!(old_clock, 1);

    // Node-b leaves; remaining nodes trigger re-election.
    let d2_a = node_a.start_election();
    let d2_c = node_c.start_election();
    node_a.handle_message(&d2_a);
    node_a.handle_message(&d2_c);
    node_c.handle_message(&d2_a);
    node_c.handle_message(&d2_c);

    if let ElectionState::Electing { ref mut deadline, .. } = node_a.state {
        *deadline = std::time::Instant::now();
    }
    if let ElectionState::Electing { ref mut deadline, .. } = node_c.state {
        *deadline = std::time::Instant::now();
    }

    let _r2 = node_a.finalize_election().unwrap();
    node_c.finalize_election();

    // Clock must have incremented.
    assert!(node_a.clock > old_clock);
    assert_eq!(node_a.clock, 2);

    // Old clock events must be rejected by a router synced to the new clock.
    let mut router = GossipRouter::new("test".to_string());
    router.invalidate_session(node_a.clock);
    assert!(!router.accept(old_clock));
    assert!(router.accept(node_a.clock));
}

/// Verify that tokio::signal::ctrl_c() is a proper future that does not panic
/// when polled.  A full end-to-end SIGINT test is performed manually via
/// `run_mesh_node` integration; here we only assert type-safety and non-panic.
#[tokio::test]
async fn shutdown_signal_future_is_selectable() {
    use futures_util::FutureExt;
    // The ctrl_c future must be Send and must not panic on first poll.
    let mut fut = std::pin::pin!(tokio::signal::ctrl_c());
    // First poll registers the signal handler; it returns Pending because no
    // signal has been sent yet.  We just need to prove it does not panic.
    let _ = fut.poll_unpin(&mut std::task::Context::from_waker(
        futures_util::task::noop_waker_ref(),
    ));
}

/// Verify that a GossipRouter rejects events from a clock that is strictly
/// older than the active clock after a new election result is processed.
#[test]
fn router_drops_stale_events_after_new_election() {
    let mut router = GossipRouter::new("prod".to_string());
    router.register_all_topics();

    // Initial session at clock 3.
    router.invalidate_session(3);

    // A command arrives tagged with clock 3 — accepted.
    assert!(router.accept(3));

    // Election result with clock 5 arrives (new master elected).
    router.invalidate_session(5);

    // Old command still in flight with clock 3 — rejected.
    assert!(!router.accept(3));
    // Command tagged with clock 5 — accepted.
    assert!(router.accept(5));
    // Future command with clock 6 — accepted (router is not strict-future).
    assert!(router.accept(6));
}
