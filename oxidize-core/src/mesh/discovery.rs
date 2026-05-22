//! libp2p peer discovery with mDNS and namespace isolation.

use futures_util::StreamExt;
use libp2p::{
    gossipsub,
    identify,
    identity::Keypair,
    mdns,
    swarm::Swarm,
    PeerId, Transport,
};
use libp2p::core::upgrade::Version;
use libp2p::noise;
use libp2p::tcp::tokio::Transport as TokioTcpTransport;
use libp2p::yamux;
use serde::{Deserialize, Serialize};

use super::node::{MeshConfig, NodeCapabilities};

/// Events emitted by the discovery layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryEvent {
    Discovered {
        peer_id: PeerId,
        address: libp2p::Multiaddr,
        capabilities: NodeCapabilities,
        namespace: String,
    },
    Expired {
        peer_id: PeerId,
    },
}

/// Serialized payload attached to mDNS TXT records / identify protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryPayload {
    pub namespace: String,
    pub capabilities: NodeCapabilities,
}

/// Builds a libp2p [`Keypair`] and derived [`PeerId`] for this node.
pub fn generate_identity() -> (Keypair, PeerId) {
    let keypair = Keypair::generate_ed25519();
    let peer_id = PeerId::from(keypair.public());
    (keypair, peer_id)
}

/// Checks whether two nodes belong to the same namespace.
pub fn same_namespace(a: &str, b: &str) -> bool {
    a == b
}

/// Discovery service wrapping a libp2p swarm with mDNS.
pub struct DiscoveryService {
    pub local_peer_id: PeerId,
    pub namespace: String,
}

impl DiscoveryService {
    pub fn new(peer_id: PeerId, namespace: String) -> Self {
        Self {
            local_peer_id: peer_id,
            namespace,
        }
    }

    /// Build the discovery payload for this node.
    pub fn payload(&self, capabilities: &NodeCapabilities) -> DiscoveryPayload {
        DiscoveryPayload {
            namespace: self.namespace.clone(),
            capabilities: capabilities.clone(),
        }
    }

    /// Filter a peer payload: returns `true` if the peer is in the same namespace.
    pub fn accept_peer(&self, payload: &DiscoveryPayload) -> bool {
        same_namespace(&self.namespace, &payload.namespace)
    }
}

/// Creates a libp2p swarm configured for mesh use.
///
/// The swarm enables TCP + Noise + Yamux and mDNS v2 for local discovery.
/// Topics are namespaced so that different namespaces cannot see each other's messages.
pub fn build_swarm(
    keypair: &Keypair,
    namespace: &str,
    agent_version: String,
) -> Result<
    Swarm<crate::mesh::gossip::MeshBehaviour>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    use libp2p::swarm::Config as SwarmConfig;

    let peer_id = PeerId::from(keypair.public());

    // TCP + Noise + Yamux
    let noise_config = noise::Config::new(keypair)?;
    let transport = TokioTcpTransport::new(libp2p::tcp::Config::default().nodelay(true))
        .upgrade(Version::V1)
        .authenticate(noise_config)
        .multiplex(yamux::Config::default())
        .boxed();

    // GossipSub
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .max_transmit_size(2usize.pow(20)) // 1 MiB
        .validate_messages()
        .build()
        .map_err(|e| format!("gossipsub config: {e}"))?;

    let mut behaviour = crate::mesh::gossip::MeshBehaviour {
        mdns: mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?,
        gossipsub: gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_config,
        )?,
        identify: libp2p::identify::Behaviour::new(
            libp2p::identify::Config::new(
                "/oxidize/mesh/0.1.0".to_string(),
                keypair.public(),
            )
            .with_agent_version(agent_version),
        ),
    };

    // Subscribe to all 6 topics under the given namespace
    for topic in crate::mesh::gossip::TopicKind::all() {
        let t = gossipsub::IdentTopic::new(topic.topic_name(namespace));
        behaviour.gossipsub.subscribe(&t)?;
    }

    let swarm = Swarm::new(
        transport,
        behaviour,
        peer_id,
        SwarmConfig::with_tokio_executor()
            .with_idle_connection_timeout(std::time::Duration::from_secs(60)),
    );

    Ok(swarm)
}

/// Build a future that resolves on the first shutdown signal (Ctrl-C or SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let sigterm = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let sigterm = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = sigterm => {},
    }
}

/// Publish a serializable payload on a mesh topic, wrapping it in a
/// [`MeshEnvelope`] tagged with the given election clock.
fn publish_envelope<T: serde::Serialize>(
    swarm: &mut Swarm<crate::mesh::gossip::MeshBehaviour>,
    namespace: &str,
    kind: crate::mesh::gossip::TopicKind,
    clock: u64,
    payload: &T,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data = crate::mesh::gossip::MeshEnvelope::pack(clock, payload)?;
    let topic = gossipsub::IdentTopic::new(kind.topic_name(namespace));
    let _ = swarm.behaviour_mut().gossipsub.publish(topic, data);
    Ok(())
}

/// Run a mesh node: build swarm, listen on `mesh_port`, drive the event loop,
/// converge on a leader via Bully election, and gracefully shut down on
/// SIGINT / SIGTERM.
///
/// The node:
/// 1. Starts an election when ≥2 peers are discovered.
/// 2. Broadcasts `ElectionMessage::Declare` on `ELECTION_MESSAGES`.
/// 3. Waits for the election timeout (default 3 s), then finalizes.
/// 4. Broadcasts `ElectionMessage::Result` so every peer converges.
/// 5. Invalidates the GossipSub session so stale events are dropped.
/// 6. On shutdown, emits a `CONNECTION_MESSAGES` disconnect and closes the swarm.
pub async fn run_mesh_node(mesh_port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use libp2p::swarm::SwarmEvent;
    use libp2p::gossipsub::Event as GossipsubEvent;
    use std::time::Duration;

    let namespace = MeshConfig::default_namespace();
    let (keypair, peer_id) = generate_identity();
    let local_peer_id_str = peer_id.to_string();
    println!(
        "oxidize mesh node starting…\n  PeerId: {}\n  namespace: {}\n  mDNS discovery started",
        peer_id, namespace
    );

    let capabilities = MeshConfig::default().capabilities;
    let discovery = DiscoveryService::new(peer_id, namespace.clone());
    let payload_json = serde_json::to_string(&discovery.payload(&capabilities)).unwrap_or_default();

    let mut swarm = build_swarm(&keypair, &namespace, payload_json)?;
    let mut router = crate::mesh::gossip::GossipRouter::new(namespace.clone());
    router.register_all_topics();

    let mut topology = crate::mesh::topology::TopologyGraph::new();
    topology.local_peer_id = Some(local_peer_id_str.clone());

    let election_timeout = Duration::from_secs(3);
    let mut election = crate::mesh::election::BullyElection::new(
        local_peer_id_str.clone(),
        0,
        capabilities.clone(),
        election_timeout,
    );

    let listen_addr: std::net::SocketAddr = if mesh_port == 0 {
        "0.0.0.0:0".parse().unwrap()
    } else {
        format!("0.0.0.0:{}", mesh_port).parse().unwrap()
    };
    let multiaddr: libp2p::Multiaddr = format!(
        "/ip4/{}/tcp/{}",
        listen_addr.ip(),
        listen_addr.port()
    )
    .parse()
    .map_err(|e| format!("invalid multiaddr: {e}"))?;

    swarm.listen_on(multiaddr)?;

    let mut printed = false;
    let mut known_peers = std::collections::HashSet::<PeerId>::new();

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    // Main event loop.
    'outer: loop {
        // Only arm the periodic timer when there is actual work to do.
        let needs_timer = matches!(election.state, crate::mesh::election::ElectionState::Electing { .. })
            || (matches!(election.state, crate::mesh::election::ElectionState::Idle)
                && known_peers.len() >= 2);

        let maybe_timer = if needs_timer {
            tokio::time::sleep(Duration::from_millis(100))
        } else {
            // Sleep for a very long time when nothing needs polling.
            tokio::time::sleep(Duration::from_secs(3600))
        };
        tokio::pin!(maybe_timer);

        tokio::select! {
            _ = &mut shutdown => {
                println!("Received SIGINT/SIGTERM, shutting down…");
                break 'outer;
            }
            event = swarm.select_next_some() => {
                match &event {
                    SwarmEvent::NewListenAddr { address, .. } if !printed => {
                        println!("  listening on: {}", address);
                        printed = true;
                        println!("mesh node ready — waiting for peers (Ctrl-C to exit)");
                    }
                    SwarmEvent::Behaviour(b) => match b {
                        crate::mesh::gossip::MeshEvent::Mdns(mdns::Event::Discovered(list)) => {
                            for (discovered_peer, addr) in list {
                                if *discovered_peer == discovery.local_peer_id {
                                    continue;
                                }
                                let _ = swarm.dial(addr.clone());
                            }
                        }
                        crate::mesh::gossip::MeshEvent::Mdns(mdns::Event::Expired(list)) => {
                            for (expired_peer, _addr) in list {
                                if known_peers.remove(expired_peer) {
                                    let pid = expired_peer.to_string();
                                    topology.remove_node(&pid);
                                    println!("Peer {} expired / disconnected", pid);
                                }
                            }
                        }
                        crate::mesh::gossip::MeshEvent::Identify(identify::Event::Received { peer_id: remote_peer, info, .. }) => {
                            let protocol_version = &info.protocol_version;
                            if !known_peers.contains(remote_peer)
                                && protocol_version.starts_with("/oxidize/mesh/")
                                && let Ok(payload) = serde_json::from_str::<DiscoveryPayload>(&info.agent_version)
                                && discovery.accept_peer(&payload)
                            {
                                known_peers.insert(*remote_peer);
                                let peer_id_str = remote_peer.to_string();
                                topology.add_or_update_node(&peer_id_str, payload.capabilities.clone());
                                println!(
                                    "Discovered peer {} (namespace match) device={} memory={}B can_shard={}",
                                    remote_peer,
                                    payload.capabilities.device_type,
                                    payload.capabilities.memory_bytes,
                                    payload.capabilities.can_shard
                                );
                            }
                        }
                        crate::mesh::gossip::MeshEvent::Gossipsub(GossipsubEvent::Message { message, .. }) => {
                            if let Some(kind) = router.resolve(&message.topic) {
                                match kind {
                                    crate::mesh::gossip::TopicKind::ElectionMessages => {
                                        if let Ok((clock, inner)) = crate::mesh::gossip::MeshEnvelope::unpack(&message.data) {
                                            if !router.accept(clock) {
                                                println!("Stale election message dropped (clock {} < active {})", clock, router.active_clock);
                                                continue;
                                            }
                                            if let Ok(msg) = serde_json::from_slice::<crate::mesh::election::ElectionMessage>(&inner) {
                                                let old_clock = election.clock;
                                                election.handle_message(&msg);
                                                if election.clock != old_clock {
                                                    router.invalidate_session(election.clock);
                                                }
                                                if let crate::mesh::election::ElectionState::Elected { clock, master } = &election.state {
                                                    println!("Election result accepted: master={} clock={}", master, clock);
                                                }
                                            }
                                        }
                                    }
                                    crate::mesh::gossip::TopicKind::ConnectionMessages => {
                                        let source = message.source.map(|p| p.to_string()).unwrap_or_default();
                                        println!("Connection message from {} ({} bytes)", source, message.data.len());
                                    }
                                    _ => {
                                        let source = message.source.map(|p| p.to_string()).unwrap_or_default();
                                        println!(
                                            "GossipSub message on {:?} from {} ({} bytes)",
                                            kind, source, message.data.len()
                                        );
                                    }
                                }
                            }
                        }
                        crate::mesh::gossip::MeshEvent::Gossipsub(GossipsubEvent::Subscribed { peer_id: remote_peer, topic }) => {
                            if let Some(kind) = router.resolve(topic) {
                                println!("Peer {} subscribed to {:?}", remote_peer, kind);
                            }
                        }
                        crate::mesh::gossip::MeshEvent::Gossipsub(GossipsubEvent::Unsubscribed { peer_id: remote_peer, topic }) => {
                            if let Some(kind) = router.resolve(topic) {
                                println!("Peer {} unsubscribed from {:?}", remote_peer, kind);
                            }
                        }
                        _ => {}
                    }
                    _ => {}
                }
            }
            _ = &mut maybe_timer => {
                // Election timeout handling.
                if let crate::mesh::election::ElectionState::Electing { .. } = election.state
                    && election.is_timed_out()
                    && let Some(result) = election.finalize_election()
                {
                    publish_envelope(&mut swarm, &namespace, crate::mesh::gossip::TopicKind::ElectionMessages, election.clock, &result)?;
                    router.invalidate_session(election.clock);
                    println!(
                        "Election finalized: master={} clock={}",
                        election.current_master().unwrap_or("?"),
                        election.clock
                    );
                }

                // Auto-trigger election when we have ≥2 peers and are idle.
                if matches!(election.state, crate::mesh::election::ElectionState::Idle)
                    && known_peers.len() >= 2
                {
                    let declare = election.start_election();
                    publish_envelope(&mut swarm, &namespace, crate::mesh::gossip::TopicKind::ElectionMessages, election.clock, &declare)?;
                    println!("Election started (clock={})", election.clock);
                }
            }
        }
    }

    // Graceful shutdown: emit disconnect and give the swarm a moment to flush.
    let disconnect = crate::mesh::gossip::GossipMessage {
        topic: crate::mesh::gossip::TopicKind::ConnectionMessages,
        payload: local_peer_id_str.clone().into_bytes(),
        source_peer_id: Some(local_peer_id_str.clone()),
    };
    if let Ok(disconnect_data) = serde_json::to_vec(&disconnect) {
        let topic = gossipsub::IdentTopic::new(
            crate::mesh::gossip::TopicKind::ConnectionMessages.topic_name(&namespace)
        );
        let _ = swarm.behaviour_mut().gossipsub.publish(topic, disconnect_data);
    }

    // Brief drain so the disconnect message has a chance to hit the wire.
    let drain_deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(drain_deadline) => break,
            _ = swarm.select_next_some() => {},
        }
    }

    println!("swarm closed");
    println!("ring disconnected");
    println!("mesh node shutdown complete");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn identity_generates_unique_peer_ids() {
        let (kp1, id1) = generate_identity();
        let (kp2, id2) = generate_identity();
        assert_ne!(id1, id2);
        assert_eq!(id1, PeerId::from(kp1.public()));
        assert_eq!(id2, PeerId::from(kp2.public()));
    }

    #[test]
    fn same_namespace_matches_exactly() {
        assert!(same_namespace("prod", "prod"));
        assert!(!same_namespace("prod", "dev"));
        assert!(same_namespace("", ""));
    }

    #[test]
    fn discovery_service_filters_cross_namespace() {
        let (_, peer_id) = generate_identity();
        let svc = DiscoveryService::new(peer_id, "prod".to_string());
        let payload_same = DiscoveryPayload {
            namespace: "prod".to_string(),
            capabilities: NodeCapabilities::default(),
        };
        let payload_diff = DiscoveryPayload {
            namespace: "dev".to_string(),
            capabilities: NodeCapabilities::default(),
        };
        assert!(svc.accept_peer(&payload_same));
        assert!(!svc.accept_peer(&payload_diff));
    }

    #[test]
    fn discovery_payload_serializes_roundtrip() {
        let payload = DiscoveryPayload {
            namespace: "test-ns".to_string(),
            capabilities: NodeCapabilities {
                device_type: "cpu".to_string(),
                memory_bytes: 16_000_000_000,
                cpu_threads: 16,
                can_shard: true,
                tags: {
                    let mut m = HashMap::new();
                    m.insert("foo".to_string(), "bar".to_string());
                    m
                },
            },
        };
        let json = serde_json::to_string(&payload).unwrap();
        let back: DiscoveryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, back);
    }
}
