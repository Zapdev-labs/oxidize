//! GossipSub topic definitions and message routing for the mesh control plane.

use libp2p::{
    gossipsub::{self, TopicHash},
    identify,
    mdns,
    swarm::NetworkBehaviour,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The six GossipSub topics used by the mesh control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TopicKind {
    GlobalEvents,
    LocalEvents,
    Commands,
    ElectionMessages,
    ConnectionMessages,
    DownloadCommands,
}

impl TopicKind {
    /// String identifier used for GossipSub topic creation.
    pub fn as_str(&self) -> &'static str {
        match self {
            TopicKind::GlobalEvents => "oxidize/mesh/global_events",
            TopicKind::LocalEvents => "oxidize/mesh/local_events",
            TopicKind::Commands => "oxidize/mesh/commands",
            TopicKind::ElectionMessages => "oxidize/mesh/election_messages",
            TopicKind::ConnectionMessages => "oxidize/mesh/connection_messages",
            TopicKind::DownloadCommands => "oxidize/mesh/download_commands",
        }
    }

    /// All six topics.
    pub fn all() -> [TopicKind; 6] {
        [
            TopicKind::GlobalEvents,
            TopicKind::LocalEvents,
            TopicKind::Commands,
            TopicKind::ElectionMessages,
            TopicKind::ConnectionMessages,
            TopicKind::DownloadCommands,
        ]
    }
}

/// A message received on a GossipSub topic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GossipMessage {
    pub topic: TopicKind,
    pub payload: Vec<u8>,
    pub source_peer_id: Option<String>,
}

/// Combined libp2p network behaviour for mesh nodes.
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "MeshEvent")]
pub struct MeshBehaviour {
    pub mdns: mdns::tokio::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
    pub identify: identify::Behaviour,
}

/// Events emitted by [`MeshBehaviour`] into the swarm loop.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum MeshEvent {
    Mdns(mdns::Event),
    Gossipsub(gossipsub::Event),
    Identify(identify::Event),
}

impl From<mdns::Event> for MeshEvent {
    fn from(event: mdns::Event) -> Self {
        MeshEvent::Mdns(event)
    }
}

impl From<gossipsub::Event> for MeshEvent {
    fn from(event: gossipsub::Event) -> Self {
        MeshEvent::Gossipsub(event)
    }
}

impl From<identify::Event> for MeshEvent {
    fn from(event: identify::Event) -> Self {
        MeshEvent::Identify(event)
    }
}

/// Router that tracks subscriptions and routes inbound messages.
#[derive(Debug, Default)]
pub struct GossipRouter {
    /// Map from topic hash to the known [`TopicKind`].
    pub topics: HashMap<TopicHash, TopicKind>,
}

impl GossipRouter {
    /// Register all six topics so inbound messages can be mapped to [`TopicKind`].
    pub fn register_all_topics(&mut self) {
        for kind in TopicKind::all() {
            let hash = gossipsub::IdentTopic::new(kind.as_str()).hash();
            self.topics.insert(hash, kind);
        }
    }

    /// Number of registered topics.
    pub fn topic_count(&self) -> usize {
        self.topics.len()
    }

    /// Map a GossipSub topic hash to our [`TopicKind`], if known.
    pub fn resolve(&self, hash: &TopicHash) -> Option<TopicKind> {
        self.topics.get(hash).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_kind_all_returns_six() {
        assert_eq!(TopicKind::all().len(), 6);
    }

    #[test]
    fn topic_kind_as_str_unique() {
        let mut seen = std::collections::HashSet::new();
        for t in TopicKind::all() {
            assert!(seen.insert(t.as_str()));
        }
    }

    #[test]
    fn gossip_router_registers_all_six_topics() {
        let mut router = GossipRouter::default();
        router.register_all_topics();
        assert_eq!(router.topic_count(), 6);
    }

    #[test]
    fn gossip_router_resolves_known_topic() {
        let mut router = GossipRouter::default();
        router.register_all_topics();
        let hash = gossipsub::IdentTopic::new(TopicKind::Commands.as_str()).hash();
        assert_eq!(router.resolve(&hash), Some(TopicKind::Commands));
    }

    #[test]
    fn gossip_router_resolves_unknown_to_none() {
        let router = GossipRouter::default();
        let hash = gossipsub::IdentTopic::new("unknown/topic").hash();
        assert_eq!(router.resolve(&hash), None);
    }

    #[test]
    fn gossip_message_serializes_roundtrip() {
        let msg = GossipMessage {
            topic: TopicKind::ElectionMessages,
            payload: vec![1, 2, 3],
            source_peer_id: Some("12D3KooW".to_string()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: GossipMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn mesh_event_from_variants() {
        // Just exercise the From impls compile and execute.
        let mdns_event = MeshEvent::from(mdns::Event::Discovered(vec![]));
        assert!(matches!(mdns_event, MeshEvent::Mdns(_)));
    }
}
