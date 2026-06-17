package mesh

import (
	"encoding/json"
	"net/http"
	"time"
)

// Runtime routes mesh chat requests across TCP peers when configured.
type Runtime struct {
	Engine    *MeshChatEngine
	Transport *TcpTransport
	Local     MeshNode
}

// NewRuntime constructs a mesh runtime with a gossip engine and TCP transport.
func NewRuntime(local MeshNode) *Runtime {
	engine := NewMeshChatEngine(local)
	engine.Router.Update(local)
	transport := NewTcpTransport(local.Addr)
	return &Runtime{Engine: engine, Transport: transport, Local: local}
}

// StartListen binds the TCP transport for inbound mesh RPCs.
func (rt *Runtime) StartListen() error {
	if rt.Transport == nil {
		return nil
	}
	return rt.Transport.Listen()
}

// RouteCompletion executes locally or forwards to the first healthy peer.
func (rt *Runtime) RouteCompletion(model, prompt string, localGenerate func(string, string) (string, error)) (string, error) {
	if rt == nil || rt.Engine == nil {
		return "", ErrMeshUnavailable
	}
	peers := rt.Engine.Router.Peers()
	for _, peer := range peers {
		if !peer.Healthy || peer.ID == rt.Local.ID || peer.Addr == "" {
			continue
		}
		if rt.Transport == nil {
			continue
		}
		req := MeshRequest{Kind: "completion", Model: model, Prompt: prompt, NodeID: rt.Local.ID}
		payload, err := json.Marshal(req)
		if err != nil {
			continue
		}
		if err := rt.Transport.Send(peer.Addr, payload); err != nil {
			continue
		}
		if msg := rt.Transport.RecvWait(defaultMeshTimeout); msg != nil {
			var resp MeshResponse
			if json.Unmarshal(msg, &resp) == nil && resp.OK {
				return resp.Text, nil
			}
		}
	}
	if localGenerate == nil {
		return "", ErrMeshUnavailable
	}
	return localGenerate(model, prompt)
}

// HandleHTTP serves mesh RPC payloads received over TCP (called from accept loop hooks).
func (rt *Runtime) HandleHTTP(w http.ResponseWriter, model, prompt string, localGenerate func(string, string) (string, error)) {
	text, err := rt.RouteCompletion(model, prompt, localGenerate)
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"model": model,
		"choices": []map[string]any{{
			"index": 0,
			"message": map[string]any{
				"role":    "assistant",
				"content": text,
			},
			"finish_reason": "stop",
		}},
	})
}

var ErrMeshUnavailable = &meshError{Message: "mesh runtime is not configured"}

type meshError struct{ Message string }

func (e *meshError) Error() string { return e.Message }

const defaultMeshTimeout = 2 * time.Second
