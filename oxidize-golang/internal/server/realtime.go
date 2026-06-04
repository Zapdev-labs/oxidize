package server

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
)

const websocketGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

func (a *application) realtime(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" || !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		http.Error(w, "websocket upgrade required", http.StatusBadRequest)
		return
	}
	conn, rw, err := http.NewResponseController(w).Hijack()
	if err != nil {
		http.Error(w, "websocket hijack failed", http.StatusInternalServerError)
		return
	}
	defer conn.Close()
	acceptHash := sha1.Sum([]byte(key + websocketGUID))
	_, _ = rw.WriteString("HTTP/1.1 101 Switching Protocols\r\n")
	_, _ = rw.WriteString("Upgrade: websocket\r\n")
	_, _ = rw.WriteString("Connection: Upgrade\r\n")
	_, _ = rw.WriteString("Sec-WebSocket-Accept: " + base64.StdEncoding.EncodeToString(acceptHash[:]) + "\r\n\r\n")
	if err := rw.Flush(); err != nil {
		return
	}
	ws := &websocketConn{conn: conn, rw: rw}
	_ = ws.writeJSON(map[string]any{"type": "session.created", "session": map[string]any{"modalities": []string{"text"}}})
	for {
		payload, opcode, err := ws.readFrame()
		if err != nil {
			return
		}
		if opcode == 0x8 {
			return
		}
		if opcode != 0x1 {
			continue
		}
		a.handleRealtimeEvent(ws, payload)
	}
}

func (a *application) handleRealtimeEvent(ws *websocketConn, payload []byte) {
	var event map[string]any
	if err := json.Unmarshal(payload, &event); err != nil {
		_ = ws.writeJSON(map[string]any{"type": "error", "error": map[string]any{"message": "malformed realtime event"}})
		return
	}
	switch event["type"] {
	case "session.update":
		session, _ := event["session"].(map[string]any)
		_ = ws.writeJSON(map[string]any{"type": "session.updated", "session": session})
	case "conversation.item.create":
		item, _ := event["item"].(map[string]any)
		_ = ws.writeJSON(map[string]any{"type": "conversation.item.created", "item": item})
	case "response.create":
		_ = ws.writeJSON(map[string]any{"type": "response.created", "response": map[string]any{"status": "in_progress"}})
		_ = ws.writeJSON(map[string]any{"type": "error", "error": map[string]any{"message": "no model loaded"}})
	case "response.cancel":
		_ = ws.writeJSON(map[string]any{"type": "response.done", "response": map[string]any{"status": "cancelled"}})
	default:
		_ = ws.writeJSON(map[string]any{"type": "error", "error": map[string]any{"message": "unsupported realtime event"}})
	}
}

type websocketConn struct {
	conn net.Conn
	rw   *bufio.ReadWriter
}

func (ws *websocketConn) readFrame() ([]byte, byte, error) {
	header := make([]byte, 2)
	if _, err := io.ReadFull(ws.rw, header); err != nil {
		return nil, 0, err
	}
	opcode := header[0] & 0x0f
	masked := header[1]&0x80 != 0
	length := uint64(header[1] & 0x7f)
	switch length {
	case 126:
		var ext [2]byte
		if _, err := io.ReadFull(ws.rw, ext[:]); err != nil {
			return nil, 0, err
		}
		length = uint64(binary.BigEndian.Uint16(ext[:]))
	case 127:
		var ext [8]byte
		if _, err := io.ReadFull(ws.rw, ext[:]); err != nil {
			return nil, 0, err
		}
		length = binary.BigEndian.Uint64(ext[:])
	}
	var mask [4]byte
	if masked {
		if _, err := io.ReadFull(ws.rw, mask[:]); err != nil {
			return nil, 0, err
		}
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(ws.rw, payload); err != nil {
		return nil, 0, err
	}
	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return payload, opcode, nil
}

func (ws *websocketConn) writeJSON(v any) error {
	payload, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return ws.writeText(payload)
}

func (ws *websocketConn) writeText(payload []byte) error {
	header := []byte{0x81}
	switch {
	case len(payload) < 126:
		header = append(header, byte(len(payload)))
	case len(payload) <= 65535:
		header = append(header, 126, byte(len(payload)>>8), byte(len(payload)))
	default:
		header = append(header, 127, 0, 0, 0, 0, byte(len(payload)>>24), byte(len(payload)>>16), byte(len(payload)>>8), byte(len(payload)))
	}
	if _, err := ws.conn.Write(header); err != nil {
		return err
	}
	_, err := ws.conn.Write(payload)
	return err
}
