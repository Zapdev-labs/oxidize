package mesh

import (
	"encoding/binary"
	"errors"
	"io"
	"net"
	"sync"
	"time"
)

const tcpReadTimeout = 30 * time.Second

// TcpTransport provides length-prefixed TCP messaging for mesh nodes.
type TcpTransport struct {
	Addr     string
	listener net.Listener
	mu       sync.Mutex
	inbox    chan []byte
	closed   bool
}

// NewTcpTransport constructs a transport bound to addr (host:port).
func NewTcpTransport(addr string) *TcpTransport {
	return &TcpTransport{Addr: addr, inbox: make(chan []byte, 64)}
}

// Listen binds and accepts inbound connections in the background.
func (t *TcpTransport) Listen() error {
	ln, err := net.Listen("tcp", t.Addr)
	if err != nil {
		return err
	}
	t.mu.Lock()
	t.listener = ln
	t.mu.Unlock()
	go t.acceptLoop(ln)
	return nil
}

// Dial connects to a remote mesh peer and reads messages into the inbox.
func (t *TcpTransport) Dial(addr string) error {
	conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return err
	}
	go t.readConn(conn)
	return nil
}

// Send writes a length-prefixed frame to addr.
func (t *TcpTransport) Send(addr string, msg []byte) error {
	conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return err
	}
	defer conn.Close()
	return writeFrame(conn, msg)
}

// Recv returns the next message or nil if none are queued.
func (t *TcpTransport) Recv() []byte {
	select {
	case m := <-t.inbox:
		return m
	default:
		return nil
	}
}

// RecvWait blocks until a message arrives or the transport closes.
func (t *TcpTransport) RecvWait(timeout time.Duration) []byte {
	select {
	case m := <-t.inbox:
		return m
	case <-time.After(timeout):
		return nil
	}
}

// Close shuts down the listener.
func (t *TcpTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	if t.listener != nil {
		return t.listener.Close()
	}
	return nil
}

func (t *TcpTransport) acceptLoop(ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			t.mu.Lock()
			closed := t.closed
			t.mu.Unlock()
			if closed {
				return
			}
			continue
		}
		go t.readConn(conn)
	}
}

func (t *TcpTransport) readConn(conn net.Conn) {
	defer conn.Close()
	for {
		_ = conn.SetReadDeadline(time.Now().Add(tcpReadTimeout))
		msg, err := readFrame(conn)
		if err != nil {
			return
		}
		select {
		case t.inbox <- msg:
		default:
		}
	}
}

func writeFrame(w io.Writer, payload []byte) error {
	if len(payload) > 1<<28 {
		return errors.New("mesh: frame too large")
	}
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(payload)))
	if _, err := w.Write(header); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func readFrame(r io.Reader) ([]byte, error) {
	var header [4]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(header[:])
	if n == 0 || n > 1<<28 {
		return nil, errors.New("mesh: invalid frame length")
	}
	payload := make([]byte, n)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}
	return payload, nil
}

// MeshRequest is a JSON mesh RPC envelope.
type MeshRequest struct {
	Kind   string `json:"kind"`
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	NodeID string `json:"node_id"`
}

// MeshResponse is returned by mesh generation routing.
type MeshResponse struct {
	OK    bool   `json:"ok"`
	Text  string `json:"text,omitempty"`
	Error string `json:"error,omitempty"`
}
