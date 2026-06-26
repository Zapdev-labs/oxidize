package mesh

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"time"
)

// maxRingFrame is the upper bound on a single ring frame (256 MiB) to reject
// corrupt length prefixes.
const maxRingFrame = 256 << 20

// FullDuplexTcpTransport mirrors DualTcpTransport: it sends on one stream and
// receives on another, using little-endian 4-byte length-prefixed framing
// (matching the Rust ring TCP wire format, which uses LE u32).
type FullDuplexTcpTransport struct {
	sendMu sync.Mutex
	recvMu sync.Mutex
	send   net.Conn
	recv   net.Conn
}

// NewFullDuplexTcpTransport wraps a send and receive connection.
func NewFullDuplexTcpTransport(send, recv net.Conn) *FullDuplexTcpTransport {
	return &FullDuplexTcpTransport{send: send, recv: recv}
}

// SendToRight writes a length-prefixed frame to the send stream.
func (t *FullDuplexTcpTransport) SendToRight(ctx context.Context, data []byte) error {
	t.sendMu.Lock()
	defer t.sendMu.Unlock()
	if dl, ok := ctx.Deadline(); ok {
		_ = t.send.SetWriteDeadline(dl)
		defer t.send.SetWriteDeadline(time.Time{})
	}
	var header [4]byte
	binary.LittleEndian.PutUint32(header[:], uint32(len(data)))
	if _, err := t.send.Write(header[:]); err != nil {
		return &RingError{Kind: "io", Message: err.Error()}
	}
	if _, err := t.send.Write(data); err != nil {
		return &RingError{Kind: "io", Message: err.Error()}
	}
	return nil
}

// RecvFromLeft reads a length-prefixed frame from the receive stream.
func (t *FullDuplexTcpTransport) RecvFromLeft(ctx context.Context) ([]byte, error) {
	t.recvMu.Lock()
	defer t.recvMu.Unlock()
	if dl, ok := ctx.Deadline(); ok {
		_ = t.recv.SetReadDeadline(dl)
		defer t.recv.SetReadDeadline(time.Time{})
	}
	var header [4]byte
	if _, err := io.ReadFull(t.recv, header[:]); err != nil {
		return nil, &RingError{Kind: "io", Message: err.Error()}
	}
	n := binary.LittleEndian.Uint32(header[:])
	if n > maxRingFrame {
		return nil, &RingError{Kind: "io", Message: "ring frame too large"}
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(t.recv, buf); err != nil {
		return nil, &RingError{Kind: "io", Message: err.Error()}
	}
	return buf, nil
}

// Close shuts down both underlying connections.
func (t *FullDuplexTcpTransport) Close() error {
	var firstErr error
	if t.send != nil {
		if err := t.send.Close(); err != nil {
			firstErr = err
		}
	}
	if t.recv != nil {
		if err := t.recv.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// tcpRing bundles backends and their transports so callers can clean up.
type tcpRing struct {
	Backends   []*RingBackend
	transports []*FullDuplexTcpTransport
}

// Close tears down all transports in the ring.
func (r *tcpRing) Close() error {
	var firstErr error
	for _, t := range r.transports {
		if err := t.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// CreateTcpRing spawns a TCP ring of numRanks backends on localhost ephemeral
// ports. Each rank opens an outbound connection to its right neighbour and
// accepts an inbound connection from its left neighbour, wrapping both in a
// FullDuplexTcpTransport. Mirrors create_tcp_ring.
func CreateTcpRing(ctx context.Context, numRanks int) ([]*RingBackend, func() error, error) {
	listeners := make([]net.Listener, numRanks)
	addrs := make([]string, numRanks)
	for i := 0; i < numRanks; i++ {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			for j := 0; j < i; j++ {
				_ = listeners[j].Close()
			}
			return nil, nil, &RingError{Kind: "io", Message: err.Error()}
		}
		listeners[i] = ln
		addrs[i] = ln.Addr().String()
	}

	recvStreams := make([]net.Conn, numRanks)
	sendStreams := make([]net.Conn, numRanks)
	errCh := make(chan error, numRanks*2)
	var wg sync.WaitGroup

	// Accept inbound connections (recv direction) per rank.
	for rank := 0; rank < numRanks; rank++ {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			conn, err := listeners[rank].Accept()
			_ = listeners[rank].Close()
			if err != nil {
				errCh <- &RingError{Kind: "io", Message: err.Error()}
				return
			}
			recvStreams[rank] = conn
		}(rank)
	}

	// Dial right neighbour (send direction) per rank, with backoff.
	for rank := 0; rank < numRanks; rank++ {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			right := addrs[(rank+1)%numRanks]
			conn, err := dialWithBackoff(ctx, right, 5, 5*time.Millisecond)
			if err != nil {
				errCh <- err
				return
			}
			sendStreams[rank] = conn
		}(rank)
	}

	wg.Wait()
	close(errCh)
	if err := <-errCh; err != nil {
		for _, c := range recvStreams {
			if c != nil {
				_ = c.Close()
			}
		}
		for _, c := range sendStreams {
			if c != nil {
				_ = c.Close()
			}
		}
		return nil, nil, err
	}

	backends := make([]*RingBackend, numRanks)
	transports := make([]*FullDuplexTcpTransport, numRanks)
	for rank := 0; rank < numRanks; rank++ {
		t := NewFullDuplexTcpTransport(sendStreams[rank], recvStreams[rank])
		transports[rank] = t
		backends[rank] = NewRingBackend(rank, numRanks, t)
	}
	ring := &tcpRing{Backends: backends, transports: transports}
	return backends, ring.Close, nil
}

// dialWithBackoff dials addr with exponential backoff and jitter, mirroring the
// perfWins requirement for retry resilience during ring bootstrap.
func dialWithBackoff(ctx context.Context, addr string, attempts int, base time.Duration) (net.Conn, error) {
	var lastErr error
	delay := base
	for i := 0; i < attempts; i++ {
		d := net.Dialer{Timeout: 5 * time.Second}
		conn, err := d.DialContext(ctx, "tcp", addr)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		// Exponential backoff with deterministic jitter (no global rand state).
		jitter := time.Duration(int64(delay) / 4)
		sleep := delay + jitter
		select {
		case <-ctx.Done():
			return nil, &RingError{Kind: "io", Message: ctx.Err().Error()}
		case <-time.After(sleep):
		}
		delay *= 2
	}
	return nil, &RingError{Kind: "io", Message: lastErr.Error()}
}
