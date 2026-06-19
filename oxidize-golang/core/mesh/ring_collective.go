package mesh

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
)

// RingError mirrors RingError. It captures the failure modes of a ring
// collective operation.
type RingError struct {
	Kind    string
	Message string
}

func (e *RingError) Error() string {
	if e.Message != "" {
		return "ring: " + e.Kind + ": " + e.Message
	}
	return "ring: " + e.Kind
}

func errWrongChunkSize(expected, actual int) *RingError {
	return &RingError{Kind: "wrong_chunk_size", Message: fmt.Sprintf("expected multiple of %d, got remainder %d", expected, actual)}
}

func errByteLengthMismatch(expected, actual int) *RingError {
	return &RingError{Kind: "byte_length_mismatch", Message: fmt.Sprintf("expected %d bytes, got %d", expected, actual)}
}

// RingNodeTransport is the abstract ring transport: each rank sends to its
// right neighbour and receives from its left neighbour. It mirrors the Rust
// RingTransport trait. Context cancellation aborts a blocked send/recv.
type RingNodeTransport interface {
	SendToRight(ctx context.Context, data []byte) error
	RecvFromLeft(ctx context.Context) ([]byte, error)
}

// RingBackend mirrors RingBackend. It performs ring all-reduce (AllSum) and
// ring all-gather (AllGather) over a RingNodeTransport.
type RingBackend struct {
	Rank      int
	NumRanks  int
	Transport RingNodeTransport
}

// NewRingBackend constructs a ring backend for a given rank.
func NewRingBackend(rank, numRanks int, transport RingNodeTransport) *RingBackend {
	return &RingBackend{Rank: rank, NumRanks: numRanks, Transport: transport}
}

// AllSum performs ring all-reduce, summing data in-place so every rank ends up
// with the same element-wise sum. len(data) must be divisible by NumRanks.
func (r *RingBackend) AllSum(ctx context.Context, data []float32) error {
	if r.NumRanks <= 1 {
		return nil
	}
	if len(data)%r.NumRanks != 0 {
		return errWrongChunkSize(r.NumRanks, len(data)%r.NumRanks)
	}

	chunkSize := len(data) / r.NumRanks
	sendBuf := make([]float32, chunkSize)
	recvF32 := make([]float32, chunkSize)

	// Scatter-reduce: N-1 steps.
	sendChunk := r.Rank
	recvChunk := (r.Rank + r.NumRanks - 1) % r.NumRanks
	for step := 0; step < r.NumRanks-1; step++ {
		sendOff := sendChunk * chunkSize
		copy(sendBuf, data[sendOff:sendOff+chunkSize])
		recvBytes, err := r.exchange(ctx, f32SliceToBytes(sendBuf))
		if err != nil {
			return err
		}
		if err := bytesToF32SliceInto(recvBytes, recvF32); err != nil {
			return err
		}
		recvOff := recvChunk * chunkSize
		for i := 0; i < chunkSize; i++ {
			data[recvOff+i] += recvF32[i]
		}
		sendChunk = (sendChunk + r.NumRanks - 1) % r.NumRanks
		recvChunk = (recvChunk + r.NumRanks - 1) % r.NumRanks
	}

	// All-gather: N-1 steps.
	sendChunk = (r.Rank + 1) % r.NumRanks
	recvChunk = r.Rank
	for step := 0; step < r.NumRanks-1; step++ {
		sendOff := sendChunk * chunkSize
		copy(sendBuf, data[sendOff:sendOff+chunkSize])
		recvBytes, err := r.exchange(ctx, f32SliceToBytes(sendBuf))
		if err != nil {
			return err
		}
		if err := bytesToF32SliceInto(recvBytes, recvF32); err != nil {
			return err
		}
		recvOff := recvChunk * chunkSize
		copy(data[recvOff:recvOff+chunkSize], recvF32)
		sendChunk = (sendChunk + r.NumRanks - 1) % r.NumRanks
		recvChunk = (recvChunk + r.NumRanks - 1) % r.NumRanks
	}

	return nil
}

// AllGather performs ring all-gather: each rank contributes one chunk (data)
// and receives the concatenation of all chunks into out. len(out) must equal
// len(data) * NumRanks.
func (r *RingBackend) AllGather(ctx context.Context, data []float32, out []float32) error {
	if r.NumRanks <= 1 {
		copy(out, data)
		return nil
	}
	chunkSize := len(data)
	if len(out) != chunkSize*r.NumRanks {
		return errWrongChunkSize(chunkSize*r.NumRanks, len(out))
	}

	sendBuf := make([]float32, chunkSize)
	recvF32 := make([]float32, chunkSize)

	localOffset := r.Rank * chunkSize
	copy(out[localOffset:localOffset+chunkSize], data)

	sendChunk := r.Rank
	recvChunk := (r.Rank + r.NumRanks - 1) % r.NumRanks
	for step := 0; step < r.NumRanks-1; step++ {
		sendOff := sendChunk * chunkSize
		copy(sendBuf, out[sendOff:sendOff+chunkSize])
		recvBytes, err := r.exchange(ctx, f32SliceToBytes(sendBuf))
		if err != nil {
			return err
		}
		if err := bytesToF32SliceInto(recvBytes, recvF32); err != nil {
			return err
		}
		recvOff := recvChunk * chunkSize
		copy(out[recvOff:recvOff+chunkSize], recvF32)
		sendChunk = (sendChunk + r.NumRanks - 1) % r.NumRanks
		recvChunk = (recvChunk + r.NumRanks - 1) % r.NumRanks
	}

	return nil
}

// exchange performs a concurrent send-to-right / recv-from-left, mirroring the
// Rust tokio::join! pattern so the ring does not deadlock.
func (r *RingBackend) exchange(ctx context.Context, send []byte) ([]byte, error) {
	type recvResult struct {
		data []byte
		err  error
	}
	recvCh := make(chan recvResult, 1)
	go func() {
		data, err := r.Transport.RecvFromLeft(ctx)
		recvCh <- recvResult{data, err}
	}()
	sendErr := r.Transport.SendToRight(ctx, send)
	res := <-recvCh
	if sendErr != nil {
		return nil, sendErr
	}
	if res.err != nil {
		return nil, res.err
	}
	return res.data, nil
}

// PipelineSend forwards activations to the next pipeline stage (right
// neighbour). Mirrors pipeline_send.
func PipelineSend(ctx context.Context, ring *RingBackend, activations []float32) error {
	return ring.Transport.SendToRight(ctx, f32SliceToBytes(activations))
}

// PipelineRecv receives activations from the previous pipeline stage (left
// neighbour). Mirrors pipeline_recv.
func PipelineRecv(ctx context.Context, ring *RingBackend, numFloats int) ([]float32, error) {
	bytes, err := ring.Transport.RecvFromLeft(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]float32, numFloats)
	if err := bytesToF32SliceInto(bytes, out); err != nil {
		return nil, err
	}
	return out, nil
}

// TensorParallelAllSum performs a tensor-parallel all_sum over the ring.
// Mirrors tensor_parallel_all_sum.
func TensorParallelAllSum(ctx context.Context, ring *RingBackend, partial []float32) error {
	return ring.AllSum(ctx, partial)
}

// TensorParallelAllGather gathers outputs from all ranks. Mirrors
// tensor_parallel_all_gather.
func TensorParallelAllGather(ctx context.Context, ring *RingBackend, partial, out []float32) error {
	return ring.AllGather(ctx, partial, out)
}

// f32SliceToBytes converts a slice of float32 into little-endian bytes,
// matching the Rust framing.
func f32SliceToBytes(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// bytesToF32SliceInto converts little-endian bytes into a pre-allocated
// float32 slice.
func bytesToF32SliceInto(bytes []byte, out []float32) error {
	if len(bytes) != len(out)*4 {
		return errByteLengthMismatch(len(out)*4, len(bytes))
	}
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
	}
	return nil
}

// bytesToF32Slice converts little-endian bytes into a freshly allocated
// float32 slice.
func bytesToF32Slice(bytes []byte) ([]float32, error) {
	if len(bytes)%4 != 0 {
		return nil, errByteLengthMismatch((len(bytes)/4+1)*4, len(bytes))
	}
	out := make([]float32, len(bytes)/4)
	if err := bytesToF32SliceInto(bytes, out); err != nil {
		return nil, err
	}
	return out, nil
}

// ChannelRingTransport is an in-memory ring transport used for tests, mirroring
// the Rust ChannelTransport.
type ChannelRingTransport struct {
	RightTx chan<- []byte
	LeftRx  <-chan []byte
}

// SendToRight pushes data onto the right neighbour's inbound channel.
func (c *ChannelRingTransport) SendToRight(ctx context.Context, data []byte) error {
	select {
	case c.RightTx <- data:
		return nil
	case <-ctx.Done():
		return &RingError{Kind: "io", Message: ctx.Err().Error()}
	}
}

// RecvFromLeft pops data from the left neighbour's channel.
func (c *ChannelRingTransport) RecvFromLeft(ctx context.Context) ([]byte, error) {
	select {
	case data, ok := <-c.LeftRx:
		if !ok {
			return nil, &RingError{Kind: "io", Message: "channel closed"}
		}
		return data, nil
	case <-ctx.Done():
		return nil, &RingError{Kind: "io", Message: ctx.Err().Error()}
	}
}

// CreateMockRing builds a mock ring of numRanks backends connected via
// channels. Mirrors create_mock_ring.
func CreateMockRing(numRanks int) []*RingBackend {
	channels := make([]chan []byte, numRanks)
	for i := range channels {
		channels[i] = make(chan []byte, 64)
	}
	backends := make([]*RingBackend, numRanks)
	for rank := 0; rank < numRanks; rank++ {
		leftRank := (rank + numRanks - 1) % numRanks
		transport := &ChannelRingTransport{
			// rank sends to its right neighbour's inbound channel.
			RightTx: channels[(rank+1)%numRanks],
			// rank receives from its own inbound channel (fed by its left).
			LeftRx: channels[rank],
		}
		_ = leftRank
		backends[rank] = NewRingBackend(rank, numRanks, transport)
	}
	return backends
}
