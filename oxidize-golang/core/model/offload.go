package model

import (
	"errors"
	"math"
	"sort"
)

// OffloadPolicy controls how a model is partitioned between CPU and GPU.
type OffloadPolicy int

// Recognised policies.
const (
	OffloadPolicyLayerMajor OffloadPolicy = iota
	OffloadPolicyRowMajor
	OffloadPolicyPipeline
)

// String returns a human-readable label.
func (p OffloadPolicy) String() string {
	switch p {
	case OffloadPolicyLayerMajor:
		return "layer-major"
	case OffloadPolicyRowMajor:
		return "row-major"
	case OffloadPolicyPipeline:
		return "pipeline"
	}
	return "unknown"
}

// DeviceMemory represents a single device's memory budget.
type DeviceMemory struct {
	DeviceID int
	Backend  string
	Bytes    int64
}

// LayerOffloadPlan mirrors LayerOffloadPlan.
type LayerOffloadPlan struct {
	Layers []LayerAssignment
}

// LayerAssignment describes where a single layer lives.
type LayerAssignment struct {
	LayerIndex int
	DeviceID   int
	Backend    string
	Bytes      int64
}

// TotalBytes returns the sum of bytes across all layer assignments.
func (p *LayerOffloadPlan) TotalBytes() int64 {
	var total int64
	for _, l := range p.Layers {
		total += l.Bytes
	}
	return total
}

// ByDevice groups assignments by device id.
func (p *LayerOffloadPlan) ByDevice() map[int]int64 {
	out := map[int]int64{}
	for _, l := range p.Layers {
		out[l.DeviceID] += l.Bytes
	}
	return out
}

// MultiGpuOffloadPlan mirrors MultiGpuOffloadPlan.
type MultiGpuOffloadPlan struct {
	Stages    []PipelineStage
	Policy    OffloadPolicy
	TotalBytes int64
}

// PipelineStage mirrors PipelineStage.
type PipelineStage struct {
	StageID     int
	LayerRange  [2]int
	DeviceID    int
	Backend     string
	Bytes       int64
	MicroBatch  int
}

// NewMultiGpuOffloadPlan creates an empty plan with the given policy.
func NewMultiGpuOffloadPlan(policy OffloadPolicy) *MultiGpuOffloadPlan {
	return &MultiGpuOffloadPlan{Policy: policy}
}

// AddStage appends a stage to the plan.
func (p *MultiGpuOffloadPlan) AddStage(stage PipelineStage) {
	if stage.MicroBatch <= 0 {
		stage.MicroBatch = 1
	}
	p.Stages = append(p.Stages, stage)
	p.TotalBytes += stage.Bytes
}

// Validate ensures the plan covers all layers without gaps.
func (p *MultiGpuOffloadPlan) Validate(totalLayers int) error {
	if len(p.Stages) == 0 {
		return errors.New("offload: no stages")
	}
	sort.Slice(p.Stages, func(i, j int) bool { return p.Stages[i].LayerRange[0] < p.Stages[j].LayerRange[0] })
	for i, s := range p.Stages {
		if s.LayerRange[0] < 0 || s.LayerRange[1] > totalLayers {
			return errors.New("offload: layer range out of bounds")
		}
		if s.LayerRange[0] >= s.LayerRange[1] {
			return errors.New("offload: empty layer range")
		}
		if i > 0 && s.LayerRange[0] != p.Stages[i-1].LayerRange[1] {
			return errors.New("offload: gap or overlap between stages")
		}
	}
	return nil
}

// LayerOffloadPlanner mirrors LayerOffloadPlanner.
type LayerOffloadPlanner struct {
	Devices []DeviceMemory
	Policy  OffloadPolicy
}

// NewLayerOffloadPlanner constructs a planner.
func NewLayerOffloadPlanner(devices []DeviceMemory, policy OffloadPolicy) *LayerOffloadPlanner {
	return &LayerOffloadPlanner{Devices: devices, Policy: policy}
}

// Plan distributes `layerCount` layers across the available devices.
func (p *LayerOffloadPlanner) Plan(layerCount int, bytesPerLayer int64) *LayerOffloadPlan {
	if len(p.Devices) == 0 || layerCount == 0 {
		return &LayerOffloadPlan{}
	}
	plan := &LayerOffloadPlan{}
	switch p.Policy {
	case OffloadPolicyLayerMajor:
		per := layerCount / len(p.Devices)
		rem := layerCount % len(p.Devices)
		idx := 0
		for d := range p.Devices {
			end := idx + per
			if d < rem {
				end++
			}
			for i := idx; i < end; i++ {
				plan.Layers = append(plan.Layers, LayerAssignment{
					LayerIndex: i,
					DeviceID:   p.Devices[d].DeviceID,
					Backend:    p.Devices[d].Backend,
					Bytes:      bytesPerLayer,
				})
			}
			idx = end
		}
	case OffloadPolicyRowMajor, OffloadPolicyPipeline:
		// Treat as a contiguous block
		for i := 0; i < layerCount; i++ {
			d := p.Devices[i%len(p.Devices)]
			plan.Layers = append(plan.Layers, LayerAssignment{
				LayerIndex: i,
				DeviceID:   d.DeviceID,
				Backend:    d.Backend,
				Bytes:      bytesPerLayer,
			})
		}
	}
	return plan
}

// GpuOffloadConfig mirrors GpuOffloadConfig.
type GpuOffloadConfig struct {
	Enabled        bool
	MaxLayersOnGpu int
	FallbackToCPU  bool
	UseAsyncCopy   bool
}

// OffloadMetrics tracks offload statistics.
type OffloadMetrics struct {
	TotalTransfers int64
	BytesTransferred int64
	AvgLatencyMicros float32
}

// AvgBytesPerTransfer returns the average bytes per transfer.
func (m OffloadMetrics) AvgBytesPerTransfer() float32 {
	if m.TotalTransfers == 0 {
		return 0
	}
	return float32(m.BytesTransferred / m.TotalTransfers)
}

// silence unused import
var _ = math.Ceil
