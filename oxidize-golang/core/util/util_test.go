package util

import "testing"

func TestWebWorkerRoundTrip(t *testing.T) {
	req := WebWorkerRequest{Type: "generate", Payload: []byte(`{"x":1}`)}
	data, err := EncodeWebWorkerRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := DecodeWebWorkerRequest(data)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Type != "generate" {
		t.Fatalf("unexpected type: %s", dec.Type)
	}
}

func TestSummarise(t *testing.T) {
	results := []Result{{Name: "a", TokensPerSec: 10}, {Name: "b", TokensPerSec: 20}, {Name: "c", TokensPerSec: 30}, {Name: "d", TokensPerSec: 40}}
	s := Summarise(results)
	if s.Mean != 25 {
		t.Fatalf("expected mean 25, got %v", s.Mean)
	}
	if s.P95 <= s.Median {
		t.Fatalf("expected p95 > median, got %v vs %v", s.P95, s.Median)
	}
}

func TestPipeline(t *testing.T) {
	p := NewPipeline()
	p.Add(PipelineStep{Name: "double", Apply: func(in any) (any, error) { return in.(int) * 2, nil }, Enabled: true})
	p.Add(PipelineStep{Name: "addone", Apply: func(in any) (any, error) { return in.(int) + 1, nil }, Enabled: true})
	out, err := p.Run(2)
	if err != nil {
		t.Fatal(err)
	}
	if out.(int) != 5 {
		t.Fatalf("expected 5, got %v", out)
	}
}

func TestPipelineSkipsDisabled(t *testing.T) {
	p := NewPipeline()
	p.Add(PipelineStep{Name: "skip", Enabled: false, Apply: func(in any) (any, error) { return in, nil }})
	out, err := p.Run(7)
	if err != nil {
		t.Fatal(err)
	}
	if out.(int) != 7 {
		t.Fatalf("expected 7, got %v", out)
	}
}
