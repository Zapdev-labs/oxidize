package validation

import "testing"

func TestImplementedSuites(t *testing.T) {
	if len(ImplementedSuites()) == 0 {
		t.Fatal("expected at least one suite")
	}
}

func TestRunnerRun(t *testing.T) {
	r := NewRunner()
	r.Enable(SuiteForward)
	r.Enable(SuiteSampling)
	rep := r.Run()
	if rep.Total != 2 || rep.Passed != 2 {
		t.Fatalf("unexpected report: %+v", rep)
	}
}

func TestProbeRegistration(t *testing.T) {
	RegisterProbe(SuiteForward, func() error { return nil })
	if err := RunProbe(SuiteForward); err != nil {
		t.Fatal(err)
	}
}
