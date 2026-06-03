package metalbackend

import "testing"

func TestBuildInfo(t *testing.T) {
	info := Info()
	if info.DetectedAtBuild {
		t.Fatal("stub: metal should not be detected")
	}
}

func TestShouldUseMps(t *testing.T) {
	if !ShouldUseMpsGemv(100, 100) {
		t.Fatal("100x100 should exceed the threshold")
	}
	if ShouldUseMpsGemv(1, 1) {
		t.Fatal("1x1 should not exceed the threshold")
	}
}
