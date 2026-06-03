package mlxbackend

import "testing"

func TestBuildInfo(t *testing.T) {
	if Info().DetectedAtBuild {
		t.Fatal("stub: mlx should not be detected")
	}
}
