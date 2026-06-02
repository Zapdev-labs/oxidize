package buildinfo

import "testing"

func TestConstants(t *testing.T) {
	if Name != "oxidize-go" {
		t.Fatalf("Name = %q", Name)
	}
	if ModulePath != "github.com/Zapdev-labs/oxidize/golang" {
		t.Fatalf("ModulePath = %q", ModulePath)
	}
	if Version == "" {
		t.Fatal("Version must not be empty")
	}
}
