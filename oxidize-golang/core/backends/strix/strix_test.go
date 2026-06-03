package strixbackend

import "testing"

func TestDefaultProfile(t *testing.T) {
	p := DefaultProfile()
	if p.Mode != ModeHybrid {
		t.Fatalf("mode = %v", p.Mode)
	}
	if !p.LazyLoading {
		t.Fatal("lazy loading should be true")
	}
}

func TestShouldLazyLoadLayer(t *testing.T) {
	if !ShouldLazyLoadLayer(10, 5) {
		t.Fatal("layer 10 with 5 resident should lazy load")
	}
	if ShouldLazyLoadLayer(1, 5) {
		t.Fatal("layer 1 with 5 resident should be resident")
	}
}

func TestRdna35WorkgroupSize(t *testing.T) {
	cases := map[int]uint32{
		1024:  64,
		4096:  128,
		16384: 256,
	}
	for hidden, want := range cases {
		if got := Rdna35WorkgroupSize(hidden); got != want {
			t.Fatalf("hidden=%d = %d, want %d", hidden, got, want)
		}
	}
}
