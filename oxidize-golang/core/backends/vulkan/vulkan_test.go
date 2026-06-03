package vulkanbackend

import "testing"

func TestBuildInfo(t *testing.T) {
	if Info().DetectedAtBuild {
		t.Fatal("stub: vulkan should not be detected")
	}
}

func TestClassifyDevice(t *testing.T) {
	cases := []struct {
		vendor, device uint32
		want           DeviceClass
	}{
		{0x8086, 0x4900, DeviceIntelArc},
		{0x8086, 0x7D40, DeviceIntelArc},
		{0x8086, 0x1234, DeviceIntelIntegrated},
		{0x10DE, 0x1234, DeviceNvidia},
		{0x1002, 0x1234, DeviceAmd},
		{0x0000, 0x0000, DeviceOther},
	}
	for _, c := range cases {
		if got := ClassifyDevice(c.vendor, c.device, ""); got != c.want {
			t.Fatalf("vendor=%x device=%x = %v, want %v", c.vendor, c.device, got, c.want)
		}
	}
}

func TestPlanLayerDispatch(t *testing.T) {
	plans := PlanLayerDispatch(4, 256)
	if len(plans) != 4 {
		t.Fatalf("plans = %d", len(plans))
	}
	for _, p := range plans {
		if p.Workgroups[0] == 0 {
			t.Fatalf("workgroup 0 must be non-zero")
		}
	}
}

func TestShouldUseVulkan(t *testing.T) {
	if !ShouldUseVulkanGemv(100, 100) {
		t.Fatal("should use vulkan for large gemv")
	}
	if ShouldUseVulkanGemv(1, 1) {
		t.Fatal("should not use vulkan for small gemv")
	}
}
