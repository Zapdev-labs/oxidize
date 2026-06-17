package cudabackend

import (
	"os/exec"
	"strings"
)

// gpuPresent returns true when nvidia-smi reports at least one GPU.
func gpuPresent() bool {
	out, err := exec.Command("nvidia-smi", "-L").CombinedOutput()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "GPU ") {
			return true
		}
	}
	return false
}
