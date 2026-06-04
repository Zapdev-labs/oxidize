package cli

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/mesh"
)

func maybeRunMeshChat(ctx context.Context, opts genOptions, modelPath string, stdout, stderr io.Writer) (bool, error) {
	if !opts.Mesh {
		return false, nil
	}
	_ = ctx
	local := mesh.MeshNode{ID: "local", Addr: fmt.Sprintf("127.0.0.1:%d", opts.MeshPort), Role: "worker", Healthy: true}
	engine := mesh.NewMeshChatEngine(local)
	engine.Router.Update(local)
	transport := mesh.NewTcpTransport(local.Addr)
	_ = transport
	_, _ = fmt.Fprintf(stdout, "oxidize mesh chat (gossip engine). peers=%d. type exit to quit.\n", len(engine.Router.Peers()))
	cfgRun := opts.runConfig(modelPath)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		if _, err := io.WriteString(stdout, "> "); err != nil {
			return true, err
		}
		if !scanner.Scan() {
			return true, nil
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.EqualFold(line, "exit") || strings.EqualFold(line, "quit") {
			return true, nil
		}
		for _, peer := range engine.Router.Peers() {
			if peer.ID != local.ID {
				engine.Router.Update(peer)
			}
		}
		cfgRun.Prompt = line
		if err := generateRun(ctx, cfgRun, stdout, stderr); err != nil {
			_, _ = fmt.Fprintf(stderr, "generation failed: %v\n", err)
		}
		_, _ = io.WriteString(stdout, "\n")
	}
}
