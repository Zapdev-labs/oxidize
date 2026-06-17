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
	addr := fmt.Sprintf("127.0.0.1:%d", opts.MeshPort)
	local := mesh.MeshNode{ID: "local", Addr: addr, Role: "worker", Healthy: true}
	rt := mesh.NewRuntime(local)
	if err := rt.StartListen(); err != nil {
		return true, fmt.Errorf("mesh listen: %w", err)
	}
	for _, peer := range strings.Split(opts.MeshPeers, ",") {
		peer = strings.TrimSpace(peer)
		if peer == "" || peer == addr {
			continue
		}
		rt.Engine.Router.Update(mesh.MeshNode{ID: peer, Addr: peer, Role: "worker", Healthy: true})
		if err := rt.Transport.Dial(peer); err != nil {
			_, _ = fmt.Fprintf(stderr, "mesh: dial %s: %v\n", peer, err)
		}
	}
	_, _ = fmt.Fprintf(stdout, "oxidize mesh chat on %s (peers=%d). type exit to quit.\n", addr, len(rt.Engine.Router.Peers()))
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
		cfgRun.Prompt = line
		text, err := rt.RouteCompletion(cfgRun.ModelPath, line, func(_, prompt string) (string, error) {
			if err := generateRun(ctx, cfgRun, stdout, stderr); err != nil {
				return "", err
			}
			return prompt, nil
		})
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "mesh generation failed: %v\n", err)
			continue
		}
		if text != "" && text != line {
			_, _ = fmt.Fprintf(stdout, "%s\n", text)
		}
		_, _ = io.WriteString(stdout, "\n")
	}
}
