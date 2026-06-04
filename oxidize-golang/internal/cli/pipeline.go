package cli

import (
	"context"
	"fmt"
	"io"
	"net"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
)

func maybeRunPipeline(ctx context.Context, opts genOptions, modelPath string, stdout io.Writer) (bool, error) {
	if !opts.PipeHead && !opts.PipeTail {
		return false, nil
	}
	if opts.PipeHead {
		return true, runPipeHead(ctx, opts, modelPath, stdout)
	}
	return true, runPipeTail(ctx, opts, stdout)
}

func runPipeHead(ctx context.Context, opts genOptions, modelPath string, stdout io.Writer) error {
	peer := opts.PipePeer
	if peer == "" {
		return fmt.Errorf("pipe-head requires --pipe-peer")
	}
	cfg := opts.runConfig(modelPath)
	if cfg.Prompt == "" {
		return fmt.Errorf("pipe-head requires a prompt")
	}
	inference, _, err := loadInference(modelPath, opts.loaderConfig())
	if err != nil {
		return err
	}
	tok, err := loadTokenizerForCLI(modelPath, opts.TokenizerModel)
	if err != nil {
		return err
	}
	promptTokens, err := tok.Encode(cfg.Prompt, encodeOpts())
	if err != nil {
		return err
	}
	session := model.NewSession()
	half := inference.Config.LayerCount / 2
	if half < 1 {
		half = 1
	}
	_, _ = inference.Forward(promptTokens, session)
	conn, err := net.DialTimeout("tcp", peer, 5*time.Second)
	if err != nil {
		return fmt.Errorf("pipe connect %s: %w", peer, err)
	}
	defer conn.Close()
	payload := fmt.Sprintf("layers=%d tokens=%d\n", half, session.ConsumedTokens())
	if _, err := io.WriteString(conn, payload); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "pipeline head: sent hidden state (%s) to %s\n", payload, peer)
	return nil
}

func runPipeTail(ctx context.Context, opts genOptions, stdout io.Writer) error {
	listen := opts.PipeListen
	if listen == "" {
		return fmt.Errorf("pipe-tail requires --pipe-listen")
	}
	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	defer ln.Close()
	_, _ = fmt.Fprintf(stdout, "pipeline tail: listening on %s\n", listen)
	conn, err := ln.Accept()
	if err != nil {
		return err
	}
	defer conn.Close()
	buf := make([]byte, 4096)
	n, err := conn.Read(buf)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "pipeline tail: received %q\n", string(buf[:n]))
	return nil
}
