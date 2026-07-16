// oxidize-serve: OpenAI-compatible HTTP server over the oxidize-c C ABI.
//
//	oxidize-serve --model path.gguf [--host H --port P]
//	oxidize-serve <hf-repo-id-or-path>        (positional; HF ids are resolved)
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
)

func main() {
	model := flag.String("model", "", "GGUF path or Hugging Face repo id (also accepted positionally)")
	host := flag.String("host", "127.0.0.1", "listen host")
	port := flag.Int("port", 8080, "listen port")
	ctx := flag.Int("ctx", 0, "context length (0 = model default)")
	threads := flag.Int("threads", 0, "worker threads (0 = one per CPU)")
	id := flag.String("id", "", "model id reported by /v1/models (default: architecture)")
	flag.Parse()

	arg := *model
	if arg == "" && flag.NArg() > 0 {
		arg = flag.Arg(0)
	}
	if arg == "" {
		fmt.Fprintln(os.Stderr, "usage: oxidize-serve --model <path.gguf|hf-repo-id> [--host H --port P]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	path := arg
	if isHFRepoID(arg) {
		p, err := ResolveGGUF(ResolveOptions{Repo: arg})
		if err != nil {
			log.Fatalf("oxidize-serve: %v", err)
		}
		path = p
	}

	log.Printf("oxidize-serve: loading %s (%s, isa=%s)", path, Version(), ISA())
	m, err := OpenModel(path, *id, *ctx, *threads)
	if err != nil {
		log.Fatalf("oxidize-serve: %v", err)
	}
	defer m.Close()
	md := m.Meta()
	log.Printf("oxidize-serve: model %q arch=%s vocab=%d ctx=%d tensors=%d",
		m.ID(), md.Arch, md.Vocab, md.Ctx, md.NTensors)

	addr := net.JoinHostPort(*host, strconv.Itoa(*port))
	log.Printf("oxidize-serve: listening on http://%s (POST /v1/chat/completions, /v1/completions; conversation KV via conversation= or conversation_auto)", addr)
	if err := http.ListenAndServe(addr, NewServer(m).Handler()); err != nil {
		log.Fatalf("oxidize-serve: %v", err)
	}
}
