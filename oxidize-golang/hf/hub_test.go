package hf

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestSplitRepoAndFile(t *testing.T) {
	repo, file, err := splitRepoAndFile("org/model/Q4.gguf", "")
	if err != nil {
		t.Fatal(err)
	}
	if repo != "org/model" || file != "Q4.gguf" {
		t.Fatalf("got %q %q", repo, file)
	}
}

func TestResolveGGUFUsesCache(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/demo/tiny":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"tiny.gguf"}]}`))
		case "/demo/tiny/resolve/main/tiny.gguf":
			_, _ = w.Write([]byte("GGUF"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cache := t.TempDir()
	opts := ResolveOptions{Repo: "demo/tiny", CacheDir: cache, APIBase: srv.URL, CDNBase: srv.URL}
	path1, err := ResolveGGUF(opts)
	if err != nil {
		t.Fatal(err)
	}
	path2, err := ResolveGGUF(opts)
	if err != nil {
		t.Fatal(err)
	}
	if path1 != path2 {
		t.Fatalf("cache miss: %q vs %q", path1, path2)
	}
	data, err := os.ReadFile(path2)
	if err != nil || string(data) != "GGUF" {
		t.Fatalf("cached content = %q err=%v", data, err)
	}
}

func TestPickSingleGGUFMultiple(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"a.gguf"},{"rfilename":"b.gguf"}]}`))
	}))
	defer srv.Close()
	_, err := pickSingleGGUF("x/y", "main", srv.Client(), srv.URL)
	if err == nil {
		t.Fatal("expected error for multiple gguf files")
	}
}

func TestListGGUFFilesSorts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"z.gguf"},{"rfilename":"a.gguf"}]}`))
	}))
	defer srv.Close()
	names, err := ListGGUFFiles("m", "main", srv.Client(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 || names[0] != "a.gguf" {
		t.Fatalf("got %v", names)
	}
}
