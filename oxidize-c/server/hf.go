// Hugging Face GGUF resolver (P7b), mirrored from oxidize-golang/hf/hub.go and
// trimmed to what the server needs: repo-id -> local .gguf path, downloading to
// ~/.cache/oxidize/hf. net/http only. A local filesystem path bypasses all of
// this (see isHFRepoID).
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// isHFRepoID reports whether arg should be resolved via Hugging Face rather than
// opened as a local file. An existing path, an absolute path, or a "./" prefix
// is always local; otherwise "org/model" (optionally ".../file.gguf") is an HF id.
func isHFRepoID(arg string) bool {
	if arg == "" {
		return false
	}
	if _, err := os.Stat(arg); err == nil {
		return false // a real file/dir on disk
	}
	if filepath.IsAbs(arg) || strings.HasPrefix(arg, ".") || strings.HasPrefix(arg, "~") {
		return false
	}
	parts := strings.Split(arg, "/")
	return len(parts) >= 2 && parts[0] != "" && parts[1] != ""
}

// ResolveOptions controls Hugging Face GGUF resolution.
type ResolveOptions struct {
	Repo       string
	Revision   string
	Filename   string
	CacheDir   string
	HTTPClient *http.Client
	APIBase    string // default https://huggingface.co
	CDNBase    string // default https://huggingface.co
}

// ResolveGGUF downloads (or returns a cached) GGUF for a Hugging Face repo. Repo
// may be "org/model" or "org/model/quant.gguf" when Filename is empty.
func ResolveGGUF(opts ResolveOptions) (string, error) {
	repo, filename, err := splitRepoAndFile(opts.Repo, opts.Filename)
	if err != nil {
		return "", err
	}
	if opts.Filename != "" {
		filename = opts.Filename
	}
	if filename == "" {
		if filename, err = pickSingleGGUF(repo, opts.Revision, opts.HTTPClient, opts.APIBase); err != nil {
			return "", err
		}
	}
	rev := opts.Revision
	if rev == "" {
		rev = "main"
	}
	cache := opts.CacheDir
	if cache == "" {
		cache = defaultCacheDir()
	}
	destDir := filepath.Join(cache, strings.ReplaceAll(repo, "/", "_"))
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(destDir, filename)
	if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
		return dest, nil // cache hit
	}
	log.Printf("hf: downloading %s/%s (rev %s)", repo, filename, rev)
	tmp := dest + ".part"
	if err := downloadFile(resolveURL(opts.CDNBase, repo, rev, filename), tmp, opts.HTTPClient); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	return dest, nil
}

func splitRepoAndFile(repo, explicitFile string) (string, string, error) {
	repo = strings.TrimSpace(repo)
	if repo == "" {
		return "", "", fmt.Errorf("hf: empty repo")
	}
	if explicitFile != "" {
		return repo, explicitFile, nil
	}
	parts := strings.Split(repo, "/")
	if len(parts) >= 3 && strings.HasSuffix(strings.ToLower(parts[len(parts)-1]), ".gguf") {
		file := parts[len(parts)-1]
		return strings.Join(parts[:len(parts)-1], "/"), file, nil
	}
	return repo, "", nil
}

func defaultCacheDir() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".cache", "oxidize", "hf")
	}
	return filepath.Join(os.TempDir(), "oxidize-hf")
}

func resolveURL(base, repo, revision, filename string) string {
	if base == "" {
		base = "https://huggingface.co"
	}
	return fmt.Sprintf("%s/%s/resolve/%s/%s", strings.TrimRight(base, "/"), repo, revision, filename)
}

func pickSingleGGUF(repo, revision string, client *http.Client, apiBase string) (string, error) {
	names, err := listGGUFFiles(repo, revision, client, apiBase)
	if err != nil {
		return "", err
	}
	switch len(names) {
	case 0:
		return "", fmt.Errorf("hf: repo %q has no .gguf files", repo)
	case 1:
		return names[0], nil
	default:
		var b strings.Builder
		fmt.Fprintf(&b, "hf: repo %q has multiple .gguf files; append /<file>.gguf. Candidates:\n", repo)
		for i, n := range names {
			if i >= 25 {
				b.WriteString("  ...\n")
				break
			}
			b.WriteString("  " + n + "\n")
		}
		return "", fmt.Errorf("%s", strings.TrimSuffix(b.String(), "\n"))
	}
}

func listGGUFFiles(repo, revision string, client *http.Client, apiBase string) ([]string, error) {
	if revision == "" {
		revision = "main"
	}
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}
	if apiBase == "" {
		apiBase = "https://huggingface.co"
	}
	url := fmt.Sprintf("%s/api/models/%s", strings.TrimRight(apiBase, "/"), repo)
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("hf: list %q: %w", repo, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("hf: list %q: HTTP %d: %s", repo, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var info struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("hf: decode model info: %w", err)
	}
	var out []string
	for _, s := range info.Siblings {
		if strings.HasSuffix(strings.ToLower(s.RFilename), ".gguf") {
			out = append(out, s.RFilename)
		}
	}
	sort.Strings(out)
	return out, nil
}

func downloadFile(url, dest string, client *http.Client) error {
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("hf: download %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("hf: download %s: HTTP %d", url, resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.Copy(f, resp.Body); err != nil {
		return err
	}
	return f.Close()
}
