package hf

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const defaultRevision = "main"

// ResolveOptions controls Hugging Face GGUF resolution.
type ResolveOptions struct {
	Repo       string
	Revision   string
	Filename   string
	CacheDir   string
	HTTPClient *http.Client
	APIBase    string
	CDNBase    string
}

// ResolveGGUF downloads (or returns a cached) GGUF file for a Hugging Face repo.
// Repo may be "org/model" or "org/model/quant.gguf" when Filename is empty.
func ResolveGGUF(opts ResolveOptions) (string, error) {
	repo, filename, err := splitRepoAndFile(opts.Repo, opts.Filename)
	if err != nil {
		return "", err
	}
	if opts.Filename != "" {
		filename = opts.Filename
	}
	if filename == "" {
		filename, err = pickSingleGGUF(repo, opts.Revision, opts.HTTPClient, opts.APIBase)
		if err != nil {
			return "", err
		}
	}
	rev := opts.Revision
	if rev == "" {
		rev = defaultRevision
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
		return dest, nil
	}
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
		repo = strings.Join(parts[:len(parts)-1], "/")
		return repo, file, nil
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

type modelInfo struct {
	Siblings []struct {
		RFilename string `json:"rfilename"`
	} `json:"siblings"`
}

func pickSingleGGUF(repo, revision string, client *http.Client, apiBase string) (string, error) {
	names, err := ListGGUFFiles(repo, revision, client, apiBase)
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
		b.WriteString(fmt.Sprintf("hf: repo %q has multiple .gguf files; specify --file. Candidates:\n", repo))
		for i, n := range names {
			if i >= 25 {
				b.WriteString("  ...\n")
				break
			}
			b.WriteString("  ")
			b.WriteString(n)
			b.WriteString("\n")
		}
		return "", fmt.Errorf("%s", strings.TrimSuffix(b.String(), "\n"))
	}
}

// ListGGUFFiles returns sorted .gguf filenames in a Hugging Face model repo.
func ListGGUFFiles(repo, revision string, client *http.Client, apiBase string) ([]string, error) {
	if revision == "" {
		revision = defaultRevision
	}
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}
	if apiBase == "" {
		apiBase = "https://huggingface.co"
	}
	url := fmt.Sprintf("%s/api/models/%s", strings.TrimRight(apiBase, "/"), repo)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("hf: list %q: %w", repo, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("hf: list %q: HTTP %d: %s", repo, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var info modelInfo
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
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
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
