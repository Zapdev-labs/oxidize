package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// stubGen substitutes for *Model so the HTTP/JSON/SSE layer is tested without a
// real GGUF: emit() feeds pieces exactly as the C callback would.
type stubGen struct {
	id     string
	pieces [][]byte
}

func (g *stubGen) ID() string     { return g.id }
func (g *stubGen) Meta() Metadata { return Metadata{Arch: "stub"} }
func (g *stubGen) Generate(_ string, _ SampleParams, sink func([]byte) bool) error {
	for _, p := range g.pieces {
		if sink(p) {
			return nil
		}
	}
	return nil
}

func words(ss ...string) [][]byte {
	out := make([][]byte, len(ss))
	for i, s := range ss {
		out[i] = []byte(s)
	}
	return out
}

func srv(g Generator) *httptest.Server { return httptest.NewServer(NewServer(g).Handler()) }

func post(t *testing.T, ts *httptest.Server, path, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestHealthz(t *testing.T) {
	ts := srv(&stubGen{id: "m"})
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("healthz: %v status=%v", err, resp.StatusCode)
	}
}

func TestModels(t *testing.T) {
	ts := srv(&stubGen{id: "gemma4-test"})
	defer ts.Close()
	resp := mustGet(t, ts.URL+"/v1/models")
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string `json:"id"`
			Object string `json:"object"`
		} `json:"data"`
	}
	decodeBody(t, resp, &out)
	if out.Object != "list" || len(out.Data) != 1 || out.Data[0].ID != "gemma4-test" || out.Data[0].Object != "model" {
		t.Fatalf("models = %+v", out)
	}
}

func TestChatNonStreaming(t *testing.T) {
	ts := srv(&stubGen{id: "m", pieces: words("Hello", ", ", "world")})
	defer ts.Close()
	resp := post(t, ts, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	var out struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int         `json:"index"`
			Message      chatMessage `json:"message"`
			FinishReason string      `json:"finish_reason"`
		} `json:"choices"`
		Usage usage `json:"usage"`
	}
	decodeBody(t, resp, &out)
	if out.Object != "chat.completion" || out.Model != "m" || len(out.Choices) != 1 {
		t.Fatalf("shape: %+v", out)
	}
	c := out.Choices[0]
	if c.Message.Role != "assistant" || c.Message.Content != "Hello, world" {
		t.Fatalf("content = %q role = %q", c.Message.Content, c.Message.Role)
	}
	if c.FinishReason != "stop" {
		t.Fatalf("finish_reason = %q", c.FinishReason)
	}
	if out.Usage.CompletionTokens != 3 {
		t.Fatalf("completion_tokens = %d", out.Usage.CompletionTokens)
	}
}

func TestChatConversationEcho(t *testing.T) {
	ts := srv(&stubGen{id: "m", pieces: words("ok")})
	defer ts.Close()
	resp := post(t, ts, "/v1/chat/completions",
		`{"model":"m","conversation":"c-1","messages":[{"role":"user","content":"hi"}]}`)
	var out struct {
		Conversation string `json:"conversation"`
		Choices      []struct {
			Message chatMessage `json:"message"`
		} `json:"choices"`
	}
	decodeBody(t, resp, &out)
	if out.Conversation != "c-1" {
		t.Fatalf("conversation = %q", out.Conversation)
	}
	if len(out.Choices) != 1 || out.Choices[0].Message.Content != "ok" {
		t.Fatalf("choices = %+v", out.Choices)
	}
	del, err := http.NewRequest(http.MethodDelete, ts.URL+"/v1/conversations/c-1", nil)
	if err != nil {
		t.Fatal(err)
	}
	dresp, err := http.DefaultClient.Do(del)
	if err != nil {
		t.Fatal(err)
	}
	defer dresp.Body.Close()
	if dresp.StatusCode != 200 {
		t.Fatalf("delete status = %d", dresp.StatusCode)
	}
}

func TestChatStreamingSSE(t *testing.T) {
	ts := srv(&stubGen{id: "m", pieces: words("Hello", " ", "world")})
	defer ts.Close()
	resp := post(t, ts, "/v1/chat/completions",
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	var content, firstRole, gotDone string
	var finish string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			gotDone = data
			continue
		}
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Delta struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", data, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Fatalf("chunk object = %q", chunk.Object)
		}
		d := chunk.Choices[0]
		if d.Delta.Role != "" && firstRole == "" {
			firstRole = d.Delta.Role
		}
		content += d.Delta.Content
		if d.FinishReason != nil {
			finish = *d.FinishReason
		}
	}
	if firstRole != "assistant" {
		t.Fatalf("first delta role = %q", firstRole)
	}
	if content != "Hello world" {
		t.Fatalf("streamed content = %q", content)
	}
	if finish != "stop" {
		t.Fatalf("finish_reason = %q", finish)
	}
	if gotDone != "[DONE]" {
		t.Fatal("missing [DONE] sentinel")
	}
}

func TestCompletionsNonStreaming(t *testing.T) {
	ts := srv(&stubGen{id: "m", pieces: words("a", "b", "c")})
	defer ts.Close()
	resp := post(t, ts, "/v1/completions", `{"model":"m","prompt":"x"}`)
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	decodeBody(t, resp, &out)
	if out.Object != "text_completion" || len(out.Choices) != 1 || out.Choices[0].Text != "abc" {
		t.Fatalf("completion = %+v", out)
	}
}

// text_completion streaming: choices[].text must ALWAYS be a JSON string, never
// null — including the terminal chunk (empty delta + finish_reason). A strict
// client (e.g. the OpenAI Python SDK, which models text as a required str)
// rejects null at end-of-stream. Uses *string so a JSON null decodes to nil.
func TestCompletionsStreamTextNeverNull(t *testing.T) {
	ts := srv(&stubGen{id: "m", pieces: words("a", "b", "c")})
	defer ts.Close()
	resp := post(t, ts, "/v1/completions", `{"model":"m","stream":true,"prompt":"x"}`)
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}
	var text string
	sawFinish := false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			continue
		}
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Text         *string `json:"text"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", data, err)
		}
		if chunk.Object != "text_completion" {
			t.Fatalf("chunk object = %q", chunk.Object)
		}
		c := chunk.Choices[0]
		if c.Text == nil {
			t.Fatalf("text_completion chunk has null text (must be a string): %s", data)
		}
		text += *c.Text
		if c.FinishReason != nil {
			sawFinish = true
		}
	}
	if text != "abc" {
		t.Fatalf("streamed text = %q, want %q", text, "abc")
	}
	if !sawFinish {
		t.Fatal("no chunk carried finish_reason")
	}
}

// Stop sequences are enforced in Go (the C ABI only knows EOS/EOT): output is cut
// at the first stop and the stop text itself is not emitted.
func TestStopSequenceTrim(t *testing.T) {
	ts := srv(&stubGen{id: "m", pieces: words("Hello ", "STOP", " world")})
	defer ts.Close()
	resp := post(t, ts, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":"STOP"}`)
	var out struct {
		Choices []struct {
			Message      chatMessage `json:"message"`
			FinishReason string      `json:"finish_reason"`
		} `json:"choices"`
	}
	decodeBody(t, resp, &out)
	if got := out.Choices[0].Message.Content; got != "Hello " {
		t.Fatalf("stop-trimmed content = %q, want %q", got, "Hello ")
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish = %q", out.Choices[0].FinishReason)
	}
}

// A multi-byte rune split across two streamed fragments must arrive intact (no
// U+FFFD from JSON-encoding an incomplete UTF-8 sequence).
func TestStreamingUTF8Split(t *testing.T) {
	zhong := []byte("中") // E4 B8 AD
	ts := srv(&stubGen{id: "m", pieces: [][]byte{zhong[:1], zhong[1:], []byte("!")}})
	defer ts.Close()
	resp := post(t, ts, "/v1/chat/completions",
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()

	var content string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || strings.Contains(line, "[DONE]") {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatal(err)
		}
		content += chunk.Choices[0].Delta.Content
	}
	if content != "中!" {
		t.Fatalf("reassembled stream = %q, want %q", content, "中!")
	}
}

func TestBadJSON(t *testing.T) {
	ts := srv(&stubGen{id: "m"})
	defer ts.Close()
	resp := post(t, ts, "/v1/chat/completions", `{not json`)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

// outputFilter unit checks (the stop + UTF-8 logic that streaming relies on).
func TestOutputFilterUnit(t *testing.T) {
	// UTF-8 hold-back: a lead byte alone is retained, then released.
	f := &outputFilter{}
	if e, _ := f.push([]byte{0xE4}); len(e) != 0 {
		t.Fatalf("emitted incomplete rune: %v", e)
	}
	if e, _ := f.push([]byte{0xB8, 0xAD}); string(e) != "中" {
		t.Fatalf("completed rune = %q", e)
	}
	// Stop split across pushes: a stop straddling two fragments is never emitted,
	// and everything before it is. Byte-level hold-back means the exact split
	// point per push is an implementation detail; the invariant is the total.
	g := &outputFilter{stops: []string{"STOP"}, maxLen: 4}
	e1, s1 := g.push([]byte("hi ST"))
	if s1 {
		t.Fatal("stopped before the stop string completed")
	}
	e2, s2 := g.push([]byte("OP tail"))
	if !s2 {
		t.Fatal("stop string not detected across fragments")
	}
	got := string(e1) + string(e2)
	if got != "hi " {
		t.Fatalf("emitted %q before stop, want %q", got, "hi ")
	}
	if strings.Contains(got, "STOP") {
		t.Fatalf("stop string leaked into output: %q", got)
	}
}

// ---- cgo load-path smoke test (proves the C ABI links + error handling) ----

func TestCgoLinkAndErrors(t *testing.T) {
	if Version() == "" {
		t.Fatal("ox_version empty")
	}
	if ISA() == "" {
		t.Fatal("ox_isa empty")
	}
	t.Logf("linked %s (isa=%s)", Version(), ISA())
	// A missing model must fail cleanly with a non-empty error, not crash — this
	// exercises ox_model_open, the (char* err,size_t) convention, and cstr().
	if _, err := OpenModel("/no/such/model.gguf", "", 0, 1); err == nil {
		t.Fatal("expected error opening missing model")
	} else if !strings.Contains(err.Error(), "ox_model_open") {
		t.Fatalf("unexpected error: %v", err)
	}
	// End-to-end generation against a real GGUF is covered by the C ABI
	// acceptance test (tests/test_abi.c, run via `make test`); it needs a
	// synthesized runnable model this Go layer can't build cheaply.
}

// ---- HF resolver (P7b), mirrors oxidize-golang/hf/hub_test.go ---------------

func TestResolveGGUFUsesCache(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/demo/tiny":
			w.Write([]byte(`{"siblings":[{"rfilename":"tiny.gguf"}]}`))
		case "/demo/tiny/resolve/main/tiny.gguf":
			w.Write([]byte("GGUF"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	opts := ResolveOptions{Repo: "demo/tiny", CacheDir: t.TempDir(), APIBase: ts.URL, CDNBase: ts.URL}
	p1, err := ResolveGGUF(opts)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := ResolveGGUF(opts)
	if err != nil {
		t.Fatal(err)
	}
	if p1 != p2 {
		t.Fatalf("cache miss: %q vs %q", p1, p2)
	}
	if b, _ := os.ReadFile(p2); string(b) != "GGUF" {
		t.Fatalf("cached content = %q", b)
	}
}

func TestResolveMultipleGGUFErrors(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"siblings":[{"rfilename":"a.gguf"},{"rfilename":"b.gguf"}]}`))
	}))
	defer ts.Close()
	if _, err := ResolveGGUF(ResolveOptions{Repo: "x/y", CacheDir: t.TempDir(), APIBase: ts.URL, CDNBase: ts.URL}); err == nil {
		t.Fatal("expected error for multiple gguf files")
	}
}

func TestIsHFRepoID(t *testing.T) {
	cases := map[string]bool{
		"unsloth/gemma-3-4b-it-GGUF": true,
		"org/model/file.gguf":        true,
		"/abs/path/model.gguf":       false,
		"./rel.gguf":                 false,
		"single":                     false,
	}
	for in, want := range cases {
		if got := isHFRepoID(in); got != want {
			t.Errorf("isHFRepoID(%q) = %v, want %v", in, got, want)
		}
	}
	// An existing file is never an HF id even if it contains a slash.
	f := t.TempDir() + "/a/b" // a/b-shaped but real
	os.MkdirAll(f, 0o755)
	if isHFRepoID(f) {
		t.Errorf("existing path treated as HF id: %q", f)
	}
}

// ---- helpers ----

func mustGet(t *testing.T, url string) *http.Response {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func decodeBody(t *testing.T, resp *http.Response, v any) {
	t.Helper()
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, b)
	}
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatal(err)
	}
}
