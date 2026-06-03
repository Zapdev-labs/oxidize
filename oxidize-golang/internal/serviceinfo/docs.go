package serviceinfo

type OpenAPISpec struct {
	OpenAPI string                     `json:"openapi"`
	Info    OpenAPIInfo                `json:"info"`
	Servers []map[string]string        `json:"servers"`
	Paths   map[string]OpenAPIPathItem `json:"paths"`
}

type OpenAPIInfo struct {
	Title       string `json:"title"`
	Version     string `json:"version"`
	Description string `json:"description"`
}

type OpenAPIPathItem struct {
	Get  *OpenAPIOperation `json:"get,omitempty"`
	Post *OpenAPIOperation `json:"post,omitempty"`
}

type OpenAPIOperation struct {
	Summary string `json:"summary"`
}

func OpenAPI(version string) OpenAPISpec {
	return OpenAPISpec{
		OpenAPI: "3.1.0",
		Info: OpenAPIInfo{
			Title:       "oxidize-server API",
			Version:     version,
			Description: "OpenAI-compatible endpoints exposed by oxidize-go.",
		},
		Servers: []map[string]string{{"url": "/"}},
		Paths: map[string]OpenAPIPathItem{
			"/healthz":             {Get: &OpenAPIOperation{Summary: "Health check"}},
			"/livez":               {Get: &OpenAPIOperation{Summary: "Liveness check"}},
			"/readyz":              {Get: &OpenAPIOperation{Summary: "Readiness check"}},
			"/v1/chat/completions": {Post: &OpenAPIOperation{Summary: "Create chat completion"}},
			"/v1/completions":      {Post: &OpenAPIOperation{Summary: "Create text completion"}},
			"/v1/models":           {Get: &OpenAPIOperation{Summary: "List models"}},
			"/v1/embeddings":       {Post: &OpenAPIOperation{Summary: "Create embeddings"}},
		},
	}
}
