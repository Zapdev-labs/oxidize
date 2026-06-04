package server

import (
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

type Config struct {
	Host           string
	Port           int
	ModelsDir      string
	DefaultModel   string
	Backend        string
	MaxTokens      int
	Temperature    float32
	TopP           float32
	TopK           int
	Loader         generate.LoaderConfig
}

func (c Config) withDefaults() Config {
	if c.Host == "" {
		c.Host = "127.0.0.1"
	}
	if c.Port < 0 {
		c.Port = 0
	}
	if c.MaxTokens <= 0 {
		c.MaxTokens = 128
	}
	if c.Temperature == 0 {
		c.Temperature = 0.8
	}
	if c.TopP == 0 {
		c.TopP = 0.9
	}
	if c.Backend == "" {
		c.Backend = "cpu"
	}
	if c.Loader.Backend == "" {
		c.Loader = generate.LoaderConfig{
			Backend:       c.Backend,
			AllowFallback: true,
		}
	}
	return c
}
