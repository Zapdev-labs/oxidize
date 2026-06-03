package server

type Config struct {
	Host      string
	Port      int
	ModelsDir string
}

func (c Config) withDefaults() Config {
	if c.Host == "" {
		c.Host = "127.0.0.1"
	}
	if c.Port < 0 {
		c.Port = 0
	}
	return c
}
