package serviceinfo

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

type ModelInfo struct {
	ID           string
	Path         string
	Version      uint32
	Architecture string
}

func DefaultModelID(models []ModelInfo) string {
	if len(models) == 0 {
		return "oxidize-default"
	}
	return models[0].ID
}

func DiscoverModels(dir string) ([]ModelInfo, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	models := make([]ModelInfo, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		file, loadErr := gguf.LoadFile(path)
		if loadErr != nil {
			return nil, loadErr
		}
		models = append(models, ModelInfo{
			ID:           strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name())),
			Path:         path,
			Version:      file.Version,
			Architecture: file.Metadata["general.architecture"].String,
		})
	}
	sort.Slice(models, func(i int, j int) bool {
		return models[i].ID < models[j].ID
	})
	return models, nil
}
