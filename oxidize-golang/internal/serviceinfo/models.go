package serviceinfo

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

const minGGUFFullValidationBytes = 1024

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
		stat, statErr := os.Stat(path)
		if statErr != nil {
			return nil, statErr
		}
		if stat.Size() >= minGGUFFullValidationBytes {
			if loadErr := gguf.ValidateFile(path); loadErr != nil {
				return nil, loadErr
			}
		}
		header, loadErr := gguf.LoadMetadata(path)
		if loadErr != nil {
			return nil, loadErr
		}
		models = append(models, ModelInfo{
			ID:           strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name())),
			Path:         path,
			Version:      header.Version,
			Architecture: header.Metadata["general.architecture"].String,
		})
	}
	sort.Slice(models, func(i int, j int) bool {
		return models[i].ID < models[j].ID
	})
	return models, nil
}
