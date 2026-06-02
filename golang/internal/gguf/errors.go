package gguf

import "fmt"

func errInvalidMagic() error {
	return fmt.Errorf("invalid gguf magic")
}

func errUnsupportedVersion(version uint32) error {
	return fmt.Errorf("unsupported gguf version: %d", version)
}

func errUnexpectedEOF() error {
	return fmt.Errorf("unexpected end of file")
}

func errUnknownMetadataType(value uint32) error {
	return fmt.Errorf("unknown metadata type: %d", value)
}

func errInvalidAlignment(value uint64) error {
	return fmt.Errorf("invalid alignment: %d", value)
}

func errIntegerOverflow() error {
	return fmt.Errorf("integer overflow while parsing")
}
