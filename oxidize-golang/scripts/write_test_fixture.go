//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/Zapdev-labs/oxidize/golang/internal/testutil"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: write_test_fixture <path>")
		os.Exit(2)
	}
	if err := testutil.WriteValidFixture(os.Args[1]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
