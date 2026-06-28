package cli

import (
	"io"

	"github.com/olekukonko/tablewriter"
)

func renderTable(w io.Writer, header []string, rows [][]string) {
	table := tablewriter.NewWriter(w)
	table.SetHeader(header)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding(" ")
	table.AppendBulk(rows)
	table.Render()
}
