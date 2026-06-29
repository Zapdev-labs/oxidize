package cli

import (
	"fmt"
	"math"
	"time"
)

func humanBytes(n int64) string {
	if n < 0 {
		return "?"
	}
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	val := float64(n) / float64(div)
	suffix := "KMGTPE"[exp]
	if suffix == 'K' || suffix == 'M' {
		return fmt.Sprintf("%.1f %ciB", val, suffix)
	}
	return fmt.Sprintf("%.1f %ciB", val, suffix)
}

func humanTime(t time.Time, empty string) string {
	if t.IsZero() {
		return empty
	}
	now := time.Now()
	diff := now.Sub(t)
	switch {
	case diff < time.Minute:
		return "just now"
	case diff < time.Hour:
		m := int(diff.Minutes())
		if m == 1 {
			return "1 minute ago"
		}
		return fmt.Sprintf("%d minutes ago", m)
	case diff < 24*time.Hour:
		h := int(diff.Hours())
		if h == 1 {
			return "1 hour ago"
		}
		return fmt.Sprintf("%d hours ago", h)
	case diff < 7*24*time.Hour:
		d := int(diff.Hours() / 24)
		if d == 1 {
			return "1 day ago"
		}
		return fmt.Sprintf("%d days ago", d)
	case diff < 30*24*time.Hour:
		w := int(diff.Hours() / (24 * 7))
		if w == 1 {
			return "1 week ago"
		}
		return fmt.Sprintf("%d weeks ago", w)
	case diff < 365*24*time.Hour:
		mo := int(diff.Hours() / (24 * 30))
		if mo == 1 {
			return "1 month ago"
		}
		return fmt.Sprintf("%d months ago", mo)
	default:
		y := int(diff.Hours() / (24 * 365))
		if y == 1 {
			return "1 year ago"
		}
		return fmt.Sprintf("%d years ago", y)
	}
}

func pullPercent(downloaded, total int64) int {
	if total <= 0 {
		return 0
	}
	return int(math.Min(100, math.Round(float64(downloaded)/float64(total)*100)))
}
