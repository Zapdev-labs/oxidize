package serviceinfo

import "fmt"

type MetricsData struct {
	RequestsTotal    int
	RequestsInflight int
}

func MetricsSnapshot(data MetricsData) string {
	return fmt.Sprintf(
		"# HELP oxidize_requests_total Total requests.\n# TYPE oxidize_requests_total counter\noxidize_requests_total %d\n# HELP oxidize_requests_in_flight Requests in flight.\n# TYPE oxidize_requests_in_flight gauge\noxidize_requests_in_flight %d\n",
		data.RequestsTotal,
		data.RequestsInflight,
	)
}
