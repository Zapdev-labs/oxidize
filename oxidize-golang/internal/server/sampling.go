package server

import "github.com/Zapdev-labs/oxidize/golang/internal/api"

const samplingUnset = -1

func samplingFromChat(req api.ChatCompletionRequest) (temperature, topP float32, topK int) {
	temperature, topP, topK = samplingUnset, samplingUnset, 0
	if req.Temperature != nil {
		temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		topP = float32(*req.TopP)
	}
	if req.TopK != nil {
		topK = *req.TopK
	}
	return temperature, topP, topK
}

func samplingFromCompletion(req api.CompletionRequest) (temperature, topP float32, topK int) {
	temperature, topP, topK = samplingUnset, samplingUnset, 0
	if req.Temperature != nil {
		temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		topP = float32(*req.TopP)
	}
	if req.TopK != nil {
		topK = *req.TopK
	}
	return temperature, topP, topK
}
