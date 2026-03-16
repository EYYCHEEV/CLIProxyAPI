package responses

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func parseOpenAIResponsesSSEEvent(t *testing.T, chunk string) (string, gjson.Result) {
	t.Helper()

	lines := strings.Split(chunk, "\n")
	if len(lines) < 2 {
		t.Fatalf("unexpected SSE chunk: %q", chunk)
	}

	event := strings.TrimSpace(strings.TrimPrefix(lines[0], "event:"))
	dataLine := strings.TrimSpace(strings.TrimPrefix(lines[1], "data:"))
	if !gjson.Valid(dataLine) {
		t.Fatalf("invalid SSE data JSON: %q", dataLine)
	}
	return event, gjson.Parse(dataLine)
}

func collectOpenAIResponsesEvents(t *testing.T, chunks []string) []string {
	t.Helper()

	request := []byte(`{"model":"glm-5","stream":true}`)
	var param any
	var out []string
	for _, chunk := range chunks {
		out = append(out, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(chunk), &param)...)
	}
	return out
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_TrailingUsageAfterFinishReason(t *testing.T) {
	request := []byte(`{"model":"glm-5","stream":true}`)
	var param any
	var out []string

	out = append(out, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1773664499,"model":"glm-5","choices":[{"index":0,"delta":{"content":"baseline ok"},"finish_reason":null}]}`), &param)...)
	finishOnly := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1773664499,"model":"glm-5","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}`), &param)
	out = append(out, finishOnly...)
	for _, chunk := range finishOnly {
		event, _ := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.completed" {
			t.Fatalf("response.completed emitted on finish_reason chunk before trailing usage arrived")
		}
	}
	usageChunk := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1773664499,"model":"glm-5","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":42,"total_tokens":53,"completion_tokens_details":{"reasoning_tokens":38},"prompt_tokens_details":{"cached_tokens":0}}}`), &param)
	out = append(out, usageChunk...)
	out = append(out, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: [DONE]`), &param)...)

	completedCount := 0
	completedPos := -1
	usagePos := -1
	for i, chunk := range out {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.completed" {
			completedCount++
			completedPos = i
			if data.Get("response.usage.total_tokens").Int() != 53 {
				t.Fatalf("total tokens = %d, want 53", data.Get("response.usage.total_tokens").Int())
			}
			if data.Get("response.usage.output_tokens_details.reasoning_tokens").Int() != 38 {
				t.Fatalf("reasoning tokens = %d, want 38", data.Get("response.usage.output_tokens_details.reasoning_tokens").Int())
			}
		}
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			usagePos = i
		}
	}

	if completedCount != 1 {
		t.Fatalf("response.completed count = %d, want 1", completedCount)
	}
	if completedPos == -1 {
		t.Fatalf("missing response.completed event")
	}
	if usagePos == -1 {
		t.Fatalf("missing message done event")
	}
	if completedPos <= usagePos {
		t.Fatalf("response.completed should be after message finalization: messageDone=%d completed=%d", usagePos, completedPos)
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_UsageBeforeFinishReason(t *testing.T) {
	out := collectOpenAIResponsesEvents(t, []string{
		`data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1773664500,"model":"glm-5","choices":[{"index":0,"delta":{"content":"baseline ok"},"finish_reason":null}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`,
		`data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1773664500,"model":"glm-5","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}`,
	})

	completedCount := 0
	for _, chunk := range out {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event != "response.completed" {
			continue
		}
		completedCount++
		if data.Get("response.usage.input_tokens").Int() != 7 {
			t.Fatalf("input tokens = %d, want 7", data.Get("response.usage.input_tokens").Int())
		}
		if data.Get("response.usage.output_tokens").Int() != 2 {
			t.Fatalf("output tokens = %d, want 2", data.Get("response.usage.output_tokens").Int())
		}
		if data.Get("response.usage.total_tokens").Int() != 9 {
			t.Fatalf("total tokens = %d, want 9", data.Get("response.usage.total_tokens").Int())
		}
	}

	if completedCount != 1 {
		t.Fatalf("response.completed count = %d, want 1", completedCount)
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_NoUsageFallsBackToZero(t *testing.T) {
	out := collectOpenAIResponsesEvents(t, []string{
		`data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1773664501,"model":"glm-5","choices":[{"index":0,"delta":{"content":"baseline ok"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1773664501,"model":"glm-5","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	})

	completedCount := 0
	for _, chunk := range out {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event != "response.completed" {
			continue
		}
		completedCount++
		if data.Get("response.usage.input_tokens").Int() != 0 {
			t.Fatalf("input tokens = %d, want 0", data.Get("response.usage.input_tokens").Int())
		}
		if data.Get("response.usage.output_tokens").Int() != 0 {
			t.Fatalf("output tokens = %d, want 0", data.Get("response.usage.output_tokens").Int())
		}
		if data.Get("response.usage.total_tokens").Int() != 0 {
			t.Fatalf("total tokens = %d, want 0", data.Get("response.usage.total_tokens").Int())
		}
	}

	if completedCount != 1 {
		t.Fatalf("response.completed count = %d, want 1", completedCount)
	}
}
