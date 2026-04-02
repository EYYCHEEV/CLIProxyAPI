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

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_RequestFieldsPreservedOnDoneCompletion(t *testing.T) {
	request := []byte(`{"model":"glm-5","stream":true,"instructions":"keep this","metadata":{"ticket":"123"},"temperature":0.25,"store":true}`)
	var param any

	events := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":1773664502,"model":"glm-5","choices":[{"index":0,"delta":{"content":"baseline ok"},"finish_reason":null}]}`), &param)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":1773664502,"model":"glm-5","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}`), &param)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "glm-5", request, request, []byte(`data: [DONE]`), &param)...)

	completedCount := 0
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event != "response.completed" {
			continue
		}
		completedCount++
		if data.Get("response.instructions").String() != "keep this" {
			t.Fatalf("instructions = %q, want %q", data.Get("response.instructions").String(), "keep this")
		}
		if data.Get("response.metadata.ticket").String() != "123" {
			t.Fatalf("metadata.ticket = %q, want %q", data.Get("response.metadata.ticket").String(), "123")
		}
		if data.Get("response.temperature").Float() != 0.25 {
			t.Fatalf("temperature = %v, want %v", data.Get("response.temperature").Float(), 0.25)
		}
		if !data.Get("response.store").Bool() {
			t.Fatalf("store = false, want true")
		}
	}

	if completedCount != 1 {
		t.Fatalf("response.completed count = %d, want 1", completedCount)
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningOnlyStopSynthesizesMessage(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi","object":"chat.completion.chunk","created":1773664503,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"Model: Kimi\nPWD: /tmp/workdir"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi","object":"chat.completion.chunk","created":1773664503,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":2,"total_tokens":13}}`),
		&param,
	)...)

	var messageDone gjson.Result
	var completed gjson.Result
	messagePartAdded := false
	messageTextDone := false
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.content_part.added" && strings.HasPrefix(data.Get("item_id").String(), "msg_") {
			messagePartAdded = true
		}
		if event == "response.output_text.done" && strings.HasPrefix(data.Get("item_id").String(), "msg_") {
			messageTextDone = true
		}
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			messageDone = data
		}
		if event == "response.completed" {
			completed = data
		}
	}

	if !messageDone.Exists() {
		t.Fatalf("expected synthesized message item.done event")
	}
	if !messagePartAdded {
		t.Fatalf("expected synthesized message content_part.added event")
	}
	if !messageTextDone {
		t.Fatalf("expected synthesized message output_text.done event")
	}
	if got := messageDone.Get("item.content.0.text").String(); got != "Model: Kimi\nPWD: /tmp/workdir" {
		t.Fatalf("message text = %q, want %q", got, "Model: Kimi\nPWD: /tmp/workdir")
	}
	if !completed.Exists() {
		t.Fatalf("expected response.completed event")
	}
	outputTypes := completed.Get("response.output.#.type").Array()
	if len(outputTypes) != 2 {
		t.Fatalf("response.output len = %d, want 2", len(outputTypes))
	}
	if outputTypes[0].String() != "reasoning" || outputTypes[1].String() != "message" {
		t.Fatalf("response.output types = %v, want [reasoning message]", outputTypes)
	}
	if got := completed.Get("response.output.0.summary.0.text").String(); got != "Model: Kimi\nPWD: /tmp/workdir" {
		t.Fatalf("completed reasoning text = %q, want %q", got, "Model: Kimi\nPWD: /tmp/workdir")
	}
	if got := completed.Get("response.output.1.content.0.text").String(); got != "Model: Kimi\nPWD: /tmp/workdir" {
		t.Fatalf("completed message text = %q, want %q", got, "Model: Kimi\nPWD: /tmp/workdir")
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningOnlyStopWithTrailingUsageSynthesizesMessage(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-trailing","object":"chat.completion.chunk","created":1773664505,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"deferred final text"},"finish_reason":null}]}`),
		&param,
	)...)
	finishOnly := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-trailing","object":"chat.completion.chunk","created":1773664505,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}]}`),
		&param,
	)
	events = append(events, finishOnly...)
	for _, chunk := range finishOnly {
		event, _ := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.completed" {
			t.Fatalf("response.completed emitted on Kimi stop chunk before trailing usage arrived")
		}
	}
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-trailing","object":"chat.completion.chunk","created":1773664505,"model":"kimi-k2.5","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2,"total_tokens":13}}`),
		&param,
	)...)

	messageDoneCount := 0
	completedCount := 0
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			messageDoneCount++
			if got := data.Get("item.content.0.text").String(); got != "deferred final text" {
				t.Fatalf("message text = %q, want %q", got, "deferred final text")
			}
		}
		if event != "response.completed" {
			continue
		}
		completedCount++
		if data.Get("response.usage.total_tokens").Int() != 13 {
			t.Fatalf("total tokens = %d, want 13", data.Get("response.usage.total_tokens").Int())
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != "deferred final text" {
			t.Fatalf("completed message text = %q, want %q", got, "deferred final text")
		}
	}

	if messageDoneCount != 1 {
		t.Fatalf("message done count = %d, want 1", messageDoneCount)
	}
	if completedCount != 1 {
		t.Fatalf("response.completed count = %d, want 1", completedCount)
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningOnlyStopWithoutUsageCompletesOnDone(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-done","object":"chat.completion.chunk","created":1773664510,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"done-final-text"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-done","object":"chat.completion.chunk","created":1773664510,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: [DONE]`),
		&param,
	)...)

	foundCompleted := false
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event != "response.completed" {
			continue
		}
		foundCompleted = true
		if data.Get("response.usage.total_tokens").Int() != 0 {
			t.Fatalf("total tokens = %d, want 0", data.Get("response.usage.total_tokens").Int())
		}
		if got := data.Get("response.output.0.summary.0.text").String(); got != "done-final-text" {
			t.Fatalf("completed reasoning text = %q, want %q", got, "done-final-text")
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != "done-final-text" {
			t.Fatalf("completed message text = %q, want %q", got, "done-final-text")
		}
	}

	if !foundCompleted {
		t.Fatalf("expected response.completed event")
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningAndContentDoesNotDuplicateMessage(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-content","object":"chat.completion.chunk","created":1773664506,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"internal-only"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-content","object":"chat.completion.chunk","created":1773664506,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"real final text"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-content","object":"chat.completion.chunk","created":1773664506,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}`),
		&param,
	)...)

	messageDoneCount := 0
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			messageDoneCount++
			if got := data.Get("item.content.0.text").String(); got != "real final text" {
				t.Fatalf("message text = %q, want %q", got, "real final text")
			}
		}
		if event != "response.completed" {
			continue
		}
		outputTypes := data.Get("response.output.#.type").Array()
		if len(outputTypes) != 2 || outputTypes[0].String() != "reasoning" || outputTypes[1].String() != "message" {
			t.Fatalf("response.output types = %v, want [reasoning message]", outputTypes)
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != "real final text" {
			t.Fatalf("completed message text = %q, want %q", got, "real final text")
		}
	}

	if messageDoneCount != 1 {
		t.Fatalf("message done count = %d, want 1", messageDoneCount)
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningAccumulationSynthesizesFullMessage(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-accum","object":"chat.completion.chunk","created":1773664511,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"first-"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-accum","object":"chat.completion.chunk","created":1773664511,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"second"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-accum","object":"chat.completion.chunk","created":1773664511,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}`),
		&param,
	)...)

	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			if got := data.Get("item.content.0.text").String(); got != "first-second" {
				t.Fatalf("message text = %q, want %q", got, "first-second")
			}
		}
		if event != "response.completed" {
			continue
		}
		if got := data.Get("response.output.0.summary.0.text").String(); got != "first-second" {
			t.Fatalf("completed reasoning text = %q, want %q", got, "first-second")
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != "first-second" {
			t.Fatalf("completed message text = %q, want %q", got, "first-second")
		}
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningOnlyStopSameChunkSynthesizesMessage(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any

	events := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-same-chunk","object":"chat.completion.chunk","created":1773664507,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"same chunk final text","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`),
		&param,
	)

	foundCompleted := false
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event != "response.completed" {
			continue
		}
		foundCompleted = true
		if got := data.Get("response.output.0.summary.0.text").String(); got != "same chunk final text" {
			t.Fatalf("completed reasoning text = %q, want %q", got, "same chunk final text")
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != "same chunk final text" {
			t.Fatalf("completed message text = %q, want %q", got, "same chunk final text")
		}
	}

	if !foundCompleted {
		t.Fatalf("expected response.completed event")
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningPromotionPreservesWhitespace(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any
	rawReasoning := "  padded final text  \n"

	events := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-whitespace","object":"chat.completion.chunk","created":1773664508,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"  padded final text  \n","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}`),
		&param,
	)

	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			if got := data.Get("item.content.0.text").String(); got != rawReasoning {
				t.Fatalf("message text = %q, want %q", got, rawReasoning)
			}
		}
		if event != "response.completed" {
			continue
		}
		if got := data.Get("response.output.0.summary.0.text").String(); got != rawReasoning {
			t.Fatalf("completed reasoning text = %q, want %q", got, rawReasoning)
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != rawReasoning {
			t.Fatalf("completed message text = %q, want %q", got, rawReasoning)
		}
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_KimiReasoningPromotionRequiresMatchingChoiceIndex(t *testing.T) {
	request := []byte(`{"model":"kimi-k2.5","stream":true}`)
	var param any

	events := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"kimi-k2.5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-kimi-multichoice","object":"chat.completion.chunk","created":1773664509,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"reasoning_content":"choice-zero-reasoning"},"finish_reason":null},{"index":1,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`),
		&param,
	)

	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			t.Fatalf("unexpected synthesized message for mismatched choice index")
		}
		if event != "response.completed" {
			continue
		}
		outputTypes := data.Get("response.output.#.type").Array()
		if len(outputTypes) != 1 || outputTypes[0].String() != "reasoning" {
			t.Fatalf("response.output types = %v, want [reasoning]", outputTypes)
		}
		if got := data.Get("response.output.0.summary.0.text").String(); got != "choice-zero-reasoning" {
			t.Fatalf("completed reasoning text = %q, want %q", got, "choice-zero-reasoning")
		}
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_GenericReasoningOnlyStopSynthesizesMessage(t *testing.T) {
	request := []byte(`{"model":"glm-5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"glm-5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-glm","object":"chat.completion.chunk","created":1773664504,"model":"glm-5","choices":[{"index":0,"delta":{"reasoning_content":"internal-only"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"glm-5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-glm","object":"chat.completion.chunk","created":1773664504,"model":"glm-5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`),
		&param,
	)...)

	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			if got := data.Get("item.content.0.text").String(); got != "internal-only" {
				t.Fatalf("message text = %q, want %q", got, "internal-only")
			}
		}
		if event != "response.completed" {
			continue
		}
		if got := data.Get("response.output.0.summary.0.text").String(); got != "internal-only" {
			t.Fatalf("completed reasoning text = %q, want %q", got, "internal-only")
		}
		if got := data.Get("response.output.1.content.0.text").String(); got != "internal-only" {
			t.Fatalf("completed message text = %q, want %q", got, "internal-only")
		}
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_StopWithoutReasoningDoesNotSynthesizeMessage(t *testing.T) {
	request := []byte(`{"model":"glm-5","stream":true}`)
	var param any

	events := ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"glm-5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-empty-stop","object":"chat.completion.chunk","created":1773664512,"model":"glm-5","choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`),
		&param,
	)

	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		if event == "response.output_item.done" && data.Get("item.type").String() == "message" {
			t.Fatalf("unexpected synthesized message without reasoning text")
		}
		if event != "response.completed" {
			continue
		}
		if data.Get("response.output").Exists() {
			t.Fatalf("unexpected response.output for empty stop payload")
		}
	}
}

func TestConvertOpenAIChatCompletionsResponseToOpenAIResponses_InterleavedReasoningUsesDistinctChoiceState(t *testing.T) {
	request := []byte(`{"model":"glm-5","stream":true}`)
	var param any
	var events []string

	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"glm-5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-interleaved","object":"chat.completion.chunk","created":1773664513,"model":"glm-5","choices":[{"index":0,"delta":{"reasoning_content":"alpha"},"finish_reason":null}]}`),
		&param,
	)...)
	events = append(events, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(
		context.Background(),
		"glm-5",
		request,
		request,
		[]byte(`data: {"id":"chatcmpl-interleaved","object":"chat.completion.chunk","created":1773664513,"model":"glm-5","choices":[{"index":1,"delta":{"reasoning_content":"beta"},"finish_reason":null}]}`),
		&param,
	)...)

	added := map[string]int{}
	deltas := map[string]string{}
	for _, chunk := range events {
		event, data := parseOpenAIResponsesSSEEvent(t, chunk)
		switch event {
		case "response.output_item.added":
			if data.Get("item.type").String() != "reasoning" {
				continue
			}
			added[data.Get("item.id").String()] = int(data.Get("output_index").Int())
		case "response.reasoning_summary_text.delta":
			deltas[data.Get("item_id").String()] = data.Get("delta").String()
		}
	}

	if len(added) != 2 {
		t.Fatalf("reasoning item count = %d, want 2", len(added))
	}
	if len(deltas) != 2 {
		t.Fatalf("reasoning delta count = %d, want 2", len(deltas))
	}
	idx0Found := false
	idx1Found := false
	for id, idx := range added {
		switch idx {
		case 0:
			idx0Found = true
			if got := deltas[id]; got != "alpha" {
				t.Fatalf("reasoning delta for index 0 = %q, want %q", got, "alpha")
			}
		case 1:
			idx1Found = true
			if got := deltas[id]; got != "beta" {
				t.Fatalf("reasoning delta for index 1 = %q, want %q", got, "beta")
			}
		default:
			t.Fatalf("unexpected reasoning output index %d", idx)
		}
	}
	if !idx0Found || !idx1Found {
		t.Fatalf("missing reasoning output indices 0/1: idx0=%v idx1=%v", idx0Found, idx1Found)
	}
}
