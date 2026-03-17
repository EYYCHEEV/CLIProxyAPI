package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestOpenAICompatExecutorCompactPassthrough(t *testing.T) {
	var gotPath string
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		gotBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response.compaction","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}
	payload := []byte(`{"model":"shared-alias","input":[{"role":"user","content":"hi"}]}`)
	resp, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.1-codex-max",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Alt:          "responses/compact",
		Stream:       false,
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/responses/compact" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/responses/compact")
	}
	if !gjson.GetBytes(gotBody, "input").Exists() {
		t.Fatalf("expected input in body")
	}
	if got := gjson.GetBytes(gotBody, "model").String(); got != "gpt-5.1-codex-max" {
		t.Fatalf("model = %q, want %q", got, "gpt-5.1-codex-max")
	}
	if gjson.GetBytes(gotBody, "messages").Exists() {
		t.Fatalf("unexpected messages in body")
	}
	if string(resp.Payload) != `{"id":"resp_1","object":"response.compaction","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}` {
		t.Fatalf("payload = %s", string(resp.Payload))
	}
}

func TestOpenAICompatExecutorNativeResponsesNonStream(t *testing.T) {
	var gotPath string
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		gotBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"READY"}]}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "alibaba-coding-plan",
		Attributes: map[string]string{
			"base_url":    server.URL + "/v1",
			"api_key":     "test",
			"compat_name": "alibaba-coding-plan",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello"}`)
	resp, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "qwen3.5-plus",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/responses" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/responses")
	}
	if !gjson.GetBytes(gotBody, "input").Exists() {
		t.Fatalf("expected input in native responses body")
	}
	if got := gjson.GetBytes(gotBody, "model").String(); got != "qwen3.5-plus" {
		t.Fatalf("model = %q, want %q", got, "qwen3.5-plus")
	}
	if gjson.GetBytes(gotBody, "messages").Exists() {
		t.Fatalf("unexpected messages in native responses body")
	}
	if !gjson.GetBytes(resp.Payload, "output.0.content.0.text").Exists() {
		t.Fatalf("expected responses payload passthrough")
	}
}

func TestOpenAICompatExecutorNativeResponsesResolvesCompatConfigFromExecutorProvider(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Attributes: map[string]string{
			"base_url": server.URL + "/v1",
			"api_key":  "test",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello"}`)
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "qwen3.5-plus",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/responses" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/responses")
	}
}

func TestOpenAICompatExecutorNativeResponsesPrefersCompatNameOverProviderKey(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "provider-key-target",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: false},
				},
			},
			{
				Name: "compat-name-target",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("executor-provider", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "auth-provider",
		Attributes: map[string]string{
			"base_url":     server.URL + "/v1",
			"api_key":      "test",
			"compat_name":  "compat-name-target",
			"provider_key": "provider-key-target",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello"}`)
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "qwen3.5-plus",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/responses" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/responses")
	}
}

func TestOpenAICompatExecutorResolveUsageProviderPrefersCompatName(t *testing.T) {
	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{Name: "provider-key-target"},
			{Name: "compat-name-target"},
		},
	}
	executor := NewOpenAICompatExecutor("executor-provider", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "auth-provider",
		Attributes: map[string]string{
			"compat_name":  "compat-name-target",
			"provider_key": "provider-key-target",
		},
	}
	if got := executor.resolveUsageProvider(auth); got != "compat-name-target" {
		t.Fatalf("usage provider = %q, want %q", got, "compat-name-target")
	}
}

func TestOpenAICompatExecutorNativeResponsesStream(t *testing.T) {
	var gotPath string
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		gotBody = body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n"))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "alibaba-coding-plan",
		Attributes: map[string]string{
			"base_url":    server.URL + "/v1",
			"api_key":     "test",
			"compat_name": "alibaba-coding-plan",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello","stream":true}`)
	streamResp, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "qwen3.5-plus",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       true,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}
	if gotPath != "/v1/responses" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/responses")
	}
	if gjson.GetBytes(gotBody, "stream_options.include_usage").Exists() {
		t.Fatalf("unexpected chat streaming usage shim in native responses body")
	}
	if !gjson.GetBytes(gotBody, "input").Exists() {
		t.Fatalf("expected input in native responses stream body")
	}
	if got := gjson.GetBytes(gotBody, "model").String(); got != "qwen3.5-plus" {
		t.Fatalf("model = %q, want %q", got, "qwen3.5-plus")
	}
	count := 0
	for chunk := range streamResp.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		count++
	}
	if count == 0 {
		t.Fatalf("expected stream chunks")
	}
}

func TestOpenAICompatExecutorNativeResponsesPrefersUpstreamModelOverAlias(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","output":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
					{Name: "glm-5", Alias: "shared-alias", NativeResponses: false},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "alibaba-coding-plan",
		Attributes: map[string]string{
			"base_url":    server.URL + "/v1",
			"api_key":     "test",
			"compat_name": "alibaba-coding-plan",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello"}`)
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "glm-5",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/chat/completions")
	}
}

func TestOpenAICompatExecutorNativeResponsesAmbiguousAliasFallsBackToChat(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"READY"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
					{Name: "glm-5", Alias: "shared-alias", NativeResponses: false},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "alibaba-coding-plan",
		Attributes: map[string]string{
			"base_url":    server.URL + "/v1",
			"api_key":     "test",
			"compat_name": "alibaba-coding-plan",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello"}`)
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "shared-alias",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/chat/completions")
	}
}

func TestOpenAICompatExecutorNativeResponsesMixedAliasPoolFallsBackEvenForNativeMember(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"READY"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
					{Name: "glm-5", Alias: "shared-alias", NativeResponses: false},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "alibaba-coding-plan",
		Attributes: map[string]string{
			"base_url":    server.URL + "/v1",
			"api_key":     "test",
			"compat_name": "alibaba-coding-plan",
		},
	}
	payload := []byte(`{"model":"shared-alias","input":"hello"}`)
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "qwen3.5-plus",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: "shared-alias",
		},
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/chat/completions")
	}
}

func TestOpenAICompatExecutorNativeResponsesMixedAliasPoolWithoutMetadataFallsBackToChat(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"READY"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{
			{
				Name: "alibaba-coding-plan",
				Models: []config.OpenAICompatibilityModel{
					{Name: "qwen3.5-plus", Alias: "shared-alias", NativeResponses: true},
					{Name: "glm-5", Alias: "shared-alias", NativeResponses: false},
				},
			},
		},
	}
	executor := NewOpenAICompatExecutor("alibaba-coding-plan", cfg)
	auth := &cliproxyauth.Auth{
		Provider: "alibaba-coding-plan",
		Attributes: map[string]string{
			"base_url": server.URL + "/v1",
			"api_key":  "test",
		},
	}
	payload := []byte(`{"model":"qwen3.5-plus","input":"hello"}`)
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "qwen3.5-plus",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q, want %q", gotPath, "/v1/chat/completions")
	}
}
