package config

import "testing"

func TestSanitizeOpenAICompatibility_TrimsModelsAndDropsIncomplete(t *testing.T) {
	cfg := &Config{
		OpenAICompatibility: []OpenAICompatibility{
			{
				Name:    " test-provider ",
				BaseURL: " https://example.com/v1 ",
				Models: []OpenAICompatibilityModel{
					{Name: " qwen3.5-plus ", Alias: " qwen3.5-plus ", NativeResponses: true},
					{Name: " glm-5 ", Alias: " glm-5 "},
					{Name: " kimi-k2.5 ", Alias: " "},
					{Name: " ", Alias: " orphan-alias "},
				},
			},
		},
	}

	cfg.SanitizeOpenAICompatibility()

	if len(cfg.OpenAICompatibility) != 1 {
		t.Fatalf("expected 1 provider, got %d", len(cfg.OpenAICompatibility))
	}
	provider := cfg.OpenAICompatibility[0]
	if provider.Name != "test-provider" {
		t.Fatalf("provider name = %q, want %q", provider.Name, "test-provider")
	}
	if provider.BaseURL != "https://example.com/v1" {
		t.Fatalf("provider base-url = %q, want %q", provider.BaseURL, "https://example.com/v1")
	}
	if len(provider.Models) != 2 {
		t.Fatalf("expected 2 sanitized models, got %d", len(provider.Models))
	}
	if provider.Models[0].Name != "qwen3.5-plus" || provider.Models[0].Alias != "qwen3.5-plus" || !provider.Models[0].NativeResponses {
		t.Fatalf("first model mismatch: %+v", provider.Models[0])
	}
	if provider.Models[1].Name != "glm-5" || provider.Models[1].Alias != "glm-5" || provider.Models[1].NativeResponses {
		t.Fatalf("second model mismatch: %+v", provider.Models[1])
	}
}
