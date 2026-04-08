package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadConfigRejectsTopLevelAPIKeys(t *testing.T) {
	t.Parallel()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	body := strings.Join([]string{
		`remote-management:`,
		`  secret-key: ""`,
		`api-keys:`,
		`  - old-style-key`,
	}, "\n")
	if err := os.WriteFile(configPath, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := LoadConfigOptional(configPath, false)
	if err == nil {
		t.Fatal("expected deprecated top-level api-keys to fail")
	}
	if !strings.Contains(err.Error(), deprecatedTopLevelAPIKeysMsg) {
		t.Fatalf("error = %v, want %q", err, deprecatedTopLevelAPIKeysMsg)
	}
}

func TestLoadConfigOptionalRejectsTopLevelAPIKeysInOptionalMode(t *testing.T) {
	t.Parallel()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	body := strings.Join([]string{
		`api-keys:`,
		`  - old-style-key`,
	}, "\n")
	if err := os.WriteFile(configPath, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := LoadConfigOptional(configPath, true)
	if err == nil {
		t.Fatal("expected deprecated top-level api-keys to fail in optional mode")
	}
	if !strings.Contains(err.Error(), deprecatedTopLevelAPIKeysMsg) {
		t.Fatalf("error = %v, want %q", err, deprecatedTopLevelAPIKeysMsg)
	}
}
