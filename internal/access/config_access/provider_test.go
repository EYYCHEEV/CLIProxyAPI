package configaccess

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	sdkaccess "github.com/router-for-me/CLIProxyAPI/v6/sdk/access"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v6/sdk/config"
)

func TestRegisterSupportsEnvBackedKeys(t *testing.T) {
	t.Setenv("STRONK_GATEWAY_PUBLIC_API_KEY", "bundle-public-key")
	Register(&sdkconfig.SDKConfig{
		APIKeyEnvs: []string{"STRONK_GATEWAY_PUBLIC_API_KEY"},
	})
	t.Cleanup(func() {
		sdkaccess.UnregisterProvider(sdkaccess.AccessProviderTypeConfigAPIKey)
	})

	manager := sdkaccess.NewManager()
	manager.SetProviders(sdkaccess.RegisteredProviders())

	successReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	successReq.Header.Set("Authorization", "Bearer bundle-public-key")
	result, authErr := manager.Authenticate(context.Background(), successReq)
	if authErr != nil {
		t.Fatalf("expected env-backed auth to succeed, got %v", authErr)
	}
	if result == nil || result.Principal != "bundle-public-key" {
		t.Fatalf("expected principal bundle-public-key, got %#v", result)
	}

	wrongReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	wrongReq.Header.Set("Authorization", "Bearer wrong-key")
	_, authErr = manager.Authenticate(context.Background(), wrongReq)
	if !sdkaccess.IsAuthErrorCode(authErr, sdkaccess.AuthErrorCodeInvalidCredential) {
		t.Fatalf("expected invalid credential error, got %#v", authErr)
	}
}

func TestRegisterFailsClosedWhenEnvKeyIsMissing(t *testing.T) {
	Register(&sdkconfig.SDKConfig{
		APIKeyEnvs: []string{"STRONK_GATEWAY_PUBLIC_API_KEY"},
	})
	t.Cleanup(func() {
		sdkaccess.UnregisterProvider(sdkaccess.AccessProviderTypeConfigAPIKey)
	})

	manager := sdkaccess.NewManager()
	manager.SetProviders(sdkaccess.RegisteredProviders())

	noCredsReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	_, authErr := manager.Authenticate(context.Background(), noCredsReq)
	if !sdkaccess.IsAuthErrorCode(authErr, sdkaccess.AuthErrorCodeNoCredentials) {
		t.Fatalf("expected missing credentials error, got %#v", authErr)
	}

	wrongReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	wrongReq.Header.Set("Authorization", "Bearer wrong-key")
	_, authErr = manager.Authenticate(context.Background(), wrongReq)
	if !sdkaccess.IsAuthErrorCode(authErr, sdkaccess.AuthErrorCodeInvalidCredential) {
		t.Fatalf("expected invalid credential error, got %#v", authErr)
	}
}
