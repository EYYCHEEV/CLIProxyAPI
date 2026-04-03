// config_reload.go implements debounced configuration hot reload.
// It detects material changes and reloads clients when the config changes.
package watcher

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"github.com/joho/godotenv"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/watcher/diff"
	"gopkg.in/yaml.v3"

	log "github.com/sirupsen/logrus"
)

func (w *Watcher) stopConfigReloadTimer() {
	w.configReloadMu.Lock()
	if w.configReloadTimer != nil {
		w.configReloadTimer.Stop()
		w.configReloadTimer = nil
	}
	w.configReloadMu.Unlock()
}

func (w *Watcher) scheduleConfigReload() {
	w.configReloadMu.Lock()
	defer w.configReloadMu.Unlock()
	if w.configReloadTimer != nil {
		w.configReloadTimer.Stop()
	}
	w.configReloadTimer = time.AfterFunc(configReloadDebounce, func() {
		w.configReloadMu.Lock()
		w.configReloadTimer = nil
		w.configReloadMu.Unlock()
		w.reloadConfigIfChanged()
	})
}

func (w *Watcher) reloadConfigIfChanged() {
	data, err := os.ReadFile(w.configPath)
	if err != nil {
		log.Errorf("failed to read config file for hash check: %v", err)
		return
	}
	if len(data) == 0 {
		log.Debugf("ignoring empty config file write event")
		return
	}
	sum := sha256.Sum256(data)
	newHash := hex.EncodeToString(sum[:])
	newBundleEnvHash := w.bundleEnvHash()

	w.clientsMutex.RLock()
	currentHash := w.lastConfigHash
	currentBundleEnvHash := w.lastBundleEnvHash
	w.clientsMutex.RUnlock()

	if currentHash != "" && currentHash == newHash && currentBundleEnvHash == newBundleEnvHash {
		log.Debugf("config file content unchanged (config/env hash match), skipping reload")
		return
	}
	log.Infof("config file changed, reloading: %s", w.configPath)
	if w.reloadConfig() {
		finalHash := newHash
		finalBundleEnvHash := newBundleEnvHash
		if updatedData, errRead := os.ReadFile(w.configPath); errRead == nil && len(updatedData) > 0 {
			sumUpdated := sha256.Sum256(updatedData)
			finalHash = hex.EncodeToString(sumUpdated[:])
		} else if errRead != nil {
			log.WithError(errRead).Debug("failed to compute updated config hash after reload")
		}
		finalBundleEnvHash = w.bundleEnvHash()
		w.clientsMutex.Lock()
		w.lastConfigHash = finalHash
		w.lastBundleEnvHash = finalBundleEnvHash
		w.clientsMutex.Unlock()
		w.persistConfigAsync()
	}
}

func (w *Watcher) reloadConfig() bool {
	w.reloadBundleEnv()
	log.Debug("=========================== CONFIG RELOAD ============================")
	log.Debugf("starting config reload from: %s", w.configPath)

	newConfig, errLoadConfig := config.LoadConfig(w.configPath)
	if errLoadConfig != nil {
		log.Errorf("failed to reload config: %v", errLoadConfig)
		return false
	}

	if w.mirroredAuthDir != "" {
		newConfig.AuthDir = w.mirroredAuthDir
	} else {
		if resolvedAuthDir, errResolveAuthDir := util.ResolveAuthDir(newConfig.AuthDir); errResolveAuthDir != nil {
			log.Errorf("failed to resolve auth directory from config: %v", errResolveAuthDir)
		} else {
			newConfig.AuthDir = resolvedAuthDir
		}
	}

	w.clientsMutex.Lock()
	var oldConfig *config.Config
	_ = yaml.Unmarshal(w.oldConfigYaml, &oldConfig)
	w.oldConfigYaml, _ = yaml.Marshal(newConfig)
	w.config = newConfig
	w.clientsMutex.Unlock()

	var affectedOAuthProviders []string
	if oldConfig != nil {
		_, affectedOAuthProviders = diff.DiffOAuthExcludedModelChanges(oldConfig.OAuthExcludedModels, newConfig.OAuthExcludedModels)
	}

	util.SetLogLevel(newConfig)
	if oldConfig != nil && oldConfig.Debug != newConfig.Debug {
		log.Debugf("log level updated - debug mode changed from %t to %t", oldConfig.Debug, newConfig.Debug)
	}

	if oldConfig != nil {
		details := diff.BuildConfigChangeDetails(oldConfig, newConfig)
		if len(details) > 0 {
			log.Debugf("config changes detected:")
			for _, d := range details {
				log.Debugf("  %s", d)
			}
		} else {
			log.Debugf("no material config field changes detected")
		}
	}

	authDirChanged := oldConfig == nil || oldConfig.AuthDir != newConfig.AuthDir
	retryConfigChanged := oldConfig != nil && (oldConfig.RequestRetry != newConfig.RequestRetry || oldConfig.MaxRetryInterval != newConfig.MaxRetryInterval || oldConfig.MaxRetryCredentials != newConfig.MaxRetryCredentials)
	forceAuthRefresh := oldConfig != nil && (oldConfig.ForceModelPrefix != newConfig.ForceModelPrefix || !reflect.DeepEqual(oldConfig.OAuthModelAlias, newConfig.OAuthModelAlias) || retryConfigChanged)

	log.Infof("config successfully reloaded, triggering client reload")
	w.reloadClients(authDirChanged, affectedOAuthProviders, forceAuthRefresh)
	return true
}

func (w *Watcher) reloadBundleEnv() {
	if w == nil || w.configPath == "" {
		return
	}

	for _, envPath := range w.bundleEnvCandidates() {
		info, statErr := os.Stat(envPath)
		if statErr != nil || info.IsDir() {
			continue
		}
		if err := godotenv.Overload(envPath); err != nil {
			log.WithError(err).Warnf("failed to reload bundle env from %s", envPath)
		} else {
			log.Debugf("reloaded bundle env from: %s", envPath)
		}
		return
	}
}

func (w *Watcher) bundleEnvCandidates() []string {
	if w == nil || w.configPath == "" {
		return nil
	}

	configDir := filepath.Dir(w.configPath)
	parentDir := filepath.Dir(configDir)
	candidates := []string{filepath.Join(configDir, ".env")}
	if filepath.Base(configDir) == "generated" {
		if parentEnv := filepath.Join(parentDir, ".env"); parentEnv != candidates[0] {
			candidates = append([]string{parentEnv}, candidates...)
		}
	} else if parentEnv := filepath.Join(parentDir, ".env"); parentEnv != candidates[0] {
		candidates = append(candidates, parentEnv)
	}
	return candidates
}

func (w *Watcher) bundleEnvHash() string {
	for _, envPath := range w.bundleEnvCandidates() {
		info, statErr := os.Stat(envPath)
		if statErr != nil || info.IsDir() {
			continue
		}
		data, readErr := os.ReadFile(envPath)
		if readErr != nil {
			log.WithError(readErr).Debugf("failed to read bundle env for hash check: %s", envPath)
			return ""
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:])
	}
	return ""
}
