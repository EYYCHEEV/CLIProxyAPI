package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

// keysTabModel displays provider key configuration.
type keysTabModel struct {
	client   *Client
	viewport viewport.Model
	gemini   []map[string]any
	claude   []map[string]any
	codex    []map[string]any
	vertex   []map[string]any
	openai   []map[string]any
	err      error
	width    int
	height   int
	ready    bool
	status   string
}

type keysDataMsg struct {
	gemini []map[string]any
	claude []map[string]any
	codex  []map[string]any
	vertex []map[string]any
	openai []map[string]any
	err    error
}

type keyActionMsg struct {
	action string
	err    error
}

func newKeysTabModel(client *Client) keysTabModel {
	return keysTabModel{
		client: client,
	}
}

func (m keysTabModel) Init() tea.Cmd {
	return m.fetchKeys
}

func (m keysTabModel) fetchKeys() tea.Msg {
	result := keysDataMsg{}
	var err error
	result.gemini, err = m.client.GetGeminiKeys()
	if err != nil {
		result.err = err
		return result
	}
	result.claude, err = m.client.GetClaudeKeys()
	if err != nil {
		result.err = err
		return result
	}
	result.codex, err = m.client.GetCodexKeys()
	if err != nil {
		result.err = err
		return result
	}
	result.vertex, err = m.client.GetVertexKeys()
	if err != nil {
		result.err = err
		return result
	}
	result.openai, err = m.client.GetOpenAICompat()
	if err != nil {
		result.err = err
		return result
	}
	return result
}

func (m keysTabModel) Update(msg tea.Msg) (keysTabModel, tea.Cmd) {
	switch msg := msg.(type) {
	case localeChangedMsg:
		m.viewport.SetContent(m.renderContent())
		return m, nil
	case keysDataMsg:
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.err = nil
			m.gemini = msg.gemini
			m.claude = msg.claude
			m.codex = msg.codex
			m.vertex = msg.vertex
			m.openai = msg.openai
		}
		m.viewport.SetContent(m.renderContent())
		return m, nil

	case keyActionMsg:
		if msg.err != nil {
			m.status = errorStyle.Render("✗ " + msg.err.Error())
		} else {
			m.status = successStyle.Render("✓ " + msg.action)
		}
		m.viewport.SetContent(m.renderContent())
		return m, m.fetchKeys

	case tea.KeyMsg:
		switch msg.String() {
		case "r":
			m.status = ""
			return m, m.fetchKeys
		default:
			var cmd tea.Cmd
			m.viewport, cmd = m.viewport.Update(msg)
			return m, cmd
		}
	}

	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	return m, cmd
}

func (m *keysTabModel) SetSize(w, h int) {
	m.width = w
	m.height = h
	if !m.ready {
		m.viewport = viewport.New(w, h)
		m.viewport.SetContent(m.renderContent())
		m.ready = true
	} else {
		m.viewport.Width = w
		m.viewport.Height = h
	}
}

func (m keysTabModel) View() string {
	if !m.ready {
		return T("loading")
	}
	return m.viewport.View()
}

func (m keysTabModel) renderContent() string {
	var sb strings.Builder

	sb.WriteString(titleStyle.Render(T("keys_title")))
	sb.WriteString("\n")
	sb.WriteString(helpStyle.Render(T("keys_help")))
	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("─", m.width))
	sb.WriteString("\n")

	if m.err != nil {
		sb.WriteString(errorStyle.Render(T("error_prefix") + m.err.Error()))
		sb.WriteString("\n")
		return sb.String()
	}

	// ━━━ Provider keys (read-only display) ━━━
	renderProviderKeys(&sb, "Gemini API Keys", m.gemini)
	renderProviderKeys(&sb, "Claude API Keys", m.claude)
	renderProviderKeys(&sb, "Codex API Keys", m.codex)
	renderProviderKeys(&sb, "Vertex API Keys", m.vertex)

	if len(m.openai) > 0 {
		renderSection(&sb, "OpenAI Compatibility", len(m.openai))
		for i, entry := range m.openai {
			name := getString(entry, "name")
			baseURL := getString(entry, "base-url")
			prefix := getString(entry, "prefix")
			info := name
			if prefix != "" {
				info += " (prefix: " + prefix + ")"
			}
			if baseURL != "" {
				info += " → " + baseURL
			}
			sb.WriteString(fmt.Sprintf("  %d. %s\n", i+1, info))
		}
		sb.WriteString("\n")
	}

	if m.status != "" {
		sb.WriteString(m.status)
		sb.WriteString("\n")
	}

	return sb.String()
}

func renderSection(sb *strings.Builder, title string, count int) {
	header := fmt.Sprintf("%s (%d)", title, count)
	sb.WriteString(tableHeaderStyle.Render("  " + header))
	sb.WriteString("\n")
}

func renderProviderKeys(sb *strings.Builder, title string, keys []map[string]any) {
	if len(keys) == 0 {
		return
	}
	renderSection(sb, title, len(keys))
	for i, key := range keys {
		apiKey := getString(key, "api-key")
		prefix := getString(key, "prefix")
		baseURL := getString(key, "base-url")
		info := maskKey(apiKey)
		if prefix != "" {
			info += " (prefix: " + prefix + ")"
		}
		if baseURL != "" {
			info += " → " + baseURL
		}
		sb.WriteString(fmt.Sprintf("  %d. %s\n", i+1, info))
	}
	sb.WriteString("\n")
}

func maskKey(key string) string {
	if len(key) <= 8 {
		return strings.Repeat("*", len(key))
	}
	return key[:4] + strings.Repeat("*", len(key)-8) + key[len(key)-4:]
}
