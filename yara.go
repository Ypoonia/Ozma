package ozma

import (
	"embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

//go:embed src/doc_analyse/detection/default.yara
var yaraAssets embed.FS

const DefaultYaraRulesFile = "default.yara"

type YaraGlossaryError struct{ Message string }

func (e YaraGlossaryError) Error() string { return e.Message }

type compiledPattern struct {
	Name    string
	Regexp  *regexp.Regexp
	Pattern string
}

type yaraRuleMeta struct {
	RuleID                string
	Category              string
	Severity              string
	Weight                float64
	RouteHint             string
	RequiresLLMValidation bool
	Reason                string
}

type compiledRule struct {
	Name      string
	Meta      yaraRuleMeta
	Patterns  []compiledPattern
	Condition string
}

type YaraDetector struct {
	compiled   []compiledRule
	ruleSource string
	command    string
	useNative  bool
}

func NewYaraDetector() (*YaraDetector, error) {
	source, err := yaraAssets.ReadFile("src/doc_analyse/detection/default.yara")
	if err != nil {
		return nil, YaraGlossaryError{Message: "Default YARA rules failed to load."}
	}
	if command, err := exec.LookPath("yara"); err == nil {
		return &YaraDetector{ruleSource: string(source), command: command, useNative: true}, nil
	}
	compiled, err := CompileYaraRules(string(source))
	if err != nil {
		return nil, err
	}
	return &YaraDetector{compiled: compiled, ruleSource: string(source)}, nil
}

func NewNativeYaraDetector(command string) (*YaraDetector, error) {
	if command == "" {
		command = "yara"
	}
	resolved, err := exec.LookPath(command)
	if err != nil {
		return nil, YaraGlossaryError{Message: "real YARA backend requires the yara CLI in PATH"}
	}
	source, err := yaraAssets.ReadFile("src/doc_analyse/detection/default.yara")
	if err != nil {
		return nil, YaraGlossaryError{Message: "Default YARA rules failed to load."}
	}
	return &YaraDetector{ruleSource: string(source), command: resolved, useNative: true}, nil
}

func YaraDetectorFromFile(path string) (*YaraDetector, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, YaraGlossaryError{Message: fmt.Sprintf("Could not read YARA rules file '%s': %v", path, err)}
	}
	if command, err := exec.LookPath("yara"); err == nil {
		return &YaraDetector{ruleSource: string(b), command: command, useNative: true}, nil
	}
	compiled, err := CompileYaraRules(string(b))
	if err != nil {
		return nil, err
	}
	return &YaraDetector{compiled: compiled, ruleSource: string(b)}, nil
}

func CompileYaraRules(source string) ([]compiledRule, error) {
	blocks := splitRuleBlocks(source)
	rules := []compiledRule{}
	for _, block := range blocks {
		rule, err := parseRuleBlock(block)
		if err != nil {
			return nil, err
		}
		if rule.Name != "" {
			rules = append(rules, rule)
		}
	}
	return rules, nil
}

func (d *YaraDetector) Detect(chunk TextChunk) ([]DetectionFinding, error) {
	if strings.TrimSpace(chunk.Text) == "" {
		return nil, nil
	}
	if d == nil {
		return nil, YaraGlossaryError{Message: "Default YARA rules failed to load. Supply custom rules via YaraDetectorFromFile()."}
	}
	if d.useNative {
		return d.detectNative(chunk)
	}
	if len(d.compiled) == 0 {
		return nil, YaraGlossaryError{Message: "Default YARA rules failed to load. Supply custom rules via YaraDetectorFromFile()."}
	}
	byteToChar, _ := chunk.Metadata["byte_to_char"].([]int)
	if byteToChar == nil {
		byteToChar = BuildByteToChar(chunk.Text)
	}
	findings := []DetectionFinding{}
	for _, rule := range d.compiled {
		matchesByName := map[string][]regexpMatch{}
		for _, pattern := range rule.Patterns {
			for _, loc := range pattern.Regexp.FindAllStringIndex(chunk.Text, -1) {
				span := chunk.Text[loc[0]:loc[1]]
				if strings.TrimSpace(span) == "" {
					continue
				}
				matchesByName[pattern.Name] = append(matchesByName[pattern.Name], regexpMatch{span: span, startByte: loc[0]})
			}
		}
		if !ruleConditionMatches(rule, matchesByName) {
			continue
		}
		for _, matches := range matchesByName {
			for _, match := range matches {
				charOffset := match.startByte
				if match.startByte >= 0 && match.startByte < len(byteToChar) {
					charOffset = byteToChar[match.startByte]
				}
				score := (*float64)(nil)
				if rule.Meta.Weight > 0 {
					v := rule.Meta.Weight / 100
					if v > 1 {
						v = 1
					}
					score = &v
				}
				findings = append(findings, BuildFinding(
					chunk,
					match.span,
					rule.Meta.Category,
					rule.Meta.Severity,
					rule.Meta.Reason,
					rule.Meta.RuleID,
					chunk.StartChar+charOffset,
					chunk.StartChar+charOffset+utf8.RuneCountInString(match.span),
					score,
					rule.Meta.RequiresLLMValidation,
					Metadata{
						"detector":    "YaraDetector",
						"yara_rule":   rule.Meta.RuleID,
						"yara_weight": rule.Meta.Weight,
						"route_hint":  rule.Meta.RouteHint,
					},
				))
			}
		}
	}
	return finalizeFindings(findings), nil
}

func (d *YaraDetector) detectNative(chunk TextChunk) ([]DetectionFinding, error) {
	if d.command == "" {
		return nil, YaraGlossaryError{Message: "real YARA backend requires the yara CLI in PATH"}
	}
	tmpDir, err := os.MkdirTemp("", "ozma-yara-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	rulesPath := filepath.Join(tmpDir, "rules.yara")
	chunkPath := filepath.Join(tmpDir, "chunk.txt")
	if err := os.WriteFile(rulesPath, []byte(d.ruleSource), 0o600); err != nil {
		return nil, err
	}
	if err := os.WriteFile(chunkPath, []byte(chunk.Text), 0o600); err != nil {
		return nil, err
	}
	cmd := exec.Command(d.command, "-s", "-m", rulesPath, chunkPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok && exit.ExitCode() == 1 {
			return nil, nil
		}
		return nil, YaraGlossaryError{Message: fmt.Sprintf("YARA CLI failed: %v: %s", err, strings.TrimSpace(string(output)))}
	}
	return parseYaraCLIOutput(string(output), chunk)
}

func parseYaraCLIOutput(output string, chunk TextChunk) ([]DetectionFinding, error) {
	byteToChar, _ := chunk.Metadata["byte_to_char"].([]int)
	if byteToChar == nil {
		byteToChar = BuildByteToChar(chunk.Text)
	}
	headerRE := regexp.MustCompile(`^(\S+)(?:\s+\[(.*)\])?\s+`)
	stringRE := regexp.MustCompile(`^\s*0x([0-9A-Fa-f]+):\$\w+:\s?(.*)$`)
	var current *yaraRuleMeta
	findings := []DetectionFinding{}
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		if m := stringRE.FindStringSubmatch(line); len(m) == 3 && current != nil {
			offset, _ := strconv.ParseInt(m[1], 16, 64)
			span := m[2]
			if strings.TrimSpace(span) == "" {
				continue
			}
			charOffset := int(offset)
			if charOffset >= 0 && charOffset < len(byteToChar) {
				charOffset = byteToChar[charOffset]
			}
			var score *float64
			if current.Weight > 0 {
				v := current.Weight / 100
				if v > 1 {
					v = 1
				}
				score = &v
			}
			findings = append(findings, BuildFinding(chunk, span, current.Category, current.Severity, current.Reason, current.RuleID, chunk.StartChar+charOffset, chunk.StartChar+charOffset+utf8.RuneCountInString(span), score, current.RequiresLLMValidation, Metadata{
				"detector":    "YaraDetector",
				"yara_rule":   current.RuleID,
				"yara_weight": current.Weight,
				"route_hint":  current.RouteHint,
			}))
			continue
		}
		if m := headerRE.FindStringSubmatch(line); len(m) >= 2 {
			meta := parseYaraMetadataInline("")
			meta.RuleID = m[1]
			if len(m) > 2 {
				meta = parseYaraMetadataInline(m[2])
				if meta.RuleID == "" {
					meta.RuleID = m[1]
				}
			}
			current = &meta
		}
	}
	return finalizeFindings(findings), nil
}

type regexpMatch struct {
	span      string
	startByte int
}

func splitRuleBlocks(source string) []string {
	re := regexp.MustCompile(`\brule\s+\w+\s*\{`)
	locs := re.FindAllStringIndex(source, -1)
	blocks := []string{}
	for i, loc := range locs {
		end := len(source)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		blocks = append(blocks, source[loc[0]:end])
	}
	return blocks
}

func parseRuleBlock(block string) (compiledRule, error) {
	nameMatch := regexp.MustCompile(`\brule\s+(\w+)\s*\{`).FindStringSubmatch(block)
	if len(nameMatch) < 2 {
		return compiledRule{}, nil
	}
	name := nameMatch[1]
	meta := parseYaraMeta(sectionBetween(block, "meta:", "strings:"), name)
	patterns, err := parseYaraStrings(sectionBetween(block, "strings:", "condition:"))
	if err != nil {
		return compiledRule{}, err
	}
	condition := strings.TrimSuffix(strings.TrimSpace(sectionUntilClose(block, "condition:")), "}")
	return compiledRule{Name: name, Meta: meta, Patterns: patterns, Condition: condition}, nil
}

func sectionBetween(block, startMarker, endMarker string) string {
	start := strings.Index(block, startMarker)
	if start < 0 {
		return ""
	}
	start += len(startMarker)
	end := strings.Index(block[start:], endMarker)
	if end < 0 {
		return block[start:]
	}
	return block[start : start+end]
}

func sectionUntilClose(block, startMarker string) string {
	start := strings.Index(block, startMarker)
	if start < 0 {
		return ""
	}
	return block[start+len(startMarker):]
}

func parseYaraMeta(section, ruleName string) yaraRuleMeta {
	values := map[string]string{}
	lineRE := regexp.MustCompile(`(?m)^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$`)
	for _, m := range lineRE.FindAllStringSubmatch(section, -1) {
		values[strings.ToLower(m[1])] = strings.TrimSpace(m[2])
	}
	get := func(key, fallback string) string {
		raw := values[strings.ToLower(key)]
		if raw == "" {
			return fallback
		}
		if strings.HasPrefix(raw, `"`) && strings.HasSuffix(raw, `"`) {
			return strings.Trim(raw, `"`)
		}
		return raw
	}
	weight, _ := strconv.ParseFloat(get("weight", "0"), 64)
	return yaraRuleMeta{
		RuleID:                get("rule_id", ruleName),
		Category:              get("category", "unknown"),
		Severity:              get("severity", "medium"),
		Weight:                weight,
		RouteHint:             get("route_hint", "evidence"),
		RequiresLLMValidation: parseBool(get("requires_llm_validation", "false")),
		Reason:                get("reason", "YARA rule matched."),
	}
}

func parseYaraMetadataInline(section string) yaraRuleMeta {
	meta := yaraRuleMeta{
		Category:  "unknown",
		Severity:  "medium",
		RouteHint: "evidence",
		Reason:    "YARA rule matched.",
	}
	fieldRE := regexp.MustCompile(`([A-Za-z_]\w*)=("[^"]*"|[^,\s]+)`)
	for _, m := range fieldRE.FindAllStringSubmatch(section, -1) {
		key := strings.ToLower(m[1])
		value := strings.Trim(m[2], `"`)
		switch key {
		case "rule_id":
			meta.RuleID = value
		case "category":
			meta.Category = value
		case "severity":
			meta.Severity = value
		case "weight":
			meta.Weight, _ = strconv.ParseFloat(value, 64)
		case "route_hint":
			meta.RouteHint = value
		case "requires_llm_validation":
			meta.RequiresLLMValidation = parseBool(value)
		case "reason":
			meta.Reason = value
		}
	}
	return meta
}

func parseYaraStrings(section string) ([]compiledPattern, error) {
	stringRE := regexp.MustCompile(`(?is)\$([A-Za-z_]\w*)\s*=\s*(/(?:[^/\\]|\\.)*/[A-Za-z]*|"(?:[^"\\]|\\.)*")(?:\s+nocase)?`)
	patterns := []compiledPattern{}
	for _, m := range stringRE.FindAllStringSubmatch(section, -1) {
		name, raw, full := m[1], m[2], strings.ToLower(m[0])
		nocase := strings.Contains(full, "nocase")
		var expr string
		if strings.HasPrefix(raw, "/") {
			last := strings.LastIndex(raw, "/")
			expr = raw[1:last]
			flags := raw[last+1:]
			nocase = nocase || strings.Contains(flags, "i")
		} else {
			unquoted, err := strconv.Unquote(raw)
			if err != nil {
				unquoted = strings.Trim(raw, `"`)
			}
			expr = regexp.QuoteMeta(unquoted)
		}
		if nocase {
			expr = "(?i)" + expr
		}
		re, err := regexp.Compile(expr)
		if err != nil {
			return nil, YaraGlossaryError{Message: fmt.Sprintf("Failed to compile YARA pattern %s: %v", name, err)}
		}
		patterns = append(patterns, compiledPattern{Name: name, Regexp: re, Pattern: raw})
	}
	return patterns, nil
}

func ruleConditionMatches(rule compiledRule, matches map[string][]regexpMatch) bool {
	condition := strings.ToLower(rule.Condition)
	if strings.Contains(condition, "any of them") {
		return len(matches) > 0
	}
	if strings.Contains(condition, "all of them") {
		return len(matches) >= len(rule.Patterns)
	}
	for _, pattern := range rule.Patterns {
		condition = strings.ReplaceAll(condition, "$"+strings.ToLower(pattern.Name), fmt.Sprintf("%t", len(matches[pattern.Name]) > 0))
	}
	condition = strings.ReplaceAll(condition, "(", " ")
	condition = strings.ReplaceAll(condition, ")", " ")
	orParts := strings.Split(condition, " or ")
	for _, orPart := range orParts {
		andParts := strings.Split(orPart, " and ")
		ok := true
		sawBool := false
		for _, part := range andParts {
			part = strings.TrimSpace(part)
			if part == "true" {
				sawBool = true
				continue
			}
			if part == "false" {
				sawBool = true
				ok = false
			}
		}
		if sawBool && ok {
			return true
		}
	}
	return len(matches) > 0 && !strings.Contains(condition, "true") && !strings.Contains(condition, "false")
}

func parseBool(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "true", "yes", "1", "on":
		return true
	default:
		return false
	}
}
