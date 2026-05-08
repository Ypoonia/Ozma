package ozma

import (
	"embed"
	"fmt"
	"os"
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
	compiled []compiledRule
}

func NewYaraDetector() (*YaraDetector, error) {
	source, err := yaraAssets.ReadFile("src/doc_analyse/detection/default.yara")
	if err != nil {
		return nil, YaraGlossaryError{Message: "Default YARA rules failed to load."}
	}
	compiled, err := CompileYaraRules(string(source))
	if err != nil {
		return nil, err
	}
	return &YaraDetector{compiled: compiled}, nil
}

func YaraDetectorFromFile(path string) (*YaraDetector, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, YaraGlossaryError{Message: fmt.Sprintf("Could not read YARA rules file '%s': %v", path, err)}
	}
	compiled, err := CompileYaraRules(string(b))
	if err != nil {
		return nil, err
	}
	return &YaraDetector{compiled: compiled}, nil
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
	if d == nil || len(d.compiled) == 0 {
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
