package ozma

import (
	"regexp"
	"strings"

	"golang.org/x/text/unicode/norm"
)

var (
	spaceCollapse   = regexp.MustCompile(`[ \t]+`)
	newlineCollapse = regexp.MustCompile(`\n{3,}`)
	zeroWidthRunes  = []string{"\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"}
)

func NormalizeForDetection(text string) string {
	text = norm.NFKC.String(text)
	for _, r := range zeroWidthRunes {
		text = strings.ReplaceAll(text, r, "")
	}
	text = spaceCollapse.ReplaceAllString(text, " ")
	text = newlineCollapse.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}
