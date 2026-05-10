/*
  YARA rules for prompt-injection detection.
  One rule per trigger pattern. Each rule exports metadata used to
  build DetectionFinding objects.

  Meta fields per rule:
    rule_id    — stable identifier for findings
    category   — logical grouping (instruction_override, secret_exfiltration, etc.)
    severity   — critical | high | medium | low (used for score if weight not set)
    weight     — 0-100, direct signal contribution to CheapRouter yara_score
    route_hint — evidence | review | hold (soft hint to router, not authoritative)
    requires_llm_validation — true/false, advisory flag for external callers
    reason     — human-readable explanation of why the rule fired
*/

rule instruction_override {
  meta:
    rule_id    = "instruction_override"
    category   = "instruction_override"
    severity   = "high"
    weight     = 40
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "Attempts to override existing instructions."
  strings:
    $a = /ignore\s+(all\s+)?previous\s+instructions?/ nocase
    $b = /disregard\s+(all\s+)?previous\s+instructions?/ nocase
    $c = /forget\s+(all\s+)?previous\s+instructions?/ nocase
  condition:
    $a or $b or $c
}

rule system_override {
  meta:
    rule_id    = "system_override"
    category   = "instruction_override"
    severity   = "high"
    weight     = 40
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "References higher-priority system or developer instructions."
  strings:
    $a = /system\s+(override|instruction|prompt)s?/ nocase
    $b = /developer\s+(override|instruction|prompt)s?/ nocase
    $verb = /(\bignore\b|\bdisregard\b|\boverride\b|\bbypass\b|\breplace\b)/ nocase
  condition:
    ($a or $b) and $verb
}

rule hidden_prompt_exfiltration {
  meta:
    rule_id    = "hidden_prompt_exfiltration"
    category   = "secret_exfiltration"
    severity   = "critical"
    weight     = 50
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "Requests hidden prompts, hidden instructions, or private tool schema details."
  strings:
    /* exfil verb THEN system/developer prompt (within 80 chars) */
    $a = /\b(show|print|reveal|dump|repeat|leak|output|expose|tell)\b[\s\S]{0,80}\b(system|developer)\s+(prompt|message|instructions?)\b/ nocase
    /* system/developer prompt THEN exfil verb (within 80 chars) */
    $b = /\b(system|developer)\s+(prompt|message|instructions?)\b[\s\S]{0,80}\b(show|print|reveal|dump|repeat|leak|output|expose|tell)\b/ nocase
    /* exfil verb THEN hidden instructions (within 80 chars) */
    $c = /\b(show|print|reveal|dump|list|expose)\b[\s\S]{0,80}\bhidden\s+instructions?\b/ nocase
    /* exfil verb THEN private tool schemas (within 80 chars) */
    $d = /\b(show|print|reveal|dump|list|expose)\b[\s\S]{0,80}\bprivate\s+tool\s+schemas?\b/ nocase
    /* corpus wording: include available prompt/schema material in output */
    $e = /\b(include|collect|copy|extract|export|list)\b[\s\S]{0,120}\b(any\s+available\s+)?(system|developer)\s+prompts?\b/ nocase
    $f = /\b(include|collect|copy|extract|export|list)\b[\s\S]{0,120}\bhidden\s+instructions?\b/ nocase
    $g = /\b(include|collect|copy|extract|export|list)\b[\s\S]{0,120}\bprivate\s+tool\s+schemas?\b/ nocase
    $h = /\b(all_internal_prompts\.xlsx|hidden_tool_schemas\.csv)\b/ nocase
  condition:
    any of them
}

rule credential_exfiltration {
  meta:
    rule_id    = "credential_exfiltration"
    category   = "secret_exfiltration"
    severity   = "critical"
    weight     = 50
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "Requests credentials, environment values, or secret exports."
  strings:
    /* exfil verb THEN credential term (within 80 chars) */
    $a = /\b(show|print|dump|leak|reveal|export|extract|tell)\b[\s\S]{0,80}\bapi\s+keys?\b/ nocase
    $b = /\bapi\s+keys?\b[\s\S]{0,80}\b(show|print|dump|leak|reveal|export|extract|tell)\b/ nocase
    /* exfil verb THEN database connection strings (within 80 chars) */
    $c = /\b(show|print|dump|leak|reveal|export|extract|tell)\b[\s\S]{0,80}\bdatabase\s+connection\s+strings?\b/ nocase
    $d = /\bdatabase\s+connection\s+strings?\b[\s\S]{0,80}\b(show|print|dump|leak|reveal|export|extract|tell)\b/ nocase
    /* exfil verb THEN environment variables (within 80 chars) */
    $e = /\b(show|print|dump|leak|reveal|export|extract|tell)\b[\s\S]{0,80}\benvironment\s+variables?\b/ nocase
    $f = /\benvironment\s+variables?\b[\s\S]{0,80}\b(show|print|dump|leak|reveal|export|extract|tell)\b/ nocase
    /* exfil verb THEN secrets file (within 80 chars) */
    $g = /\b(show|print|dump|leak|reveal|export|extract|tell)\b[\s\S]{0,80}\bsecrets?\.(csv|xlsx)\b/ nocase
    $h = /\bsecrets?\.(csv|xlsx)\b[\s\S]{0,80}\b(show|print|dump|leak|reveal|export|extract|tell)\b/ nocase
    /* corpus wording: include available credentials in final output */
    $i = /\b(include|collect|copy|extract|export|list)\b[\s\S]{0,120}\b(api\s+keys?|database\s+connection\s+strings?|environment\s+variables?)\b/ nocase
    $j = /\benvironment\s+variables?\b[\s\S]{0,80}\b(exports?|extracts?|prints?|dumps?|leaks?|reveals?|shows?|tells?)\b/ nocase
  condition:
    any of them
}

rule tool_hijack {
  meta:
    rule_id    = "tool_hijack"
    category   = "tool_hijack"
    severity   = "high"
    weight     = 35
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "Attempts to make an agent call tools from uploaded content."
  strings:
    $a = /call\s+(the\s+)?(available\s+)?write\s+tool/ nocase
    $b = /use\s+(the\s+)?(available\s+)?write\s+tool/ nocase
    $c = /invoke\s+(the\s+)?(available\s+)?write\s+tool/ nocase
    $d = /run\s+(the\s+)?(available\s+)?write\s+tool/ nocase
  condition:
    $a or $b or $c or $d
}

rule unsafe_mutation_request {
  meta:
    rule_id    = "unsafe_mutation_request"
    category   = "unsafe_action"
    severity   = "high"
    weight     = 25
    route_hint = "review"
    requires_llm_validation = true
    reason     = "Requests write, delete, export, send, or mutation behavior."
  strings:
    /* subject (assistant/agent/model) within 80 chars of mutation verb */
    $a = /\b(assistant|chatgpt|model|agent|you)\b[\s\S]{0,80}\b(write|update|delete|remove|create|export|send)\b[\s\S]{0,80}\b(file|database|record|email|message|url|endpoint|csv|xlsx)\b/ nocase
    /* tool/function call within 60 chars of mutation verb */
    $b = /\b(use|call|invoke|run)\b[\s\S]{0,60}\b(tool|function|api|endpoint)\b[\s\S]{0,80}\b(write|update|delete|remove|create|export|send)\b/ nocase
    /* mutation verb within 60 chars of target object */
    $c = /\b(write|update|delete|remove|create|export|send)\b[\s\S]{0,60}\b(file|database|record|email|message|url|endpoint|csv|xlsx)\b/ nocase
  condition:
    any of them
}

rule safety_bypass {
  meta:
    rule_id    = "safety_bypass"
    category   = "safety_bypass"
    severity   = "high"
    weight     = 40
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "Attempts to control safety classification."
  strings:
    $a = /return\s+(this\s+)?(as\s+)?(safe|benign)/ nocase
    $b = /mark\s+(this\s+)?(as\s+)?(safe|benign)/ nocase
    $c = /classify\s+(this\s+)?(as\s+)?(safe|benign)/ nocase
    $d = /safe_to_forward\s*=\s*true/ nocase
  condition:
    $a or $b or $c or $d
}

rule concealment {
  meta:
    rule_id    = "concealment"
    category   = "concealment"
    severity   = "medium"
    weight     = 15
    route_hint = "review"
    requires_llm_validation = false
    reason     = "Attempts to hide or suppress evidence of the instruction."
  strings:
    $a = /do\s+not\s+(reveal|mention|explain|classify)/ nocase
    $b = /don't\s+(reveal|mention|explain|classify)/ nocase
  condition:
    $a or $b
}

rule authority_claim {
  meta:
    rule_id    = "authority_claim"
    category   = "instruction_override"
    severity   = "high"
    weight     = 30
    route_hint = "hold"
    requires_llm_validation = true
    reason     = "Claims authority over system, developer, or policy instructions."
  strings:
    $a = /this\s+document\s+wins/ nocase
    $b = /new\s+authority/ nocase
    $c = /mandatory\s+instruction/ nocase
    $d = /supersedes?\s+any\s+previous\s+instruction/ nocase
  condition:
    $a or $b or $c or $d
}
