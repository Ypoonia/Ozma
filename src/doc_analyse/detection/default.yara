/*
  YARA rules for prompt-injection detection.
  One rule per trigger pattern. Each rule exports metadata used to
  build DetectionFinding objects.
*/

rule instruction_override {
  meta:
    rule_id    = "instruction_override"
    category   = "instruction_override"
    severity   = "high"
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
    reason     = "Requests hidden prompts, hidden instructions, or private tool schema details."
  strings:
    $a = /system\s+prompt/ nocase
    $b = /developer\s+prompt/ nocase
    $c = /hidden\s+instructions?/ nocase
    $d = /private\s+tool\s+schemas?/ nocase
    $exfil = /(\bshow\b|\bdump\b|\bleak\b|\breveal\b|\bexfiltrate\b|\bextract\b|\bprint\b|\btell\b|\bsend\b)/ nocase
  condition:
    ($a or $b or $c or $d) and $exfil
}

rule credential_exfiltration {
  meta:
    rule_id    = "credential_exfiltration"
    category   = "secret_exfiltration"
    severity   = "critical"
    reason     = "Requests credentials, environment values, or secret exports."
  strings:
    $a = /api\s+keys?/ nocase
    $b = /database\s+connection\s+strings?/ nocase
    $c = /environment\s+variables?/ nocase
    $d = /secrets?\.(csv|xlsx)/ nocase
    $action = /(\bshow\b|\bdump\b|\bleak\b|\breveal\b|\bprint\b|\bextract\b|\bexfiltrate\b|\bsend\b|\bexport\b|\bforward\b)/ nocase
  condition:
    ($a or $b or $c or $d) and $action
}

rule tool_hijack {
  meta:
    rule_id    = "tool_hijack"
    category   = "tool_hijack"
    severity   = "high"
    reason     = "Attempts to make an agent call tools from uploaded content."
  strings:
    $a = /call\s+(the\s+)?(available\s+)?write\s+tool/ nocase
    $b = /use\s+(the\s+)?(available\s+)?write\s+tool/ nocase
    $c = /invoke\s+(the\s+)?(available\s+)?write\s+tool/ nocase
    $d = /run\s+(the\s+)?(available\s+)?write\s+tool/ nocase
  condition:
    $a or $b or $c or $d
}

rule write_operation {
  meta:
    rule_id    = "write_operation"
    category   = "unsafe_action"
    severity   = "high"
    reason     = "Requests write, delete, export, or mutation behavior."
  strings:
    $a = /write_/ nocase
    $b = /\bupdate\b/ nocase
    $c = /\bdelete\b/ nocase
    $d = /\bremove\b/ nocase
    $e = /mark\s+all/ nocase
    $f = /set\s+the\s+hierarchy/ nocase
    $g = /create\s+(a\s+)?(csv|excel|xlsx|file)/ nocase
  condition:
    $a or $b or $c or $d or $e or $f or $g
}

rule safety_bypass {
  meta:
    rule_id    = "safety_bypass"
    category   = "safety_bypass"
    severity   = "high"
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
    reason     = "Claims authority over system, developer, or policy instructions."
  strings:
    $a = /this\s+document\s+wins/ nocase
    $b = /new\s+authority/ nocase
    $c = /mandatory\s+instruction/ nocase
    $d = /supersedes?\s+any\s+previous\s+instruction/ nocase
  condition:
    $a or $b or $c or $d
}
