import { cpSync, existsSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, "..");
const outputs = ["dist"];
const assets = [
  ["src/detection/default.yara", "default.yara"],
  ["src/prompt/system.md", "system.md"],
  ["src/prompt/classification.md", "classification.md"],
  ["src/prompt/classifieragent.md", "classifieragent.md"],
];

for (const output of outputs) {
  const outputDir = resolve(projectRoot, output);
  if (!existsSync(outputDir)) {
    continue;
  }

  mkdirSync(outputDir, { recursive: true });
  for (const [source, target] of assets) {
    cpSync(resolve(projectRoot, source), resolve(outputDir, target));
  }
}
