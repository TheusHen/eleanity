# PowerShell counterpart for local/Windows CI of tiny-model parity.
$ErrorActionPreference = "Stop"

$Model = if ($env:ELEANITY_CI_MODEL) { $env:ELEANITY_CI_MODEL } else { "HuggingFaceTB/SmolLM2-135M-Instruct" }
$Backends = if ($env:ELEANITY_CI_BACKENDS) { $env:ELEANITY_CI_BACKENDS } else { "transformers,transformers" }
$Policy = if ($env:ELEANITY_CI_POLICY) { $env:ELEANITY_CI_POLICY } else { "strict" }
$OutDir = if ($env:ELEANITY_CI_OUT) { $env:ELEANITY_CI_OUT } else { ".eleanity/ci-out" }
$TokenizerOnly = $env:ELEANITY_CI_TOKENIZER_ONLY -eq "1"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "==> doctor"
uv run eleanity doctor --format json | Tee-Object -FilePath "$OutDir/doctor.json"

Write-Host "==> pull $Model"
try {
  uv run eleanity pull $Model --tokenizer-only 2>$null | Tee-Object -FilePath "$OutDir/pull.json"
} catch {
  uv run eleanity pull $Model | Tee-Object -FilePath "$OutDir/pull.json"
}

$argsList = @(
  "compare", "--model", $Model, "--backends", $Backends, "--policy", $Policy,
  "--format", "json", "--no-parallel", "--no-gates"
)
if ($TokenizerOnly) {
  $argsList += @("--tokenizer-only", "--observe", "artifact,template,special_tokens,tokens")
} else {
  $argsList += @("--observe", "artifact,template,special_tokens,tokens,generation")
}

Write-Host "==> eleanity $($argsList -join ' ')"
$stderr = "$OutDir/compare.stderr"
$stdout = "$OutDir/compare.json"
& uv run eleanity @argsList 1>$stdout 2>$stderr
$code = $LASTEXITCODE
if (Test-Path $stderr) { Get-Content $stderr }

python -c @"
import json, pathlib, os, sys
raw = pathlib.Path(r'$OutDir/compare.json').read_text(encoding='utf-8', errors='replace')
idx = raw.find('{')
payload = json.loads(raw[idx:]) if idx >= 0 else {}
summary = payload.get('summary') or {}
diagnosis = payload.get('diagnosis') or {}
coverage = payload.get('coverage') or diagnosis.get('coverage') or {}
values = {
    'exit_code': $code,
    'run_id': summary.get('run_id') or payload.get('run_id'),
    'status': summary.get('status') or diagnosis.get('status'),
    'impact': summary.get('impact'),
    'first_divergence': summary.get('first_divergence') or diagnosis.get('first_divergence'),
    'confidence': summary.get('confidence') if summary.get('confidence') is not None else diagnosis.get('confidence'),
    'coverage_percent': summary.get('coverage_percent') or coverage.get('required_coverage_percent'),
    'verified_layers': payload.get('verified_layers') or diagnosis.get('verified_layers'),
    'gates_passed': summary.get('gates_passed'),
    'reproduction_command': payload.get('reproduction_command'),
    'timings': payload.get('timings') or payload.get('timings_ms'),
}
pathlib.Path(r'$OutDir/values.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
print('=== ELEANITY VALUES ===')
print(json.dumps(values, indent=2))
"@

Write-Host "==> done ($OutDir/values.json)"
exit 0
