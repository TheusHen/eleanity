#!/usr/bin/env bash
# Download a tiny local model and run eleanity compare; print machine-readable values.
set -euo pipefail

MODEL="${ELEANITY_CI_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
BACKENDS="${ELEANITY_CI_BACKENDS:-transformers,transformers}"
POLICY="${ELEANITY_CI_POLICY:-strict}"
OUT_DIR="${ELEANITY_CI_OUT:-.eleanity/ci-out}"
TOKENIZER_ONLY="${ELEANITY_CI_TOKENIZER_ONLY:-0}"

mkdir -p "$OUT_DIR"

echo "==> doctor"
uv run eleanity doctor --format json | tee "$OUT_DIR/doctor.json"

echo "==> pull $MODEL"
uv run eleanity pull "$MODEL" --tokenizer-only 2>/dev/null || uv run eleanity pull "$MODEL" | tee "$OUT_DIR/pull.json"

ARGS=(compare --model "$MODEL" --backends "$BACKENDS" --policy "$POLICY" --format json --no-parallel --no-gates)
if [[ "$TOKENIZER_ONLY" == "1" ]]; then
  ARGS+=(--tokenizer-only --observe artifact,template,special_tokens,tokens)
else
  ARGS+=(--observe artifact,template,special_tokens,tokens,generation)
fi

echo "==> eleanity ${ARGS[*]}"
set +e
uv run eleanity "${ARGS[@]}" > "$OUT_DIR/compare.json" 2>"$OUT_DIR/compare.stderr"
CODE=$?
set -e
cat "$OUT_DIR/compare.stderr" || true

# Extract run_id from JSON or quiet fallback
OUT_DIR="$OUT_DIR" CODE="$CODE" python - <<'PY'
import json, sys, pathlib, os
path = pathlib.Path(os.environ["OUT_DIR"]) / "compare.json"
out = pathlib.Path(os.environ["OUT_DIR"])
code = int(os.environ["CODE"])
raw = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
# JSON may be preceded by log lines — find first {
idx = raw.find("{")
payload = {}
if idx >= 0:
    try:
        payload = json.loads(raw[idx:])
    except json.JSONDecodeError:
        payload = {}
summary = payload.get("summary") or {}
diagnosis = payload.get("diagnosis") or {}
coverage = payload.get("coverage") or diagnosis.get("coverage") or {}
run_id = summary.get("run_id") or diagnosis.get("run_id") or payload.get("run_id")
values = {
    "exit_code": code,
    "run_id": run_id,
    "status": summary.get("status") or diagnosis.get("status"),
    "formal_status": summary.get("formal_status") or diagnosis.get("formal_status"),
    "impact": summary.get("impact") or (diagnosis.get("impact") or {}).get("impact"),
    "first_divergence": summary.get("first_divergence") or diagnosis.get("first_divergence"),
    "confidence": summary.get("confidence") if summary.get("confidence") is not None else diagnosis.get("confidence"),
    "coverage_percent": summary.get("coverage_percent") or coverage.get("required_coverage_percent"),
    "verified_layers": payload.get("verified_layers") or diagnosis.get("verified_layers"),
    "not_verified_layers": payload.get("not_verified_layers") or diagnosis.get("not_verified_layers"),
    "gates_passed": summary.get("gates_passed"),
    "tolerance_reasons": payload.get("tolerance_reasons") or diagnosis.get("tolerance_reasons"),
    "reproduction_command": payload.get("reproduction_command"),
    "timings": payload.get("timings") or payload.get("timings_ms"),
}
(out / "values.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
print("=== ELEANITY VALUES ===")
print(json.dumps(values, indent=2))
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
        for k, v in values.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                fh.write(f"{k}={json.dumps(v)}\n")
            else:
                fh.write(f"{k}={v}\n")
sys.exit(0 if code in (0, 1) else code)  # allow DIVERGENT as captured result
PY

# Copy run dir if present
if [[ -f "$OUT_DIR/values.json" ]]; then
  RUN_ID=$(python -c "import json; print(json.load(open('$OUT_DIR/values.json')).get('run_id') or '')")
  if [[ -n "$RUN_ID" && -d ".eleanity/runs/$RUN_ID" ]]; then
    cp -a ".eleanity/runs/$RUN_ID" "$OUT_DIR/run" || true
    uv run eleanity report "$RUN_ID" --format text > "$OUT_DIR/report.txt" || true
  fi
fi

echo "==> done (values in $OUT_DIR/values.json)"
