# End-to-end benchmark workflow

English | [简体中文](END_TO_END_zh-CN.md)

## 1. Freeze and validate prompts

The first public release consumes `data/prompts/t2av_605.jsonl` together with
`data/prompts/i2av_lat_143.jsonl`. `validate-release-suite` checks their exact
totals, evaluator memberships, pool assignments, and I2AV conditioning assets.

## 2. Export generation requests

`acoustitrace export-generation-requests` writes one model-neutral JSON object
per prompt. A generator adapter consumes:

```text
prompt_id, task, prompt, evaluator_membership,
conditioning_asset_id and/or conditioning_asset_path,
output_subdir, output_filename, output_relpath
```

T2AV adapters use text only. I2AV adapters use the prompt and the referenced
conditioning image. Model-specific seeds, duration, resolution, API job IDs,
and provider revisions belong in adapter provenance rather than in the frozen
prompt text.

The generator must write to the supplied `output_relpath`. Canonical names are
`t2av/<prompt_id>.mp4` and `i2av_lat/<prompt_id>.mp4`; do not rewrite or
re-index the ID or add a model prefix.

## 3. Return generated outputs

Each attempted prompt must produce one output-manifest row, including failed
attempts:

```text
prompt_id,task,model,video_path,status,error,seed,generator_revision
```

`status=success` requires a video path. Failed or unreadable outputs remain in
the manifest so validity coverage can be reported. Validate with:

```bash
acoustitrace validate-outputs outputs.csv --prompts prompts.jsonl --check-files
```

With the one-command `--videos-dir` interface, no output manifest is needed,
but the canonical MP4 layout is enforced. Custom locations or other containers
are supported only through an explicit `--outputs` manifest.

## 4. Normalize media

Before evaluator inference, use a lossless or high-quality intermediate and
record the source codec. Preserve the original soundtrack. Do not replace
missing audio with silence, because that changes an invalid generation into a
scored acoustic failure.

## 5. Extract structured evidence

Route each prompt to its evaluator memberships:

| Evaluator | Evidence backend |
|---|---|
| Range Attenuation | receiver depth/mask track plus audio-level curve |
| Approach Gain | receiver depth/mask track plus native approach readout |
| Lateral Stability | constant-range lateral track plus level stability |
| Motion--Loudness | AV event localization, visual ordering, event RMS |
| Impact Decay | matched impact window and decay-envelope fit |
| Causality Violation | clustered AV events and onset delays |
| Log Attack Time | generated/reference impact attack duration |
| RT60 Consistency | audio apparent RT60 and visual RT60 estimate |

Backend scripts live in `experiments/evaluator_backends/`. They output the
structured fields listed in `docs/EVIDENCE_SCHEMA.md`.

## 6. Score and aggregate

Evidence JSONL is mapped to a row per model/prompt/evaluator with
`acoustitrace score-batch`. Then `acoustitrace aggregate` reports conditional
means, valid/attempted counts, coverage, and percentile bootstrap intervals.

`python evaluate.py` orchestrates these stages without changing the backend
score definitions. Each heavy backend writes structured evidence first; score
mapping and validity-aware aggregation happen only after that auditable
boundary. See `docs/DEVELOPMENT_MACHINE.md` for the public directory layout and
single command.
