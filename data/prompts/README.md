# Public release prompt manifests

The first open-source evaluator release contains exactly:

```text
t2av_605.jsonl
i2av_lat_143.jsonl
```

This is the rebalanced public release suite, not the submitted-paper prompt
assignment distribution. Newly produced scores must be labelled accordingly
and must not be substituted into the frozen paper leaderboard.

Each JSONL row uses this schema:

```json
{
  "prompt_id": "stable-id",
  "task": "t2av",
  "prompt_text": "Generation prompt",
  "category": "receiver_distance",
  "evaluator_membership": ["range_attenuation"],
  "conditioning_asset_id": "",
  "conditioning_asset_path": "",
  "evaluator_inputs": {},
  "split": "benchmark"
}
```

The canonical generated filename is exactly `<prompt_id>.mp4`. T2AV files go
under `submissions/<MODEL>/t2av/`; the public I2AV-LAT files go under
`submissions/<MODEL>/i2av_lat/`. IDs are case-sensitive and must not be
re-indexed or decorated with a model prefix. `export-generation-requests`
includes the exact `output_filename` and `output_relpath` for every row.

The release profile is:

| pool | count | evaluator membership |
|---|---:|---|
| Receiver Distance | 90 | Range Attenuation |
| Receiver Observer | 90 | Approach Gain and Lateral Stability |
| Motion--Loudness | 90 | Motion--Loudness |
| Impact/Source Generation | 90 | Impact Decay |
| Causality | 79 | Causality Violation |
| RT60 | 166 | RT60 Consistency |
| I2AV LAT | 143 | Log Attack Time |

I2AV-LAT rows require a conditioning asset for generation and an evaluator-only
reference-audio path. Receiver rows require evaluator-only detection targets.
`evaluator_inputs` is retained by the evaluator but removed from exported model
generation requests.

I2AV conditioning images are stored under `data/i2av_conditioning_assets/`.
They can be supplied as a prebuilt asset package or recreated from decoded
frame index 1 of the selected Greatest Hits videos. Reference audio used only
by the LAT evaluator is stored under `data/references/i2av_lat/`; both locations
are already encoded in `i2av_lat_143.jsonl`.

`data/references/i2av_lat_sources.jsonl` maps every public prompt ID to the
corresponding Greatest Hits sample ID and selected event window. It contains no
private filesystem paths. Recover the evaluator-only audio from the official
dataset with:

```bash
python scripts/download_greatest_hits_subset.py
python scripts/prepare_i2av_lat_references.py \
  --create-missing-conditioning-images
```

The second command creates any missing conditioning PNG from decoded frame
index 1 (the second frame), verifies that frame, and extracts the first five
seconds of reference audio.

The manifests are converted without resampling from the frozen current-605 T2AV
selection table and the official 143-row I2AV-LAT generation manifest. The
maintainer-side conversion is reproducible with
`scripts/build_public_release_manifests.py`.

Validate the pair together:

```bash
acoustitrace validate-release-suite \
  data/prompts/t2av_605.jsonl data/prompts/i2av_lat_143.jsonl
```

The original paper distribution remains documented as an audit artifact; it is
not the default public runner profile.
