# Reproducibility Boundary

## Included

- all paper-defined score mappings and validity-aware aggregation;
- raw-media backend source for the eight evaluator dimensions;
- receiver tracking and applicability gates for Range, Approach, and Lateral;
- event parsing, matching, Motion--Loudness, Impact, LAT, and Causality logic;
- final Sabine-guided Qwen3-VL visual RT60 inference runtime;
- cell-wise bootstrap intervals and matched-valid aggregation;
- the portable differentiable range-guidance objective; and
- machine checks for prompt counts, model support, dataset splits, key
  evaluator thresholds, and submission anonymization.

## External Inputs

Running the raw-media pipelines requires separately obtained OV-AVEL, FlexSED,
Qwen3-VL, Grounded-SAM, and Video-Depth-Anything installations and weights.
Large source media, generated model outputs, and provider-managed proprietary
APIs are not bundled with the code repository. The released visual RT60
checkpoint is hosted separately at
`https://huggingface.co/Missouter/AcoustiTrace-RT60`. Full leaderboard
regeneration additionally requires the corresponding generated media and any
third-party inputs documented in the installation guide.

The first public release does not include internal data-collection, controlled
validation, BRAS/STARSS case-study, or model-training workspaces. RT60 LoRA and
physics-head fine-tuning code is listed on the public roadmap for a later
release.

## Expected Score CSV

Aggregation input uses one row per attempted model-prompt-evaluator result:

```text
sample_id,model,task,evaluator,valid,score
```

Invalid rows have `valid=false` and an empty `score`. They remain in the
validity denominator and are excluded from the conditional mean.

## Range Attenuation

The receiver backend resolves the sign of the relative VDA depth coordinate
and searches propagation exponents over 0.40 s windows with a 0.05 s stride.
The reported Range score is the best finite local-window R2 for the sample.
