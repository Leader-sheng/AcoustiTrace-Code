# Release preparation status

## Included

- the public 605-prompt T2AV and 143-prompt I2AV Log Attack Time manifests;
- eight paper-aligned evaluator mappings, validity gates, scoring, aggregation,
  bootstrap confidence intervals, and matched-valid comparison;
- raw-media Receiver Observer, Source Mechanics, Causality, Log Attack Time,
  and visual RT60 adapters;
- reproducible third-party checkout metadata and environment preflight checks;
- RT60 inference runtime, with the released evaluator weights linked from the
  main README;
- canonical submission naming and output-manifest validation;
- all 143 I2AV-LAT conditioning PNGs and five-second reference WAVs;
- regression, release-contract, and repository-hygiene tests; and
- citation metadata; and
- Apache License 2.0 coverage for the source code.

I2AV Log Attack Time assets are bundled so users do not need to download
Greatest Hits during normal setup. The released source map and reconstruction
scripts remain available for provenance and recovery. Attribution, release
transformations, and the assets' CC BY 4.0 terms are recorded in
`data/ASSET_NOTICE.md`.

## Validation

- both release manifests pass the exact 605+143 contract validator;
- the Python regression suite passes 48 tests;
- all five native backend groups completed a joint real-media GPU run with ten
  assignments per dimension: 80 score rows, 64 valid rows, 16 protocol-defined
  applicability failures, zero backend errors, and overall status `success`.

Evaluator-invalid rows caused by documented applicability gates are retained as
coverage outcomes and are not backend failures or zero scores.

## External runtime assets

The repository intentionally does not vendor the Qwen3-VL base model,
AcoustiTrace RT60 evaluator weights, third-party evaluator checkpoints, generated
submissions, or Greatest Hits source media. Their public locations and expected
paths are documented in `README.md`, `models/README.md`, and
`docs/DEVELOPMENT_MACHINE.md`.

## Licensing

The source code is released under Apache License 2.0. Model weights retain the
separate license shown on their Hugging Face repository. Bundled benchmark
assets and third-party dependencies retain their documented upstream terms.
