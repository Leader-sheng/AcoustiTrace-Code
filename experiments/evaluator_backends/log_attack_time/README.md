# Log Attack Time backend

This directory contains the I2AV-only LAT measurement described in the
supplement. It uses 22,050 Hz audio, 1,024-sample RMS frames, a 128-sample hop,
Gaussian smoothing with sigma 1, a [-0.05, 0.30] s attack search interval, and
a 0.05 s pre-onset baseline. The score is
`100 * exp(-abs(log(T_generated) - log(T_reference)) / 0.35)`.

The input CSV requires `sample_id`, `generated_path`, and `reference_path`.
Optional `generated_onset_sec` and `reference_onset_sec` fields bypass automatic
onset localization. The reference audio is used only by the evaluator and is
not provided to the generator.
