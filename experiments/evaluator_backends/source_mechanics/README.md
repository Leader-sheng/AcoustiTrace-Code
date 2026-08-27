# Source-mechanics backend

The pipeline parses OV-AVEL visual events and FlexSED audio events, associates
events within 0.35 s while enforcing a 0.12 s event gap, and extracts 22,050 Hz
audio windows from -0.05 to +0.70 s around each event. Impact Decay fits the
post-peak interval from +0.02 to +0.50 s and emits the continuous decay-shape
readout used by the public score API.

Motion--Loudness splits localized events into two time-contiguous clusters and
uses Qwen3-VL to judge the visually stronger cluster from motion alone. For a
valid Qwen judgment that does not agree with the audio ordering, script `11`
applies the 8 fps frame-difference fallback. The final no-margin score is the
percentage of valid visual judgments whose selected cluster has the higher
maximum event RMS level.

OV-AVEL, FlexSED, Qwen3-VL, and their weights are external dependencies.
