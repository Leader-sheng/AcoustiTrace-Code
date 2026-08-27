# Time and causality backend

This package filters audio and visual detections at confidence 0.20, clusters
them with 0.20 s and 0.30 s gaps, and performs normal-mode association in the
[-0.08, 0.70] s delay interval. A delay up to 0.25 s is marked synchronous for
association diagnostics.

Association tolerance and scoring are deliberately separate: every matched
event with `audio_onset - visual_time < -0.001` s is a causality violation. The
1 ms margin is the minimum timestamp increment used to implement a strict
temporal-precedence test; it does not imply 1 ms event-localization accuracy.
The sample score is 100 times one minus the violation fraction. The strict
and relaxed modes are retained only as diagnostics; paper results use normal
mode.
