# Receiver/observer backend

The receiver pipeline runs scripts `01`--`05` to produce VDA depth,
Grounded-SAM source masks, audio-level curves, relative source trajectories,
and base readouts. Script `07` computes the Unified v2 local-window scores;
script `08` then applies the track- and task-specific visual validity gates.
Script `10` computes the sign-aware Range Attenuation score from the same
aligned audio and trajectory caches.

Grounded-SAM runs as a persistent in-process runtime: GroundingDINO and SAM are
loaded once and reused across all sampled keyframes. Set
`gsam.persistent_runtime: false` only when diagnosing compatibility with the
upstream one-image command-line demo.

The configuration uses 0.40 s windows with a 0.05 s stride and 0.35 s depth
smoothing. Range first resolves the sign of the relative VDA depth coordinate,
then searches the propagation exponent and local window that best explain the
observed SPL change. Approach and Lateral use their Unified v2 [0,1] local-
window readouts before the final 0--100 mapping. The VDA coordinate is relative
and is not presented as metric distance.

Install Video-Depth-Anything and Grounded-SAM under `third_party/` or edit the
two repository paths in `configs/receiver_observer_eval_config.yaml`.
