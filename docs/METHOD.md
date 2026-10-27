# Method And Evaluation Contracts

## System Boundary

CIRA augments the full PointWorld dynamics predictor. It does not replace the
scene encoder with a rigid-object estimator. The inherited network encodes
initial scene points and a provided future robot trajectory. Predictions remain
full-scene point flows, including static background and potentially nonrigid motion.

The implementation resides in `cira_pointworld/adapter.py`; common feature and
robot-role contracts come from `pointworld_interaction/adapter.py`.

## Relational Feedback

For each scene point and future prediction step, the branch selects local robot
neighbors separately for each arm. Geometric offsets, learned point features and
robot palm/finger roles form relation features. Gated messages are pooled with
mean/max statistics, then update a recurrent per-point state.

The branch reads the model's current predicted scene geometry for the following
query. It does not receive future scene ground-truth positions as model input.
The original heads predict displacement relative to the initial scene position;
the branch must not accidentally accumulate those absolute-from-initial
displacements as incremental velocity.

A zero-initialized projection preserves the base model's initial prediction.
This is an initialization property, not a guarantee after training. Tests cover
constructed examples of padding, source partitions, gradient flow and feedback.
Mechanism tests use a substituted small PTv3 implementation and are not
full pretrained-model accuracy measurements.

## Controls

`baseline` leaves the original prediction path unchanged.
`cira_no_feedback` retains the relation module and capacity but removes the
receiver-geometry feedback path. `cira` enables that path.
Known future robot motion remains available; no claim of strict temporal
causality or intervention identification follows from these controls.

## Persistent Interaction Constraints

The persistent branch lives in `constraint_transport.py` and its bridge.
It carries interaction references, estimates constraint strength/release and
combines geometric references with neural prediction. Candidate integration
uses an FP32 geometry solve where required.

The code does not initialize arbitrary far-away background points as interacting
references solely to make them move with the robot. Nevertheless, background
drift remains empirically unresolved. Persistence can improve moving-point
prediction while worsening static points, so dynamic EPE alone is insufficient.

Action-effect transport is included as a distinct comparison, not as a
demonstrated improvement. Its negative common-camera results are retained.

## Data And Metrics

The recorded eight-task split contains 179,578 training, 23,605 validation and
23,068 test windows. Recorded episodes are 1,211 / 156 / 156. These counts
describe the audited research split, not assets distributed in this repository.

EPE is computed using original point correspondences and existence/motion masks.
Published candidate summaries first average windows within episode, episodes
within task and tasks equally, then average two training seeds. Values in the
main README are millimeters; internal inherited metrics often use meters.
Rows without a finite value in a stratum are excluded, not filled with zero.

The common-camera preprocessing retry refuses camera crops that remove all
scene points. It does not fabricate points or silently substitute different
samples per model. Full reproduction requires the original audited manifests.

Candidate checkpoint selection minimizes half dynamic plus half static
task/episode-macro validation EPE, with earliest-tie selection. Older baseline
and CIRA closed-loop weights used different selection rules; their comparison
does not isolate a module effect under identical model selection.

## Closed-Loop Protocol

The long-bar pilot refreshes RGB-D and forward-kinematic robot observations,
scores 64 fixed candidate plans with a world model, executes a short selected
prefix and repeats. It is finite-candidate MPC, not MPPI, reinforcement learning
or a hardware controller.

Physical success requires simultaneous goal, settling and safety checks. The
long-bar goal contract uses 10 mm position tolerance, 5-degree yaw tolerance
with pi-periodic bar orientation, tilt limits and settling-speed checks.
Visible object masks are privileged simulator instance masks. Three shared
physical initial states remain only three independent scenes regardless of
the number of model/seed attempts.

The newer long-bar/drawer campaign additionally preserves full state contracts.
Only the two arm-configuration URDF path fields may differ by installation
prefix after content identity is checked. Numeric robot/object/controller state
is not generally normalized or weakened.

## Not Claimed

There is no claim of broad generalization, statistically established superiority,
calibrated safety, causal identification, complete test independence, real-robot
deployment or a completed larger campaign in this source snapshot.
Public release and CPU tests must not be interpreted as those validations.
