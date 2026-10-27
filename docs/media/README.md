# BEHAVIOR Point-Cloud Excerpts

These three videos visualize recorded scene geometry from existing
NVIDIA PointWorld-BEHAVIOR annotations, derived from BEHAVIOR.
They are **ground-truth data replays, not CIRA predictions**.

| Video | Visible Interaction | Independent Window |
| --- | --- | --- |
| `behavior-radio.mp4` | Tabletop radio | `task-0000_episode_00000410-235:246` |
| `behavior-kitchen.mp4` | Kitchen toaster | `task-0008_episode_00080460-580:591` |
| `behavior-toolbox.mp4` | Tool handling | `task-0019_episode_00192350-735:746` |

Each video contains 11 recorded states, displayed at 2.5 states per playback
second. Original timestamps/fps are not certified, so this is not a claim about
physical prediction horizons or a real-time slow-motion ratio. No interpolation
or invented future motion is used. Looping restarts the same independent clip.

Mesh-local colored points are transformed with recorded mesh poses.
Head and two wrist observations are concatenated. A fixed orthographic camera
and local crop expose the geometry; missing/occluded surfaces are not completed.
The figures do not represent complete scenes, continuous long rollouts, or
successful execution of entire tasks. The three previously inspected validation
windows were selected for illustration, not as performance evidence.

No training, model inference, simulator execution or GPU work was added.
`SOURCE_MANIFEST.json` records source member hashes, video identities and checks.
No raw point arrays, dataset archives or pretrained weights are distributed.

Source dataset: [NVIDIA PointWorld-BEHAVIOR](https://huggingface.co/datasets/nvidia/PointWorld-BEHAVIOR).
Original project: [BEHAVIOR](https://behavior.stanford.edu/).
Code: [NVIDIA PointWorld](https://github.com/NVlabs/PointWorld).
The source data and rendered excerpts retain applicable third-party terms;
the repository code license is not a license to the source dataset or its assets.
