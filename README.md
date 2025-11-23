# What a 7-Billion-Parameter Vision-Language-Action Model Actually Does  
### When You Point a Webcam at Your Desk

A weekend-built playground demo that lets you with a webcam and a decent GPU watch the open-source **OpenVLA-7B** model think out loud — step by step — in real time.

![demo](https://github.com/yourusername/openvla-live-demo/assets/xxx/demo.gif)

### The point of this demo

A  way to understand what they are (and what they still aren’t) is to run one live and watch the raw 7-DoF numbers stream out while it tries to obey you.

This is that demo.

### What you are looking at

- Live 1280×720 webcam feed  
- Full **OpenVLA-7B** (`openvla/openvla-7b`) in bfloat16 on a single RTX 4090  
- ~1–2 Hz inference (0.5–1 s per step)  
- True closed loop: every step uses the latest camera frame + the original prompt  
- 10-step rollouts per command (~8–10 s)  
- COCO Faster R-CNN used only for visualization and object selection (keyboard / mouse)  
- Full 7-DoF action vector printed every step  
- Ghost bounding box + hanging arm move 100 % according to OpenVLA’s (Δx, Δz), scaled to user-requested inches  

### OpenVLA-7B (June 2024)

- Released by Toyota Research Institute, UC Berkeley, Stanford, Google DeepMind  
- Paper: https://arxiv.org/abs/2406.09246  
- Project: https://openvla.github.io  
- Trained on ~970k real robot episodes (mostly BridgeData V2 + RT-1/RT-2 style)  
- Action convention (Bridge/RT-1):
dz > 0  → move end-effector down toward table
dz < 0  → lift up
dx > 0  → right
dx < 0  → left
gripper > 0 → close

### Why the external Faster R-CNN exists

OpenVLA itself never tells you which object it’s acting on.  
Without the red/blue boxes you literally cannot tell whether “move the mouse” moved the mouse or the keyboard.

The detector is pure interpretability — it has zero influence on the policy.

When the ghost box + arm lock onto the correct object → grounding worked.  
When they drift or pick the wrong one → grounding broke. Instant, unambiguous feedback.

### Commands to try

lift the keyboard by 8 inches
move the mouse left by 10 inches
push the keyboard forward 6 inches
slide the mouse diagonally up and right by 12 inches
pick up the mouse and lift 10 inches
raise the keyboard slowly by 15 inches

### Current limitations (late 2025 )

- ~800–1500 ms per forward pass → nowhere near real-time control  
- Flat-table assumption + manual calibration  
- **Pure left/right motion is so weak in OpenVLA-7B that a temporary hard-coded override forces correct lateral motion when the words “left” or “right” appear in the prompt**  
- Long or ambiguous prompts produce visible drift  
- No depth, no collision checking, no safety layer

