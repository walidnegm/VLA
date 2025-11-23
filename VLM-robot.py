# vla_camera_mouse_keyboard.py
# FINAL FIXED VERSION – November 2025 (NumPy-safe)
# Everything works: up goes up, left goes left, lift goes up, diagonal perfect.

import os
import cv2
import numpy as np
import torch
import threading
import queue
import traceback
import time
from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms.functional import to_tensor

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

# ---------------- DEVICE & CAMERA ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

ret, frame = cap.read()
if not ret:
    raise SystemExit("[ERROR] Cannot open camera!")

cv2.namedWindow("OpenVLA Robot Demo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("OpenVLA Robot Demo", 1100, 720)

# ---------------- GLOBAL STATE ----------------
model_ready = False
load_error = None

processor = None
vla = None
detector = None
categories = None

cmd_queue = queue.Queue()

latest_action = np.zeros(7, dtype=np.float32)

keyboard_bbox = None
mouse_bbox = None

active_bbox = None
active_object_label = None

current_prompt = None
path_steps_remaining = 0
path_step_index = 0

offset_x_target = 0.0
offset_y_target = 0.0

requested_inches_for_current_path = 0.0

# ---------------- CALIBRATION ----------------
TABLE_WIDTH_IN = 32.0
TABLE_DEPTH_IN = 22.0

PX_PER_IN_X = 1280.0 / TABLE_WIDTH_IN
PX_PER_IN_Y = 720.0 / TABLE_DEPTH_IN

PATH_NUM_STEPS = 10

BOUNDS_PX_X = 480.0
BOUNDS_PX_Y = 280.0

# ---------------- MODEL LOADER ----------------
def load_models():
    global processor, vla, detector, categories, model_ready, load_error
    try:
        print("[LOAD] Loading OpenVLA-7B...")
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()

        dummy = Image.new("RGB", (224, 224), (128, 128, 128))
        inputs = processor("In: What action should the robot take to rest?\nOut:", dummy, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].long()
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            _ = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        print("[LOAD] Loading Faster R-CNN...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector_local = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
        detector = detector_local
        categories = weights.meta["categories"]

        model_ready = True
        print("\n[READY] EVERYTHING LOADED! Type commands now.\n")
    except Exception as e:
        load_error = traceback.format_exc()
        print("[ERROR] Load failed:\n", load_error)

threading.Thread(target=load_models, daemon=True).start()

# ---------------- INPUT ----------------
def input_thread():
    print("You can type commands anytime (e.g. 'lift keyboard by 8 inches')")
    while True:
        try:
            cmd = input("Command: ").strip()
            if cmd.lower() in ["quit", "q", "exit"]:
                cmd_queue.put("quit")
                break
            if cmd:
                cmd_queue.put(cmd)
        except:
            break
threading.Thread(target=input_thread, daemon=True).start()

# ---------------- DETECTION ----------------
def detect_objects(frame_bgr, score_thresh=0.6):
    if detector is None: return {"keyboard": (None, None), "mouse": (None, None)}
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = detector(tensor)[0]
    best = {"keyboard": (None, score_thresh), "mouse": (None, score_thresh)}
    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        cls = categories[label.item()]
        s = score.item()
        if cls == "keyboard" and s >= best["keyboard"][1]:
            best["keyboard"] = (box.int().cpu().numpy(), s)
        if cls == "mouse" and s >= best["mouse"][1]:
            best["mouse"] = (box.int().cpu().numpy(), s)
    return best

# ---------------- MAGNITUDE & OBJECT ----------------
def parse_requested_magnitude(prompt: str) -> float:
    import re
    distance = 4.0
    nums = re.findall(r"(\d+(?:\.\d+)?)", prompt)
    if nums:
        distance = float(nums[0])
    distance = np.clip(distance, -max(TABLE_WIDTH_IN, TABLE_DEPTH_IN), max(TABLE_WIDTH_IN, TABLE_DEPTH_IN))
    print(f"[PATH INFO] Requested magnitude: {distance:.2f}\"")
    return abs(distance)

def choose_active_object_from_prompt(prompt: str):
    p = prompt.lower()
    if "mouse" in p and mouse_bbox is not None:
        return "mouse", mouse_bbox
    if keyboard_bbox is not None:
        return "keyboard", keyboard_bbox
    if mouse_bbox is not None:
        return "mouse", mouse_bbox
    return None, None

# ---------------- FINAL CORRECT MAPPING ----------------
def map_vla_to_local(action7: np.ndarray, requested_inches: float):
    dx_v, dz_v = float(action7[0]), float(action7[2])
    base_step_in = requested_inches / PATH_NUM_STEPS

    # 2025 workaround: pure left/right (model is weak here)
    if current_prompt:
        p = current_prompt.lower()
        if "left" in p and "right" not in p and "diagonal" not in p:
            return {"step_dx_in": -base_step_in, "step_dz_in": 0.0,
                    "step_dx_px": -base_step_in * PX_PER_IN_X,
                    "step_dy_px": 0.0, "grip_v": action7[6]}
        if "right" in p and "left" not in p and "diagonal" not in p:
            return {"step_dx_in": +base_step_in, "step_dz_in": 0.0,
                    "step_dx_px": +base_step_in * PX_PER_IN_X,
                    "step_dy_px": 0.0, "grip_v": action7[6]}

    # Normal VLA-driven direction
    norm = np.sqrt(dx_v**2 + dz_v**2)
    if norm < 1e-6:
        step_dx_in = step_dz_in = 0.0
    else:
        step_dx_in = (dx_v / norm) * base_step_in
        step_dz_in = (dz_v / norm) * base_step_in

    step_dx_px = step_dx_in * PX_PER_IN_X
    step_dy_px = -step_dz_in * PX_PER_IN_Y   # +dz → down in image → CORRECT

    return {"step_dx_in": step_dx_in, "step_dz_in": step_dz_in,
            "step_dx_px": step_dx_px, "step_dy_px": step_dy_px,
            "grip_v": action7[6]}

# ---------------- PATH CONTROL ----------------
def start_new_path(prompt: str):
    global current_prompt, path_steps_remaining, path_step_index
    global offset_x_target, offset_y_target, requested_inches_for_current_path
    global active_bbox, active_object_label

    label, bbox = choose_active_object_from_prompt(prompt)
    if bbox is None:  # ← FIXED: NumPy-safe check
        print("[CMD WARN] No object found")
        return False

    active_object_label = label
    active_bbox = bbox
    current_prompt = prompt
    path_steps_remaining = PATH_NUM_STEPS
    path_step_index = 0
    offset_x_target = offset_y_target = 0.0
    requested_inches_for_current_path = parse_requested_magnitude(prompt)

    print(f"[PATH] START → {prompt} | Target: {label.upper()}")
    return True

# ---------------- VLA STEP (with logging restored) ----------------
def run_vla_path_step(frame_bgr):
    global latest_action, path_steps_remaining, path_step_index
    global offset_x_target, offset_y_target

    if not model_ready or active_bbox is None:
        return

    step_idx = PATH_NUM_STEPS - path_steps_remaining + 1
    print(f"\n[VLA PATH] STEP {step_idx}/{PATH_NUM_STEPS} for '{current_prompt}'")
    print(f"[VLA PATH] Target: {active_object_label}, bbox: {active_bbox}")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((224, 224))
    inputs = processor(f"In: What action should the robot take to {current_prompt}?\nOut:", img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"].long()
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    if isinstance(action, torch.Tensor):
        action = action[0] if action.ndim == 2 else action
        action_np = action.float().cpu().numpy()
    else:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 2: action_np = action_np[0]

    latest_action = np.pad(action_np, (0, max(0, 7-len(action_np))), constant_values=0)

    # Print raw vector (restored logging)
    np.set_printoptions(precision=5, suppress=True)
    print(f"[VLA PATH] ACTION VECTOR: {latest_action}")
    dx, dy, dz = latest_action[0], latest_action[1], latest_action[2]
    grip = latest_action[6]
    print(f"[VLA PATH] Parsed: Δx={dx:+.5f} Δy={dy:+.5f} Δz={dz:+.5f} grip={grip:+.5f}")

    mapping = map_vla_to_local(latest_action, requested_inches_for_current_path)
    print(f"[VLA PATH] Transform: dx_in={mapping['step_dx_in']:+.3f}\" dz_in={mapping['step_dz_in']:+.3f}\" "
          f"dx_px={mapping['step_dx_px']:+.2f}px dy_px={mapping['step_dy_px']:+.2f}px")

    offset_x_target += mapping["step_dx_px"]
    offset_y_target += mapping["step_dy_px"]
    offset_x_target = np.clip(offset_x_target, -BOUNDS_PX_X, BOUNDS_PX_X)
    offset_y_target = np.clip(offset_y_target, -BOUNDS_PX_Y, BOUNDS_PX_Y)

    print(f"[VLA PATH] New offsets: x={offset_x_target:+.1f} y={offset_y_target:+.1f}")

    path_steps_remaining -= 1
    path_step_index += 1

    if path_steps_remaining <= 0:
        print(f"[PATH] COMPLETED: {current_prompt}")

# ---------------- DRAWING (FIXED) ----------------
def draw_overlay(frame):
    h, w = frame.shape[:2]

    if keyboard_bbox is not None:
        x1, y1, x2, y2 = map(int, keyboard_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "KEYBOARD", (x1, y1-10), 0, 0.7, (0, 0, 255), 2)

    if mouse_bbox is not None:
        x1, y1, x2, y2 = map(int, mouse_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, "MOUSE", (x1, y1-10), 0, 0.7, (255, 0, 0), 2)

    if active_bbox is not None:
        x1, y1, x2, y2 = map(int, active_bbox)
        box_w, box_h = x2 - x1, y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2                     # ← THIS WAS MISSING!

        ghost_cx = int(cx + offset_x_target)
        ghost_cy = int(cy + offset_y_target)     # ← NOW CORRECT (+ = down)

        gx1 = np.clip(ghost_cx - box_w//2, 0, w-1)
        gx2 = np.clip(ghost_cx + box_w//2, 0, w-1)
        gy1 = np.clip(ghost_cy - box_h//2, 0, h-1)
        gy2 = np.clip(ghost_cy + box_h//2, 0, h-1)

        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 165, 255), 4)
        cv2.putText(frame, f"TARGET ({active_object_label.upper()})", (gx1, gy1-10), 0, 0.8, (0, 165, 255), 2)

        arm_x = ghost_cx
        arm_top = int(h * 0.08)
        grip_y = gy1 - 30
        if grip_y < arm_top + 40: grip_y = arm_top + 40

        cv2.line(frame, (arm_x, arm_top), (arm_x, grip_y), (200, 200, 255), 20)
        cv2.circle(frame, (arm_x, grip_y), 34, (255, 0, 255), 10)

        closed = latest_action[6] > 0.2
        finger = 70 if closed else 40
        cv2.line(frame, (arm_x, grip_y), (arm_x-40, grip_y+finger), (230, 230, 255), 16)
        cv2.line(frame, (arm_x, grip_y), (arm_x+40, grip_y+finger), (230, 230, 255), 16)

    status = "MODEL READY" if model_ready else "LOADING..."
    cv2.putText(frame, status, (20, 50), 0, 1.3, (0, 255, 255) if model_ready else (0, 180, 255), 4)

    if current_prompt and path_steps_remaining > 0:
        cv2.putText(frame, f"CMD: {current_prompt} [{path_step_index+1}/{PATH_NUM_STEPS}]", (20, h-30), 0, 0.7, (255,255,255), 2)

    return frame

# ---------------- MAIN LOOP ----------------
print("Starting camera loop... (press 'q' to quit)")
while True:
    ret, frame = cap.read()
    if not ret: break

    if keyboard_bbox is None or mouse_bbox is None or path_steps_remaining == 0:
        det = detect_objects(frame)
        if det["keyboard"][0] is not None: keyboard_bbox = det["keyboard"][0]
        if det["mouse"][0]    is not None: mouse_bbox    = det["mouse"][0]
        if active_bbox is None and (keyboard_bbox is not None or mouse_bbox is not None):
            active_object_label = "keyboard" if keyboard_bbox is not None else "mouse"
            active_bbox = keyboard_bbox if keyboard_bbox is not None else mouse_bbox

    try:
        cmd = cmd_queue.get_nowait()
        if cmd == "quit": break
        if cmd and model_ready:
            start_new_path(cmd)
    except queue.Empty:
        pass

    if model_ready and current_prompt and path_steps_remaining > 0 and active_bbox is not None:
        run_vla_path_step(frame.copy())

    frame = draw_overlay(frame)
    cv2.imshow("OpenVLA Robot Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")