from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    print("Device:", device)

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )

    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation=None,         # no flash-attn dependency
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,           # ok, accelerate is installed
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # Dummy gray image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    instruction = "move straight up"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    with torch.no_grad():
        inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
        action = vla.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False,
        )

    if isinstance(action, torch.Tensor) and action.ndim == 2:
        action = action[0]

    print("Predicted action shape:", tuple(action.shape))
    print("First few values:", action[:7].tolist())


if __name__ == "__main__":
    main()
