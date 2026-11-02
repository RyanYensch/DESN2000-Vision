import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32, device_map={"": "cpu"})

model.eval()

OBJECT_NAME: str = "cardboard box"

def check_frame(path: str) -> bool:
    image = Image.open(path)
    encoded = model.encode_image(image)

    res = model.answer_question(encoded, f"Is there anything other than a {OBJECT_NAME} in the image?", tokeniser)

    return res.strip().lower().startswith("no")

def main():
    while True:
        triggered = check_frame("frame.jpg")
        if triggered:
            print(f"{OBJECT_NAME.upper()} DETECTED")

        time.sleep(0.5)


if __name__ == "__main__":
    main()
