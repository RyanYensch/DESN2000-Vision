import torch
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

tokeniser = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32, device_map={"": "cpu"})

model.eval()

OBJECT_NAME: str = "cardboard box"

def check_frame(frame_bgr: "cv2.Mat") -> bool:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    encoded = model.encode_image(image)

    res = model.answer_question(encoded, f"Is there a {OBJECT_NAME} in the image?", tokeniser)

    return res.strip().lower().startswith("yes")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print(f"Watching for {OBJECT_NAME}...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            if check_frame(frame):
                print(f"{OBJECT_NAME.upper()} DETECTED")

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
