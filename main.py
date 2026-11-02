import torch
import cv2
import threading
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.float32, device_map={"": "cpu"})

model.eval()

OBJECT_NAME: str = "cardboard box"

def check_frame(frame_bgr: "cv2.Mat") -> bool:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    res = model.detect(image, OBJECT_NAME)

    return len(res["objects"]) > 0

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

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if check_frame(frame):
                print(f"{OBJECT_NAME.upper()} DETECTED")

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
