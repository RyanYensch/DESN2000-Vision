import torch
import cv2
import threading
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.float32, device_map={"": "cpu"})

model.eval()

OBJECT_NAME: str = "smartphone"

latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

def bg_loop():
    while not stop_event.is_set():
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = model.detect(image, OBJECT_NAME)

        n = len(res["objects"]) 
        if n > 0:
            print(f"{OBJECT_NAME.upper()} DETECTED ({n} objects)")
        else:
            print(".", end="", flush=True)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")


    t = threading.Thread(target=bg_loop, daemon=True)
    t.start()

    print(f"Watching for {OBJECT_NAME}...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            with frame_lock:
                global latest_frame
                latest_frame = frame

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
